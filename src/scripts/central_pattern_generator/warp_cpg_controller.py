"""
GPU-Accelerated Hexapod Central Pattern Generator Controller using NVIDIA Warp. Synthesizes:
 [1] Zhang et al., "Central Pattern Generators for Locomotion Control in Hexapod Robot Legs", CCDC 2022 
 [2] Zhong et al., "Locomotion Control and Gait Planning of a Novel Hexapod Robot Using Biomimetic Neurons", IEEE TCST 2018
 [3] Yin et al., "Energy Efficiency Optimization of Hexapod Robots Based on Central Pattern Generator Control", ISRIMT 2023
"""

import numpy as np
import torch
import warp as wp
from silver2_isaac_constants import GAIT_CONFIGS
from warp_cpg_kernels import cpg_hopf_substep_kernel, cpg_kinematics_and_ik_kernel

class WarpHexapodCPGController:
    def __init__(
        self,
        leg_mounts: dict,
        link_lengths: tuple,
        canonical_to_newton_indices: np.ndarray,
        dt: float = 0.0025,
        total_period: float = 2.0,
        gait: str = "tripod",
        num_envs: int = 1,
        device: str = "cuda:0"
    ):
        self.device = device
        self.dt = dt
        self.total_period = total_period
        self.num_envs = num_envs
        self.num_legs = 6
        self.L1, self.L2, self.L3 = [float(l) for l in link_lengths]

        # Oscillator dynamics parameters [3]
        self.alpha = 1.0
        self.mu = 100.0
        self.radius = float(np.sqrt(self.mu))
        self.b = 2.0
        self.coupling_strength = 0.4
        self.coupling_weight = self.coupling_strength / (self.num_legs - 1)

        # Trajectory mapping gains
        self.k1 = 1.0
        self.k2 = 0.0
        self.k3 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.l1 = -0.01
        self.l2 = 0.005

        # Sub-stepping
        self.sub_steps = 5
        self.dt_sub = self.dt / self.sub_steps

        # ----------------------------------------------------------------------
        # Precompute Static Mount Offsets and Inverse Yaw Transforms (Eq. 5 [1])
        # ----------------------------------------------------------------------
        self.leg_names = list(leg_mounts.keys())
        mount_pos_np = np.array([leg_mounts[k]['pos'] for k in self.leg_names], dtype=np.float32)
        yaws_np = np.array([leg_mounts[k]['yaw'] for k in self.leg_names], dtype=np.float32)
        c_y, s_y = np.cos(yaws_np), np.sin(yaws_np)

        rot_z_inv_np = np.zeros((self.num_legs, 3, 3), dtype=np.float32)
        rot_z_inv_np[:, 0, 0] = c_y
        rot_z_inv_np[:, 0, 1] = s_y
        rot_z_inv_np[:, 1, 0] = -s_y
        rot_z_inv_np[:, 1, 1] = c_y
        rot_z_inv_np[:, 2, 2] = 1.0

        # Upload static kinematic tables to GPU
        self.mount_positions = wp.array(mount_pos_np, dtype=wp.float32, device=self.device)
        self.rot_z_inv = wp.array(rot_z_inv_np, dtype=wp.float32, device=self.device)
        self.canonical_to_newton = wp.array(
            canonical_to_newton_indices.astype(np.int32), 
            dtype=wp.int32, 
            device=self.device
        )

        # ----------------------------------------------------------------------
        # Preallocate Device Buffers (Zero Allocation During Loop)
        # ----------------------------------------------------------------------
        # Ping-pong buffers for Hopf Euler sub-steps
        self.x_buf_0 = wp.zeros((self.num_envs, self.num_legs), dtype=wp.float32, device=self.device)
        self.y_buf_0 = wp.zeros((self.num_envs, self.num_legs), dtype=wp.float32, device=self.device)
        self.x_buf_1 = wp.zeros((self.num_envs, self.num_legs), dtype=wp.float32, device=self.device)
        self.y_buf_1 = wp.zeros((self.num_envs, self.num_legs), dtype=wp.float32, device=self.device)

        # Joint targets buffer (18 DOFs per environment, pre-remapped for Newton)
        self.wp_joint_targets = wp.zeros((self.num_envs, 18), dtype=wp.float32, device=self.device)
        
        # Zero-copy DLPack PyTorch view passed directly to ArticulationView
        self.torch_joint_targets = wp.to_torch(self.wp_joint_targets)

        # Coupling matrix buffers
        self.cos_diff = wp.zeros((self.num_legs, self.num_legs), dtype=wp.float32, device=self.device)
        self.sin_diff = wp.zeros((self.num_legs, self.num_legs), dtype=wp.float32, device=self.device)

        # Initialize gait and seed limit cycles on GPU
        self.set_gait(gait, reset_state=True)

    def set_gait(self, gait_name: str, reset_state: bool = False):
        """Switches gait, duty factor, and uploads phase coupling matrices to GPU."""
        if gait_name not in GAIT_CONFIGS:
            raise ValueError(f"Unknown gait '{gait_name}'. Choose from: {list(GAIT_CONFIGS.keys())}")

        cfg = GAIT_CONFIGS[gait_name]
        self.active_gait = gait_name
        self.epsilon = float(cfg["epsilon"])
        self.phi_phase = np.array(cfg["phases"], dtype=np.float32)

        # Stance and swing phase speeds (Eq. 2 [3])
        self.omega_swing = float(np.pi / ((1.0 - self.epsilon) * self.total_period))
        self.omega_stance = float(np.pi / (self.epsilon * self.total_period))
        self.delta_omega = self.omega_stance - self.omega_swing

        # Coupling matrix: phi_ij = 2 * pi * (phi_i - phi_j)
        phase_diff = 2.0 * np.pi * (self.phi_phase[:, None] - self.phi_phase[None, :])
        cos_diff_np = np.cos(phase_diff).astype(np.float32)
        sin_diff_np = np.sin(phase_diff).astype(np.float32)

        # In-place upload to pinned GPU buffers
        wp.copy(self.cos_diff, wp.array(cos_diff_np, dtype=wp.float32, device=self.device))
        wp.copy(self.sin_diff, wp.array(sin_diff_np, dtype=wp.float32, device=self.device))

        if reset_state:
            theta = 2.0 * np.pi * self.phi_phase
            x_init = (self.radius * np.cos(theta)).astype(np.float32)
            y_init = (self.radius * np.sin(theta)).astype(np.float32)

            # Broadcast across all environments
            x_init_batch = np.tile(x_init, (self.num_envs, 1))
            y_init_batch = np.tile(y_init, (self.num_envs, 1))

            wp.copy(self.x_buf_0, wp.array(x_init_batch, dtype=wp.float32, device=self.device))
            wp.copy(self.y_buf_0, wp.array(y_init_batch, dtype=wp.float32, device=self.device))

    def step(
        self,
        default_feet_pos_body: wp.array2d,
        dir_angle_rad: float = 0.0,
        stride_forward: float = 0.01,
        stride_lateral: float = 0.005,
        yaw_rate: float = 0.0,
        yaw_gain: float = 0.005,
        ramp: float = 1.0
    ) -> torch.Tensor:
        """
        Executes one control cycle completely on GPU.
        Returns a zero-copy PyTorch tensor (num_envs, 18) in Newton DOF ordering.
        """
        # 1. Step Hopf oscillators on GPU using ping-pong buffers
        cur_x, cur_y = self.x_buf_0, self.y_buf_0
        next_x, next_y = self.x_buf_1, self.y_buf_1

        for _ in range(self.sub_steps):
            wp.launch(
                kernel=cpg_hopf_substep_kernel,
                dim=(self.num_envs, self.num_legs),
                inputs=[
                    cur_x, cur_y, next_x, next_y,
                    self.cos_diff, self.sin_diff,
                    self.omega_swing, self.delta_omega,
                    self.coupling_weight, self.alpha, self.mu, self.b,
                    self.dt_sub
                ],
                device=self.device
            )
            # Swap ping-pong pointers (zero overhead)
            cur_x, next_x = next_x, cur_x
            cur_y, next_y = next_y, cur_y

        # If odd number of substeps, copy back to buf_0 to maintain ownership
        if self.sub_steps % 2 != 0:
            wp.copy(self.x_buf_0, cur_x)
            wp.copy(self.y_buf_0, cur_y)

        # 2. Fused trajectory mapping, inverse kinematics, and Newton DOF remapping
        wp.launch(
            kernel=cpg_kinematics_and_ik_kernel,
            dim=(self.num_envs, self.num_legs),
            inputs=[
                self.x_buf_0,
                self.y_buf_0,
                default_feet_pos_body,
                self.mount_positions,
                self.rot_z_inv,
                self.canonical_to_newton,
                self.wp_joint_targets,
                self.L1, self.L2, self.L3,
                self.k1, self.k2, self.k3,
                self.b1, self.b2, self.l2,
                float(dir_angle_rad),
                float(stride_forward),
                float(stride_lateral),
                float(yaw_rate),
                float(yaw_gain),
                float(ramp)
            ],
            device=self.device
        )

        # Return pre-mapped zero-copy tensor directly on cuda:0
        return self.torch_joint_targets