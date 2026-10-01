# pyright: reportInvalidTypeForm = false
import warp as wp
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
from silver2_constants.isaac_constants import *
from evo_warp_cpg_kernels import *

wp.init()

class VectorizedWarpHexapodCPGController:
    def __init__(
        self,
        num_envs: int,
        leg_mounts: Dict[str, Dict[str, Any]],
        link_lengths: Tuple[float, float, float],
        dt: float = 0.01,
        gait: str = "tripod",
        device: str = "cuda:0"
    ):
        self.num_envs = int(num_envs)
        self.num_legs = 6
        self.total_threads = self.num_envs * self.num_legs
        self.device = device
        self.dt = float(dt)

        self.link_lengths = wp.vec3(
            float(link_lengths[0]),
            float(link_lengths[1]),
            float(link_lengths[2])
        )

        # 1. Mount Transforms (Shared across all env instances)
        mount_pos_host = np.zeros((self.num_legs, 3), dtype=np.float32)
        rot_z_inv_host = np.zeros((self.num_legs, 3, 3), dtype=np.float32)
        leg_names = list(leg_mounts.keys())

        for i, name in enumerate(leg_names):
            mount_pos_host[i] = leg_mounts[name]['pos']
            yaw = float(leg_mounts[name]['yaw'])
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            rot_z_inv_host[i] = np.array([
                [ c_y,  s_y, 0.0],
                [-s_y,  c_y, 0.0],
                [ 0.0,  0.0, 1.0]
            ], dtype=np.float32)

        self.mount_positions = wp.array(mount_pos_host, dtype=wp.vec3, device=self.device)
        self.rot_z_inv = wp.array(rot_z_inv_host, dtype=wp.mat33, device=self.device)

        # 2. Base Oscillator Hyperparameters
        self.alpha = 1.0
        self.mu = 100.0
        self.radius = float(np.sqrt(self.mu))
        self.epsilon_y = 0.5
        self.sub_steps = 5
        self.dt_sub = self.dt / float(self.sub_steps)

        # Sensory phase reflex toggles
        self.enable_phase_freeze = 0
        self.epsilon_window = 0.25

        # 3. Canonical Coupling Matrix
        cfg = GAIT_CONFIGS[gait]
        self.phi_phase = np.array(cfg["phases"], dtype=np.float32)
        phase_diff = 2.0 * np.pi * (self.phi_phase[:, None] - self.phi_phase[None, :])
        cos_diff = np.cos(phase_diff).astype(np.float32)
        sin_diff = np.sin(phase_diff).astype(np.float32)
        coupling_diff_host = np.stack([cos_diff, sin_diff], axis=-1)
        self.coupling_diff = wp.array(coupling_diff_host, dtype=wp.vec2, device=self.device)

        # Extract dynamic coordination groups for inter-leg phase lock
        unique_phases = []
        for phi in self.phi_phase:
            if not any(np.isclose(phi, u, atol=1e-3) for u in unique_phases):
                unique_phases.append(phi)
        self.gait_groups = [
            [i for i, phi in enumerate(self.phi_phase) if np.isclose(phi, u, atol=1e-3)]
            for u in unique_phases
        ]

        # 4. Preallocated Vectorized Device Buffers
        self.state = wp.zeros(self.total_threads, dtype=wp.vec2, device=self.device)
        self.state_next = wp.zeros(self.total_threads, dtype=wp.vec2, device=self.device)
        self.feet_pos_centroid = wp.zeros(self.total_threads, dtype=wp.vec3, device=self.device)
        self.feet_pos_local = wp.zeros(self.total_threads, dtype=wp.vec3, device=self.device)
        self.joint_targets_device = wp.zeros(self.total_threads, dtype=wp.vec3, device=self.device)
        self.contact_state = wp.zeros(self.total_threads, dtype=wp.int32, device=self.device)

        # 5. Preallocated Vectorized Genome Arrays (Length = num_envs)
        self.l2_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.b1_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.b2_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.k1_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.k2_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.k3_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.b_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.omega_swing_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.delta_omega_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.coupling_weight_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.f_touch_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.f_release_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        self.reset_all_states()

    def reset_all_states(self):
        """Initializes all oscillators onto the canonical limit cycle."""
        theta = 2.0 * np.pi * self.phi_phase
        x_single = (self.radius * np.cos(theta)).astype(np.float32)
        y_single = (self.radius * np.sin(theta)).astype(np.float32)
        state_single = np.stack([x_single, y_single], axis=-1)
        state_pop = np.tile(state_single, (self.num_envs, 1))

        self.state = wp.array(state_pop, dtype=wp.vec2, device=self.device)
        self.state_next = wp.array(state_pop, dtype=wp.vec2, device=self.device)
        self.contact_state.zero_()

    def set_population_genomes(self, genomes_torch: torch.Tensor):
        """
        Ingests population genome tensor (shape: [num_envs, 10]) on cuda:0 zero-copy:
        [l2, k1, k2, k3, epsilon, total_period, coupling_strength, b, f_touch, f_release]
        """
        assert genomes_torch.shape == (self.num_envs, 10), "Invalid genome tensor shape"

        l2 = genomes_torch[:, 0].contiguous()
        k1 = genomes_torch[:, 1].contiguous()
        k2 = genomes_torch[:, 2].contiguous()
        k3 = genomes_torch[:, 3].contiguous()
        epsilon = genomes_torch[:, 4].contiguous()
        period = genomes_torch[:, 5].contiguous()
        coupling = genomes_torch[:, 6].contiguous()
        b = genomes_torch[:, 7].contiguous()
        f_touch = genomes_torch[:, 8].contiguous()
        f_release = genomes_torch[:, 9].contiguous()

        # Compute dynamic frequencies
        omega_swing = torch.pi / ((1.0 - epsilon) * period)
        omega_stance = torch.pi / (epsilon * period)
        delta_omega = omega_stance - omega_swing
        coupling_weight = coupling / float(self.num_legs - 1)

        # Zero-copy views into Warp arrays
        self.l2_vec = wp.from_torch(l2, dtype=wp.float32)
        self.b1_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.b2_vec = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.k1_vec = wp.from_torch(k1, dtype=wp.float32)
        self.k2_vec = wp.from_torch(k2, dtype=wp.float32)
        self.k3_vec = wp.from_torch(k3, dtype=wp.float32)
        self.b_vec = wp.from_torch(b, dtype=wp.float32)
        self.omega_swing_vec = wp.from_torch(omega_swing.contiguous(), dtype=wp.float32)
        self.delta_omega_vec = wp.from_torch(delta_omega.contiguous(), dtype=wp.float32)
        self.coupling_weight_vec = wp.from_torch(coupling_weight.contiguous(), dtype=wp.float32)
        self.f_touch_vec = wp.from_torch(f_touch, dtype=wp.float32)
        self.f_release_vec = wp.from_torch(f_release, dtype=wp.float32)

    def compute_joint_targets_vec(
        self,
        default_feet_pos_body: wp.array(dtype=wp.vec3),
        dir_angle_rad: float = 0.0,
        stride_forward: float = 0.01,
        stride_lateral: float = 0.005,
        yaw_rate: float = 0.0,
        yaw_gain: float = 0.0,
        ramp: float = 1.0,
        contact_forces: Optional[wp.array(dtype=wp.vec3)] = None
    ) -> wp.array(dtype=wp.vec3):
        # 1. Sensory Phase Resetting (Case A)
        if contact_forces is not None:
            wp.launch(
                kernel=apply_phase_resetting_kernel_vec,
                dim=self.total_threads,
                inputs=[
                    self.state,
                    contact_forces,
                    self.contact_state,
                    self.radius,
                    self.f_touch_vec,
                    self.f_release_vec,
                    self.epsilon_y
                ],
                device=self.device
            )

            # Vectorized Dynamic Group Phase Lock (Agnostic to gait)
            # Enforces that if any leg in group g touches down, its group peers snap to touchdown
            st_torch = wp.to_torch(self.state).reshape(self.num_envs, self.num_legs, 2)
            for group in self.gait_groups:
                if len(group) > 1:
                    group_x = st_torch[:, group, 0]
                    group_y = st_torch[:, group, 1]
                    # Mask of envs where at least one leg in group triggered touchdown
                    is_touchdown = (group_x >= (self.radius - 0.01)) & (group_y.abs() <= 0.1)
                    triggered_envs = is_touchdown.any(dim=1)
                    if triggered_envs.any():
                        # For those envs, snap all group legs still in swing (y < 0) to [R, 0]
                        for leg_idx in group:
                            swing_mask = triggered_envs & (st_torch[:, leg_idx, 1] < 0.0)
                            st_torch[swing_mask, leg_idx, 0] = self.radius
                            st_torch[swing_mask, leg_idx, 1] = 0.0

        # 2. Step Hopf ODEs across all population oscillators
        for _ in range(self.sub_steps):
            wp.launch(
                kernel=cpg_substep_kernel_vec,
                dim=self.total_threads,
                inputs=[
                    self.state,
                    self.state_next,
                    self.coupling_diff,
                    self.contact_state,
                    self.num_legs,
                    self.dt_sub,
                    self.alpha,
                    self.mu,
                    self.b_vec,
                    self.omega_swing_vec,
                    self.delta_omega_vec,
                    self.coupling_weight_vec,
                    self.enable_phase_freeze,
                    self.epsilon_window
                ],
                device=self.device
            )
            self.state, self.state_next = self.state_next, self.state

        # 3. Omnidirectional Cartesian Trajectory Generation
        wp.launch(
            kernel=map_foot_trajectory_omnidirectional_kernel_vec,
            dim=self.total_threads,
            inputs=[
                self.state,
                default_feet_pos_body,
                self.feet_pos_centroid,
                self.k1_vec,
                self.k2_vec,
                self.k3_vec,
                self.b1_vec,
                self.b2_vec,
                self.l2_vec,
                float(dir_angle_rad),
                float(stride_forward),
                float(stride_lateral),
                float(yaw_rate),
                float(yaw_gain),
                float(ramp)
            ],
            device=self.device
        )

        # 4. Transform Centroid -> Local Base Frame
        wp.launch(
            kernel=centroid_to_leg_frame_kernel_vec,
            dim=self.total_threads,
            inputs=[
                self.feet_pos_centroid,
                self.mount_positions,
                self.rot_z_inv,
                self.feet_pos_local
            ],
            device=self.device
        )

        # 5. Vectorized Analytical IK across all joints
        wp.launch(
            kernel=inverse_kinematics_kernel_vec,
            dim=self.total_threads,
            inputs=[
                self.feet_pos_local,
                self.link_lengths,
                self.joint_targets_device
            ],
            device=self.device
        )

        return self.joint_targets_device