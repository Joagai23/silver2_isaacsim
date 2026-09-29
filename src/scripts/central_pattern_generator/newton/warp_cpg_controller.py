# pyright: reportInvalidTypeForm = false
import warp as wp
import numpy as np
from typing import Dict, Tuple, Any, Optional
from silver2_constants.isaac_constants import GAIT_CONFIGS
from warp_cpg_kernels import *

wp.init()

class WarpHexapodCPGController:
    """
    NVIDIA Warp implementation of the SILVER2 Hexapod CPG Controller.
    Executes ODE integration, trajectory generation, and inverse kinematics
    directly in GPU memory on cuda:0.
    """

    def __init__(
        self,
        leg_mounts: Dict[str, Dict[str, Any]],
        link_lengths: Tuple[float, float, float],
        dt: float = 0.01,
        total_period: float = 2.0,
        gait: str = "tripod",
        device: str = "cuda:0"
    ):
        """
        Initializes static kinematic constants, GPU transformation buffers,
        and oscillator hyperparameters.
        """
        self.device = device
        self.dt = float(dt)
        self.total_period = float(total_period)
        self.leg_names = list(leg_mounts.keys())
        self.num_legs = len(self.leg_names)
        self.link_lengths = wp.vec3(
            float(link_lengths[0]),
            float(link_lengths[1]),
            float(link_lengths[2])
        )

        # Precompute static mount offsets and inverse yaw transforms (Eq. 5 [1])
        mount_pos_host = np.zeros((self.num_legs, 3), dtype=np.float32)
        rot_z_inv_host = np.zeros((self.num_legs, 3, 3), dtype=np.float32)

        for i, name in enumerate(self.leg_names):
            mount_pos_host[i] = leg_mounts[name]['pos']
            yaw = float(leg_mounts[name]['yaw'])
            c_y, s_y = np.cos(yaw), np.sin(yaw)

            # Inverse rotation Rz(-yaw)
            rot_z_inv_host[i] = np.array([
                [ c_y,  s_y, 0.0],
                [-s_y,  c_y, 0.0],
                [ 0.0,  0.0, 1.0]
            ], dtype=np.float32)

        # Upload static geometric transforms to GPU memory
        self.mount_positions = wp.array(mount_pos_host, dtype=wp.vec3, device=self.device)
        self.rot_z_inv = wp.array(rot_z_inv_host, dtype=wp.mat33, device=self.device)

        # Oscillator Dynamics Hyperparameters [3]
        self.alpha = 1.0
        self.mu = 100.0
        self.radius = float(np.sqrt(self.mu))
        self.b = 2.0
        self.coupling_strength = 0.4

        # Trajectory Mapping Parameters
        self.k1 = 1.0
        self.k2 = 0.0
        self.k3 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.l1 = -0.01
        self.l2 = 0.005

        # Numerical Sub-Stepping
        self.sub_steps = 5
        self.dt_sub = self.dt / float(self.sub_steps)

        # Sensory Phase Resetting Thresholds (Calibrated for 26.78 kg)
        self.f_touch = 15.0
        self.f_release = 5.0
        self.epsilon_y = 0.5
        self.epsilon_window = 1.5

        # Preallocate Intermediate GPU Buffers
        self.state = wp.zeros(self.num_legs, dtype=wp.vec2, device=self.device)
        self.state_next = wp.zeros(self.num_legs, dtype=wp.vec2, device=self.device)
        self.feet_pos_centroid = wp.zeros(self.num_legs, dtype=wp.vec3, device=self.device)
        self.feet_pos_local = wp.zeros(self.num_legs, dtype=wp.vec3, device=self.device)
        self.joint_targets_device = wp.zeros(self.num_legs, dtype=wp.vec3, device=self.device)
        self.contact_state = wp.zeros(self.num_legs, dtype=wp.int32, device=self.device)

        self.set_gait(gait, reset_state=True)

    def set_gait(self, gait_name: str, reset_state: bool = False):
        """
        Switches active gait, calculates duty-factor frequencies, uploads
        pairwise coupling rotation matrix to GPU memory, and seeds limit cycle states.
        """
        if gait_name not in GAIT_CONFIGS:
            raise ValueError(f"Unknown gait '{gait_name}'. Choose from: {list(GAIT_CONFIGS.keys())}")

        cfg = GAIT_CONFIGS[gait_name]
        self.active_gait = gait_name
        self.epsilon = float(cfg["epsilon"])
        self.phi_phase = np.array(cfg["phases"], dtype=np.float32)

        # Dynamic coordination groups: cluster leg indices by matching phase offsets
        unique_phases = []
        for phi in self.phi_phase:
            if not any(np.isclose(phi, u, atol=1e-3) for u in unique_phases):
                unique_phases.append(phi)

        self.gait_groups = [
            [i for i, phi in enumerate(self.phi_phase) if np.isclose(phi, u, atol=1e-3)]
            for u in unique_phases
        ]

        print(f"[INFO] Gait '{gait_name}' loaded with {len(self.gait_groups)} dynamic coordination groups:")
        for g_idx, group in enumerate(self.gait_groups):
            print(f"  Group {g_idx} (phase = {unique_phases[g_idx]:.3f}): Legs {group}")

        # Stance and swing phase speeds (rad/s) (Eq. 2 [3])
        self.omega_swing = float(np.pi / ((1.0 - self.epsilon) * self.total_period))
        self.omega_stance = float(np.pi / (self.epsilon * self.total_period))
        self.delta_omega = self.omega_stance - self.omega_swing

        # Pairwise coupling matrix: phi_ij = 2 * pi * (phi_i - phi_j)
        phase_diff = 2.0 * np.pi * (self.phi_phase[:, None] - self.phi_phase[None, :])
        cos_diff = np.cos(phase_diff).astype(np.float32)
        sin_diff = np.sin(phase_diff).astype(np.float32)

        # Pack (cos, sin) into a (6, 6) wp.vec2 device array
        coupling_diff_host = np.stack([cos_diff, sin_diff], axis=-1)
        self.coupling_diff = wp.array(coupling_diff_host, dtype=wp.vec2, device=self.device)

        # Seed states onto limit cycle only during startup or explicit resets
        if reset_state or not hasattr(self, 'state'):
            theta = 2.0 * np.pi * self.phi_phase
            x_init = (self.radius * np.cos(theta)).astype(np.float32)
            y_init = (self.radius * np.sin(theta)).astype(np.float32)
            state_host = np.stack([x_init, y_init], axis=-1)

            self.state = wp.array(state_host, dtype=wp.vec2, device=self.device)
            self.state_next = wp.array(state_host, dtype=wp.vec2, device=self.device)

    def step_cpg(self, contact_forces: Optional[wp.array(dtype=wp.vec3)] = None):
        """
        Executes sub-stepping Forward Euler integration on GPU device memory.
        Utilizes double-buffered ping-pong updates across CUDA threads.
        If contact_forces is supplied, applies phase resetting and enables phase freeze.
        """
        enable_freeze = 0

        if contact_forces is not None:
            # Evaluate contacts and early touchdown resets
            self.apply_phase_resetting(contact_forces)

            # Tripod Phase Lock Enforcement
            st_np = self.state.numpy()
            for group in self.gait_groups:
                if len(group) > 1:
                    group_touched_down = any(
                        st_np[idx, 0] >= self.radius - 0.01 and abs(st_np[idx, 1]) <= 0.1 
                        for idx in group
                    )
                    if group_touched_down:
                        for idx in group:
                            if st_np[idx, 1] < 0.0:
                                st_np[idx] = [self.radius, 0.0]

            self.state = wp.array(st_np, dtype=wp.vec2, device=self.device)

        coupling_weight = self.coupling_strength / float(self.num_legs - 1)

        for _ in range(self.sub_steps):
            wp.launch(
                kernel=cpg_substep_kernel,
                dim=self.num_legs,
                inputs=[
                    self.state,
                    self.state_next,
                    self.coupling_diff,
                    self.contact_state,
                    self.num_legs,
                    self.dt_sub,
                    self.alpha,
                    self.mu,
                    self.b,
                    self.omega_swing,
                    self.delta_omega,
                    coupling_weight,
                    enable_freeze,
                    self.epsilon_window
                ],
                device=self.device
            )
            # Ping-pong buffer swap to avoid race conditions across threads
            self.state, self.state_next = self.state_next, self.state

    def map_foot_trajectory_omnidirectional(
        self,
        default_feet_pos_body: wp.array(dtype=wp.vec3),
        dir_angle_rad: float = 0.0,
        stride_forward: float = 0.01,
        stride_lateral: float = 0.005,
        yaw_rate: float = 0.0,
        yaw_gain: float = 0.005,
        ramp: float = 1.0
    ) -> wp.array(dtype=wp.vec3):
        """
        Launches GPU kernel to map Hopf states into Cartesian coordinates in centroid frame.
        Writes result directly into preallocated device buffer self.feet_pos_centroid.
        """
        wp.launch(
            kernel=map_foot_trajectory_omnidirectional_kernel,
            dim=self.num_legs,
            inputs=[
                self.state,
                default_feet_pos_body,
                self.feet_pos_centroid,
                self.k1,
                self.k2,
                self.k3,
                self.b1,
                self.b2,
                self.l2,
                float(dir_angle_rad),
                float(stride_forward),
                float(stride_lateral),
                float(yaw_rate),
                float(yaw_gain),
                float(ramp)
            ],
            device=self.device
        )
        return self.feet_pos_centroid

    def inverse_kinematics(
        self, 
        feet_pos_local: wp.array(dtype=wp.vec3)
    ) -> wp.array(dtype=wp.vec3):
        """
        Batched device wrapper launching IK evaluation across all legs in parallel.
        Writes resulting joint angles directly into self.joint_targets_device.
        """
        wp.launch(
            kernel=inverse_kinematics_kernel,
            dim=self.num_legs,
            inputs=[
                feet_pos_local,
                self.link_lengths,
                self.joint_targets_device
            ],
            device=self.device
        )
        return self.joint_targets_device

    def compute_default_feet_body(
        self,
        default_angles_deg: np.ndarray
    ) -> wp.array(dtype=wp.vec3):
        """
        Calculates default foot positions in the centroid frame directly on GPU memory.
        
        Args:
            default_angles_deg: np.ndarray of shape (3,) or (6, 3) in degrees.
            
        Returns:
            wp.array(shape=6, dtype=wp.vec3): Nominal foot positions on cuda:0.
        """
        angles_deg = np.array(default_angles_deg, dtype=np.float32)
        if angles_deg.ndim == 1 and angles_deg.shape[0] == 3:
            angles_deg = np.tile(angles_deg, (self.num_legs, 1))

        angles_rad_host = np.deg2rad(angles_deg).astype(np.float32)
        angles_rad_device = wp.array(
            angles_rad_host,
            dtype=wp.vec3,
            device=self.device
        )

        out_feet = wp.zeros(self.num_legs, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=compute_default_feet_body_kernel,
            dim=self.num_legs,
            inputs=[
                angles_rad_device,
                self.mount_positions,
                self.rot_z_inv,
                self.link_lengths,
                out_feet
            ],
            device=self.device
        )
        return out_feet

    def compute_joint_targets(
        self,
        default_feet_pos_body: wp.array(dtype=wp.vec3),
        dir_angle_rad: float = 0.0,
        stride_forward: float = 0.01,
        stride_lateral: float = 0.005,
        yaw_rate: float = 0.0,
        yaw_gain: float = 0.005,
        ramp: float = 1.0,
        contact_forces: Optional[wp.array(dtype=wp.vec3)] = None
    ) -> wp.array(dtype=wp.vec3):
        """
        Executes one control cycle on the GPU:
        1. Steps the coupled Hopf oscillators via Forward Euler sub-stepping.
        2. Maps limit-cycle states into Cartesian foot positions in centroid frame.
        3. Transforms Cartesian targets into local leg frames.
        4. Solves analytical 3-DOF inverse kinematics for all legs in parallel.

        Returns:
            wp.array(shape=6, dtype=wp.vec3): Commanded joint angles [coxa, femur, tibia] on cuda:0.
        """
        # 1. Step Hopf oscillator states
        self.step_cpg(contact_forces=contact_forces)

        # 2. Map oscillator phases to 3D Cartesian coordinates (Centroid Frame)
        self.map_foot_trajectory_omnidirectional(
            default_feet_pos_body=default_feet_pos_body,
            dir_angle_rad=dir_angle_rad,
            stride_forward=stride_forward,
            stride_lateral=stride_lateral,
            yaw_rate=yaw_rate,
            yaw_gain=yaw_gain,
            ramp=ramp
        )

        # 3. Transform coordinates from centroid frame to local leg base frames
        wp.launch(
            kernel=centroid_to_leg_frame_kernel,
            dim=self.num_legs,
            inputs=[
                self.feet_pos_centroid,
                self.mount_positions,
                self.rot_z_inv,
                self.feet_pos_local
            ],
            device=self.device
        )

        # 4. Solve analytical inverse kinematics per leg in parallel
        self.inverse_kinematics(self.feet_pos_local)

        return self.joint_targets_device

    def apply_phase_resetting(self, contact_forces: wp.array(dtype=wp.vec3)):
        """
        Executes sensory phase resetting kernel: applies Schmitt trigger
        and projects early touchdowns onto [R, 0.0].
        """
        wp.launch(
            kernel=apply_phase_resetting_kernel,
            dim=self.num_legs,
            inputs=[
                self.state,
                contact_forces,
                self.contact_state,
                self.radius,
                self.f_touch,
                self.f_release,
                self.epsilon_y
            ],
            device=self.device
        )