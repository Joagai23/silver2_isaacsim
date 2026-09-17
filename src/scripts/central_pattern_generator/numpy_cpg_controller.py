"""
Hexapod Central Pattern Generator Controller designed for the SILVER2 monitoring platform.
Synthesizes:
 [1] Zhang et al., CCDC 2022 (Analytical 3-DOF IK and Cartesian trajectory mapping).
 [2] Zhong et al., IEEE TCST 2018 (Centroid-leg frame transformations).
 [3] Yin et al., ISRIMT 2023 (Explicit stance/swing duty factor Hopf formulation).
"""

import numpy as np

GAIT_CONFIGS = {
    "tripod": {
        "epsilon": 0.5,
        "phases": np.array([
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5
        ])
    },
    "quadruped": {
        "epsilon": 2.0 / 3.0,
        "phases": np.array([
            0.0, 1.0 / 3.0, 2.0 / 3.0,
            2.0 / 3.0, 1.0 / 3.0, 0.0
        ])
    },
    "wave": {
        "epsilon": 5.0 / 6.0,
        "phases": np.array([
            5.0 / 6.0, 4.0 / 6.0, 3.0 / 6.0,
            2.0 / 6.0, 1.0 / 6.0, 0.0
        ])
    }
}

class HexapodCPGController:
    def __init__(self, leg_mounts, link_lengths, dt=0.01, total_period=2.0, gait="tripod"):
        """
        Args:
            leg_mounts: dict of 6 legs with 'pos': [x, y, z] and 'yaw': angle (rad)
                        relative to centroid frame in Isaac Sim conventions (X-fwd, Y-left, Z-up).
            link_lengths: tuple (L1, L2, L3) for coxa, femur, tibia.
            dt: integration time-step (s).
        """
        self.dt = dt
        self.L1, self.L2, self.L3 = link_lengths
        self.leg_mounts = leg_mounts
        self.leg_names = list(leg_mounts.keys())
        self.num_legs = len(self.leg_names)
        self.total_period = total_period

        # Precompute static mount offsets and inverse yaw transforms (Eq. 5 [1])
        self.mount_positions = np.array([leg_mounts[name]['pos'] for name in self.leg_names], dtype=np.float64)
        yaws = np.array([leg_mounts[name]['yaw'] for name in self.leg_names], dtype=np.float64)
        c_y, s_y = np.cos(yaws), np.sin(yaws)

        self.rot_z_inv = np.zeros((self.num_legs, 3, 3), dtype=np.float64)
        self.rot_z_inv[:, 0, 0] = c_y
        self.rot_z_inv[:, 0, 1] = s_y
        self.rot_z_inv[:, 1, 0] = -s_y
        self.rot_z_inv[:, 1, 1] = c_y
        self.rot_z_inv[:, 2, 2] = 1.0

        # Oscillator dynamics parameters [3]
        self.alpha = 1.0
        self.mu = 100.0
        self.radius = np.sqrt(self.mu)
        self.b = 2.0
        self.coupling_strength = 0.4

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
        self.diagonal_mask = ~np.eye(self.num_legs, dtype=bool)

        # Initialize gait and seed oscillator limit cycle
        self.set_gait(gait, reset_state=True)

    def set_gait(self, gait_name: str, reset_state: bool = False):
        """Switches gait, duty factor, and recalculates phase coupling matrices."""
        if gait_name not in GAIT_CONFIGS:
            raise ValueError(f"Unknown gait '{gait_name}'. Choose from: {list(GAIT_CONFIGS.keys())}")

        cfg = GAIT_CONFIGS[gait_name]
        self.active_gait = gait_name
        self.epsilon = cfg["epsilon"]
        self.phi_phase = cfg["phases"]

        # Stance and swing phase speeds (Eq. 2 [3])
        self.omega_swing = np.pi / ((1.0 - self.epsilon) * self.total_period)
        self.omega_stance = np.pi / (self.epsilon * self.total_period)
        self.delta_omega = self.omega_stance - self.omega_swing

        # Coupling matrix based on phase differences: phi_ij = 2 * pi * (phi_i - phi_j)
        phase_diff = 2.0 * np.pi * (self.phi_phase[:, None] - self.phi_phase[None, :])
        self.cos_diff = np.cos(phase_diff)
        self.sin_diff = np.sin(phase_diff)

        # Seed states onto limit cycle only during startup or explicit resets
        if reset_state or not hasattr(self, 'x'):
            theta = 2.0 * np.pi * self.phi_phase
            self.x = self.radius * np.cos(theta)
            self.y = self.radius * np.sin(theta)

    def step_cpg(self):
        """
        Forward Euler integration of coupled Hopf oscillators with
        dynamic duty factor frequency modulation and diffusive phase coupling.
        """
        coupling_weight = self.coupling_strength / (self.num_legs - 1)

        for _ in range(self.sub_steps):
            r2 = self.x**2 + self.y**2

            # Dual-frequency evaluation using logistic sigmoid (Eq. 2 [3])
            sigma = 1.0 / (1.0 + np.exp(-np.clip(self.b * self.y, -50.0, 50.0)))
            omega_i = self.omega_swing + sigma * self.delta_omega

            # Rotated neighbor state projections
            x_rot = self.x[None, :] * self.cos_diff - self.y[None, :] * self.sin_diff
            y_rot = self.x[None, :] * self.sin_diff + self.y[None, :] * self.cos_diff

            # Diffusive coupling: vanishes at phase lock
            c_x = np.sum((x_rot - self.x[:, None]) * self.diagonal_mask, axis=1)
            c_y = np.sum((y_rot - self.y[:, None]) * self.diagonal_mask, axis=1)

            coupling_x = coupling_weight * c_x
            coupling_y = coupling_weight * c_y

            # Limit cycle ODE integration (Eq. 3 [3])
            dx = self.alpha * (self.mu - r2) * self.x - omega_i * self.y + coupling_x
            dy = self.alpha * (self.mu - r2) * self.y + omega_i * self.x + coupling_y

            self.x += dx * self.dt_sub
            self.y += dy * self.dt_sub

    def map_foot_trajectory_omnidirectional(
        self, 
        default_feet_pos_body, 
        dir_angle_rad=0.0, 
        stride_forward=0.01, 
        stride_lateral=0.005, 
        ramp=1.0
    ):
        """
        Omnidirectional Cartesian trajectory mapping.
        Array Layout: Axis 0 = Lateral (X), Axis 1 = Longitudinal (Y), Axis 2 = Up (Z)
        """
        x_tilde = self.k1 * self.x
        y_tilde = np.where(self.y >= 0.0, self.k2 * self.y + self.b1, self.k3 * self.y + self.b2)

        l_lat = -stride_lateral * np.sin(dir_angle_rad)
        l_fwd = -stride_forward * np.cos(dir_angle_rad)

        p_centroid = default_feet_pos_body.copy()
        p_centroid[:, 0] += ramp * l_lat * x_tilde
        p_centroid[:, 1] += ramp * l_fwd * x_tilde
        p_centroid[:, 2] -= ramp * self.l2 * y_tilde

        return p_centroid

    def inverse_kinematics(self, p_leg_base):
        """
        Analytical 3-DOF IK for Coxa (yaw), Femur (pitch), Tibia (pitch).
        p_leg_base: [x, y, z] relative to leg base origin.
        """
        px, py, pz = p_leg_base

        # Joint 1: Coxa - Yaw
        theta_1 = np.arctan2(py, px)
        c1, s1 = np.cos(theta_1), np.sin(theta_1)

        # Auxiliary terms: gamma_2
        gamma_2 = px * c1 + py * s1 - self.L1

        # Joint 3: Tibia - Pitch
        cos_theta_3 = (gamma_2**2 + pz**2 - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        theta_3 = np.arccos(np.clip(cos_theta_3, -1.0, 1.0))

        # Joint 2: Femur - Pitch
        chord = np.sqrt(gamma_2**2 + pz**2)
        chord_pitch_down = np.arctan2(-pz, gamma_2)
        sin_psi = (self.L3 * np.sin(theta_3)) / chord
        psi = np.arcsin(np.clip(sin_psi, -1.0, 1.0))
        theta_2 = chord_pitch_down - psi

        return np.array([theta_1, theta_2, theta_3])
    
    def compute_joint_targets(
        self, 
        default_feet_pos_body, 
        dir_angle_rad=0.0, 
        stride_forward=0.01, 
        stride_lateral=0.005, 
        ramp=1.0
    ):
        """Executes one control cycle and outputs joint angles (6, 3)."""
        self.step_cpg()
        p_centroid = self.map_foot_trajectory_omnidirectional(
            default_feet_pos_body,
            dir_angle_rad=dir_angle_rad,
            stride_forward=stride_forward,
            stride_lateral=stride_lateral,
            ramp=ramp
        )

        p_rel = p_centroid - self.mount_positions
        p_local_batch = np.einsum('ijk,ik->ij', self.rot_z_inv, p_rel)

        joint_targets = np.zeros((self.num_legs, 3))
        for i in range(self.num_legs):
            joint_targets[i] = self.inverse_kinematics(p_local_batch[i])

        return joint_targets