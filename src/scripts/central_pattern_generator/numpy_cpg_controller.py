"""
Hexapod Central Pattern Generator Controller designed for the SILVER2 monitoring platform.
Synthesizes:
 [1] Zhang et al., "Central Pattern Generators for Locomotion Control in Hexapod Robot Legs", CCDC 2022 
 [2] Zhong et al., "Locomotion Control and Gait Planning of a Novel Hexapod Robot Using Biomimetic Neurons", IEEE TCST 2018
 [3] Yin et al., "Energy Efficiency Optimization of Hexapod Robots Based on Central Pattern Generator Control", ISRIMT 2023 
"""

import numpy as np
from silver2_isaac_constants import GAIT_CONFIGS

class NumpyHexapodCPGController:
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
        yaw_rate=0.0,
        yaw_gain=0.005,
        ramp=1.0
    ):
        """
        Omnidirectional Cartesian trajectory mapping with turning / pivoting superposition.
        Array Layout: Axis 0 = Lateral (Y), Axis 1 = Longitudinal (X), Axis 2 = Up (Z)
        
        Args:
            default_feet_pos_body: shape (6, 3) nominal foot positions in centroid frame.
            dir_angle_rad: heading angle in radians (0.0 = Pure Forward).
            stride_forward: longitudinal translation step size (m).
            stride_lateral: lateral translation step size (m).
            yaw_rate: commanded angular rate (+ for CCW / turn left, - for CW / turn right).
            yaw_gain: scaling factor mapping angular velocity to linear foot displacement.
            ramp: soft-start scaling factor [0.0, 1.0].
        """
        x_tilde = self.k1 * self.x
        y_tilde = np.where(self.y >= 0.0, self.k2 * self.y + self.b1, self.k3 * self.y + self.b2)

        # Foot anchor coordinates relative to centroid
        pos_lat = default_feet_pos_body[:, 0]
        pos_fwd = default_feet_pos_body[:, 1]

        # Base translational strokes
        l_lat_trans = -stride_lateral * np.sin(dir_angle_rad)
        l_fwd_trans = -stride_forward * np.cos(dir_angle_rad)

        # Rotational twist
        l_lat = l_lat_trans - yaw_gain * yaw_rate * pos_fwd
        l_fwd = l_fwd_trans + yaw_gain * yaw_rate * pos_lat
        
        p_centroid = default_feet_pos_body.copy()
        p_centroid[:, 0] += ramp * l_lat * x_tilde    # Axis 0: Lateral
        p_centroid[:, 1] += ramp * l_fwd * x_tilde    # Axis 1: Longitudinal
        p_centroid[:, 2] -= ramp * self.l2 * y_tilde  # Axis 2: Vertical

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

    def compute_default_feet_body(self, default_angles_deg, leg_mounts, link_lengths):
        """
        Computes nominal foot positions in the centroid frame (Eq. 2 & Eq. 4 [1]).
        
        Args:
            default_angles_deg: Array-like of joint angles [coxa, femur, tibia] in degrees.
                                Can be shape (3,) if identical across all legs, 
                                or shape (6, 3) for per-leg configurations.
            leg_mounts: Dict defining 'pos' and 'yaw' for each leg.
            link_lengths: Tuple (L1, L2, L3) in meters.
            
        Returns:
            np.ndarray: default_feet_body of shape (6, 3) in robot centroid frame.
        """
        L1, L2, L3 = link_lengths
        leg_names = list(leg_mounts.keys())
        num_legs = len(leg_names)

        angles_deg = np.array(default_angles_deg, dtype=np.float64)
        if angles_deg.ndim == 1 and angles_deg.shape[0] == 3:
            angles_deg = np.tile(angles_deg, (num_legs, 1))

        angles_rad = np.radians(angles_deg)
        default_feet_body = np.zeros((num_legs, 3), dtype=np.float64)

        for i, name in enumerate(leg_names):
            theta1, theta2, theta3 = angles_rad[i]
            
            # Forward Kinematics in Leg Base Frame (Eq. 2)
            c1, s1 = np.cos(theta1), np.sin(theta1)
            c2, s2 = np.cos(theta2), np.sin(theta2)
            c23 = np.cos(theta2 + theta3)
            s23 = np.sin(theta2 + theta3)

            p_o0 = np.array([
                L1 * c1 + L2 * c1 * c2 + L3 * c1 * c23,
                L1 * s1 + L2 * s1 * c2 + L3 * s1 * c23,
                -(L2 * s2 + L3 * s23)  # Negated for Isaac Sim Y-axis pitch
            ])

            # Centroid Frame Transformation: R_z(phi) * p_o0 + t_base (Eq. 4)
            mount = leg_mounts[name]
            p_mount = np.array(mount['pos'])
            phi = mount['yaw']

            c_phi, s_phi = np.cos(phi), np.sin(phi)
            rot_z = np.array([
                [c_phi, -s_phi, 0.0],
                [s_phi,  c_phi, 0.0],
                [  0.0,    0.0, 1.0]
            ])

            default_feet_body[i] = rot_z @ p_o0 + p_mount

        return default_feet_body
    
    def compute_joint_targets(
        self, 
        default_feet_pos_body, 
        dir_angle_rad=0.0, 
        stride_forward=0.01, 
        stride_lateral=0.005, 
        yaw_rate=0.0,
        yaw_gain=0.005,
        ramp=1.0
    ):
        """Executes one control cycle and outputs joint angles (6, 3)."""
        self.step_cpg()
        p_centroid = self.map_foot_trajectory_omnidirectional(
            default_feet_pos_body,
            dir_angle_rad=dir_angle_rad,
            stride_forward=stride_forward,
            stride_lateral=stride_lateral,
            yaw_rate=yaw_rate,
            yaw_gain=yaw_gain,
            ramp=ramp
        )

        # Transform from centroid frame to each leg's mounting frame
        p_rel = p_centroid - self.mount_positions
        p_local_batch = np.einsum('ijk,ik->ij', self.rot_z_inv, p_rel)

        # Solve analytical IK per leg
        joint_targets = np.zeros((self.num_legs, 3))
        for i in range(self.num_legs):
            joint_targets[i] = self.inverse_kinematics(p_local_batch[i])

        return joint_targets