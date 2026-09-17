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
    def __init__(self, leg_mounts, link_lengths, dt=0.01):
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

        # Precompute static mount offsets
        self.mount_positions = np.array(
            [leg_mounts[name]['pos'] for name in self.leg_names], 
            dtype=np.float64
        )

        # Precompute static inverse rotation matrices (Eq. 5)
        yaws = np.array([leg_mounts[name]['yaw'] for name in self.leg_names], dtype=np.float64)
        c_y = np.cos(yaws)
        s_y = np.sin(yaws)

        self.rot_z_inv = np.zeros((self.num_legs, 3, 3), dtype=np.float64)
        self.rot_z_inv[:, 0, 0] = c_y
        self.rot_z_inv[:, 0, 1] = s_y
        self.rot_z_inv[:, 1, 0] = -s_y
        self.rot_z_inv[:, 1, 1] = c_y
        self.rot_z_inv[:, 2, 2] = 1.0

        # CPG Hopf parameters
        self.alpha = 1.0
        self.mu = 100.0
        self.omega = np.pi

        # Trajectory mapping gains
        self.k1 = 1.0
        self.k2 = 0.0
        self.k3 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.l1 = -0.01
        self.l2 = 0.005

        # Tripod gait phase definitions (L0, L1, L2, L3, L4, L5)
        self.phi_phase = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

        # Precompute coupling matrix: phi_ij = 2 * pi * (phi_i - phi_j)
        self.phase_diff = 2.0 * np.pi * (self.phi_phase[:, None] - self.phi_phase[None, :])
        self.sin_diff = np.sin(self.phase_diff)
        self.cos_diff = np.cos(self.phase_diff)

        # Oscillator initial state
        self.x = np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        self.y = np.zeros(self.num_legs)

    def step_cpg(self):
        """
        Forward Euler integration of coupled Hopf oscillators.
        (Eq.7).
        """
        sub_steps = 5
        dt_sub = self.dt / sub_steps
        diagonal_mask = ~np.eye(self.num_legs, dtype=bool)

        for _ in range(sub_steps):
            r2 = self.x**2 + self.y**2

            coupling_x = np.sum((self.x[None, :] * self.cos_diff - self.y[None, :] * self.sin_diff) * diagonal_mask, axis=1)
            coupling_y = np.sum((self.x[None, :] * self.sin_diff + self.y[None, :] * self.cos_diff) * diagonal_mask, axis=1)

            dx = self.alpha * (self.mu - r2) * self.x - self.omega * self.y + coupling_x
            dy = self.alpha * (self.mu - r2) * self.y + self.omega * self.x + coupling_y

            self.x += dx * dt_sub
            self.y += dy * dt_sub

    def map_foot_trajectory_linear(self, default_feet_pos_body, ramp=1.0):
        """
        Maps (x_i, y_i) to Cartesian coordinates in the centroid frame.
        default_feet_pos_body: shape (6, 3) representing nominal [x0, y0, z0] per leg.
        (Eq.8, Eq.9).
        """
        x_tilde = self.k1 * self.x
        y_tilde = np.where(self.y >= 0, self.k2 * self.y + self.b1, self.k3 * self.y + self.b2)

        # Target positions in Body Centroid Frame
        # In Isaac: X is forward, Y is lateral, and Z is vertical
        p_centroid = default_feet_pos_body.copy()
        p_centroid[:, 0] = default_feet_pos_body[:, 0]  
        p_centroid[:, 1] += ramp * self.l1 * x_tilde
        p_centroid[:, 2] -= ramp * self.l2 * y_tilde

        return p_centroid

    def map_foot_trajectory_omnidirectional( self, default_feet_pos_body, dir_angle_rad=0.0, 
                                            stride_forward=0.01, stride_lateral=0.005, ramp=1.0):
        """
        Omnidirectional Cartesian trajectory mapping.
        Frame: +X = Front, +Y = Left, +Z = Up
        
        dir_angle_rad = 0.0        -> Forward (+X)
        dir_angle_rad = pi / 2     -> Right (-Y)
        dir_angle_rad = -pi / 2    -> Left (+Y)
        dir_angle_rad = pi         -> Backward (-X)
        """
        x_tilde = self.k1 * self.x
        y_tilde = np.where(self.y >= 0.0, self.k2 * self.y + self.b1, self.k3 * self.y + self.b2)

        # Decompose stroke into body frame components
        # Note: negative sign ensures positive velocity along chosen heading
        l_lat = -stride_lateral * np.sin(dir_angle_rad)
        l_fwd = -stride_forward * np.cos(dir_angle_rad)

        p_centroid = default_feet_pos_body.copy()
        p_centroid[:, 0] += ramp * l_lat * x_tilde      # Axis 0: Lateral
        p_centroid[:, 1] += ramp * l_fwd * x_tilde      # Axis 1: Longitudinal
        p_centroid[:, 2] -= ramp * self.l2 * y_tilde    # Axis 2: Vertical clearance lift (+Z)

        return p_centroid

    def inverse_kinematics(self, p_leg_base, L1, L2, L3):
        """
        Analytical 3-DOF IK for Coxa (yaw), Femur (pitch), Tibia (pitch).
        p_leg_base: [x, y, z] relative to leg base origin.
        L1, L2, L3: Link lengths (Coxa, Femur, Tibia).
        (Eq. 6)
        """
        px, py, pz = p_leg_base

        # Joint 1: Coxa - Yaw
        theta_1 = np.arctan2(py, px)
        c1 = np.cos(theta_1)
        s1 = np.sin(theta_1)

        # Auxiliary terms: gamma_2, gamma_3
        gamma_2 = px * c1 + py * s1 - L1

        # Joint 3: Tibia - Pitch
        cos_theta_3 = (gamma_2**2 + pz**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
        theta_3 = np.arccos(np.clip(cos_theta_3, -1.0, 1.0))

        # Joint 2: Femur - Pitch
        chord = np.sqrt(gamma_2**2 + pz**2)
        chord_pitch_down = np.arctan2(-pz, gamma_2)
        psi = np.arcsin(np.clip((L3 * np.sin(theta_3)) / chord, -1.0, 1.0))
        theta_2 = chord_pitch_down - psi

        return np.array([theta_1, theta_2, theta_3])
    
    def compute_joint_targets(self, default_feet_pos_body, dir_angle_rad=0.0, ramp=1.0):
        """
        Executes one control cycle and outputs joint angles (6, 3).
        (Eq.5, Eq.6)
        """
        self.step_cpg()
        p_centroid = self.map_foot_trajectory_omnidirectional(default_feet_pos_body, dir_angle_rad=dir_angle_rad, ramp=ramp)

        # Relative offset (6, 3)
        p_rel = p_centroid - self.mount_positions

        # Negative yaw rotation
        p_local_batch = np.einsum('ijk,ik->ij', self.rot_z_inv, p_rel)

        # Solve IK per leg 
        joint_targets = np.zeros((self.num_legs, 3))
        for i in range(self.num_legs):
            joint_targets[i] = self.inverse_kinematics(p_local_batch[i], self.L1, self.L2, self.L3)

        return joint_targets