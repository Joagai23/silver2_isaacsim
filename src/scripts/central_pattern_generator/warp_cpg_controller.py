import warp as wp
import numpy as np
from typing import Dict, Tuple, Any

# Ensure Warp runtime is initialized
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
        self.mount_positions = wp.array(
            mount_pos_host, 
            dtype=wp.vec3, 
            device=self.device
        )
        self.rot_z_inv = wp.array(
            rot_z_inv_host, 
            dtype=wp.mat33, 
            device=self.device
        )

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

        # ----------------------------------------------------------------------
        # 6. Initialize Gait State (Deferred to set_gait analysis)
        # ----------------------------------------------------------------------
        # self.set_gait(gait, reset_state=True)