import numpy as np

# Link lengths: Coxa, Femur, Tibia (in meters)
SILVER2_LINKS = (0.1068, 0.2232, 0.276)

# Leg mount transforms in body centroid frame (X forward, Y left, Z up)
SILVER2_MOUNTS = {
    'L0': {'pos': [0.1944,  -0.1176, -0.0336], 'yaw':  0.0},
    'L1': {'pos': [0.1944,  0.018, -0.0336], 'yaw':  0.0},
    'L2': {'pos': [0.1944,  0.1524, -0.0336], 'yaw':  0.0},
    'L3': {'pos': [-0.1944, -0.1176, -0.0336], 'yaw': np.pi},
    'L4': {'pos': [-0.1944, 0.018, -0.0336], 'yaw': np.pi},
    'L5': {'pos': [-0.1944, 0.1524, -0.0336], 'yaw': np.pi},
}

# Nominal ground contact positions in body frame
SILVER2_DEFAULT_FEET_BODY = np.array([
    [ 0.39852845, -0.32172845, -0.1507235 ],
    [ 0.48308122,  0.018,      -0.1507235 ],
    [ 0.39852845,  0.35652845, -0.1507235 ],
    [-0.39852845, -0.32172845, -0.1507235 ],
    [-0.48308122,  0.018,      -0.1507235 ],
    [-0.39852845,  0.35652845, -0.1507235 ]
])

# Verified standing angles from HAL stability test
SILVER2_STANDING_ANGLES_DEG = np.array([
    [-45.0, -45.0, 130.0],
    [  0.0, -45.0, 130.0],
    [ 45.0, -45.0, 130.0],
    [ 45.0, -45.0, 130.0],
    [  0.0, -45.0, 130.0],
    [-45.0, -45.0, 130.0],
])

SILVER2_DIRECTION_MAP = {
    # Cardinal directions
    "forward":         0.0,
    "right":           np.pi / 2.0,
    "backward":        np.pi,
    "left":           -np.pi / 2.0,

    # Diagonals
    "forward_right":   np.pi / 4.0,
    "backward_right":  3.0 * np.pi / 4.0,
    "backward_left":  -3.0 * np.pi / 4.0,
    "forward_left":   -np.pi / 4.0,
}

# Explicit Stance Parameters
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

# Dynamixel XM430 Drive Parameters (Currently Terrestrial)
K_P = 800.0 # Realistic Kp = 35.0
K_D = 25.0 # Realistic Kd = 1.5
MAX_TORQUE = 500.0 # Realistic Max. Torque = 4.8
DEFAULT_DRIVES = {
    "coxa":  {"stiffness": K_P, "damping": K_D, "max_force": MAX_TORQUE},
    "femur": {"stiffness": K_P, "damping": K_D, "max_force": MAX_TORQUE},
    "tibia": {"stiffness": K_P, "damping": K_D, "max_force": MAX_TORQUE},
}