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