import numpy as np
from numpy_cpg_controller import NumpyHexapodCPGController

def compute_default_feet_body(default_angles_deg, leg_mounts, link_lengths):
    """
    Computes nominal foot positions in the centroid frame (Eq. 2 & Eq. 4).
    
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

# Link lengths: Coxa, Femur, Tibia (in meters)
silver2_links = (0.1068, 0.2232, 0.276)

# Leg mount transforms in body centroid frame (X forward, Y left, Z up)
silver2_mounts = {
    'L0': {'pos': [0.1944,  -0.1176, -0.0336], 'yaw':  0.0},
    'L1': {'pos': [0.1944,  0.018, -0.0336], 'yaw':  0.0},
    'L2': {'pos': [0.1944,  0.1524, -0.0336], 'yaw':  0.0},
    'L3': {'pos': [-0.1944, -0.1176, -0.0336], 'yaw': np.pi},
    'L4': {'pos': [-0.1944, 0.018, -0.0336], 'yaw': np.pi},
    'L5': {'pos': [-0.1944, 0.1524, -0.0336], 'yaw': np.pi},
}

silver2_default_angles = np.array([
    [-45, -45, 130], # L0
    [0, -45, 130], # L1
    [45, -45, 130], # L2
    [45, -45, 130], # L3
    [0, -45, 130], # L4
    [-45, -45, 130], # L5
])

# Nominal ground contact positions in body frame
default_feet_body = np.array([
    [ 0.39852845, -0.32172845, -0.1507235 ],
    [ 0.48308122,  0.018,      -0.1507235 ],
    [ 0.39852845,  0.35652845, -0.1507235 ],
    [-0.39852845, -0.32172845, -0.1507235 ],
    [-0.48308122,  0.018,      -0.1507235 ],
    [-0.39852845,  0.35652845, -0.1507235 ]
])

controller = NumpyHexapodCPGController(silver2_mounts, silver2_links, dt=0.01)

# Step simulation forward for 200 ticks (2 seconds)
trajectory_history = []
for _ in range(200):
    joint_angles = controller.compute_joint_targets(default_feet_body)
    trajectory_history.append(joint_angles)

trajectory_history = np.array(trajectory_history)
print(f"Generated trajectory batch: {trajectory_history.shape} (Ticks, Legs, Joints)")

# Convert to degrees for inspection
trajectory_deg = np.degrees(trajectory_history)
leg_names = list(silver2_mounts.keys())
joint_names = ['Coxa (Yaw)', 'Femur (Pitch)', 'Tibia (Pitch)']

print("\n" + "=" * 62)
print(f"{'Leg':<5} | {'Joint':<14} | {'Min (°)':<10} | {'Max (°)':<10} | {'Range (°)':<10}")
print("=" * 62)

for l_idx, leg in enumerate(leg_names):
    for j_idx, j_name in enumerate(joint_names):
        j_min = np.min(trajectory_deg[:, l_idx, j_idx])
        j_max = np.max(trajectory_deg[:, l_idx, j_idx])
        j_rng = j_max - j_min
        print(f"{leg:<5} | {j_name:<14} | {j_min:10.2f} | {j_max:10.2f} | {j_rng:10.2f}")
    print("-" * 62)

import matplotlib.pyplot as plt

time_axis = np.arange(trajectory_history.shape[0]) * controller.dt
trajectory_deg = np.degrees(trajectory_history)
leg_names = list(silver2_mounts.keys())

fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue: Coxa, Orange: Femur, Green: Tibia
labels = ['Coxa (θ₁)', 'Femur (θ₂)', 'Tibia (θ₃)']

for i, leg in enumerate(leg_names):
    ax = axes[i]
    for j in range(3):
        ax.plot(time_axis, trajectory_deg[:, i, j], label=labels[j], color=colors[j], linewidth=1.5)
    ax.set_title(f"Leg {leg} Joint Angles")
    ax.grid(True, linestyle='--', alpha=0.6)
    if i % 2 == 0:
        ax.set_ylabel("Angle (deg)")
    if i >= 4:
        ax.set_xlabel("Time (s)")

axes[0].legend(loc='upper right')
plt.tight_layout()
plt.savefig("cpg_joint_trajectories.png", dpi=200)
plt.show()