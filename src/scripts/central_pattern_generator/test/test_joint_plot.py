import numpy as np
from numpy_cpg_controller import NumpyHexapodCPGController

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