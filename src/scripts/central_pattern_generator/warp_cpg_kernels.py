"""
GPU-Native NVIDIA Warp Kernels for Hexapod CPG Locomotion & Analytical IK.
Zero-copy execution pipeline designed for NVIDIA Isaac Sim & Newton physics backend.
"""

import warp as wp

# Warp Array Type Descriptors
WpFloatArray = wp.array(dtype=wp.float32)
WpIntArray = wp.array(dtype=wp.int32)
WpFloatArray2D = wp.array2d(dtype=wp.float32)
WpFloatArray3D = wp.array3d(dtype=wp.float32)

# Kernel 1: Coupled Hopf Oscillator Forward Euler Substep
@wp.kernel
def cpg_hopf_substep_kernel(
    x_in: WpFloatArray2D, # type: ignore
    y_in: WpFloatArray2D, # type: ignore
    x_out: WpFloatArray2D, # type: ignore
    y_out: WpFloatArray2D, # type: ignore
    cos_diff: WpFloatArray2D, # type: ignore
    sin_diff: WpFloatArray2D, # type: ignore
    omega_swing: float,
    delta_omega: float,
    coupling_weight: float,
    alpha: float,
    mu: float,
    b: float,
    dt_sub: float
):
    """
    Parallel integration step for 6 coupled Hopf oscillators per robot.
    Launched with 2D grid: dim=(num_envs, 6)
    """
    env_id, i = wp.tid()

    xi = x_in[env_id, i]
    yi = y_in[env_id, i]
    r2 = xi * xi + yi * yi

    # Dual-frequency modulation via logistic sigmoid (Eq. 2 [3])
    sigma = 1.0 / (1.0 + wp.exp(-wp.clamp(b * yi, -50.0, 50.0)))
    omega_i = omega_swing + sigma * delta_omega

    # Diffusive phase coupling with rotated neighbor projections
    cx = float(0.0)
    cy = float(0.0)
    for j in range(6):
        if i != j:
            xj = x_in[env_id, j]
            yj = y_in[env_id, j]
            cd = cos_diff[i, j]
            sd = sin_diff[i, j]
            
            # 2D phase rotation
            x_rot = xj * cd - yj * sd
            y_rot = xj * sd + yj * cd
            
            cx += (x_rot - xi)
            cy += (y_rot - yi)

    # Limit cycle ODE integration (Eq. 3 [3])
    dx = alpha * (mu - r2) * xi - omega_i * yi + coupling_weight * cx
    dy = alpha * (mu - r2) * yi + omega_i * xi + coupling_weight * cy

    x_out[env_id, i] = xi + dx * dt_sub
    y_out[env_id, i] = yi + dy * dt_sub

# Kernel 2: Fused Trajectory Mapping, Transform, IK, and DOF Remap
@wp.kernel
def cpg_kinematics_and_ik_kernel(
    x: WpFloatArray2D, # type: ignore
    y: WpFloatArray2D, # type: ignore
    default_feet_pos: WpFloatArray2D,   # type: ignore (6, 3) Layout: [Lateral(Y), Long(X), Up(Z)] 
    mount_positions: WpFloatArray2D,    # type: ignore (6, 3)
    rot_z_inv: WpFloatArray3D,          # type: ignore (6, 3, 3)
    canonical_to_newton: WpIntArray,    # type: ignore (18,)
    joint_targets: WpFloatArray2D,      # type: ignore (num_envs, 18) Output written in Newton order
    L1: float,
    L2: float,
    L3: float,
    k1: float,
    k2: float,
    k3: float,
    b1: float,
    b2: float,
    l2: float,
    dir_angle_rad: float,
    stride_forward: float,
    stride_lateral: float,
    yaw_rate: float,
    yaw_gain: float,
    ramp: float
):
    """
    Fuses operational-space trajectory calculation, frame rotation, analytical 3-DOF IK,
    and direct DOF index remapping into a single CUDA execution pass per leg.
    Launched with 2D grid: dim=(num_envs, 6)
    """
    env_id, i = wp.tid()

    xi = x[env_id, i]
    yi = y[env_id, i]

    # 1. Operational-space trajectory shaping
    x_tilde = k1 * xi
    y_tilde = float(0.0)
    if yi >= 0.0:
        y_tilde = k2 * yi + b1
    else:
        y_tilde = k3 * yi + b2

    pos_lat = default_feet_pos[i, 0]  # Axis 0: Lateral (Y)
    pos_fwd = default_feet_pos[i, 1]  # Axis 1: Longitudinal (X)
    pos_up  = default_feet_pos[i, 2]  # Axis 2: Up (Z)

    # Base translational strokes
    l_lat_trans = -stride_lateral * wp.sin(dir_angle_rad)
    l_fwd_trans = -stride_forward * wp.cos(dir_angle_rad)

    # Rotational twist superposition
    l_lat = l_lat_trans - yaw_gain * yaw_rate * pos_fwd
    l_fwd = l_fwd_trans + yaw_gain * yaw_rate * pos_lat

    # Target foot coordinates in centroid frame
    p_cent_lat = pos_lat + ramp * l_lat * x_tilde
    p_cent_fwd = pos_fwd + ramp * l_fwd * x_tilde
    p_cent_up  = pos_up  - ramp * l2 * y_tilde

    # 2. Centroid-to-Leg Frame Transform: p_local = rot_z_inv * (p_cent - mount)
    rel_0 = p_cent_lat - mount_positions[i, 0]
    rel_1 = p_cent_fwd - mount_positions[i, 1]
    rel_2 = p_cent_up  - mount_positions[i, 2]

    px = rot_z_inv[i, 0, 0] * rel_0 + rot_z_inv[i, 0, 1] * rel_1 + rot_z_inv[i, 0, 2] * rel_2
    py = rot_z_inv[i, 1, 0] * rel_0 + rot_z_inv[i, 1, 1] * rel_1 + rot_z_inv[i, 1, 2] * rel_2
    pz = rot_z_inv[i, 2, 0] * rel_0 + rot_z_inv[i, 2, 1] * rel_1 + rot_z_inv[i, 2, 2] * rel_2

    # 3. Analytical 3-DOF Inverse Kinematics
    # Joint 1: Coxa (Yaw)
    theta_1 = wp.atan2(py, px)
    c1 = wp.cos(theta_1)
    s1 = wp.sin(theta_1)

    # Auxiliary reach parameter gamma_2
    gamma_2 = px * c1 + py * s1 - L1

    # Joint 3: Tibia (Pitch)
    cos_theta_3 = (gamma_2 * gamma_2 + pz * pz - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    cos_theta_3_clamped = wp.clamp(cos_theta_3, -1.0, 1.0)
    theta_3 = wp.acos(cos_theta_3_clamped)

    # Joint 2: Femur (Pitch)
    chord = wp.sqrt(gamma_2 * gamma_2 + pz * pz)
    chord_pitch_down = wp.atan2(-pz, gamma_2)
    sin_psi = (L3 * wp.sin(theta_3)) / wp.max(chord, 1e-6)
    sin_psi_clamped = wp.clamp(sin_psi, -1.0, 1.0)
    psi = wp.asin(sin_psi_clamped)
    theta_2 = chord_pitch_down - psi

    # 4. In-Place Remap Direct to Newton Target Buffer
    can_idx_0 = i * 3 + 0
    can_idx_1 = i * 3 + 1
    can_idx_2 = i * 3 + 2

    # Direct scatter into Newton DOF ordering
    joint_targets[env_id, canonical_to_newton[can_idx_0]] = theta_1
    joint_targets[env_id, canonical_to_newton[can_idx_1]] = theta_2
    joint_targets[env_id, canonical_to_newton[can_idx_2]] = theta_3