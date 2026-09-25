import warp as wp

@wp.kernel
def cpg_substep_kernel(
    state_in: wp.array(dtype=wp.vec2),
    state_out: wp.array(dtype=wp.vec2),
    coupling_diff: wp.array2d(dtype=wp.vec2),
    num_legs: int,
    dt_sub: float,
    alpha: float,
    mu: float,
    b: float,
    omega_swing: float,
    delta_omega: float,
    coupling_weight: float
):
    i = wp.tid()

    # Read current state vector [x_i, y_i]
    st_i = state_in[i]
    xi = st_i[0]
    yi = st_i[1]
    r2 = xi * xi + yi * yi

    # Sigmoid transition for stance/swing dual-frequency evaluation (Eq. 2 [3])
    sig_arg = wp.clamp(b * yi, -50.0, 50.0)
    sigma = 1.0 / (1.0 + wp.exp(-sig_arg))
    omega_i = omega_swing + sigma * delta_omega

    # Diffusive coupling across all peer legs (j != i)
    c_x = float(0.0)
    c_y = float(0.0)
    for j in range(num_legs):
        if i != j:
            st_j = state_in[j]
            xj = st_j[0]
            yj = st_j[1]

            # Read precomputed [cos(d_phi), sin(d_phi)]
            diff = coupling_diff[i, j]
            cos_d = diff[0]
            sin_d = diff[1]

            # Rotate peer state into leg i's coordinate frame (Eq. 7 [1])
            x_rot = xj * cos_d - yj * sin_d
            y_rot = xj * sin_d + yj * cos_d

            c_x += (x_rot - xi)
            c_y += (y_rot - yi)

    coupling_x = coupling_weight * c_x
    coupling_y = coupling_weight * c_y

    # Continuous Hopf nonlinear derivatives (Eq. 3 [3])
    dx = alpha * (mu - r2) * xi - omega_i * yi + coupling_x
    dy = alpha * (mu - r2) * yi + omega_i * xi + coupling_y

    # Forward Euler sub-step update
    state_out[i] = wp.vec2(xi + dx * dt_sub, yi + dy * dt_sub)

@wp.kernel
def map_foot_trajectory_omnidirectional_kernel(
    state: wp.array(dtype=wp.vec2),
    default_feet_pos: wp.array(dtype=wp.vec3),
    out_feet_pos: wp.array(dtype=wp.vec3),
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
    i = wp.tid()

    # Read oscillator phase state [x_i, y_i]
    st = state[i]
    x_val = st[0]
    y_val = st[1]

    # Stance/swing vertical profile modulation (Eq. 8 [1])
    x_tilde = k1 * x_val
    y_tilde = float(0.0)
    if y_val >= float(0.0):
        y_tilde = k2 * y_val + b1
    else:
        y_tilde = k3 * y_val + b2

    # Nominal foot position in centroid frame: [Axis 0: Lat, Axis 1: Fwd, Axis 2: Up]
    p_def = default_feet_pos[i]
    pos_lat = p_def[0]
    pos_fwd = p_def[1]
    pos_up = p_def[2]

    # Translational stroke decomposition
    l_lat_trans = -stride_lateral * wp.sin(dir_angle_rad)
    l_fwd_trans = -stride_forward * wp.cos(dir_angle_rad)

    # Superimposed rotational yaw velocity
    l_lat = l_lat_trans - yaw_gain * yaw_rate * pos_fwd
    l_fwd = l_fwd_trans + yaw_gain * yaw_rate * pos_lat

    # Compute displaced Cartesian target (Eq. 9 [1])
    p_x = pos_lat + ramp * l_lat * x_tilde
    p_y = pos_fwd + ramp * l_fwd * x_tilde
    p_z = pos_up - ramp * l2 * y_tilde

    out_feet_pos[i] = wp.vec3(p_x, p_y, p_z)

@wp.func
def wp_inverse_kinematics(
    p_leg_base: wp.vec3,
    link_lengths: wp.vec3
) -> wp.vec3:
    """
    Device-side analytical 3-DOF inverse kinematics for a single leg.
    Executes inlined in GPU registers with zero global memory transactions.

    Args:
        p_leg_base: wp.vec3(px, py, pz) in the local leg base origin frame.
        link_lengths: wp.vec3(L1, L2, L3) segment lengths.

    Returns:
        wp.vec3(theta_1, theta_2, theta_3) in radians.
    """
    px = p_leg_base[0]
    py = p_leg_base[1]
    pz = p_leg_base[2]

    L1 = link_lengths[0]
    L2 = link_lengths[1]
    L3 = link_lengths[2]

    # Joint 1: Coxa - Yaw
    theta_1 = wp.atan2(py, px)
    c1 = wp.cos(theta_1)
    s1 = wp.sin(theta_1)

    # Auxiliary term: gamma_2 (radial distance in femur-tibia plane)
    gamma_2 = px * c1 + py * s1 - L1

    # Joint 3: Tibia - Pitch (Law of Cosines)
    r2_chord = gamma_2 * gamma_2 + pz * pz
    cos_theta_3 = (r2_chord - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    theta_3 = wp.acos(wp.clamp(cos_theta_3, -1.0, 1.0))

    # Joint 2: Femur - Pitch
    chord = wp.sqrt(r2_chord)
    chord_pitch_down = wp.atan2(-pz, gamma_2)
    sin_psi = (L3 * wp.sin(theta_3)) / wp.max(chord, 1e-6)
    psi = wp.asin(wp.clamp(sin_psi, -1.0, 1.0))
    theta_2 = chord_pitch_down - psi

    return wp.vec3(theta_1, theta_2, theta_3)

@wp.kernel
def inverse_kinematics_kernel(
    feet_pos_local: wp.array(dtype=wp.vec3),
    link_lengths: wp.vec3,
    joint_targets: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    joint_targets[i] = wp_inverse_kinematics(feet_pos_local[i], link_lengths)

@wp.func
def wp_forward_kinematics(
    angles_rad: wp.vec3,
    link_lengths: wp.vec3
) -> wp.vec3:
    """
    Computes 3-DOF leg forward kinematics in the local coxa base frame.
    
    Args:
        angles_rad: wp.vec3(theta1, theta2, theta3) in radians.
        link_lengths: wp.vec3(L1, L2, L3) segment lengths.
        
    Returns:
        wp.vec3(x_local, y_local, z_local) in meters.
    """
    theta1 = angles_rad[0]
    theta2 = angles_rad[1]
    theta3 = angles_rad[2]

    L1 = link_lengths[0]
    L2 = link_lengths[1]
    L3 = link_lengths[2]

    c1 = wp.cos(theta1)
    s1 = wp.sin(theta1)
    c2 = wp.cos(theta2)
    s2 = wp.sin(theta2)
    c23 = wp.cos(theta2 + theta3)
    s23 = wp.sin(theta2 + theta3)

    r = L1 + L2 * c2 + L3 * c23
    x_local = r * c1
    y_local = r * s1
    z_local = -(L2 * s2 + L3 * s23)

    return wp.vec3(x_local, y_local, z_local)

@wp.kernel
def compute_default_feet_body_kernel(
    angles_rad: wp.array(dtype=wp.vec3),
    mount_positions: wp.array(dtype=wp.vec3),
    rot_z_inv: wp.array(dtype=wp.mat33),
    link_lengths: wp.vec3,
    default_feet_body: wp.array(dtype=wp.vec3)
):
    i = wp.tid()

    # Local Forward Kinematics
    p_local = wp_forward_kinematics(angles_rad[i], link_lengths)

    # Centroid Frame Transformation: R_z(yaw) * p_local + p_mount
    # Since rot_z_inv[i] = R_z(-yaw), R_z(yaw) = transpose(rot_z_inv[i])
    rot_z = wp.transpose(rot_z_inv[i])
    p_centroid = rot_z * p_local + mount_positions[i]

    default_feet_body[i] = p_centroid

@wp.kernel
def centroid_to_leg_frame_kernel(
    feet_pos_centroid: wp.array(dtype=wp.vec3),
    mount_positions: wp.array(dtype=wp.vec3),
    rot_z_inv: wp.array(dtype=wp.mat33),
    feet_pos_local: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    # Relative displacement: p_rel = p_centroid - p_mount
    p_rel = feet_pos_centroid[i] - mount_positions[i]

    # Rotate into local leg frame: p_local = R_z(-yaw) * p_rel
    feet_pos_local[i] = rot_z_inv[i] * p_rel