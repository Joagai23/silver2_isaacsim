# pyright: reportInvalidTypeForm = false
import warp as wp

@wp.func
def wp_inverse_kinematics(
    feet_pos_local: wp.vec3,
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
    px = feet_pos_local[0]
    py = feet_pos_local[1]
    pz = feet_pos_local[2]

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
def cpg_substep_kernel_vec(
    state_in: wp.array(dtype=wp.vec2),
    state_out: wp.array(dtype=wp.vec2),
    coupling_diff: wp.array(ndim=2, dtype=wp.vec2),
    contact_state: wp.array(dtype=wp.int32),
    num_legs: int,
    dt_sub: float,
    alpha: float,
    mu: float,
    b_vec: wp.array(dtype=wp.float32),
    omega_swing_vec: wp.array(dtype=wp.float32),
    delta_omega_vec: wp.array(dtype=wp.float32),
    coupling_weight_vec: wp.array(dtype=wp.float32),
    enable_phase_freeze: int,
    epsilon_window: float
):
    """
    Vectorized Hopf ODE integration across N_envs * 6 oscillators.
    Diffusive coupling loops strictly within each robot's own 6-leg block.
    Supports sensory phase freezing across all parallel environments.
    """
    k = wp.tid()
    env_id = k // num_legs
    leg_id = k % num_legs
    env_offset = env_id * num_legs

    st_i = state_in[k]
    xi = st_i[0]
    yi = st_i[1]
    r2 = xi * xi + yi * yi

    # Environment-specific parameters
    b = b_vec[env_id]
    omega_swing = omega_swing_vec[env_id]
    delta_omega = delta_omega_vec[env_id]
    coupling_weight = coupling_weight_vec[env_id]

    # Sigmoid transition for stance/swing dual-frequency evaluation (Eq. 2 [3])
    sig_arg = wp.clamp(b * yi, -50.0, 50.0)
    sigma = 1.0 / (1.0 + wp.exp(-sig_arg))
    omega_i = omega_swing + sigma * delta_omega

    # Phase Freezing: Case B
    gamma_i = float(1.0)
    if enable_phase_freeze == 1:
        c_i = contact_state[k]
        if c_i == 0 and yi >= float(0.0) and yi <= epsilon_window:
            gamma_i = float(0.0)

    omega_eff = gamma_i * omega_i

    # Diffusive coupling across all peer legs (j != leg_id)
    c_x = float(0.0)
    c_y = float(0.0)
    for j in range(num_legs):
        if j != leg_id:
            peer_idx = env_offset + j
            st_j = state_in[peer_idx]
            xj = st_j[0]
            yj = st_j[1]

            diff = coupling_diff[leg_id, j]
            cos_d = diff[0]
            sin_d = diff[1]

            x_rot = xj * cos_d - yj * sin_d
            y_rot = xj * sin_d + yj * cos_d

            c_x += (x_rot - xi)
            c_y += (y_rot - yi)

    coupling_x = coupling_weight * c_x
    coupling_y = coupling_weight * c_y

    # Continuous Hopf nonlinear derivatives (Eq. 3 [3]) + Modulated Angular Speed
    dx = alpha * (mu - r2) * xi - omega_eff * yi + coupling_x
    dy = alpha * (mu - r2) * yi + omega_eff * xi + coupling_y

    # Forward Euler sub-step update
    state_out[k] = wp.vec2(xi + dx * dt_sub, yi + dy * dt_sub)

@wp.kernel
def apply_phase_resetting_kernel_vec(
    state: wp.array(dtype=wp.vec2),
    contact_forces: wp.array(dtype=wp.vec3),
    contact_state: wp.array(dtype=wp.int32),
    radius: float,
    f_touch_vec: wp.array(dtype=wp.float32),
    f_release_vec: wp.array(dtype=wp.float32),
    epsilon_y: float
):
    """
    Vectorized Schmitt trigger & phase reset across (N_envs * 6) threads.
    """
    k = wp.tid()
    env_id = k // 6

    # Genome thresholds for this environment instance
    f_touch = f_touch_vec[env_id]
    f_release = f_release_vec[env_id]

    # Normal contact force magnitude
    force_vec = contact_forces[k]
    fn = wp.abs(force_vec[2])

    # Schmitt Trigger Hysteresis Filter
    prev_c = contact_state[k]
    curr_c = prev_c

    if fn >= f_touch:
        curr_c = 1
    elif fn <= f_release:
        curr_c = 0

    contact_state[k] = curr_c

    # Read current oscillator state
    st = state[k]
    xi = st[0]
    yi = st[1]

    # Early Ground Contact: Case A
    if curr_c == 1 and yi < -epsilon_y and xi > float(0.0):
        state[k] = wp.vec2(radius, float(0.0))

@wp.kernel
def map_foot_trajectory_omnidirectional_kernel_vec(
    state: wp.array(dtype=wp.vec2),
    default_feet_pos_body: wp.array(dtype=wp.vec3),
    out_feet_pos: wp.array(dtype=wp.vec3),
    k1_vec: wp.array(dtype=wp.float32),
    k2_vec: wp.array(dtype=wp.float32),
    k3_vec: wp.array(dtype=wp.float32),
    b1_vec: wp.array(dtype=wp.float32),
    b2_vec: wp.array(dtype=wp.float32),
    l2_vec: wp.array(dtype=wp.float32),
    dir_angle_rad: float,
    stride_forward: float,
    stride_lateral: float,
    yaw_rate: float,
    yaw_gain: float,
    ramp: float
):
    k = wp.tid()
    env_id = k // 6
    leg_id = k % 6

    k1 = k1_vec[env_id]
    k2 = k2_vec[env_id]
    k3 = k3_vec[env_id]
    b1 = b1_vec[env_id]
    b2 = b2_vec[env_id]
    l2 = l2_vec[env_id]

    # Read oscillator phase state [x_i, y_i]
    st = state[k]
    xi = st[0]
    yi = st[1]

    # Stance/swing vertical profile modulation (Eq. 8 [1])
    x_tilde = k1 * xi
    y_tilde = float(0.0)
    if yi >= float(0.0):
        y_tilde = k2 * yi + b1
    else:
        y_tilde = k3 * yi + b2

    # Nominal foot position in centroid frame: [Axis 0: Lat, Axis 1: Fwd, Axis 2: Up]
    p_def = default_feet_pos_body[leg_id]
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

    out_feet_pos[k] = wp.vec3(p_x, p_y, p_z)

@wp.kernel
def centroid_to_leg_frame_kernel_vec(
    feet_pos_centroid: wp.array(dtype=wp.vec3),
    mount_positions: wp.array(dtype=wp.vec3),
    rot_z_inv: wp.array(dtype=wp.mat33),
    feet_pos_local: wp.array(dtype=wp.vec3)
):
    k = wp.tid()
    leg_id = k % 6

    # Relative displacement: p_rel = p_centroid - p_mount
    p_rel = feet_pos_centroid[k] - mount_positions[leg_id]

    # Rotate into local leg frame: p_local = R_z(-yaw) * p_rel
    feet_pos_local[k] = rot_z_inv[leg_id] * p_rel

@wp.kernel
def inverse_kinematics_kernel_vec(
    feet_pos_local: wp.array(dtype=wp.vec3),
    link_lengths: wp.vec3,
    joint_targets: wp.array(dtype=wp.vec3)
):
    k = wp.tid()
    joint_targets[k] = wp_inverse_kinematics(feet_pos_local[k], link_lengths)