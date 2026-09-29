# pyright: reportInvalidTypeForm = false
import warp as wp

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

    fn = wp.abs(contact_forces[k][2])

    curr_c = contact_state[k]
    if fn >= f_touch:
        curr_c = 1
    elif fn <= f_release:
        curr_c = 0
    contact_state[k] = curr_c

    st = state[k]
    xi = st[0]
    yi = st[1]

    # Case A: Truncate swing on early ground contact
    if curr_c == 1:
        if yi < -epsilon_y and xi > float(0.0):
            state[k] = wp.vec2(radius, float(0.0))

@wp.kernel
def cpg_substep_kernel_vec(
    state_in: wp.array(dtype=wp.vec2),
    state_out: wp.array(dtype=wp.vec2),
    coupling_diff: wp.array(ndim=2, dtype=wp.vec2),  # Shared 6x6 canonical phase differences
    num_legs: int,
    dt_sub: float,
    alpha: float,
    mu: float,
    b_vec: wp.array(dtype=wp.float32),
    omega_swing_vec: wp.array(dtype=wp.float32),
    delta_omega_vec: wp.array(dtype=wp.float32),
    coupling_weight_vec: wp.array(dtype=wp.float32)
):
    """
    Vectorized Hopf ODE integration across N_envs * 6 oscillators.
    Diffusive coupling loops strictly within the robot's own 6-leg block.
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

    # Smooth stance/swing sigmoid transition
    sig_arg = wp.clamp(b * yi, -50.0, 50.0)
    sigma = 1.0 / (1.0 + wp.exp(-sig_arg))
    omega_i = omega_swing + sigma * delta_omega

    # Intra-robot diffusive coupling across peer legs (j != leg_id)
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

    dx = alpha * (mu - r2) * xi - omega_i * yi + coupling_x
    dy = alpha * (mu - r2) * yi + omega_i * xi + coupling_y

    state_out[k] = wp.vec2(xi + dx * dt_sub, yi + dy * dt_sub)

@wp.kernel
def map_foot_trajectory_omnidirectional_kernel_vec(
    state: wp.array(dtype=wp.vec2),
    default_feet_pos_body: wp.array(dtype=wp.vec3),  # (6,) canonical baseline
    feet_pos_centroid: wp.array(dtype=wp.vec3),      # (N_envs * 6,)
    k1_vec: wp.array(dtype=wp.float32),
    k2_vec: wp.array(dtype=wp.float32),
    k3_vec: wp.array(dtype=wp.float32),
    l1_vec: wp.array(dtype=wp.float32),
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
    l1 = l1_vec[env_id]
    l2 = l2_vec[env_id]

    st = state[k]
    xi = st[0]
    yi = st[1]

    # Stance: yi >= 0 -> y_tilde = 0; Swing: yi < 0 -> y_tilde = yi
    y_tilde = float(0.0)
    if yi < float(0.0):
        y_tilde = yi

    # Effective vertical clearance
    h = l2 - l1
    delta_x = k1 * xi * stride_forward * ramp
    delta_y = k2 * xi * stride_lateral * ramp
    delta_z = (l1 + k3 * y_tilde * (h / 10.0)) * ramp

    # Yaw steering bias
    yaw_bias = yaw_rate * yaw_gain * xi * ramp
    if leg_id == 0 or leg_id == 1 or leg_id == 2:
        delta_x -= yaw_bias
    else:
        delta_x += yaw_bias

    cos_dir = wp.cos(dir_angle_rad)
    sin_dir = wp.sin(dir_angle_rad)

    dx_rot = delta_x * cos_dir - delta_y * sin_dir
    dy_rot = delta_x * sin_dir + delta_y * cos_dir

    def_p = default_feet_pos_body[leg_id]
    feet_pos_centroid[k] = wp.vec3(
        def_p[0] + dx_rot,
        def_p[1] + dy_rot,
        def_p[2] + delta_z
    )

@wp.kernel
def centroid_to_leg_frame_kernel_vec(
    feet_pos_centroid: wp.array(dtype=wp.vec3),
    mount_positions: wp.array(dtype=wp.vec3),     # (6,)
    rot_z_inv: wp.array(dtype=wp.mat33),           # (6,)
    feet_pos_local: wp.array(dtype=wp.vec3)        # (N_envs * 6,)
):
    k = wp.tid()
    leg_id = k % 6

    p_c = feet_pos_centroid[k]
    p_m = mount_positions[leg_id]
    r_inv = rot_z_inv[leg_id]

    diff = p_c - p_m
    feet_pos_local[k] = r_inv * diff

@wp.kernel
def inverse_kinematics_kernel_vec(
    feet_pos_local: wp.array(dtype=wp.vec3),
    link_lengths: wp.vec3,
    joint_targets: wp.array(dtype=wp.vec3)        # (N_envs * 6,) -> [q_coxa, q_femur, q_tibia]
):
    k = wp.tid()
    p = feet_pos_local[k]
    px = p[0]
    py = p[1]
    pz = p[2]

    l1 = link_lengths[0]
    l2 = link_lengths[1]
    l3 = link_lengths[2]

    # 1. Coxa Angle (Planar sweep)
    q_coxa = wp.atan2(py, px)

    # 2. Planar projection in femur-tibia frame
    r = wp.sqrt(px * px + py * py)
    r_prime = r - l1
    z_prime = pz

    d2 = r_prime * r_prime + z_prime * z_prime
    d = wp.sqrt(d2)

    # 3. Tibia Angle (Law of Cosines)
    cos_tibia = (d2 - l2 * l2 - l3 * l3) / (2.0 * l2 * l3)
    cos_tibia_clamped = wp.clamp(cos_tibia, -1.0, 1.0)
    sin_tibia = -wp.sqrt(wp.max(0.0, 1.0 - cos_tibia_clamped * cos_tibia_clamped))
    q_tibia = wp.atan2(sin_tibia, cos_tibia_clamped)

    # 4. Femur Angle
    alpha = wp.atan2(-z_prime, r_prime)
    cos_beta = (l2 * l2 + d2 - l3 * l3) / (2.0 * l2 * d)
    cos_beta_clamped = wp.clamp(cos_beta, -1.0, 1.0)
    beta = wp.acos(cos_beta_clamped)
    q_femur = alpha - beta

    joint_targets[k] = wp.vec3(q_coxa, q_femur, q_tibia)