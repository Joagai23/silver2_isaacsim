# pyright: reportInvalidTypeForm=false
import warp as wp
"""
Set of static hydrodynamic functions optimized with Warp.
"""
@wp.func
def calculate_submersion_and_cob(
    position: wp.vec3,
    orientation: wp.quat,
    keypoints: wp.array(dtype=wp.vec3)
):
    """
    Calculates a continuous submersion ratio based on the highest and lowest points
    of the object, preventing the "sticking" effect at the water surface.
    """
    # Initialize limits
    z_min = wp.inf
    z_max = -wp.inf

    # Initialize CoB accumulators
    sum_pos = wp.vec3(0.0, 0.0, 0.0)
    submerged_count = 0.0

    # Iterate over keypoins
    n_points = keypoints.shape[0]
    for i in range(n_points):
        # Transform local point to world space
        # World = Position + (Rotation * Local)
        local_point = keypoints[i]
        world_point = position + wp.quat_rotate(orientation, local_point)

        # Track Min/Max for ratio
        z_point = world_point[2]
        if z_point < z_min: z_min = z_point
        if z_point > z_max: z_max = z_point

        # Accumulate of CoB if underwater (z < 0)
        if z_point < 0.0:
            sum_pos += world_point
            submerged_count += 1.0

    # Calculate submersion ratio
    ratio = 0.0
    # Fully out
    if z_min >= 0.0: ratio = 0.0
    # Fully in
    elif z_max <= 0.0: ratio = 1.0
    # Partial
    else:
        total_height = z_max - z_min
        if total_height < 1.0e-6:
            if z_min < 0.0: ratio = 1.0
            else: ratio = 0.0
        else:
            ratio = -z_min / total_height
            if ratio > 1.0: ratio = 1.0

    # Center of Buoyancy
    cob = position
    if submerged_count > 0.0:
        cob = sum_pos / submerged_count

    return ratio, cob

@wp.func
def calculate_pressure_and_area(
    speed: wp.vec3,
    velocity_direction: wp.vec3,
    position: wp.vec3,
    orientation: wp.quat,
    center_of_buoyancy: wp.vec3,
    face_normals: wp.array(dtype=wp.vec3),
    face_centers: wp.array(dtype=wp.vec3),
    face_areas: wp.array(dtype=float)
):
    """
    Calculates Center of Pressure and Total Projected Area.
    Checks Z-height to ensure only underwater faces contribute to drag.
    """
    # Defaults
    cop = center_of_buoyancy
    total_projected_area = 0.0

    if speed > 1.0e-6:
        cop_weighted_sum = wp.vec3(0.0, 0.0, 0.0)
        n_faces = face_normals.shape[0]
        for i in range(n_faces):
            # Transform to world space
            # World Normal = Rotation * Local Normal
            local_normal = face_normals[i]
            world_normal = wp.quat_rotate(orientation, local_normal)

            # World Center = Position + Rotation * Local Center
            local_center = face_centers[i]
            world_center = position + wp.quat_rotate(orientation, local_center)

            # Alignment Check
            dt = wp.dot(world_normal, velocity_direction)
            alignment = -dt

            if alignment > 0.0:
                # Check if face center is underwater
                if world_center[2] < 0.0:
                    area_val = alignment * face_areas[i]
                    total_projected_area += area_val
                    cop_weighted_sum += world_center * area_val

        if total_projected_area > 1.0e-6:
            cop = cop_weighted_sum / total_projected_area

        return cop, total_projected_area

@wp.func
def calculate_hybrid_drag(
    speed: wp.vec3,
    velocity_direction: wp.vec3,
    linear_velocity: wp.vec3,
    angular_velocity: wp.vec3,
    sub_ratio: float,
    total_projected_area: float,
    water_density: float,
    volume: float,
    linear_drag: float,
    angular_drag: float,
    linear_damp: float,
    angular_damp: float
):
    """
    Calculates a hybrid drag force that combines a simple linear model (for low-speed stability)
    and a physically-based quadratic model (for high-speed realism).
    """
    # --- Linear Drag ---
    # Quadratic
    quad_drag_force = wp.vec3(0.0)
    if speed > 1.0e-6:
        drag_mag = 0.5 * water_density * (speed * speed) * linear_drag * total_projected_area
        quad_drag_force = -drag_mag * velocity_direction
    # Damping
    LOW_SPEED_THRESHOLD = 0.2
    lin_damp_scale = 1.0
    if speed < LOW_SPEED_THRESHOLD:
        lin_damp_scale = speed / LOW_SPEED_THRESHOLD
    linear_damping_force = -linear_damp * linear_velocity * lin_damp_scale
    # Combined Linear Drag
    total_drag_force = (quad_drag_force + linear_damping_force) * sub_ratio

    # --- Angular Drag ---
    angular_speed = wp.length(angular_velocity)
    # Quadratic
    quad_drag_torque = wp.vec3(0.0)
    if angular_speed > 1.0e-6:
        angular_direction = angular_velocity / angular_speed
        ang_mag = 0.5 * water_density * (angular_speed * angular_speed) * angular_drag * volume
        quad_drag_torque = -ang_mag * angular_direction
    # Damping
    ang_damp_scale = 1.0
    if angular_speed < LOW_SPEED_THRESHOLD:
        ang_damp_scale = angular_speed / LOW_SPEED_THRESHOLD
    linear_ang_torque = -angular_damp * angular_velocity * ang_damp_scale
    total_drag_torque = (quad_drag_torque + linear_ang_torque) * sub_ratio

    return total_drag_force, total_drag_torque

@wp.func
def calculate_lift(
    speed: wp.vec3,
    velocity_direction: wp.vec3,
    orientation: wp.quat,
    sub_ratio: float,
    projected_area: float,
    water_density: float,
    lift_coeff_param: float
):
    """
    Calculates a simplified lift force based on the direction of incoming fluid (object's velocity) and the cube's
    "up" direction.
    """
    if speed > 1.0e-6 and sub_ratio > 1.0e-9:
        # Extract "Up" vector from quaternion
        up_vector = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))

        # Angle of attack
        dt = -(wp.dot(up_vector, velocity_direction))
        if dt > 1.0: dt = 1.0
        if dt < -1.0: dt = -1.0
        angle_of_attack = wp.asin(dt)

        # Lift Magnitude
        lift_coefficient = wp.sin(2.0 * angle_of_attack)
        lift_magnitude = 0.5 * water_density * (speed * speed) * lift_coefficient * projected_area * lift_coeff_param

        # Lift Direction
        lift_axis = wp.cross(velocity_direction, up_vector)
        norm_lift_axis = wp.length(lift_axis)
        if norm_lift_axis > 1.0e-6:
            lift_axis = lift_axis / norm_lift_axis
            lift_dir = wp.cross(lift_axis, velocity_direction)
        
        lift_force = lift_magnitude * lift_dir * sub_ratio
        return lift_force
    
@wp.func
def calculate_added_mass(
    linear_accel: wp.vec3,
    angular_accel: wp.vec3,
    orientation: wp.quat,
    sub_ratio: float,
    added_mass_matrix: wp.array(dtype=wp.vec3) # [0] is linear diagonal, [1] is angular diagonal
):
    added_mass_force = wp.vec3(0.0)
    added_mass_torque = wp.vec3(0.0)

    if sub_ratio > 1.0e-9:
        # Rotate world-space accelerations into the object's local (body) frame
        linear_accel_local = wp.quat_rotate(orientation, linear_accel)
        angular_accel_local = wp.quat_rotate(orientation, angular_accel)

        # Construct 6D Vector
        added_mass_linear_diagonal = added_mass_matrix[0]
        added_mass_angular_diagonal = added_mass_matrix[1]

        # Calculate forces and torques in the local frame: Fi = -Mij * Aj
        force_local = -1.0 * wp.cw_mul(added_mass_linear_diagonal, linear_accel_local)
        torque_local = -1.0 * wp.cw_mul(added_mass_angular_diagonal, angular_accel_local)

        # Rotate the calculated forces and torques back into the world frame to be applied
        added_mass_force = wp.quat_rotate(orientation, force_local) * sub_ratio
        added_mass_torque = wp.quat_rotate(orientation, torque_local) * sub_ratio

    return added_mass_force, added_mass_torque

@wp.kernel
def solve_hydrodynamics_kernel(
    # Inputs
    wp_position: wp.array(dtype=wp.vec3),
    wp_orientation: wp.array(dtype=wp.quat),
    wp_linear_velocity: wp.array(dtype=wp.vec3),
    wp_angular_velocity: wp.array(dtype=wp.vec3),
    wp_linear_acceleration: wp.array(dtype=wp.vec3),
    wp_angular_acceleration: wp.array(dtype=wp.vec3),
    # Constants
    wp_keypoints: wp.array(dtype=wp.vec3),
    wp_normals: wp.array(dtype=wp.vec3),
    wp_centers: wp.array(dtype=wp.vec3),
    wp_areas: wp.array(dtype=float),
    wp_params: wp.array(dtype=float),
    wp_added_mass: wp.array(dtype=wp.vec3), # [lin_coeff, ang_coeff]
    # Outputs
    out_buoyancy_force: wp.array(dtype=wp.vec3),
    out_drag_force: wp.array(dtype=wp.vec3),
    out_lift_force: wp.array(dtype=wp.vec3),
    out_drag_torque: wp.array(dtype=wp.vec3),
    out_added_mass_force: wp.array(dtype=wp.vec3),
    out_added_mass_torque: wp.array(dtype=wp.vec3),
    out_center_of_buoyancy: wp.array(dtype=wp.vec3),
    out_center_of_pressure: wp.array(dtype=wp.vec3)
):
    # Thread Indexing
    tid = wp.tid()

    # Unpack Physics
    position = wp_position[tid]
    orientation = wp_orientation[tid]
    linear_velocity = wp_linear_velocity[tid]
    angular_velocity = wp_angular_velocity[tid]
    linear_acceleration = wp_linear_acceleration[tid]
    angular_acceleration = wp_angular_acceleration[tid]

    # Unpack Parameters [density, gravity, lin_drag, ang_drag, lin_damp, ang_damp, lift, volume]
    water_density = wp_params[0]
    gravity = wp_params[1]
    linear_drag = wp_params[2]
    angular_drag = wp_params[3]
    linear_damp = wp_params[4]
    angular_damp = wp_params[5]
    lift = wp_params[6]
    volume = wp_params[7]

    # Submersion and Buoyancy
    sub_ratio, center_of_buoyancy = calculate_submersion_and_cob(position, orientation, wp_keypoints)

    # Initialize Outputs
    buoyancy_force = wp.vec3(0.0)
    drag_force = wp.vec3(0.0)
    lift_force = wp.vec3(0.0)
    drag_torque = wp.vec3(0.0)
    added_mass_force = wp.vec3(0.0)
    added_mass_torque = wp.vec3(0.0)
    center_of_pressure = center_of_buoyancy

    # Only calculate physics if object touches water
    if sub_ratio > 1.0e-9:
        # Buoyancy Force
        buoyancy_force = wp.vec3(0.0, 0.0, water_density * (sub_ratio * volume) * gravity)

        # Flow Dynamics
        speed = wp.length(linear_velocity)
        if speed > 1.0e-6: velocity_direction = linear_velocity / speed
        else: velocity_direction = wp.vec3(0.0)

        # Center of Pressure and Projected Area
        center_of_pressure, total_projected_area = calculate_pressure_and_area(
            speed, velocity_direction, position, orientation, center_of_buoyancy,
            wp_normals, wp_centers, wp_areas
        )

        # Drag
        drag_force, drag_torque = calculate_hybrid_drag(
            speed, velocity_direction, linear_velocity, angular_velocity,
            sub_ratio, total_projected_area, water_density, volume,
            linear_drag, angular_drag, linear_damp, angular_damp
        )

        # Lift
        lift_force = calculate_lift(
            speed, velocity_direction, orientation, sub_ratio, total_projected_area,
            water_density, lift
        )

        # Added Mass
        added_mass_force, added_mass_torque = calculate_added_mass(
            linear_acceleration, angular_acceleration, orientation,
            sub_ratio, wp_added_mass
        )
    
    # Outputs
    out_buoyancy_force[tid] = buoyancy_force
    out_drag_force[tid] = drag_force
    out_lift_force[tid] = lift_force
    out_drag_torque[tid] = drag_torque
    out_added_mass_force[tid] = added_mass_force
    out_added_mass_torque[tid] = added_mass_torque
    out_center_of_buoyancy[tid] = center_of_buoyancy
    out_center_of_pressure[tid] = center_of_pressure