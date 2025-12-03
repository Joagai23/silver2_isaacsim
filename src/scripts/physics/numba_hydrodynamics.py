import numpy as np
from numba import njit

"""
Set of static hydrodynamic functions optimized with Numba.
"""

@njit(cache=True, fastmath=True)
def quaternion_to_matrix(quaternion):
    """
    Converts quaternion [x, y, z, w] to a 3x3 rotation matrix without
    using Euler angles.
    """
    x, y, z, w = quaternion

    # Pre-calculate products
    x2 = x + x
    y2 = y + y
    z2 = z + z

    xx = x * x2
    xy = x * y2
    xz = x * z2

    yy = y * y2
    yz = y * z2
    zz = z * z2
    
    wx = w * x2
    wy = w * y2
    wz = w * z2

    # Allocate matrix directly
    matrix = np.empty((3, 3), dtype=np.float64)

    # Row 1
    matrix[0, 0] = 1.0 - (yy + zz)
    matrix[0, 1] = xy - wz
    matrix[0, 2] = xz + wy
    
    # Row 2
    matrix[1, 0] = xy + wz
    matrix[1, 1] = 1.0 - (xx + zz)
    matrix[1, 2] = yz - wx
    
    # Row 3
    matrix[2, 0] = xz - wy
    matrix[2, 1] = yz + wx
    matrix[2, 2] = 1.0 - (xx + yy)
    
    return matrix

@njit(cache=True, fastmath=True)
def analyze_submersion_and_cob(world_keypoints, position):
    """
    Calculates a continuous submersion ratio based on the highest and lowest points
    of the object, preventing the "sticking" effect at the water surface.
    """
    n_points = world_keypoints.shape[0]
    
    # Initialize limits
    z_min = np.inf
    z_max = -np.inf
    
    # Initialize CoB accumulators
    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
    submerged_count = 0

    for i in range(n_points):
        # Read point once
        px = world_keypoints[i, 0]
        py = world_keypoints[i, 1]
        pz = world_keypoints[i, 2]
        
        # Track Min/Max for Ratio
        if pz < z_min: z_min = pz
        if pz > z_max: z_max = pz
        
        # Accumulate for CoB if underwater
        if pz < 0:
            sum_x += px
            sum_y += py
            sum_z += pz
            submerged_count += 1

    # Ratio Logic (Linear approximation)
    if z_min >= 0: return 0.0, position # Fully out
    if z_max <= 0: return 1.0, position # Fully in based on bounds logic
    
    # Calculate height fraction
    total_height = z_max - z_min
    if total_height < 1e-6:
        ratio = 1.0 if z_min < 0 else 0.0
    else:
        ratio = -z_min / total_height
        if ratio > 1.0: ratio = 1.0

    # CoB Calculation
    if submerged_count == 0:
        cob = position
    else:
        inv_c = 1.0 / submerged_count
        cob = np.array([sum_x * inv_c, sum_y * inv_c, sum_z * inv_c])

    return ratio, cob

@njit(cache=True, fastmath=True)
def calculate_pressure_and_area(speed, vel_dir, center_of_buoyancy, rot_mat, local_face_normals, local_face_centers, position, face_areas):
    """
    Calculates Center of Pressure and Total Projected Area.
    Checks Z-height to ensure only underwater faces contribute to drag.
    """
    # Defaults
    total_projected_area = 0.0
    center_of_pressure = center_of_buoyancy
    
    # Only calculate detailed drag if moving
    if speed > 1e-6:
        # Transform Normals & Centers to World Space
        world_face_normals = (rot_mat @ local_face_normals.T).T
        world_face_centers = (rot_mat @ local_face_centers.T).T + position
        
        # Calculate alignment with flow
        dot_products = world_face_normals @ vel_dir
        
        cop_weighted_sum = np.zeros(3)
        
        for i in range(len(dot_products)):
            # Is face opposing flow?
            alignment = -dot_products[i] 
            
            if alignment > 0:
                # Only count this face if its center is underwater (Z < 0)
                if world_face_centers[i, 2] < 0:
                    area_val = alignment * face_areas[i]
                    total_projected_area += area_val
                    cop_weighted_sum += world_face_centers[i] * area_val
        
        # Calculate final CoP
        if total_projected_area > 1e-6:
            center_of_pressure = cop_weighted_sum / total_projected_area

        return center_of_pressure, total_projected_area

@njit(cache=True, fastmath=True)
def calculate_hybrid_drag(speed, vel_dir, sub_ratio, water_density, total_projected_area, total_volume, 
                        linear_drag_coeff, linear_damping, linear_vel,
                        angular_drag_coeff, angular_damping, angular_vel):
    """
    Calculates a hybrid drag force that combines a simple linear model (for low-speed stability)
    and a physically-based quadratic model (for high-speed realism).
    """
    # Linear Quadratic Drag
    LOW_SPEED_THRESHOLD = 0.2
    quad_drag_force = np.zeros(3)
    if speed > 1e-6:
        drag_mag = 0.5 * water_density * (speed**2) * linear_drag_coeff * total_projected_area
        quad_drag_force = -drag_mag * vel_dir
    # Linear Damping
    lin_damp_scale = 1.0
    if speed < LOW_SPEED_THRESHOLD:
        lin_damp_scale = speed / LOW_SPEED_THRESHOLD 
    linear_damping_force = -linear_damping * linear_vel * lin_damp_scale  
    # Combined Linear Drag
    total_linear_drag_force = (quad_drag_force + linear_damping_force) * sub_ratio

    # Angular Quadratic Drag
    ang_speed = np.linalg.norm(angular_vel)
    quad_drag_torque = np.zeros(3)  
    if ang_speed > 1e-6:
        ang_dir = angular_vel / ang_speed
        ang_mag = 0.5 * water_density * (ang_speed**2) * angular_drag_coeff * total_volume
        quad_drag_torque = -ang_mag * ang_dir   
    # Angular Damping
    ang_damp_scale = 1.0
    if ang_speed < LOW_SPEED_THRESHOLD:
        ang_damp_scale = ang_speed / LOW_SPEED_THRESHOLD
    linear_ang_torque = -angular_damping * angular_vel * ang_damp_scale   
    # Combined Angular Drag 
    total_angular_drag_torque = (quad_drag_torque + linear_ang_torque) * sub_ratio

    return total_linear_drag_force, total_angular_drag_torque

@njit(cache=True, fastmath=True)
def calculate_lift(speed, vel_dir, rot_mat, total_projected_area, 
                   water_density, lift_coeff_param, sub_ratio):
    """
    Calculates a simplified lift force based on the direction of incoming fluid (object's velocity) and the cube's
    "up" direction.
    """
    # Early exit
    if speed < 1e-6 or sub_ratio <= 1e-9:
        return np.zeros(3)
    
    # Calculate angle of attack
    up_vector = rot_mat[:, 2]
    dot_product = -(up_vector[0]*vel_dir[0] + up_vector[1]*vel_dir[1] + up_vector[2]*vel_dir[2])
    if dot_product > 1.0: dot_product = 1.0
    elif dot_product < -1.0: dot_product = -1.0
    angle_of_attack = np.arcsin(dot_product)

    # Calculate lift magnitude (flat plate approximation)
    lift_coefficient = np.sin(2 * angle_of_attack)
    lift_magnitude = 0.5 * water_density * (speed**2) * lift_coefficient * total_projected_area
    lift_magnitude *= lift_coeff_param

    # Lift direction Perpendicular to velocity and within plane velocity/up)
    lift_axis = np.cross(vel_dir, up_vector)
    norm_lift_axis = np.linalg.norm(lift_axis)
    if norm_lift_axis < 1e-6:
        return np.zeros(3)
    lift_axis /= norm_lift_axis

    # The lift direction is perpendicular to both the flow and this new axis.
    lift_direction = np.cross(lift_axis, vel_dir)

    return lift_magnitude * lift_direction * sub_ratio

@njit(cache=True, fastmath=True)
def calculate_added_mass(sub_ratio, lin_acc, ang_acc, rot_mat, added_mass_matrix):
    """
    Calculates added mass forces and torques using a 6x6 matrix in the body frame.
    """
    # Early exit
    if sub_ratio <= 1e-9:
        return np.zeros(3), np.zeros(3)
    
    # Rotate world-space accelerations into the object's local (body) frame
    linear_accel_local = rot_mat.T @ lin_acc
    angular_accel_local = rot_mat.T @ ang_acc

    # Construct 6D Vector
    accel_6d = np.empty(6, dtype=np.float64)
    accel_6d[0] = linear_accel_local[0]
    accel_6d[1] = linear_accel_local[1]
    accel_6d[2] = linear_accel_local[2]
    accel_6d[3] = angular_accel_local[0]
    accel_6d[4] = angular_accel_local[1]
    accel_6d[5] = angular_accel_local[2]

    # Calculate forces and torques in the local frame: Fi = -Mij * Aj
    force_torque_vector_local = -1.0 * (added_mass_matrix @ accel_6d)

    # Split the 6-DOF result back into 3D force and torque vectors
    force_local = force_torque_vector_local[:3]
    torque_local = force_torque_vector_local[3:]

    # Rotate the calculated forces and torques back into the world frame to be applied
    force_world = rot_mat @ force_local
    torque_world = rot_mat @ torque_local

    # Scale using submersion ratio and return
    return force_world * sub_ratio, torque_world * sub_ratio

@njit(cache=True, fastmath=True)
def solve_hydrodynamics(
    position, orientation_quat, linear_vel, angular_vel, linear_accel, angular_accel,
    total_volume, water_density, gravity,
    linear_drag_coeff, angular_drag_coeff, linear_damping, angular_damping, lift_coeff,
    local_keypoints, local_face_centers, face_areas, local_face_normals, added_mass_matrix
):
    """
    Main static merhod to calculate hydrodynamic forces using advanced models:
    - Multi-point buoyancy with Center of Buoyancy calculation.
    - Hybrid drag (linear + quadratic) with dynamic cross-sectional area and Center of Pressure.
    - Simplified lift force model.
    - 6x6 Added Mass matrix for realistic inertial effects.
    """
    # Orientation & Geometry
    rot_mat = quaternion_to_matrix(orientation_quat)
    world_keypoints = (rot_mat @ local_keypoints.T).T + position

    # Submersion & Buoyancy
    sub_ratio, center_of_buoyancy = analyze_submersion_and_cob(world_keypoints, position)

    # If not in water, return
    if sub_ratio <= 1e-9:
        zeros = np.zeros(3)
        return zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, 0.0

    # Buoyancy Force
    buoyancy_force = np.array([0.0, 0.0, water_density * (sub_ratio * total_volume) * gravity])

    # Flow dynamics
    speed = np.linalg.norm(linear_vel)
    if speed > 1e-6:
        vel_dir = linear_vel / speed
    else:
        vel_dir = np.zeros(3)

    # Center of pressure
    center_of_pressure, total_projected_area = calculate_pressure_and_area(
        speed, vel_dir, center_of_buoyancy, rot_mat, 
        local_face_normals, local_face_centers, position, face_areas
    )

    # Drag
    drag_force, drag_torque = calculate_hybrid_drag(
        speed, vel_dir, sub_ratio, water_density, total_projected_area, total_volume,
        linear_drag_coeff, linear_damping, linear_vel, angular_drag_coeff, angular_damping, angular_vel
    )

    # Lift
    lift_force = calculate_lift(
        speed, vel_dir, rot_mat, total_projected_area,
        water_density, lift_coeff, sub_ratio
    )
    
    # Added Mass
    added_mass_force, added_mass_torque = calculate_added_mass(
        sub_ratio, linear_accel, angular_accel, rot_mat, added_mass_matrix
    )

    return buoyancy_force, drag_force, lift_force, drag_torque, added_mass_force, added_mass_torque, center_of_buoyancy, center_of_pressure, sub_ratio