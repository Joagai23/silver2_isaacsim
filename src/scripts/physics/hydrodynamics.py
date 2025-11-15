import numpy as np
import math

class Hydrodynamics:
    """
    A class to calculate hydrodynamic forces using advanced models:
    - Multi-point buoyancy with Center of Buoyancy calculation.
    - Hybrid drag (linear + quadratic) with dynamic cross-sectional area and Center of Pressure.
    - Simplified lift force model.
    - 6x6 Added Mass matrix for realistic inertial effects.
    """
    def __init__(self, width, depth, height, linear_drag_coefficient, angular_drag_coefficient, linear_damping, 
                 angular_damping, water_density, gravity, linear_mass_coeff, angular_mass_coeff, lift_coefficient):
        # Cube properties
        self.width = width
        self.depth = depth
        self.height = height
        self.total_volume = width * depth * height
        # Environment properties
        self.water_density = water_density
        self.gravity = gravity
        # Physics coefficients
        self.linear_drag_coefficient = linear_drag_coefficient
        self.angular_drag_coefficient = angular_drag_coefficient
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.lift_coefficient = lift_coefficient
        # Pre-calculated geometry
        self._face_areas, self._local_face_normals = self._create_cube_facepoints()
        self._local_keypoints, self._local_face_centers = self._create_cube_keypoints()
        self._added_mass_matrix = self._create_added_mass_matrix(width, depth, height, self.total_volume, 
                                                                 water_density, linear_mass_coeff, angular_mass_coeff)

    def calculate_hydrodynamic_forces(self, position, orientation_quat, linear_vel, angular_vel, linear_accel, angular_accel):
        """
        Calculates and returns all hydrodynamic forces, torques, and points of application.
        """
        roll, pitch, yaw = self._quaternion_to_euler(orientation_quat)
        rotation_matrix = self._get_rotation_matrix(roll, pitch, yaw)
        world_keypoints = self._get_world_corners(position, rotation_matrix)

        if world_keypoints is None:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0

        submerged_keypoints = world_keypoints[world_keypoints[:, 2] < 0]
        submersion_ratio = self._calculate_submerged_ratio(world_keypoints)

        if submersion_ratio <= 0:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0

        buoyancy_force = self._calculate_buoyancy(submersion_ratio)
        drag_force, drag_torque = self._calculate_hybrid_drag(submersion_ratio, linear_vel, angular_vel, rotation_matrix)
        lift_force = self._calculate_lift(submersion_ratio, linear_vel, rotation_matrix)
        added_mass_force, added_mass_torque = self._calculate_added_mass(submersion_ratio, linear_accel, angular_accel, rotation_matrix)

        center_of_buoyancy, center_of_pressure = self._calculate_centers_of_pressure(submerged_keypoints, position, rotation_matrix, linear_vel)
        
        return buoyancy_force, drag_force, lift_force, drag_torque, added_mass_force, added_mass_torque, center_of_buoyancy, center_of_pressure, submersion_ratio
    
    def _calculate_buoyancy(self, submersion_ratio):
        """
            Calculates the buoyancy force vector based on submergence and orientation.
        """
        submerged_volume = submersion_ratio * self.total_volume
        buoyancy_magnitude = self.water_density * submerged_volume * self.gravity   
        buoyancy_vector = np.array([0.0, 0.0, buoyancy_magnitude])

        return buoyancy_vector
    
    def _calculate_hybrid_drag(self, submersion_ratio, linear_velocity, angular_velocity, rotation_matrix):
        """
        Calculates a hybrid drag force that combines a simple linear model (for low-speed stability)
        and a physically-based quadratic model (for high-speed realism).
        """
        # A speed threshold below which damping is softened to prevent oscillation
        LOW_SPEED_THRESHOLD = 0.2

        ## LINEAR VELOCITY
        # Quadratic Drag
        speed = np.linalg.norm(linear_velocity)
        if speed < 1e-6:
            quadratic_drag_force = np.zeros(3)
        else:
            velocity_direction = linear_velocity / speed
            dynamic_area = self._get_dynamic_cross_sectional_area(rotation_matrix, velocity_direction)
            drag_magnitude = 0.5 * self.water_density * (speed ** 2) * self.linear_drag_coefficient * dynamic_area
            quadratic_drag_force = -drag_magnitude * velocity_direction
        # Linear Drag
        damping_scale_factor = min(1.0, speed / LOW_SPEED_THRESHOLD)
        linear_drag_force = -self.linear_damping * linear_velocity * damping_scale_factor
        # Combine forces and scale them based on object submersion
        drag_force = (quadratic_drag_force + linear_drag_force) * submersion_ratio

        ## ANGULAR VELOCITY
        # Quadratic Drag
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed < 1e-6:
            angular_quadratic_drag_torque = np.zeros(3)
        else:
            angular_velocity_direction = angular_velocity / angular_speed
            angular_drag_magnitude = 0.5 * self.water_density * (angular_speed ** 2) * self.angular_drag_coefficient * self.total_volume
            angular_quadratic_drag_torque = -angular_drag_magnitude * angular_velocity_direction
        # Linear Drag
        angular_damping_scale_factor = min(1.0, angular_speed / LOW_SPEED_THRESHOLD)
        linear_angular_drag_torque = -self.angular_damping * angular_velocity * angular_damping_scale_factor
        drag_torque = (angular_quadratic_drag_torque + linear_angular_drag_torque) * submersion_ratio

        return drag_force, drag_torque

    def _calculate_lift(self, submersion_ratio, linear_velocity, rotation_matrix):
        """
        Calculates a simplified lift force based on the direction of incoming fluid (object's velocity) and the cube's
        "up" direction.
        """
        speed = np.linalg.norm(linear_velocity)
        if speed < 1e-6:
            return np.zeros(3)
        velocity_direction = linear_velocity / speed

        # The object's 'up' vector in world coordinates
        up_vector = rotation_matrix[:, 2]
        # Calculate Angle of Attack
        dot_product = -np.dot(up_vector, velocity_direction)
        angle_of_attack = np.arcsin(np.clip(dot_product, -1.0, 1.0))

        # Calculate Lift Magnitude using a flat plate approximation
        calculated_lift_coefficient = np.sin(2 * angle_of_attack)
        cross_sectional_area = self._get_dynamic_cross_sectional_area(rotation_matrix, velocity_direction)
        lift_magnitude = 0.5 * self.water_density * (speed ** 2) * calculated_lift_coefficient * cross_sectional_area
        lift_magnitude *= self.lift_coefficient 

        # Calculate Lift Direction (Perpendicular to velocity and within plane velocity/up)
        lift_axis = np.cross(velocity_direction, up_vector)
        norm_lift_axis = np.linalg.norm(lift_axis)
        if norm_lift_axis < 1e-6:
            return np.zeros(3)
        lift_axis /= norm_lift_axis

        # The lift direction is perpendicular to both the flow and this new axis.
        lift_direction = np.cross(lift_axis, velocity_direction)
        lift_force = lift_magnitude * lift_direction * submersion_ratio
        
        return lift_force

    def _calculate_added_mass(self, submersion_ratio, linear_accel_world, angular_accel_world, rotation_matrix):
        """
        Calculates added mass forces and torques using a 6x6 matrix in the body frame.
        """
        # The inverse of a rotation matrix is its transpose
        rotation_matrix_transpose = rotation_matrix.T

        # Rotate world-space accelerations into the object's local (body) frame
        linear_accel_local = rotation_matrix_transpose @ linear_accel_world
        angular_accel_local = rotation_matrix_transpose @ angular_accel_world

        # Create a 6-DOF acceleration vector
        accel_vector_local = np.concatenate([linear_accel_local, angular_accel_local])

        # Calculate forces and torques in the local frame: Fi = -Mij * Aj
        force_torque_vector_local = -self._added_mass_matrix @ accel_vector_local

        # Split the 6-DOF result back into 3D force and torque vectors
        force_local = force_torque_vector_local[:3]
        torque_local = force_torque_vector_local[3:]

        # Rotate the calculated forces and torques back into the world frame to be applied
        force_world = rotation_matrix @ force_local
        torque_world = rotation_matrix @ torque_local

        # Scale using submersion ratio
        force_world *= submersion_ratio
        torque_world *= submersion_ratio
        
        return force_world, torque_world

    def _calculate_centers_of_pressure(self, submerged_keypoints, position, rotation_matrix, linear_velocity):
        """
        Calculates the Center of Buoyancy (CoB) and Center of Pressure (CoP)
        by finding the average position of the submerged keypoints.
        CoB is the geometric center of the submerged volume.
        CoP is the area-weighted center of the faces resisting motion.

        Args:
            submerged_keypoints (np.array): An (N, 3) array of the object's submerged (Z < 0) keypoints in world space.
            position (np.array): The object's world position.
            rotation_matrix (np.array): The object's 3x3 rotation matrix.
            velocity_direction (np.array): The normalized direction of the object's linear velocity.

        Returns:
            (np.array, np.array): A tuple containing the (center_of_buoyancy, center_of_pressure).
        """
        # --- 1. Calculate Center of Buoyancy (CoB) ---
        if submerged_keypoints.shape[0] == 0:
            return position, position     
        # Center of Buoyancy is the average of submerged points' position
        center_of_buoyancy = np.mean(submerged_keypoints, axis=0)

        # --- 2. Calculate Center of Pressure (CoP) for Drag ---
        # Transform local face properties to world space
        world_face_normals = (rotation_matrix @ self._local_face_normals.T).T
        world_face_centers = (rotation_matrix @ self._local_face_centers.T).T + position
        # Calculate the projected area of each face
        speed = np.linalg.norm(linear_velocity)
        if speed < 1e-6:
            velocity_direction = np.zeros(3)
        else:
            velocity_direction = linear_velocity / speed
        projected_areas = np.maximum(0, -np.dot(world_face_normals, velocity_direction.flatten())) * self._face_areas
        total_projected_area = np.sum(projected_areas)

        if total_projected_area < 1e-6:
            # If no area is facing the flow, CoP is at the CoB as a fallback.
            center_of_pressure = center_of_buoyancy
        else:
            # The CoP is the weighted average of the face centers, with the projected area as the weight.
            weighted_centers = world_face_centers * projected_areas[:, np.newaxis]
            center_of_pressure = np.sum(weighted_centers, axis=0) / total_projected_area

        return center_of_buoyancy, center_of_pressure

    def _calculate_submerged_ratio(self, world_keypoints):
        """
        Calculates a continuous submersion ratio based on the highest and lowest points
        of the object, preventing the "sticking" effect at the water surface.
        """
        if world_keypoints.shape[0] == 0:
            return 0.0

        z_coords = world_keypoints[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)

        # Fully above water
        if z_min >= 0:  
            return 0.0
        # Fully submerged
        if z_max <= 0:
            return 1.0
        
        # Partially submerged
        total_effective_height = z_max - z_min
        submerged_height = -z_min

        # Check and prevent zero division  for thin objects
        if total_effective_height < 1e-6:
            if z_min >= 0:
                return 0.0
            else:
                return 1.0

        return submerged_height / total_effective_height

    def _get_dynamic_cross_sectional_area(self, rotation_matrix, velocity_direction):
        """
        Calculates the projected area of the cube facing the velocity vector.
        """
        # Transform the local face normals into world space
        world_face_normals = (rotation_matrix @ self._local_face_normals.T).T

        # Ensure the velocity direction is a 1D array to prevent shape mismatch errors
        velocity_direction_flat = velocity_direction.flatten()
        
        # Calculate the dot product between each world face normal and the velocity direction.
        # We take the absolute value and only consider faces pointing against the flow (dot product < 0).
        projected_areas = np.maximum(0, -np.dot(world_face_normals, velocity_direction_flat)) * self._face_areas
        
        # The total cross-sectional area is the sum of these projected areas
        return np.sum(projected_areas)

    def _quaternion_to_euler(self, quaternion):
        """
        This implementation assumes normalized quaternion.
        Converts to Euler angles in 3-2-1 sequence.

        Args:
            quaternion (float[4]): The object's orientation as a quaternion [x, y, z, w].

        Returns:
            float[3]: Rotation values in Euler coordinate system.
        """
        x, y, z, w = quaternion
        # Roll
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        # Pitch
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0) 
        pitch_y = np.arcsin(t2)
        # Yaw
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z
    
    def _get_rotation_matrix(self, roll, pitch, yaw):
        """
        Generate a rotation matrix for the rotation values of the 3-axis.

        Args:
            roll (float): rotational value in the X-axis.
            pitch (float): rotational value in the Y-axis.
            yaw (float): rotational value in the Z-axis.

        Returns:
            np.array: A 3x3 matrix representing the rotation of the object in the world coordinate frame.
        """
        c_r, s_r = math.cos(roll), math.sin(roll)
        c_p, s_p = math.cos(pitch), math.sin(pitch)
        c_y, s_y = math.cos(yaw), math.sin(yaw)

        return np.array([
            [c_y*c_p,   c_y*s_p*s_r - s_y*c_r,  c_y*s_p*c_r + s_y*s_r],
            [s_y*c_p,   s_y*s_p*s_r + c_y*c_r,  s_y*s_p*c_r - c_y*s_r],
            [-s_p,      c_p*s_r,                c_p*c_r]
        ])
    
    def _get_world_corners(self, position, rotation_matrix):
        """
        Calculates the world coordinates of the 27 keypoints of the cube.

        Args:
            position (float[3]): Position coordinates of the center of the object in the world.
            rotation_matrix (np.array): A 3x3 matrix representing the rotation of the object in the world coordinate frame.
        
        Returns:
            np.array: An (27, 3) numpy array of the key positions in world space, or None.
        """
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position

        local_corners_homogeneous = np.hstack([self._local_keypoints, np.ones((27, 1))])
        world_corners_homogeneous = (transform_matrix @ local_corners_homogeneous.T).T

        return world_corners_homogeneous[:, :3]

    def _create_cube_keypoints(self):
        """
        Generates a 27x3 matrix containing local key position vertices for volume calculations.
        Generates a 6x3 matrix containing the local center point of each cube face.
        """
        x = self.width / 2.0
        y = self.depth / 2.0
        z = self.height / 2.0

        local_keypoints = np.array([
            # Top Layer
            [-x, +y, +z], [0, +y, +z], [+x, +y, +z],
            [-x, 0, +z], [0, 0, +z], [+x, 0, +z],
            [-x, -y, +z], [0, -y, +z], [+x, -y, +z],

            # Middle Layer
            [-x, +y, 0], [0, +y, 0], [+x, +y, 0],
            [-x, 0, 0], [0, 0, 0], [+x, 0, 0],
            [-x, -y, 0], [0, -y, 0], [+x, -y, 0],

            # Bottom Layer
            [-x, +y, -z], [0, +y, -z], [+x, +y, -z],
            [-x, 0, -z], [0, 0, -z], [+x, 0, -z],
            [-x, -y, -z], [0, -y, -z], [+x, -y, -z]
        ])

        local_face_centers = np.array([
            [x, 0, 0], [-x, 0, 0], # Right (+X), Left (-X)
            [0, y, 0], [0, -y, 0], # Front (+Y), Back (-Y)
            [0, 0, z], [0, 0, -z] # Top (+Z), Bottom (-Z)
        ])

        return local_keypoints, local_face_centers
    
    def _create_cube_facepoints(self):
        """
        Define the areas of the cube's faces and the normals for each face in local space
        """
        face_areas = np.array([
            self.depth * self.height,  # Right face (+X)
            self.depth * self.height,  # Left face (-X)
            self.width * self.height,  # Front face (+Y)
            self.width * self.height,  # Back face (-Y)
            self.width * self.depth,   # Top face (+Z)
            self.width * self.depth,   # Bottom face (-Z)
        ])
        local_face_normals = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

        return face_areas, local_face_normals
    
    def _create_added_mass_matrix(self, width, depth, height, volume, water_density, linear_mass_coeff, angular_mass_coeff):
        """
        Construct the 6x6 added mass matrix from the exposed diagonal terms
        """ 
        added_mass_diag = np.array([
            # Linear resistance to acceleration is proportional to mass
            volume * linear_mass_coeff * water_density,  # Surge (X)
            volume * linear_mass_coeff * water_density,  # Sway (Y) 
            volume * linear_mass_coeff * water_density,   # Heave (Z) 
            
            # Angular terms are proportional to the area moment of inertia
            volume * (depth**2 + height**2) * angular_mass_coeff * water_density, # Roll 
            volume * (width**2 + height**2) * angular_mass_coeff * water_density, # Pitch 
            volume * (width**2 + depth**2) * angular_mass_coeff * water_density  # Yaw
        ])
        return np.diag(added_mass_diag)