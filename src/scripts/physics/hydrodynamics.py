import numpy as np
import math

class Hydrodynamics:
    """
    A class to calculate hydrodynamic forces for an object (Buoyancy and Drag).
    """
    def __init__(self, width, depth, height, drag_coefficient, angular_drag_coefficient, linear_damping, angular_damping, water_density, gravity):
        self.width = width
        self.depth = depth
        self.height = height
        self.water_density = water_density
        self.gravity = gravity
        self.drag_coefficient = drag_coefficient
        self.angular_drag_coefficient = angular_drag_coefficient
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.total_volume = width * depth * height

        self._face_areas, self._local_face_normals = self._create_cube_facepoints()
        self._local_keypoints, self._local_face_centers = self._create_cube_keypoints()

    def calculate_hydrodynamic_forces(self, position, orientation_quat, linear_vel, angular_vel):
        """
        Calculates and returns all hydrodynamic forces and torques acting on the object.
        """
        roll, pitch, yaw = self._quaternion_to_euler(orientation_quat)
        rotation_matrix = self._get_rotation_matrix(roll, pitch, yaw)
        world_corners = self._get_world_corners(position, rotation_matrix)

        if world_corners is None:
            zeros = np.zeros(3)
            return zeros, zeros

        submerged_count = np.sum(world_corners[:, 2] < 0)
        submersion_ratio = submerged_count / 27

        buoyancy_force = self._calculate_buoyancy(submersion_ratio)
        drag_force, drag_torque = self._calculate_hybrid_drag(submersion_ratio, linear_vel, angular_vel, rotation_matrix)
        lift_force = self._calculate_lift(submersion_ratio, linear_vel, rotation_matrix)

        center_of_buoyancy, center_of_pressure = self._calculate_centers_of_pressure(world_corners, position, rotation_matrix, linear_vel)
        
        return buoyancy_force, drag_force, lift_force, drag_torque, center_of_buoyancy, center_of_pressure
    
    def _calculate_buoyancy(self, submersion_ratio):
        """
            Calculates the buoyancy force vector based on submergence and orientation.

            Args:
                submersion_ratio (float): Approximate value of the object that is underwater (Z < 0.0).

            Returns:
                np.array: A 3D vector representing the buoyancy force in the world coordinate frame.
        """
        submerged_volume = submersion_ratio * self.total_volume
        buoyancy_magnitude = self.water_density * submerged_volume * self.gravity   
        buoyancy_vector = np.array([0.0, 0.0, buoyancy_magnitude])

        return buoyancy_vector
    
        """
        Calculates drag force and torque using the quadratic drag formula.
        Fd = 0.5 * rho * u^2 * Cd * A
        """
        # Linear Drag
        speed = np.linalg.norm(linear_velocity)
        
        if speed < 1e-6:
            drag_force = np.zeros(3)
        else:
            velocity_direction = linear_velocity / speed
            dynamic_area = self._get_dynamic_cross_sectional_area(rotation_matrix, velocity_direction)

            drag_magnitude = 0.5 * self.water_density * (speed ** 2) * self.drag_coefficient * dynamic_area
            drag_force = -drag_magnitude * velocity_direction
            drag_force *= submersion_ratio

        # Angular Drag (simplified quadratic model)
        angular_speed = np.linalg.norm(angular_velocity)

        if angular_speed < 1e-6:
            drag_torque = np.zeros(3)
        else:
            angular_velocity_dir = angular_velocity / angular_speed
            torque_magnitude = self.angular_drag_coefficient * (angular_speed ** 2)
            drag_torque = -torque_magnitude * angular_velocity_dir
            drag_torque *= submersion_ratio

        return drag_force, drag_torque
    
    def _calculate_hybrid_drag(self, submersion_ratio, linear_velocity, angular_velocity, rotation_matrix):
        """
        Calculates a hybrid drag force that combines a simple linear model (for low-speed stability)
        and a physically-based quadratic model (for high-speed realism).
        """
        # A speed threshold below which damping is softened to prevent oscillation
        LOW_SPEED_THRESHOLD = 0.2

        # Quadratic Drag (Dominant at high speeds)
        speed = np.linalg.norm(linear_velocity)
        if speed < 1e-6:
            quadratic_drag_force = np.zeros(3)
        else:
            velocity_direction = linear_velocity / speed
            dynamic_area = self._get_dynamic_cross_sectional_area(rotation_matrix, velocity_direction)
            drag_magnitude = 0.5 * self.water_density * (speed ** 2) * self.drag_coefficient * dynamic_area
            quadratic_drag_force = -drag_magnitude * velocity_direction
        # Linear Drag (Provides stability at lower speeds)
        linear_damping_coeff = self.drag_coefficient * self.linear_damping
        damping_scale_factor = min(1.0, speed / LOW_SPEED_THRESHOLD)
        linear_drag_force = -linear_damping_coeff * linear_velocity * damping_scale_factor
        # Combine forces and scale them based on object submersion
        drag_force = (quadratic_drag_force + linear_drag_force) * submersion_ratio

        # Angular Drag (Hybrid model)
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed < 1e-6:
            drag_torque = np.zeros(3)
        else:
            angular_velocity_dir = angular_velocity / angular_speed
            quadratic_torque_magnitude = self.angular_drag_coefficient * (angular_speed ** 2)
            linear_torque_magnitude = self.angular_damping * angular_speed
            drag_torque = -(quadratic_torque_magnitude + linear_torque_magnitude) * angular_velocity_dir
            drag_torque *= submersion_ratio

        return drag_force, drag_torque

    def _calculate_lift(self, submersion_ratio, linear_velocity, rotation_matrix):
        """
        Calculates a simplified lift force.
        This model assumes lift is generated primarily by the pitch angle.
        """
        speed = np.linalg.norm(linear_velocity)
        if speed < 1e-6:
            return np.zeros(3)
        velocity_direction = linear_velocity / speed

        # Calculate Angle of Attack
        up_vector = rotation_matrix[:, 2]
        world_up = np.array([0, 0, 1])
        dot_product = np.dot(up_vector, world_up)
        dot_product_clipped = np.clip(dot_product, -1.0, 1.0)
        attack_angle = np.arccos(dot_product_clipped)

        # Calculate Lift Magnitude
        lift_coefficient = np.sin(2 * attack_angle) # Use 2 Amplifying heuristic for now
        cross_sectional_area = self._get_dynamic_cross_sectional_area(rotation_matrix, velocity_direction)
        lift_magnitude = 0.5 * self.water_density * (speed ** 2) * lift_coefficient * cross_sectional_area

        # Calculate Lift Direction (Perpendicular to velocity)
        right_vector = rotation_matrix[:, 0]
        lift_direction = np.cross(velocity_direction, right_vector)
        norm_lift_direction = lift_direction / (np.linalg.norm(lift_direction) + 1e-6)

        # Calculate and return final Lift Force
        lift_force = lift_magnitude * norm_lift_direction * submersion_ratio
        if np.any(np.isnan(lift_force)):
            return np.zeros(3)
        
        return lift_force

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
    
    def _calculate_centers_of_pressure(self, world_keypoints, position, rotation_matrix, linear_velocity):
        """
        Calculates the Center of Buoyancy (CoB) and Center of Pressure (CoP)
        by finding the average position of the submerged keypoints.
        CoB is the geometric center of the submerged volume.
        CoP is the area-weighted center of the faces resisting motion.

        Args:
            world_keypoints (np.array): An (N, 3) array of the object's keypoints in world space.
            position (np.array): The object's world position.
            rotation_matrix (np.array): The object's 3x3 rotation matrix.
            velocity_direction (np.array): The normalized direction of the object's linear velocity.

        Returns:
            (np.array, np.array): A tuple containing the (center_of_buoyancy, center_of_pressure).
        """
        # --- 1. Calculate Center of Buoyancy (CoB) ---
        submerged_points = world_keypoints[world_keypoints[:,2] < 0]
        if submerged_points.shape[0] == 0:
            return position, position     
        # Center of Buoyancy is the average of submerged points' position
        center_of_buoyancy = np.mean(submerged_points, axis=0)

        # --- 2. Calculate Center of Pressure (CoP) for Drag ---
        # Transform local face properties to world space
        world_face_normals = (rotation_matrix @ self._local_face_normals.T).T
        world_face_centers = (rotation_matrix @ self._local_face_centers.T).T + position
        # Calculate the projected area of each face
        speed = np.linalg.norm(linear_velocity)
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