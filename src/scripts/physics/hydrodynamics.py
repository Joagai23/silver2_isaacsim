import numpy as np
import math

class Hydrodynamics:
    """
    A class to calculate hydrodynamic forces for an object (Buoyancy and Drag).
    """
    def __init__(self, width, depth, height, drag_coefficient, angular_drag_coefficient, water_density, gravity):
        self.width = width
        self.depth = depth
        self.height = height
        self.water_density = water_density
        self.gravity = gravity
        self.drag_coefficient = drag_coefficient
        self.angular_drag_coefficient = angular_drag_coefficient
        self.total_volume = width * depth * height
        self._face_areas, self._local_face_normals = self._create_cube_facepoints()
        self._local_keypoints = self._create_cube_keypoints()

    def calculate_hydrodynamic_forces(self, position, orientation_quat, linear_vel, angular_vel):
        """
        Calculates and returns all hydrodynamic forces and torques acting on the object.
        """
        roll, pitch, yaw = self._quaternion_to_euler(orientation_quat)
        rotation_matrix = self._get_rotation_matrix(roll, pitch, yaw)
        world_corners = self._get_world_corners(position, rotation_matrix)
        submerged_count = np.sum(world_corners[:, 2] < 0)
        submersion_ratio = submerged_count / 27

        buoyancy_force = self._calculate_buoyancy(submersion_ratio)
        drag_force, drag_torque = self._calculate_quadratic_drag(submersion_ratio, linear_vel, angular_vel, rotation_matrix)
        total_force = buoyancy_force + drag_force
        
        return total_force, drag_torque
    
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
    
    def _calculate_linear_drag(self, submersion_ratio, linear_velocity, angular_velocity):
        """
        Calculates drag force and torque based on submersion level.

            Args:
                submersion_ratio (float): Approximate value of the object that is underwater (Z < 0.0).
                linear_velocity (float[3]): Current rate of change of an object's displacement along a straight line.
                angular_velocity (float[3]): Current rate of change of an object's angular displacement over time.

            Returns:
                drag_force (np.array): A 3D vector representing the resistance force against the object's translational motion.
                drag_torque (np.array): A 3D vector representing the resistance force against the object's rotational motion.
        """
        current_linear_damping = self.max_linear_damping * submersion_ratio
        current_angular_damping = self.max_angular_damping * submersion_ratio
        
        # Add a small constant damping for when the object is in the air
        current_linear_damping += 0.01
        current_angular_damping += 0.01

        drag_force = -current_linear_damping * linear_velocity
        drag_torque = -current_angular_damping * angular_velocity

        return drag_force, drag_torque
    
    def _calculate_quadratic_drag(self, submersion_ratio, linear_velocity, angular_velocity, rotation_matrix):
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
        Generates a 27x3 matrix containing key position vertices for volume calculations.
        """
        x = self.width / 2.0
        y = self.depth / 2.0
        z = self.height / 2.0

        return np.array([
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