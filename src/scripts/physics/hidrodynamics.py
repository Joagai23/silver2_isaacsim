import numpy as np
import math

class Hydrodynamics:
    """
    A class to calculate hydrodynamic forces for an object (Buoyancy and Drag).
    """
    def __init__(self, total_volume, total_height, water_density=1000.0, gravity=9.81):
        self.total_volume = total_volume
        self.total_height = total_height
        self.water_density = water_density
        self.gravity = gravity
        # Works for now as the cube is orthogonal --> must be changed later on
        s = total_height / 2.0
        self._local_corners = np.array([
            [s, s, s], [-s, s, s], [s, -s, s], [-s, -s, s],
            [s, s, -s], [-s, s, -s], [s, -s, -s], [-s, -s, -s]
        ])
    
    def calculate_buoyancy(self, position, quaternion):
        """
            Calculates the buoyancy force vector based on submergence and orientation.

            Args:
                position (float[3]): Position coordinates of the center of the object in the world.
                quaternion (float[4]): The object's orientation as a quaternion [x, y, z, w].

            Returns:
                np.array: A 3D vector representing the buoyancy force in the world coordinate frame.
        """
        roll, pitch, yaw = self._quaternion_to_euler(quaternion)
        rotation_matrix = self._get_rotation_matrix(roll, pitch, yaw)
        world_corners = self._get_world_corners(position, rotation_matrix)
        submerged_count = np.sum(world_corners[:, 2] < 0)
        submerged_ratio = submerged_count / 8.0
        submerged_volume = submerged_ratio * self.total_volume
        buoyancy_magnitude = self.water_density * submerged_volume * self.gravity   
        buoyancy_vector = np.array([0.0, 0.0, buoyancy_magnitude])

        return buoyancy_vector

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
        Calculates the world coordinates of the 8 corners of the cube.

        Args:
            position (float[3]): Position coordinates of the center of the object in the world.
            rotation_matrix (np.array): A 3x3 matrix representing the rotation of the object in the world coordinate frame.
        
        Returns:
            np.array: An (8, 3) numpy array of the corner positions in world space, or None.
        """
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position

        local_corners_homogeneous = np.hstack([self._local_corners, np.ones((8, 1))])
        world_corners_homogeneous = (transform_matrix @ local_corners_homogeneous.T).T

        return world_corners_homogeneous[:, :3]