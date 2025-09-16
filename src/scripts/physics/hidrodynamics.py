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

    def calculate_buoyancy(self, z_position, quaternion):
        """
            Calculates the buoyancy force vector based on submergence and orientation.

            Args:
                z_position (float): The vertical position (Z-axis) of the object's center in the world.
                quaternion (float[4]): The object's orientation as a quaternion [x, y, z, w].

            Returns:
                np.array: A 3D vector representing the buoyancy force in the world coordinate frame.
        """
        roll, pitch, yaw = self._quaternion_to_euler(quaternion)
        submerged_ratio = self._calculate_submerged_ratio(z_position)
        submerged_volume = submerged_ratio * self.total_volume
        buoyancy_magnitude = self.water_density * submerged_volume * self.gravity
        rotation_matrix = self._get_rotation_matrix(roll, pitch, yaw)
        buoyancy_vector = np.array([0.0, 0.0, buoyancy_magnitude])
        rotation_force_vector = rotation_matrix @ buoyancy_vector

        return rotation_force_vector
    
    def _calculate_submerged_ratio(self, z_position):
        """
        Calculates the submerged ratio of an object by its height.
        This method is accurate for objects with a uniform cross-sectional area.
        This method assumes the fluid surface is at z = 0.
        
        Args:
            total_height (float): The total vertical height of the object.
            z_position (float): The Z-coordinate of the center of the object.

        Returns:
            float: The submerged ratio (a value between 0.0 and 1.0).
        """
        if self.total_height <= 0:
            print("Error: Total height must be a positive number.")
            return 0
        
        half_height = self.total_height / 2.0
        bottom_z_position = z_position - half_height

        submerged_height = min(self.total_height, max(0, -bottom_z_position))
        submerged_ratio = submerged_height / self.total_height

        return submerged_ratio

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
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = np.sqrt(1.0 + 2.0 * (w * y - x * z))
        cosp = np.sqrt(1.0 - 2.0 * (w * y - x * z))
        pitch_y = 2.0 * np.arctan2(sinp, cosp) - (math.pi / 2.0)
        # Yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(siny_cosp, cosy_cosp)

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