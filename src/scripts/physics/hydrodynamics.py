import numpy as np
from .numba_hydrodynamics import solve_hydrodynamics

class Hydrodynamics:
    """
    A lightweight wrapper that stores physical constants and calls the 
    Numba-optimized static function for calculation.
    """
    def __init__(self, width, depth, height, linear_drag_coefficient, angular_drag_coefficient, linear_damping, 
                 angular_damping, water_density, gravity, linear_mass_coeff, angular_mass_coeff, lift_coefficient):
        # Cube properties
        self.width = width
        self.depth = depth
        self.height = height
        self.total_volume = width * depth * height
        
        # Environment & Physics
        self.water_density = water_density
        self.gravity = gravity
        self.linear_drag_coefficient = linear_drag_coefficient
        self.angular_drag_coefficient = angular_drag_coefficient
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.lift_coefficient = lift_coefficient
        
        # Pre-calculated geometry (converted to float64 numpy arrays for Numba)
        self._face_areas, self._local_face_normals = self._create_cube_facepoints()
        self._local_keypoints, self._local_face_centers = self._create_cube_keypoints()
        self._added_mass_matrix = self._create_added_mass_matrix(
            width, depth, height, self.total_volume, 
            water_density, linear_mass_coeff, angular_mass_coeff
        )

    def calculate_hydrodynamic_forces(self, position, orientation_quat, linear_vel, angular_vel, linear_accel, angular_accel):
        """
        Calculates and returns all hydrodynamic forces using the compiled Numba function.
        """
        return solve_hydrodynamics(
            # Dynamic State
            position.astype(np.float64), 
            orientation_quat.astype(np.float64), 
            linear_vel.astype(np.float64), 
            angular_vel.astype(np.float64), 
            linear_accel.astype(np.float64), 
            angular_accel.astype(np.float64),
            # Constants
            self.total_volume,
            self.water_density, self.gravity,
            self.linear_drag_coefficient, self.angular_drag_coefficient,
            self.linear_damping, self.angular_damping, self.lift_coefficient,
            self._local_keypoints, self._local_face_centers, 
            self._face_areas, self._local_face_normals, self._added_mass_matrix
        )
    
    def _create_cube_keypoints(self):
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
        ], dtype=np.float64)

        local_face_centers = np.array([
            [x, 0, 0], [-x, 0, 0], # Right (+X), Left (-X)
            [0, y, 0], [0, -y, 0], # Front (+Y), Back (-Y)
            [0, 0, z], [0, 0, -z]  # Top (+Z), Bottom (-Z)
        ], dtype=np.float64)

        return local_keypoints, local_face_centers
    
    def _create_cube_facepoints(self):
        face_areas = np.array([
            self.depth * self.height,  # Right face (+X)
            self.depth * self.height,  # Left face (-X)
            self.width * self.height,  # Front face (+Y)
            self.width * self.height,  # Back face (-Y)
            self.width * self.depth,   # Top face (+Z)
            self.width * self.depth,   # Bottom face (-Z)
        ], dtype=np.float64)
        
        local_face_normals = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.float64)

        return face_areas, local_face_normals
    
    def _create_added_mass_matrix(self, width, depth, height, volume, water_density, linear_mass_coeff, angular_mass_coeff):
        added_mass_diag = np.array([
            # Linear
            volume * linear_mass_coeff * water_density,  
            volume * linear_mass_coeff * water_density, 
            volume * linear_mass_coeff * water_density,   
            # Angular
            volume * (depth**2 + height**2) * angular_mass_coeff * water_density, 
            volume * (width**2 + height**2) * angular_mass_coeff * water_density, 
            volume * (width**2 + depth**2) * angular_mass_coeff * water_density  
        ], dtype=np.float64)
        return np.diag(added_mass_diag)