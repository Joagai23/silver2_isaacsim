import warp as wp
import numpy as np
from .warp_hydrodynamics import solve_hydrodynamics_kernel

class WarpHydrodynamicsWrapper:
    """
    A GPU-accelerated wrapper that stores physical constants in Warp arrays
    and calls the Warp kernel for calculation.
    """
    def __init__(self, width, depth, height, linear_drag_coefficient, angular_drag_coefficient, linear_damping, 
                 angular_damping, water_density, gravity, linear_mass_coeff, angular_mass_coeff, lift_coefficient,
                 device="cuda:0"):
        
        self.device = device
        self.graph = None

        # Pre-allocate input buffers
        self.in_position = wp.zeros(1, dtype=wp.vec3, device=device)
        self.in_orientation = wp.zeros(1, dtype=wp.quat, device=device)
        self.in_lin_vel = wp.zeros(1, dtype=wp.vec3, device=device)
        self.in_ang_vel = wp.zeros(1, dtype=wp.vec3, device=device)
        self.in_lin_acc = wp.zeros(1, dtype=wp.vec3, device=device)
        self.in_ang_acc = wp.zeros(1, dtype=wp.vec3, device=device)
        
        # Geometry Generation
        x, y, z = width/2.0, depth/2.0, height/2.0
        
        # Keypoints
        np_keypoints = np.array([
            [-x, y, z], [0, y, z], [x, y, z],
            [-x, 0, z], [0, 0, z], [x, 0, z],
            [-x, -y, z], [0, -y, z], [x, -y, z],
            [-x, y, 0], [0, y, 0], [x, y, 0],
            [-x, 0, 0], [0, 0, 0], [x, 0, 0],
            [-x, -y, 0], [0, -y, 0], [x, -y, 0],
            [-x, y, -z], [0, y, -z], [x, y, -z],
            [-x, 0, -z], [0, 0, -z], [x, 0, -z],
            [-x, -y, -z], [0, -y, -z], [x, -y, -z]
        ], dtype=np.float32)
        
        # Faces
        np_face_areas = np.array([depth*height, depth*height, width*height, width*height, width*depth, width*depth], dtype=np.float32)
        np_local_face_normals = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], dtype=np.float32)
        np_local_face_centers = np.array([[x,0,0], [-x,0,0], [0,y,0], [0,-y,0], [0,0,z], [0,0,-z]], dtype=np.float32)
        
        # GPU Allocations
        self.wp_keypoints = wp.from_numpy(np_keypoints, dtype=wp.vec3, device=device)
        self.wp_areas = wp.from_numpy(np_face_areas, dtype=float, device=device)
        self.wp_normals = wp.from_numpy(np_local_face_normals, dtype=wp.vec3, device=device)
        self.wp_centers = wp.from_numpy(np_local_face_centers, dtype=wp.vec3, device=device)
        
        # Parameters Vector
        volume = width * depth * height
        params_list = [water_density, gravity, linear_drag_coefficient, angular_drag_coefficient, 
                       linear_damping, angular_damping, lift_coefficient, volume]
        self.wp_params = wp.from_numpy(np.array(params_list, dtype=np.float32), dtype=float, device=device)
        
        # Added Mass
        added_mass_linear_value = volume * linear_mass_coeff * water_density
        added_mass_linear_vector = [added_mass_linear_value, added_mass_linear_value, added_mass_linear_value]
        added_mass_angular_value_x = volume * (depth**2 + height**2) * angular_mass_coeff * water_density
        added_mass_angular_value_y = volume * (width**2 + height**2) * angular_mass_coeff * water_density
        added_mass_angular_value_z = volume * (width**2 + depth**2) * angular_mass_coeff * water_density
        added_mass_angular_vector = [added_mass_angular_value_x, added_mass_angular_value_y, added_mass_angular_value_z]
        np_added_mass = np.array([added_mass_linear_vector, added_mass_angular_vector], dtype=np.float32)
        self.wp_added_mass = wp.from_numpy(np_added_mass, dtype=wp.vec3, device=device)
        
        # Output Buffers
        self.out_buoyancy_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_drag_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_lift_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_drag_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_added_mass_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_added_mass_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_center_of_buoyancy = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_center_of_pressure = wp.zeros(1, dtype=wp.vec3, device=device)

    def calculate_hydrodynamic_forces(self, np_position, np_orientation, np_linear_velocity, 
                                      np_angular_velocity, np_linear_acceleration, np_angular_acceleration):
        """
        Runs the Warp kernel. 
        Inputs MUST be Warp Arrays already on the GPU.
        Returns a tuple of Warp Arrays.
        """
        # Assign values from allocated memory
        self.in_position.assign(np_position)
        self.in_orientation.assign(np_orientation)
        self.in_lin_vel.assign(np_linear_velocity)
        self.in_ang_vel.assign(np_angular_velocity)
        self.in_lin_acc.assign(np_linear_acceleration)
        self.in_ang_acc.assign(np_angular_acceleration)

        # Use CUDA graphs
        if self.graph is None:
            wp.capture_begin(device=self.device)
            wp.launch(
                kernel=solve_hydrodynamics_kernel,
                dim=1,
                inputs=[
                    self.in_position, self.in_orientation, self.in_lin_vel, self.in_ang_vel, self.in_lin_acc, self.in_ang_acc,
                    self.wp_keypoints, self.wp_normals, self.wp_centers, self.wp_areas,
                    self.wp_params, self.wp_added_mass
                ],
                outputs=[
                    self.out_buoyancy_force, self.out_drag_force, self.out_lift_force, self.out_drag_torque,
                    self.out_added_mass_force, self.out_added_mass_torque, self.out_center_of_buoyancy, self.out_center_of_pressure
                ],
                device=self.device
            )
            self.graph = wp.capture_end(device=self.device)   
            wp.capture_launch(self.graph)
        else:
            wp.capture_launch(self.graph)
        
        # Return GPU pointers directly
        return (self.out_buoyancy_force, self.out_drag_force, self.out_lift_force, self.out_drag_torque, 
                self.out_added_mass_force, self.out_added_mass_torque, self.out_center_of_buoyancy, self.out_center_of_pressure)