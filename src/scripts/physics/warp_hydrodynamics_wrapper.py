import warp as wp
import torch
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
        keypoints = [
            [-x, y, z], [0, y, z], [x, y, z],
            [-x, 0, z], [0, 0, z], [x, 0, z],
            [-x, -y, z], [0, -y, z], [x, -y, z],
            [-x, y, 0], [0, y, 0], [x, y, 0],
            [-x, 0, 0], [0, 0, 0], [x, 0, 0],
            [-x, -y, 0], [0, -y, 0], [x, -y, 0],
            [-x, y, -z], [0, y, -z], [x, y, -z],
            [-x, 0, -z], [0, 0, -z], [x, 0, -z],
            [-x, -y, -z], [0, -y, -z], [x, -y, -z]
        ]
        
        # Faces
        face_areas = [depth*height, depth*height, width*height, width*height, width*depth, width*depth]
        local_face_normals = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
        local_face_centers = [[x,0,0], [-x,0,0], [0,y,0], [0,-y,0], [0,0,z], [0,0,-z]]
        
        # GPU Allocations
        self.wp_keypoints = wp.array(keypoints, dtype=wp.vec3, device=device)
        self.wp_areas = wp.array(face_areas, dtype=float, device=device)
        self.wp_normals = wp.array(local_face_normals, dtype=wp.vec3, device=device)
        self.wp_centers = wp.array(local_face_centers, dtype=wp.vec3, device=device)
        
        # Parameters Vector
        volume = width * depth * height
        params_list = [water_density, gravity, linear_drag_coefficient, angular_drag_coefficient, 
                       linear_damping, angular_damping, lift_coefficient, volume]
        self.wp_params = wp.array(params_list, dtype=float, device=device)
        
        # Added Mass
        added_mass_linear_value = volume * linear_mass_coeff * water_density
        added_mass_linear_vector = [added_mass_linear_value, added_mass_linear_value, added_mass_linear_value]
        added_mass_angular_value_x = volume * (depth**2 + height**2) * angular_mass_coeff * water_density
        added_mass_angular_value_y = volume * (width**2 + height**2) * angular_mass_coeff * water_density
        added_mass_angular_value_z = volume * (width**2 + depth**2) * angular_mass_coeff * water_density
        added_mass_angular_vector = [added_mass_angular_value_x, added_mass_angular_value_y, added_mass_angular_value_z]

        added_mass = [added_mass_linear_vector, added_mass_angular_vector]
        self.wp_added_mass = wp.array(added_mass, dtype=wp.vec3, device=device)
        
        # Output Buffers
        self.out_buoyancy_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_drag_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_lift_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_drag_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_added_mass_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_added_mass_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_center_of_buoyancy = wp.zeros(1, dtype=wp.vec3, device=device)
        self.out_center_of_pressure = wp.zeros(1, dtype=wp.vec3, device=device)

    def calculate_hydrodynamic_forces(self, t_position, t_orientation, t_linear_velocity, 
                                      t_angular_velocity, t_linear_acceleration, t_angular_acceleration):
        """
        Takes PyTorch tensors, wraps them zero-copy, feeds to CUDA Graph, returns PyTorch tensors.
        """
        # Wrap Torch Tensors
        w_position = wp.from_torch(t_position, dtype=wp.vec3)
        w_orientation = wp.from_torch(t_orientation, dtype=wp.quat) 
        w_linear_velocity = wp.from_torch(t_linear_velocity, dtype=wp.vec3)
        w_angular_velocity = wp.from_torch(t_angular_velocity, dtype=wp.vec3)
        w_linear_acceleration = wp.from_torch(t_linear_acceleration, dtype=wp.vec3)
        w_angular_acceleration = wp.from_torch(t_angular_acceleration, dtype=wp.vec3)

        # Assign values from allocated memory
        self.in_position.assign(w_position)
        self.in_orientation.assign(w_orientation)
        self.in_lin_vel.assign(w_linear_velocity)
        self.in_ang_vel.assign(w_angular_velocity)
        self.in_lin_acc.assign(w_linear_acceleration)
        self.in_ang_acc.assign(w_angular_acceleration)

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
        
        # Return GPU pointers as Torch Tensors
        return (
            wp.to_torch(self.out_buoyancy_force),
            wp.to_torch(self.out_drag_force),
            wp.to_torch(self.out_lift_force),
            wp.to_torch(self.out_drag_torque), 
            wp.to_torch(self.out_added_mass_force),
            wp.to_torch(self.out_added_mass_torque),
            wp.to_torch(self.out_center_of_buoyancy),
            wp.to_torch(self.out_center_of_pressure)
        )