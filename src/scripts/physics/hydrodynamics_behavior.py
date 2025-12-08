import carb
import omni.kit.window.property
import numpy as np
import omni.physx
import warp as wp
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)
from omni.kit.scripting import BehaviorScript
from pxr import Sdf, UsdPhysics
from isaacsim.core.prims import RigidPrim
from .warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

class HydrodynamicsBehavior(BehaviorScript):
    """
    Behavior script that applies buoyancy, drag, lift, and added mass forces to a rigid body.
    Hydrodynamics are modeled based on a cube with uniformed material density.
    """
    BEHAVIOR_NS = "hydrodynamicsBehavior"

    VARIABLES_TO_EXPOSE = [
        # General Properties
        {"attr_name": "waterDensity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1025.0, "doc": "Density of the fluid in kg/m^3."},
        {"attr_name": "gravity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 9.81, "doc": "Gravitational acceleration in m/s^2."},
        # Object Dimensions
        {"attr_name": "xDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local X-axis (m)."},
        {"attr_name": "yDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Y-axis (m)."},
        {"attr_name": "zDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Z-axis (m)."},
        # Drag Properties
        {"attr_name": "linearDragCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.2, "doc": "Quadratic linear drag coefficient (Cd)."},
        {"attr_name": "angularDragCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.8, "doc": "Quadratic angular drag coefficient."},
        {"attr_name": "linearDamping", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 300.0, "doc": "Linear damping multiplier for low-speed stability."},
        {"attr_name": "angularDamping", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 150.0, "doc": "Linear angular damping for low-speed stability."},
        # Added Mass Properties
        {"attr_name": "linearAddedMassCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.05, "doc": "Added mass coefficient for surge, sway, and heave acceleration."},
        {"attr_name": "angularAddedMassCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.02, "doc": "Added mass coefficient for roll, pitch, and yaw acceleration."},
        # Lift Properties
        {"attr_name": "liftCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "A multiplier for the overall strength of the lift force."}
    ]

    def on_init(self):
        self._hydro_calculator = None
        self._rigid_prim = None
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)
        
        # Physics-based update
        self._physx_subscription = None
        # Initialize Warp Global
        try:
            wp.init()
        except Exception as e:
            carb.log_error(f"Failed to initialize NVIDIA Warp: {e}")

        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        self._reset()
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        self._setup()
        physx_interface = omni.physx.get_physx_interface()
        self._physx_subscription = physx_interface.subscribe_physics_step_events(self._on_physics_step)

    def on_stop(self):
        self._reset()

    # FixedUpdate
    def _on_physics_step(self, delta_time:float):
        if delta_time <= 1e-6 or self._rigid_prim is None:
            return
        self._apply_behavior(delta_time)

    def _setup(self):
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"HydrodynamicsBehavior on prim {self.prim_path} requires a RigidBody component.")
            return
        
        self._rigid_prim = RigidPrim(str(self.prim_path))

        self._hydro_calculator = WarpHydrodynamicsWrapper(
            width=self._get_exposed_variable("xDimension"), 
            depth=self._get_exposed_variable("yDimension"), 
            height=self._get_exposed_variable("zDimension"),
            linear_drag_coefficient=self._get_exposed_variable("linearDragCoefficient"),
            angular_drag_coefficient=self._get_exposed_variable("angularDragCoefficient"),
            linear_damping=self._get_exposed_variable("linearDamping"),
            angular_damping=self._get_exposed_variable("angularDamping"),
            water_density=self._get_exposed_variable("waterDensity"), 
            gravity=self._get_exposed_variable("gravity"),
            linear_mass_coeff = self._get_exposed_variable("linearAddedMassCoefficient"),
            angular_mass_coeff = self._get_exposed_variable("angularAddedMassCoefficient"),
            lift_coefficient=self._get_exposed_variable("liftCoefficient"),
            device="cuda:0"
        )

        # Fetch mass to avoid explosions
        self._mass = self._rigid_prim.get_masses()[0]
        carb.log_info(f"HydrodynamicsBehavior (Warp GPU) initialized for {self.prim_path}")

    def _apply_behavior(self, delta_time):
        # Get CPU State (Update to View Tensors for RL environment)
        positions, orientations = self._rigid_prim.get_world_poses()
        linear_velocity = self._rigid_prim.get_linear_velocities()[0]
        angular_velocity = self._rigid_prim.get_angular_velocities()[0]

        # Estimate acceleration
        linear_acceleration = (linear_velocity - self._last_linear_velocity) / delta_time
        angular_acceleration = (angular_velocity - self._last_angular_velocity) / delta_time
        
        # Prepare Numpy inputs
        np_pos = positions
        np_orientation = np.array([[orientations[0][1], orientations[0][2], orientations[0][3], orientations[0][0]]], dtype=np.float32)
        np_lin_vel = np.array([linear_velocity], dtype=np.float32)
        np_ang_vel = np.array([angular_velocity], dtype=np.float32)
        np_lin_acc = np.array([linear_acceleration], dtype=np.float32)
        np_ang_acc = np.array([angular_acceleration], dtype=np.float32)
        """
        wp_position = wp.from_numpy(positions, dtype=wp.vec3, device="cuda:0")
        orientation_quat = np.array([[orientations[0][1], orientations[0][2], orientations[0][3], orientations[0][0]]], dtype=np.float32)
        wp_orientation = wp.from_numpy(orientation_quat, dtype=wp.quat, device="cuda:0")
        wp_linear_velocity = wp.from_numpy(np.array([linear_velocity]), dtype=wp.vec3, device="cuda:0")
        wp_angular_velocity = wp.from_numpy(np.array([angular_velocity]), dtype=wp.vec3, device="cuda:0")
        wp_linear_acceleration = wp.from_numpy(np.array([linear_acceleration]), dtype=wp.vec3, device="cuda:0")
        wp_angular_acceleration = wp.from_numpy(np.array([angular_acceleration]), dtype=wp.vec3, device="cuda:0")
        """

        # Run GPU Kernel 
        (wp_buoyancy_force, wp_drag_force, wp_lift_force, wp_drag_torque, wp_added_mass_force, 
         wp_added_mass_torque, wp_center_of_buoyancy, wp_center_of_pressure) = self._hydro_calculator.calculate_hydrodynamic_forces(
            np_pos, np_orientation, np_lin_vel, np_ang_vel, 
            np_lin_acc, np_ang_acc
        )

        # Readback Results 
        np_buoyancy_force = wp_buoyancy_force.numpy()[0]
        np_drag_force = wp_drag_force.numpy()[0]
        np_lift_force = wp_lift_force.numpy()[0]
        np_added_mass_force = wp_added_mass_force.numpy()[0]
        np_drag_torque = wp_drag_torque.numpy()[0]
        np_added_mass_torque = wp_added_mass_torque.numpy()[0]
        np_center_of_buoyancy = wp_center_of_buoyancy.numpy()[0]
        np_center_of_pressure = wp_center_of_pressure.numpy()[0]

        # Calculate torques generated by off-center forces
        np_position = positions[0]
        torque_from_buoyancy = np.cross(np_center_of_buoyancy - np_position, np_buoyancy_force)
        torque_from_drag = np.cross(np_center_of_pressure - np_position, np_drag_force)
        torque_from_lift = np.cross(np_center_of_pressure - np_position, np_lift_force)

        # Calculate final force ad torque
        net_force = np_buoyancy_force + np_drag_force + np_lift_force + np_added_mass_force
        net_torque = torque_from_buoyancy + torque_from_drag + torque_from_lift + np_drag_torque + np_added_mass_torque

        # Safety Clamp (Prevent Explosions)
        MAX_ACCEL = 500.0 
        max_force = self._mass * MAX_ACCEL
        force_mag = np.linalg.norm(net_force)
        if force_mag > max_force:
            scale = max_force / force_mag
            net_force *= scale
            net_torque *= scale
        
        # Apply to Simulation
        self._rigid_prim.apply_forces_and_torques_at_pos(
            forces=np.expand_dims(net_force, axis=0),
            torques=np.expand_dims(net_torque, axis=0),
            positions=np.expand_dims(np_position, axis=0),
            is_global=True,
            indices=np.array([0])
        )

        # Update stored velocities for the next frame
        self._last_linear_velocity = linear_velocity
        self._last_angular_velocity = angular_velocity

    def _reset(self):
        self._hydro_calculator = None
        self._rigid_prim = None
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)
        self._physx_subscription = None

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)