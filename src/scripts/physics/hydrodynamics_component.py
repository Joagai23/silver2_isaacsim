import carb
import omni.kit.window.property
import numpy as np
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
from .hydrodynamics import Hydrodynamics

# Log Drag
import csv
import datetime
import os


class HydrodynamicsComponent(BehaviorScript):
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
        {"attr_name": "width", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local X-axis (m)."},
        {"attr_name": "depth", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Y-axis (m)."},
        {"attr_name": "height", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Z-axis (m)."},
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
        # Log Drag
        script_directory = os.path.dirname(__file__)
        self._log_file_path = os.path.join(script_directory, "torque_log.csv")

        self._hydro_calculator = None
        self._rigid_prim = None
        self._last_linear_velocity = np.zeros(3)
        self._last_angular_velocity = np.zeros(3)
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        self._reset()
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        self._setup()

    def on_stop(self):
        self._reset()
    
    def on_update(self, current_time: float, delta_time: float):
        if delta_time <= 1e-6 or self._rigid_prim is None:
            return
        self._apply_behavior(delta_time)
    
    def _setup(self):
        # Log Drag
        header = ['timestamp', 'angular_velocity', 'torque_from_buoyancy', 'drag_torque', 'net_torque', 
                  'position_z', 'submerssion_ratio', 'center_of_buoyancy', 'added_mass_force', 'added_mass_torque'
                  'lift_force', 'torque_from_lift', 'orientation_quaternion']
        try:
            with open(self._log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            carb.log_info(f"Log file created at {self._log_file_path}")
        except Exception as e:
            carb.log_error(f"Failed to create log file: {e}")

        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"HydrodynamicsComponent on prim {self.prim_path} requires a RigidBody component.")
            return
        
        self._rigid_prim = RigidPrim(str(self.prim_path))

        self._hydro_calculator = Hydrodynamics(
            width=self._get_exposed_variable("width"), 
            depth=self._get_exposed_variable("depth"), 
            height=self._get_exposed_variable("height"),
            linear_drag_coefficient=self._get_exposed_variable("linearDragCoefficient"),
            angular_drag_coefficient=self._get_exposed_variable("angularDragCoefficient"),
            linear_damping=self._get_exposed_variable("linearDamping"),
            angular_damping=self._get_exposed_variable("angularDamping"),
            water_density=self._get_exposed_variable("waterDensity"), 
            gravity=self._get_exposed_variable("gravity"),
            linear_mass_coeff = self._get_exposed_variable("linearAddedMassCoefficient"),
            angular_mass_coeff = self._get_exposed_variable("angularAddedMassCoefficient"),
            lift_coefficient=self._get_exposed_variable("liftCoefficient")
        )
        carb.log_info(f"HydrodynamicsComponent initialized for {self.prim_path}")
    
    def _apply_behavior(self, delta_time):
        # Get current state from the simulator
        positions, orientations = self._rigid_prim.get_world_poses()
        position = positions[0]
        orientation_quat = orientations[0]
        linear_velocity = self._rigid_prim.get_linear_velocities()[0]
        angular_velocity = self._rigid_prim.get_angular_velocities()[0]

        # Estimate acceleration
        linear_acceleration = (linear_velocity - self._last_linear_velocity) / delta_time
        angular_acceleration = (angular_velocity - self._last_angular_velocity) / delta_time
        
        # Get hydrodynamic forces, torques. and centers of pressure
        (buoyancy_force, drag_force, lift_force, 
         drag_torque, added_mass_force, added_mass_torque, 
         center_of_buoyancy, center_of_pressure, submersion_ratio) = self._hydro_calculator.calculate_hydrodynamic_forces(
            position=position,
            orientation_quat= np.array([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]]),
            linear_vel=linear_velocity,
            angular_vel=angular_velocity,
            linear_accel=linear_acceleration,
            angular_accel=angular_acceleration
        )

        # Calculate torques generated by off-center forces
        # Assuming the prim's origin is its center of mass
        torque_from_buoyancy = np.cross(center_of_buoyancy - position, buoyancy_force)
        torque_from_drag = np.cross(center_of_pressure - position, drag_force)
        torque_from_lift = np.cross(center_of_buoyancy - position, lift_force)

        # Sum forces and torques to get net values
        net_force = buoyancy_force + drag_force + added_mass_force + lift_force
        net_torque = torque_from_buoyancy + torque_from_drag + drag_torque + added_mass_torque + torque_from_lift
        
        # Apply the calculated forces and torques
        self._rigid_prim.apply_forces_and_torques_at_pos(
            forces=np.expand_dims(net_force, axis=0),
            torques=np.expand_dims(net_torque, axis=0),
            is_global=True,
            indices=np.array([0])
        )

        # Update stored velocities for the next frame
        self._last_linear_velocity = linear_velocity
        self._last_angular_velocity = angular_velocity

        # Log Drag
        log_row = [
            datetime.datetime.now().isoformat(),
            angular_velocity,
            torque_from_buoyancy,
            drag_torque,
            net_torque,
            position[2],
            submersion_ratio,
            center_of_buoyancy,
            added_mass_force,
            added_mass_torque,
            lift_force,
            torque_from_lift,
            orientation_quat
        ]
        try:
            with open(self._log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)
        except Exception as e:
            carb.log_warn(f"Failed to write to log file: {e}")

    def _reset(self):
        self._hydro_calculator = None
        self._rigid_prim = None
        self._last_linear_velocity = np.zeros(3)
        self._last_angular_velocity = np.zeros(3)

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)