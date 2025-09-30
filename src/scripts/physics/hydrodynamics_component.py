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
"""
# Log Drag
import csv
import datetime
import os
"""

class HydrodynamicsComponent(BehaviorScript):
    """
    Behavior script that applies buoyancy and drag forces to a rigid body.
    Hydrodynamics are modeled based on a cube with uniformed material density.
    """
    BEHAVIOR_NS = "hydrodynamicsBehavior"

    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "waterDensity",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1000.0,
            "doc": "Density of the fluid in kg/m^3. Default is 1000 for water.",
        },
        {
            "attr_name": "gravity",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 9.81,
            "doc": "Gravitational acceleration in m/s^2.",
        },
        {
            "attr_name": "width",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1.0,
            "doc": "Lenght of the object in the X-axis in m.",
        },
        {
            "attr_name": "depth",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1.0,
            "doc": "Lenght of the object in the Y-axis in m.",
        },
        {
            "attr_name": "height",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1.0,
            "doc": "Lenght of the object in the Z-axis in m.",
        },
        {
            "attr_name": "dragCoefficient",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 0.8,
            "doc": "Quantification of drag resistance when fully submerged.",
        },
        {
            "attr_name": "angularDragCoefficient",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 0.5,
            "doc": "Quantification of angular drag resistance when fully submerged.",
        },
    ]

    def on_init(self):
        """
        # Log Drag
        script_directory = os.path.dirname(__file__)
        self._log_file_path = os.path.join(script_directory, "drag_log.csv")
        """

        """Called when the script is assigned to a prim."""
        self._hydro_calculator = None
        self._rigid_prim = None

        # Expose the variables as USD attributes
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)

        # Refresh the property windows to show the exposed variables
        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        """Called when the script is unassigned from a prim."""
        self._reset()
        # Exposed variables should be removed if the script is no longer assigned to the prim
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        """Called when `play` is pressed."""
        self._setup()

    def on_stop(self):
        """Called when `stop` is pressed."""
        self._reset()

    def on_update(self, current_time: float, delta_time: float):
        """Called on per frame update events that occur when `playing`."""
        if delta_time <= 0:
            return
        if self._rigid_prim is None:
            return

        self._apply_behavior()

    def _reset(self):
        self._hydro_calculator = None
        self._rigid_prim = None

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
    
    def _setup(self):
        """
        # Log Drag
        header = ['timestamp', 
                  'linear_drag_x', 'linear_drag_y', 'linear_drag_z',
                  'angular_drag_x', 'angular_drag_y', 'angular_drag_z']
        try:
            with open(self._log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            carb.log_info(f"Log file created at {self._log_file_path}")
        except Exception as e:
            carb.log_error(f"Failed to create log file: {e}")
        """

        """Initialize physics objects and read UI values"""
        # Check if the prim has physics enabled
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"HydrodynamicsComponent on prim {self.prim_path} requires a RigidBody component.")
            return
        
        # Get prim path
        self._rigid_prim = RigidPrim(str(self.prim_path))

        # Fetch the exposed attributes
        water_density = self._get_exposed_variable("waterDensity")
        gravity = self._get_exposed_variable("gravity")
        width = self._get_exposed_variable("width")
        depth = self._get_exposed_variable("depth")
        height = self._get_exposed_variable("height")
        drag_coefficient = self._get_exposed_variable("dragCoefficient")
        angular_drag_coefficient = self._get_exposed_variable("angularDragCoefficient")

        self._hydro_calculator = Hydrodynamics(
            width=width,
            depth=depth,
            height=height,
            drag_coefficient=drag_coefficient,
            angular_drag_coefficient=angular_drag_coefficient,
            water_density=water_density, 
            gravity=gravity
        )
        carb.log_info(f"HydrodynamicsComponent initialized for {self.prim_path}")

    def _apply_behavior(self):
        """Calculates and applies the buoyancy force."""
        # Get the poses for all prims in the view (in this case, just the primitive object)
        positions, orientations = self._rigid_prim.get_world_poses()
        # The result is an array, so we must index it to get the first (and only) element
        position = positions[0]
        orientation_quat = orientations[0]
        # Fetch linear and angular velocities
        linear_velocity = self._rigid_prim.get_linear_velocities()
        angular_velocity = self._rigid_prim.get_angular_velocities()

        # Reorder quat for our function and calculate force
        quat_xyzw = np.array([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        
        # Get hydrodynamic forces and torques
        total_force, total_torque = self._hydro_calculator.calculate_hydrodynamic_forces(
            position=position,
            orientation_quat=quat_xyzw,
            linear_vel=linear_velocity,
            angular_vel=angular_velocity
        )
        """
        # Log Drag
        log_row = [
            datetime.datetime.now().isoformat(),
            drag_force[0][0],
            drag_force[0][1],
            drag_force[0][2],
            total_torque[0],
            total_torque[1],
            total_torque[2],
        ]
        try:
            with open(self._log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)
        except Exception as e:
            carb.log_warn(f"Failed to write to log file: {e}")
        """
        # Apply the calculated force
        self._rigid_prim.apply_forces_and_torques_at_pos(
            forces=np.expand_dims(total_force, axis=0),
            torques=np.expand_dims(total_torque, axis=0),
            is_global=True
        )