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
            "default_value": 0.7,
            "doc": "Quantification of angular drag resistance when fully submerged.",
        },
        {
            "attr_name": "dragCoefficient",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 0.8,
            "doc": "Quantification of drag resistance when fully submerged.",
        },
        {
            "attr_name": "linearDamping",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 50,
            "doc": "Damping heuristic to approximate linear drag forces",
        },
        {
            "attr_name": "angularDamping",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 100,
            "doc": "Damping heuristic to approximate angular drag forces",
        }
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
        linear_damping = self._get_exposed_variable("linearDamping")
        angular_damping = self._get_exposed_variable("angularDamping")

        self._hydro_calculator = Hydrodynamics(
            width=width,
            depth=depth,
            height=height,
            drag_coefficient=drag_coefficient,
            angular_drag_coefficient=angular_drag_coefficient,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
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
        buoyancy_force, drag_force, lift_force, drag_torque, center_of_buoyancy, center_of_pressure = self._hydro_calculator.calculate_hydrodynamic_forces(
            position=position,
            orientation_quat=quat_xyzw,
            linear_vel=linear_velocity,
            angular_vel=angular_velocity
        )

        # Calculate torques generated by off-center forces
        torque_from_buoyancy = np.cross(center_of_buoyancy - position, buoyancy_force)
        torque_from_drag = np.cross(center_of_pressure - position, drag_force)
        torque_from_lift = np.cross(center_of_buoyancy - position, lift_force)

        # Sum forces and torques to get net values
        net_force = buoyancy_force + drag_force + lift_force
        net_torque = torque_from_buoyancy + torque_from_drag + torque_from_lift + drag_torque
        
        # Apply the calculated force
        self._rigid_prim.apply_forces_and_torques_at_pos(
            forces=np.expand_dims(net_force, axis=0),
            torques=np.expand_dims(net_torque, axis=0),
            is_global=True
        )