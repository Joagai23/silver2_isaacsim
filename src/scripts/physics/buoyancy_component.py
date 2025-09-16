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
from pxr import PhysxSchema
from isaacsim.core.prims import RigidPrim
from .hidrodynamics import Hydrodynamics

class BuoyancyComponent(BehaviorScript):
    """
    Behavior script that applies buoyancy and damping forces to a rigid body.
    """
    BEHAVIOR_NS = "buoyancyBehavior"

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
            "attr_name": "volume",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1.0,
            "doc": "Volume of the object in m^3.",
        },
        {
            "attr_name": "height",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 1.0,
            "doc": "Total height of the object in m.",
        },
        {
            "attr_name": "maxLinearDamping",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 100.0,
            "doc": "Max damping when fully submerged.",
        },
        {
            "attr_name": "maxAngularDamping",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 50.0,
            "doc": "Max angular damping when fully submerged.",
        },
    ]

    def on_init(self):
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

    def _setup(self):
        """Initialize physics objects and read UI values"""
         # Check if the prim has physics enabled
        if not self.prim.HasAPI(UsdPhysics.CollisionAPI):
            carb.log_warn(f"BuoyancyComponent on prim {self.prim_path} requires a Rigid Body component.")
            return
        
        # Get prim paths and APIs
        self._rigid_prim = RigidPrim(str(self.prim_path))
        """stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(self.prim_path))
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)"""

        # Fetch the exposed attributes
        water_density = self._get_exposed_variable("waterDensity")
        gravity = self._get_exposed_variable("gravity")
        volume = self._get_exposed_variable("volume")
        height = self._get_exposed_variable("height")
        max_linear_damping = self._get_exposed_variable("maxLinearDamping")
        max_angular_damping = self._get_exposed_variable("maxAngularDamping")

        # Set prim physix attributes
        """angular_damping_attr = physx_api.GetAngularDampingAttr()
        angular_damping_attr.Set(angular_damping)
        linear_damping_attr = physx_api.GetLinearDampingAttr()
        linear_damping_attr.Set(linear_damping)"""

        self._hydro_calculator = Hydrodynamics(
            total_volume=volume, 
            total_height=height,
            max_linear_damping=max_linear_damping,
            max_angular_damping=max_angular_damping,
            water_density=water_density, 
            gravity=gravity
        )
        carb.log_info(f"BuoyancyComponent initialized for {self.prim_path}")

    def _reset(self):
        self._hydro_calculator = None
        self._rigid_prim = None
        self._view = None

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
        
        # Apply the calculated force
        self._rigid_prim.apply_forces_and_torques_at_pos(
            forces=np.expand_dims(total_force, axis=0),
            torques=np.expand_dims(total_torque, axis=0),
            is_global=True
        )

    def _reset(self):
        """Clears cached objects."""
        self._hydro_calculator = None
        self._rigid_prim = None

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)