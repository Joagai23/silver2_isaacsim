import numpy as np
from omni.kit.scripting import BehaviorScript
from omni.kit.commands import execute
from omni.kit.window.property import get_window
from pxr import Gf, Sdf
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)

class AnimateWaterBehavior(BehaviorScript):
    """
    Behavior script that animates the motion of the Ocean surface by updating its shader.
    """
    BEHAVIOR_NS = "animateWaterBehavior"

    VARIABLES_TO_EXPOSE = [
        {"attr_name": "xDirection", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Direction of current in the X-axis."},
        {"attr_name": "yDirection", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Direction of current in the Y-axis."},
        {"attr_name": "xVelocity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.02, "doc": "Magnitude of current in the X-axis."},
        {"attr_name": "yVelocity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.02, "doc": "Magnitude of current in the Y-axis."},
        {"attr_name": "oceanMaterial", "attr_type": Sdf.ValueTypeNames.String, "default_value": "/World/Environment/Looks/LightBlueOceanWater", "doc": "Material containing ocean shader."} 
    ]
    def on_init(self):
        # Read and process exposed variables
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        get_window().request_rebuild()
        self._read_exposed_variables()

    def on_destroy(self):
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            get_window().request_rebuild()

    def on_play(self):
        # Initialize Local Translate
        self._translate = Gf.Vec2f(0.0, 0.0)

    def on_update(self, current_time: float, delta_time: float):
        # Save translate value before update
        self._prev_value = self._translate
        # Calculate new translate
        self._translate[0] += self._x_direction * self._x_velocity * delta_time
        self._translate[1] += self._y_direction * self._y_velocity * delta_time
        # Update Visual Property
        execute(
            "ChangeProperty",
            prop_path=f"{self._ocean_material}/Shader.inputs:texture_translate",
            value=self._translate,
            prev=self._prev_value
        )

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
    
    def _read_exposed_variables(self):
        self._x_direction = np.clip(float(self._get_exposed_variable("xDirection")), -1.0, 1.0)
        self._y_direction = np.clip(float(self._get_exposed_variable("yDirection")), -1.0, 1.0)
        self._x_velocity = self._get_exposed_variable("xVelocity")
        self._y_velocity = self._get_exposed_variable("yVelocity")
        self._ocean_material = str(self._get_exposed_variable("oceanMaterial"))