import carb
import omni.kit.window.property
import sys
import os

from isaacsim.replicator.behavior.utils.behavior_utils import create_exposed_variables, get_exposed_variable, check_if_exposed_variables_should_be_removed, remove_exposed_variables
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from omni.kit.scripting import BehaviorScript
from pxr import Sdf, UsdGeom, Gf
from isaacsim.replicator.behavior.utils.scene_utils import get_world_location
import numpy as np
import warp as wp

# Full path to OCEANSIM sensors folder
OCEANSIM_SENSORS_PATH = os.path.expanduser("~/Documents/isaac-sim/extsUser/OceanSim/isaacsim/oceansim/sensors")
if OCEANSIM_SENSORS_PATH not in sys.path:
    sys.path.append(OCEANSIM_SENSORS_PATH)
    print(f"Added OceanSim sensors path: {OCEANSIM_SENSORS_PATH}")

# Import the UW_Camera
try:
    from UW_Camera import UW_Camera
except ImportError as e:
    carb.log_error("Failed to import UW_Camera. Please check the OCEANSIM_SENSORS_PATH in the script.")
    carb.log_error(f"Details: {e}")
    UW_Camera = None

class UnderwaterCameraBehavior(BehaviorScript):
    """
    Behavior script that applies the OceanSim underwater rendering effect to the camera it is attached to..
    """
    BEHAVIOR_NS = "underwaterCameraBehavior"

    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "resolutionW",
            "attr_type": Sdf.ValueTypeNames.Int,
            "default_value": 1280,
            "doc": "The width of the camera's render output.",
        },
        {
            "attr_name": "resolutionH",
            "attr_type": Sdf.ValueTypeNames.Int,
            "default_value": 720,
            "doc": "The height of the camera's render output.",
        },
        {
            "attr_name": "backscatterValue",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.063, 0.278, 0.345),
            "doc": "Underwater processing Backscatter Values.",
        },
        {
            "attr_name": "backscatterCoeff",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.04, 0.18, 0.22),
            "doc": "Underwater processing Backscatter Coefficients.",
        },
        {
            "attr_name": "attenCoeff",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.10, 0.03, 0.02),
            "doc": "Underwater processing Attenuation Coefficients.",
        }
    ]

    def on_init(self):
        self._uw_camera = None
        if UW_Camera is None:
            return
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        omni.kit.window.property.get_window().request_rebuild()
        self._backscatter_value = wp.vec3f(self._get_exposed_variable("backscatterValue"))
        self._backscatter_coeff = wp.vec3f(self._get_exposed_variable("backscatterCoeff"))
        self._attenuation_coeff = wp.vec3f(self._get_exposed_variable("attenCoeff"))
        self._height_state = -2

    def on_destroy(self):
        self._cleanup()
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        if UW_Camera is None:
            carb.log_error("Cannot start UnderwaterCameraBehavior because UW_Camera class could not be imported.")
            return
        if not self.prim.IsA(UsdGeom.Camera):
            carb.log_warn(f"UnderwaterCameraBehavior should be attached to a Camera prim, but it is on a {self.prim.GetTypeName()}.")
            return
        
        # Get parameters from the UI
        width = self._get_exposed_variable("resolutionW")
        height = self._get_exposed_variable("resolutionH")
        yaml_path = self._get_exposed_variable("uwYamlPath")

        if width is None:
            width = 1280
            carb.log_warn("Resolution width was not ready, falling back to default 1280.")
        if height is None:
            height = 720
            carb.log_warn("Resolution height was not ready, falling back to default 720.")
        if not yaml_path:
            yaml_path = None
        
        self._uw_camera = UW_Camera(
            prim_path=str(self.prim_path),
            name=f"{self.prim.GetName()}_UW_Sensor",
            resolution=(width, height)
        )
        self._uw_camera.initialize(viewport=True)
        self._switch_camera()
        carb.log_info(f"UnderwaterCameraBehavior initialized for {self.prim_path}")

    def on_stop(self):
        self._cleanup()

    def on_update(self, current_time: float, delta_time: float):
        self._switch_camera()
        if self._uw_camera:
            self._uw_camera.render()

    def _switch_camera(self):
        current_height_state = np.sign(get_world_location(self.prim)[2])
        # Check change in state
        if current_height_state != 0 and current_height_state != self._height_state:
            # Change to above-water
            if current_height_state > 0:
                self._height_state = 1
                self._uw_camera._backscatter_coeff = wp.vec3f(0.0, 0.0, 0.0)
                self._uw_camera._backscatter_value = wp.vec3f(0.0, 0.0, 0.0)
                self._uw_camera._atten_coeff = wp.vec3f(0.0, 0.0, 0.0)
            # Change to underwater
            else:
                self._height_state = -1
                self._uw_camera._backscatter_coeff = self._backscatter_coeff
                self._uw_camera._backscatter_value = self._backscatter_value
                self._uw_camera._atten_coeff = self._attenuation_coeff

    def _cleanup(self):
        """Closes the UW_Camera sensor and releases its resources."""
        if self._uw_camera:
            self._uw_camera.close()
            self._uw_camera = None
            carb.log_info(f"UnderwaterCameraBehavior cleaned up for {self.prim_path}")

    def _get_exposed_variable(self, attr_name):
        """Helper to get the value of an exposed UI variable."""
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)