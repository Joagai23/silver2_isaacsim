import omni.kit.window.property
import numpy as np
import itertools
from omni.kit.scripting import BehaviorScript
from pxr import Gf, Sdf, Usd, UsdPhysics
from isaacsim.replicator.behavior.utils.behavior_utils import create_exposed_variables, get_exposed_variable, check_if_exposed_variables_should_be_removed, remove_exposed_variables
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from .custom_keyboard_cmd import custom_keyboard_cmd
from isaacsim.replicator.behavior.utils.scene_utils import get_world_location, get_world_rotation, set_location, set_orientation, set_rotation_with_ops
from .ui_widget import UIWidget
from .look_at import LookAt

class DynamicCameraBehavior(BehaviorScript):
    """
    Behavior script that controls a component containing a Camera.
    The Dynamic Object (DB) can either move freely of look at one of the rigidprims in the scene.
    It switches camera rendering depending on its height value.
    """
    BEHAVIOR_NS = "dynamicCameraBehavior"
    VARIABLES_TO_EXPOSE = [
        {"attr_name": "linearVelocity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 10.0, "doc": "Magnitude of the camera's linear velocity."},
        {"attr_name": "angularVelocity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 15.0, "doc": "Magnitude of the camera's angular velocity."},
        {"attr_name": "worldPrim", "attr_type": Sdf.ValueTypeNames.String, "default_value": "/World", "doc": "Primitive containing all target bodies."},
    ]

    def on_init(self):
        if custom_keyboard_cmd is None:
            return
        # Read and process exposed variables
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        omni.kit.window.property.get_window().request_rebuild()
        self._read_exposed_variables()
        # Override keyboard class init parameters when scene starts
        self._define_input_controller()
        # I have to initialize it because even within exception-catching it breaks
        self._current_target = None
        self._overlay_script = UIWidget()
        self._look_at_script = LookAt(camera_prim=self.prim, stage=self.stage)

    def on_destroy(self):
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        # Fetch list of all targets on the scene
        self._fetch_rigidbody_targets()

    def on_update(self, current_time: float, delta_time: float):
        # Get user input
        position_input, rotation_input, switch_tracking, change_target  = self._get_input()
        # Update Movement and Torque
        self._calculate_movement_and_torque(position_input, rotation_input, change_target, switch_tracking, delta_time)

    def _define_input_controller(self):
        self._input_cmd = custom_keyboard_cmd(
            base_command = np.array([0.0, 0.0, 0.0]),
            input_keyboard_mapping = {
                # Right command
                "D": [1.0, 0.0, 0.0],
                # Left command
                "A": [-1.0, 0.0, 0.0],
                # Up command
                "W": [0.0, 1.0, 0.0],
                # Down command
                "S": [0.0, -1.0, 0.0],
                # Forward command
                "UP": [0.0, 0.0, -1.0],
                # Backward command
                "DOWN": [0.0, 0.0, 1.0],
            },
            toggle_command = int(1.0),
            toggle_keyboard_mapping = {
                # Activate or deactivate focusing a target
                "T": int(-1.0)
            }
        )
        self._torque_cmd = custom_keyboard_cmd(
            base_command=np.array([0.0, 0.0, 0.0]),
            input_keyboard_mapping={
                # yaw command (left)
                "J": [0.0, 0.0, 1.0],
                # yaw command (right)
                "L": [0.0, 0.0, -1.0],
                # pitch command (up)
                "I": [0.0, 1.0, 0.0],
                # pitch command (down)
                "K": [0.0, -1.0, 0.0],
                # row command (left)
                "LEFT": [-1.0, 0.0, 0.0],
                # row command (negative)
                "RIGHT": [1.0, 0.0, 0.0],
                },
            toggle_command = int(1.0),
            toggle_keyboard_mapping = {
                # Switch target command
                "TAB": int(-1.0)
            }
        )
        self._toggle_cmd = self._input_cmd._toggle_command
        self._tab_cmd = self._torque_cmd._toggle_command

    def _get_input(self) -> tuple[Gf.Vec3f, Gf.Vec3f, bool, bool]:
        """
        Process keyboard user input and compute any change in toggle key (default = T) or tab (TAB)
        If T or Tab toggle is registered True is returned. Otherwise its false.
        """
        toggle_cmd = int(self._input_cmd._toggle_command)
        has_toggled = False
        tab_cmd = int(self._torque_cmd._toggle_command)
        has_tabbed = False

        if self._toggle_cmd is not toggle_cmd:
            self._toggle_cmd = toggle_cmd
            has_toggled = True
        if self._tab_cmd is not tab_cmd:
            self._tab_cmd = tab_cmd
            has_tabbed = True
        
        return Gf.Vec3f(*self._input_cmd._base_command), Gf.Vec3f(*self._torque_cmd._base_command), has_toggled, has_tabbed
    
    def _get_local_vectors(self, local_quaternion) -> tuple[Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]:
        """
        Compute world matrix and extract right, forward, right and upwards vectors
        """
        rotation_matrix = Gf.Matrix3d(local_quaternion)
        right_vector = rotation_matrix.GetRow(0).GetNormalized()
        upwards_vector = rotation_matrix.GetRow(1).GetNormalized()
        forward_vector = rotation_matrix.GetRow(2).GetNormalized()

        return right_vector, forward_vector, upwards_vector
    
    def _get_next_target(self):
        # Update current target to next one in the cycle
        if not self._target_iterator or self._current_target is None:
            return
        else:
            self._current_target = next(self._target_iterator)
            self._last_known_target = self._current_target
        self._external_script_target_logic(self._current_target)

    def _toggle_targets(self):
        # Activate or deactivate target tracking
        # If activated target is first object in list (if exists)
        if self._current_target is None and self._target_iterator:
            self._current_target = self._last_known_target
        else:
            self._current_target = None
        self._external_script_target_logic(self._current_target)

    def _calculate_movement_and_torque(self, position_input:Gf.Vec3f, rotation_input:Gf.Vec3f, change_target:bool, 
                                       switch_tracking:bool, delta_time:float):
        # Get current position and orientation
        current_position = get_world_location(self.prim)
        current_orientation_quat = get_world_rotation(self.prim).GetQuat()
        
        # Get orientation vectors
        right, forward, upward = self._get_local_vectors(current_orientation_quat)

        # Movement Logic
        moveDirection:Gf.Vec3f = right * position_input[0] + forward * position_input[2] + upward * position_input[1]
        moveDirection = moveDirection.GetNormalized()
        position_offset = current_position + moveDirection * self._linear_velocity * delta_time
        set_location(self.prim, position_offset)

        # Toggle Logic
        if change_target:
            self._get_next_target()
        if switch_tracking:
            self._toggle_targets()

        # Rotation Logic
        try:
            if self._current_target is None:
                # Manual Rotation
                roll_quat = Gf.Rotation(forward, rotation_input[0] * self._angular_velocity * delta_time).GetQuat()
                pitch_quat = Gf.Rotation(right, rotation_input[1] * self._angular_velocity * delta_time).GetQuat()
                yaw_quat = Gf.Rotation(upward, rotation_input[2] * self._angular_velocity * delta_time).GetQuat()
                new_orientation = yaw_quat * pitch_quat * roll_quat * current_orientation_quat
                set_orientation(self.prim, new_orientation)
            else:
                # Automated Rotation
                look_at_rotation, new_position = self._look_at_script.follow_target(delta_time)
                if look_at_rotation:
                    set_rotation_with_ops(self.prim, look_at_rotation)
                if new_position:
                    set_location(self.prim, new_position)
        except NameError as e:
            print(e)

    def _fetch_rigidbody_targets(self):
        """
        Finds all immediate children of /World that are either a rigid body
        or contain a rigid body in their hierarchy.
        """
        world_prim = self._stage.GetPrimAtPath(self._world_prim_path)

        if not world_prim:
            print(f"Error: {self._world_prim_path} prim not found.")
            self._targets = []
            self._target_iterator = None
            return
        
        self._targets = []
        for child_prim in world_prim.GetChildren():
            rigid_bodies = self._find_rigid_bodies(child_prim)
            
            # We only care about targets that actually have rigid bodies
            if rigid_bodies:
                self._targets.append({
                    "parent": child_prim.GetPath(),
                    "rigid_bodies": rigid_bodies
                })

        self._current_target = None
        if self._targets:
            self._target_iterator = itertools.cycle(self._targets)
            self._current_target = next(self._target_iterator)
            self._last_known_target = self._current_target
            # Load active instance to UI Widget and Camera Controller
            self._external_script_target_logic(self._current_target)
        else:
            self._target_iterator = None
            print(f"No targets with rigid bodies found under {self._world_prim_path}")
    
    def _find_rigid_bodies(self, prim: Usd.Prim) -> list[Usd.Prim]:
        """
        Recursively finds all descendant prims that have the UsdPhysics.RigidBodyAPI.
        """
        rigid_bodies_found = []

        def _recursive_find(current_prim):
            if not current_prim or not current_prim.IsValid():
                return
            
            # Check if this prim is a rigid body
            if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies_found.append(current_prim.GetPath())
            
            # Continue searching all children
            for child in current_prim.GetChildren():
                _recursive_find(child)

        _recursive_find(prim)
        return rigid_bodies_found

    def _get_exposed_variable(self, attr_name):
        """Helper to get the value of an exposed UI variable."""
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
    
    def _read_exposed_variables(self):
        self._linear_velocity = self._get_exposed_variable("linearVelocity")
        self._angular_velocity = self._get_exposed_variable("angularVelocity")
        self._world_prim_path = str(self._get_exposed_variable("worldPrim"))

    def _external_script_target_logic(self, current_target_data):
        if current_target_data is None:
            self._overlay_script.set_overlay_text("None")
            self._look_at_script.set_target(None)
        else:
            target_path_str = str(current_target_data["parent"])
            self._overlay_script.set_overlay_text(target_path_str)
            # Pass the whole info dictionary to the LookAt script
            self._look_at_script.set_target(current_target_data)
