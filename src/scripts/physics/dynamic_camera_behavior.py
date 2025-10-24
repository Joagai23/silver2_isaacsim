import omni.kit.window.property
import numpy as np
from omni.kit.scripting import BehaviorScript
from pxr import Gf
from isaacsim.replicator.behavior.utils.behavior_utils import create_exposed_variables, get_exposed_variable, check_if_exposed_variables_should_be_removed, remove_exposed_variables
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from .custom_keyboard_cmd import custom_keyboard_cmd
from isaacsim.replicator.behavior.utils.scene_utils import get_world_location, get_world_rotation, set_location, set_orientation
from omni.usd import get_world_transform_matrix

class DynamicCameraBehavior(BehaviorScript):
    BEHAVIOR_NS = "dynamicCameraBehavior"
    VARIABLES_TO_EXPOSE = []
    def on_init(self):
        if custom_keyboard_cmd is None:
            return

        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        # Override keyboard class init parameters when scene starts
        self._define_input_controller()

    def on_update(self, current_time: float, delta_time: float):
        # Get user input
        position_input, rotation_input, change_target, switch_tracking = self._get_input()
        print(position_input)
        # Update Movement and Torque
        self._calculate_movement_and_torque(position_input, rotation_input, delta_time)

    def _define_input_controller(self):
        self._input_cmd = custom_keyboard_cmd(
            base_command = np.array([0.0, 0.0, 0.0]),
            input_keyboard_mapping = {
                # forward command
                "D": [1.0, 0.0, 0.0],
                # backward command
                "A": [-1.0, 0.0, 0.0],
                # leftward command
                "W": [0.0, 1.0, 0.0],
                # rightward command
                "S": [0.0, -1.0, 0.0],
                # rise command
                "UP": [0.0, 0.0, 1.0],
                # sink command
                "DOWN": [0.0, 0.0, -1.0],
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
    
    def _get_local_vectors(self) -> tuple[Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]:
        """
        Compute world matrix and extract forward, right and upwards vectors
        """
        world_transform_matrix = get_world_transform_matrix(self.prim)
        rotation_matrix = Gf.Matrix3d(world_transform_matrix.ExtractRotationMatrix())

        forward_vector = rotation_matrix.GetColumn(1)
        right_vector = rotation_matrix.GetColumn(0)
        upwards_vector = rotation_matrix.GetColumn(2)

        return forward_vector, right_vector, upwards_vector
    
    def _calculate_movement_and_torque(self, position_input:Gf.Vec3f, rotation_input:Gf.Vec3f, delta_time:float):
        # Get current position and orientation
        current_position = get_world_location(self.prim)
        current_orientation_quat = get_world_rotation(self.prim).GetQuat()
        # Get orientation vectors
        forward, right, upward = self._get_local_vectors()
        
        # Normalize vectors
        forward = forward.GetNormalized()
        right = right.GetNormalized()
        upward = upward.GetNormalized()

        # Movement Logic
        moveDirection:Gf.Vec3f = right * position_input[0] + forward * position_input[1] + upward * position_input[2]
        moveDirection = moveDirection.GetNormalized()
        movement_speed = 1.0
        position_offset = current_position + moveDirection * movement_speed * delta_time
        set_location(self.prim, position_offset)

        # Rotation Logic
        rotation_speed = 10.0
        roll_quat = Gf.Rotation(forward, rotation_input[0] * rotation_speed * delta_time).GetQuat()
        pitch_quat = Gf.Rotation(right, rotation_input[1] * rotation_speed * delta_time).GetQuat()
        yaw_quat = Gf.Rotation(upward, rotation_input[2] * rotation_speed * delta_time).GetQuat()
        new_orientation = yaw_quat * pitch_quat * roll_quat * current_orientation_quat
        set_orientation(self.prim, new_orientation)
    
    def _get_exposed_variable(self, attr_name):
        """Helper to get the value of an exposed UI variable."""
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
