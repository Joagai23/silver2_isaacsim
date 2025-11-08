from pxr import Gf, Usd
from isaacsim.replicator.behavior.utils.scene_utils import (
    calculate_look_at_rotation,
    get_rotation_op_and_value,
    get_world_location
)
import carb
import numpy as np

class LookAt:
    def __init__(self, camera_prim, stage:Usd.Stage, follow_distance=10.0, follow_damping=5.0):
        self._up_axis = Gf.Vec3d(0.0, 0.0, 1.0)
        self._follow_distance = follow_distance
        self._follow_damping = follow_damping
        self._camera_prim = camera_prim
        self._stage = stage
        self._target_info = None

        self._set_camera()

    def _set_camera(self):
        # Save the initial rotation op and value of the camera
        self._initial_rotations = get_rotation_op_and_value(self._camera_prim)

    def _get_target_location(self):
        """
        Fetches the target location.
        If the target has rigid children, it returns their average position (centroid).
        Otherwise, it returns the target prim's own position.
        """
        if not self._target_info:
            return Gf.Vec3d(0.0, 0.0, 0.0)

        rigid_bodies = self._target_info["rigid_bodies"]
        
        if not rigid_bodies:
            # Fallback: No rigid children found, just return the static parent's location.
            return get_world_location(self._stage.GetPrimAtPath(self._target_info["parent"]))
        else:
            # We have rigid children! Calculate their average position.
            positions = []
            for child_prim in rigid_bodies:
                positions.append(get_world_location(self._stage.GetPrimAtPath(child_prim)))
            
            if not positions:
                return get_world_location(self._stage.GetPrimAtPath(self._target_info["parent"])) # Fallback

            # Return the average location (centroid) of all rigid parts.
            mean = np.mean(positions, axis=0)
            return Gf.Vec3d(mean[0], mean[1], mean[2])
    
    def set_target(self, target_data):
        """
        Sets the target to follow.
        target_info is a dictionary: {"parent": Usd.Prim, "rigid_bodies": list[Usd.Prim]}
        """
        self._target_info = target_data

    def follow_target(self, delta_time):
        if self._target_info:
            target_location = self._get_target_location()
            eye = get_world_location(self._camera_prim)
            # Location logic
            vec_from_target = eye - target_location
            current_distance = Gf.GetLength(vec_from_target)
            if current_distance > self._follow_distance and delta_time > 0.0:
                dir_from_target = Gf.GetNormalized(vec_from_target)
                desired_pos = target_location + dir_from_target * self._follow_distance
                alpha = self._follow_damping * delta_time
                clamped_alpha = min(1.0, max(0.0, alpha))
                new_camera_location = Gf.Lerp(clamped_alpha, eye, desired_pos)
            else:
                new_camera_location = None
            # Orientation logic
            look_at_rotation = calculate_look_at_rotation(eye, target_location, self._up_axis)
            return look_at_rotation, new_camera_location
        return None, None