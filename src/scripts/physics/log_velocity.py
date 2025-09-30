import carb
from omni.kit.scripting import BehaviorScript
import csv
import datetime
import os
from pxr import UsdPhysics
from isaacsim.core.prims import RigidPrim

class LogVelocity(BehaviorScript):
    def on_init(self):
        script_directory = os.path.dirname(__file__)
        self._log_file_path = os.path.join(script_directory, "velocity_log.csv")

    def on_play(self):
        self._setup()

        header = ['timestamp', 
                  'z_position', 'linear_velocity_z', 'angular_velocity_z',
                  'x_position', 'linear_velocity_x', 'angular_velocity_x',
                  'y_position', 'linear_velocity_y', 'angular_velocity_y']
        try:
            with open(self._log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            carb.log_info(f"Log file created at {self._log_file_path}")
        except Exception as e:
            carb.log_error(f"Failed to create log file: {e}")

    def on_stop(self):
        self._rigid_prim = None

    def on_update(self, current_time: float, delta_time: float):
        positions, _ = self._rigid_prim.get_world_poses()
        position = positions[0]
        linear_velocity = self._rigid_prim.get_linear_velocities()
        angular_velocity = self._rigid_prim.get_angular_velocities()

        log_row = [
            datetime.datetime.now().isoformat(),
            position[2],
            linear_velocity[0][2],
            angular_velocity[0][2],
            position[0],
            linear_velocity[0][0],
            angular_velocity[0][0],
            position[1],
            linear_velocity[0][1],
            angular_velocity[0][1]
        ]
        try:
            with open(self._log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)
        except Exception as e:
            carb.log_warn(f"Failed to write to log file: {e}")

    def _setup(self):
        # Check if the prim has physics enabled
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"HydrodynamicsComponent on prim {self.prim_path} requires a RigidBody component.")
            return
        
        # Get prim path
        self._rigid_prim = RigidPrim(str(self.prim_path))