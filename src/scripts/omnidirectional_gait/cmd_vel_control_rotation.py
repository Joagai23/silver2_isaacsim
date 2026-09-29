import signal
import time
import math
import threading
from typing import Union

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped

from silver2_constants.isaac_constants import (
    SILVER2_STANDING_ANGLES_DEG,
    SILVER2_CANONICAL_JOINT_NAMES
)
from omnidirectional_gait_controller import OmnidirectionalGaitController

should_quit = False

def sigint_handler(signum, frame):
    global should_quit
    should_quit = True
    rclpy.try_shutdown()

signal.signal(signal.SIGINT, sigint_handler)

class OmnidirectionalGaitNode(Node):
    def __init__(self):
        super().__init__('omnidirectional_gait_controller')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.group = ReentrantCallbackGroup()

        # Configurable ROS 2 Parameters
        self.declare_parameter('gait_type', 'tripod')
        self.declare_parameter('step_height', 0.10)
        self.declare_parameter('stance_height', 0.30)
        self.declare_parameter('gait_period', 2.0)
        self.declare_parameter('device', 'cpu')

        gait_type = self.get_parameter('gait_type').get_parameter_value().string_value
        step_height = self.get_parameter('step_height').get_parameter_value().double_value
        stance_height = self.get_parameter('stance_height').get_parameter_value().double_value
        gait_period = self.get_parameter('gait_period').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value

        # Modern PyTorch Controller
        self.controller = OmnidirectionalGaitController(
            gait_type=gait_type,
            step_height=step_height,
            stance_height=stance_height,
            gait_period=gait_period,
            device=self.device
        )

        # Canonical standing pose in radians
        standing_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
        self.Q_standing = torch.deg2rad(torch.as_tensor(standing_deg, dtype=torch.float32, device=self.device))
        self.Q_current = self.Q_standing.clone()

        # Precompute default standing foot positions in Chassis (Body) Frame [6, 3]
        local_standing_feet = self.controller.batch_leg_for_kine(self.Q_standing.reshape(6, 3))
        self.default_feet_body = self.controller.local_to_body_frame(local_standing_feet)

        # Pre-allocated ROS 2 JointState Command Message
        self.joint_order = SILVER2_CANONICAL_JOINT_NAMES
        self.cmd_msg = JointState()
        self.cmd_msg.name = self.joint_order
        self.cmd_msg.position = [0.0] * 18

        # State & Telemetry Tracking Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.is_moving = False
        self.current_sec = 0
        self.current_nanosec = 0

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.normalized_phase = 0.0
        self.ctrl_dt = 0.02
        self.state = "STANDBY"

        # Canonical joint polarity signs [18]
        self.joint_signs = np.array([-1.0, -1.0, 1.0] * 6, dtype=np.float32)
        self.joint_signs_torch = torch.from_numpy(self.joint_signs).to(device=self.device)

        # QoS & ROS 2 Communications
        sim_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_subscriber_callback, sim_qos,
            callback_group=self.group
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, '/silver2/pose',
            self.pose_callback, sim_qos,
            callback_group=self.group
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel',
            self.cmd_vel_callback, sim_qos,
            callback_group=self.group
        )

        self.pid_pos_publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.control_timer = self.create_timer(
            self.ctrl_dt, 
            self.control_loop_callback, 
            callback_group=self.group
        )

    def pose_callback(self, msg: PoseStamped):
        """
        Extracts global position and evaluates motion state from Isaac Sim pose.
        Filters contact chatter and accounts for in-place yaw rotations.
        """
        self.current_sec = msg.header.stamp.sec
        self.current_nanosec = msg.header.stamp.nanosec
        
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

        # Extract yaw angle from orientation quaternion (x, y, z, w)
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Handle initialization on first callback
        if not hasattr(self, '_pose_initialized'):
            self.last_x = self.current_x
            self.last_y = self.current_y
            self.last_yaw = current_yaw
            self._pose_initialized = True
            self.is_moving = False
            return

        # Linear and angular displacement between frames
        d_linear = math.hypot(self.current_x - self.last_x, self.current_y - self.last_y)
        d_yaw = abs(math.atan2(math.sin(current_yaw - self.last_yaw), math.cos(current_yaw - self.last_yaw)))

        # Thresholds: 0.5 mm linear displacement or 0.005 rad rotation
        self.is_moving = (d_linear > 0.0005) or (d_yaw > 0.005)

        self.last_x = self.current_x
        self.last_y = self.current_y
        self.last_yaw = current_yaw

    def cmd_vel_callback(self, msg: Twist):
        """
        Stores latest incoming velocity commands. 
        Deadband filtering prevents jitter when joystick is released.
        """
        vx = 0.0 if abs(msg.linear.x) < 1e-3 else float(msg.linear.x)
        vy = 0.0 if abs(msg.linear.y) < 1e-3 else float(msg.linear.y)
        wz = 0.0 if abs(msg.angular.z) < 1e-3 else float(msg.angular.z)

        # Scale velocity limits for SILVER2 physical safety
        self.cmd_vx = np.clip(vx, -0.40, 0.40)
        self.cmd_vy = np.clip(vy, -0.30, 0.30)
        self.cmd_wz = np.clip(wz, -0.50, 0.50)

    def joint_state_subscriber_callback(self, msg: JointState):
        """
        Parses incoming joint states from Isaac Sim, updates self.Q_current,
        and computes group-wise mean joint efforts.
        """
        # Initialize cached index mapping on first message
        if not hasattr(self, '_joint_indices_cached'):
            name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
            
            # Map canonical joint order -> indices in msg.position
            self._canonical_indices = []
            for name in self.joint_order:
                if name in name_to_idx:
                    self._canonical_indices.append(name_to_idx[name])
                else:
                    self.get_logger().error(f"Required joint '{name}' not found in /joint_states!")
                    return
            self._canonical_indices = np.array(self._canonical_indices, dtype=np.int64)
            self._joint_indices_cached = True

        # Extract joint positions in canonical order
        positions = np.array(msg.position, dtype=np.float32)
        canonical_raw = positions[self._canonical_indices]

        # Apply sign inversion to convert Isaac Sim angles -> Kinematic model angles
        canonical_kine = canonical_raw * self.joint_signs
        self.Q_current = torch.from_numpy(canonical_kine).to(
            device=self.device, dtype=torch.float32
        )

    def publish_joint_setpoint(self, pos_array: Union[torch.Tensor, np.ndarray, list], timestep: float = 0.0):
        """
        Converts kinematic joint targets to Isaac Sim drive polarities,
        publishes to /joint_command, and safely synchronizes with simulation time.

        Args:
            pos_array: Joint angles [18] or [6, 3] in radians (kinematic convention).
            timestep: Duration (in seconds) to pace this setpoint. If 0.0, returns immediately.
        """
        # Convert input to 1D float32 numpy array [18]
        if isinstance(pos_array, torch.Tensor):
            positions_kine = pos_array.detach().reshape(-1).cpu().numpy().astype(np.float32)
        elif isinstance(pos_array, np.ndarray):
            positions_kine = pos_array.reshape(-1).astype(np.float32)
        else:
            positions_kine = np.array(pos_array, dtype=np.float32).flatten()

        # Validation with early return
        if len(positions_kine) != len(self.joint_order):
            self.get_logger().error(
                f"Dimension mismatch: expected {len(self.joint_order)} joints, "
                f"got {len(positions_kine)} positions. Aborting setpoint publication."
            )
            return

        # Vectorized polarity conversion -> Isaac Sim drive coordinates
        isaac_positions = positions_kine * self.joint_signs

        # Update preallocated message and publish
        now = self.get_clock().now()
        self.cmd_msg.header.stamp = now.to_msg()
        self.cmd_msg.position = isaac_positions.tolist()
        self.pid_pos_publisher.publish(self.cmd_msg)

        # Non-blocking simulation-time sync (only if timestep > 0)
        if timestep > 0.0:
            duration = rclpy.duration.Duration(seconds=timestep)
            target_time = now + duration
            wall_start = time.monotonic()
            max_wall_wait = max(timestep * 3.0, 0.5)

            while rclpy.ok() and not should_quit:
                current_time = self.get_clock().now()
                if current_time >= target_time:
                    break
                if (time.monotonic() - wall_start) > max_wall_wait:
                    self.get_logger().warn("Simulation clock hitch detected during setpoint wait.")
                    break
                time.sleep(0.001)

    def change_configuration_loop(self, q_target: torch.Tensor, duration: float = 3.0):
        """
        Executes a 3-phase statically stable transition between current posture
        and desired standing posture.
        """
        self.get_logger().info("Initiating 3-phase posture transition...")

        result = self.controller.change_configuration(
            q_des=q_target,
            q_cur=self.Q_current,
            gait_t=duration,
            lift_height=0.05
        )

        if isinstance(result, bool):
            self.get_logger().info("Already at desired posture.")
            return

        Qq, _, admiss, n_steps, ctrl_dt = result

        if not admiss.all():
            self.get_logger().warn("Configuration change outside reachable workspace! Aborting.")
            return

        for step in range(n_steps):
            if should_quit or not rclpy.ok():
                break
            # Qq is [18, n_steps] in canonical kinematic coordinates
            step_targets = Qq[:, step]
            self.publish_joint_setpoint(step_targets, timestep=ctrl_dt)

        self.get_logger().info("Posture transition complete.")

    def control_loop_callback(self):
        """
        Non-blocking 50 Hz timer callback. Computes instantaneous analytical
        joint targets from the current cmd_vel without lookup tables.
        """
        if self.state == "STANDBY":
            return

        # Check if velocity is zero (Neutral standing stance)
        is_zero_cmd = (abs(self.cmd_vx) < 1e-3 and 
                       abs(self.cmd_vy) < 1e-3 and 
                       abs(self.cmd_wz) < 1e-3)

        if is_zero_cmd:
            self.normalized_phase = 0.0
            self.publish_joint_setpoint(self.Q_standing, timestep=0.0)
            return

        # Dynamic cycle period scaling based on command magnitude
        vel_mag = float(np.hypot(self.cmd_vx, self.cmd_vy))
        rot_mag = abs(self.cmd_wz) * 0.25
        norm_speed = np.clip(max(vel_mag, rot_mag) / 0.40, 0.0, 1.0)
        self.controller.T = float(np.interp(norm_speed, [0.0, 1.0], [3.0, 1.4]))

        # Integrate normalized phase increment directly: d_tau = dt / T
        self.normalized_phase = (self.normalized_phase + self.ctrl_dt / self.controller.T) % 1.0
        equiv_t = self.normalized_phase * self.controller.T

        # Compute instantaneous 3D Cartesian foot targets in Body Frame
        feet_targets_body = self.controller.compute_foot_trajectories(
            t=equiv_t,
            vx=self.cmd_vx,
            vy=self.cmd_vy,
            wz=self.cmd_wz,
            default_feet_body=self.default_feet_body
        )

        # Transform Body Frame -> Local Coxa Mount Frames
        feet_targets_local = self.controller.body_to_local_frame(feet_targets_body)

        # Evaluate Analytical Closed-Form Inverse Kinematics
        q_targets, is_admissible = self.controller.batch_leg_inv_kine(feet_targets_local)

        if not is_admissible.all():
            self.get_logger().warn("Command requested foot targets outside kinematic workspace!", throttle_duration_sec=1.0)
            return

        # Publish to /joint_command
        self.publish_joint_setpoint(q_targets, timestep=0.0)

def main(args=None):
    rclpy.init(args=args)
    node = OmnidirectionalGaitNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # Start executor in background thread so /clock and /joint_states advance immediately
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for initial /joint_states to cache indices
    node.get_logger().info("Waiting for initial /joint_states from Isaac Sim...")
    while rclpy.ok() and not hasattr(node, '_joint_indices_cached') and not should_quit:
        time.sleep(0.05)

    # Stand up smoothly from initial rest pose
    node.change_configuration_loop(node.Q_standing, duration=3.0)
    node.state = "ACTIVE"
    node.get_logger().info("SILVER2 Omnidirectional Gait Controller is active and listening on /cmd_vel.")

    try:
        while rclpy.ok() and not should_quit:
            time.sleep(0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()