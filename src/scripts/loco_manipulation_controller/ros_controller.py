import subprocess
import os
import numpy as np
import numpy.typing as npt
import sys, time, signal
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class RosController(Node):

    def __init__(self):
        super().__init__('ros_controller_node')
        self.__define_ros_variables()
        self.__define_ros_topics()
        self.__define_joint_limits()

    def __source_ros_environment(self, setup_path="/opt/ros/jazzy/setup.bash"):
        print(f"Attempting to source {setup_path}...")

        command = f"source {setup_path} && env"  
        result = subprocess.run(
            ['bash', '-c', command], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            print("Error: Could not source the file. Check the path.")
            return False
            
        for line in result.stdout.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

                if key == "PYTHONPATH":
                    for path in value.split(':'):
                        if path not in sys.path:
                            sys.path.append(path)
                
        print("Success! ROS environment variables are now loaded in Python.")
        return True
    
    def __define_ros_topics(self):
        print("Creating ROS subscriptions and publishers...")
        try:
            self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_subscriber_callback, 10)
            #self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
            self.pid_pos_publisher = self.create_publisher(JointState, '/joint_command', 10) 
            self.get_logger().info("ROS topics initialized successfully.")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize ROS topics: {str(e)}")
            raise e

    def __define_ros_variables(self):
        self.__Q_current = np.zeros(18)
        self.joint_order = [
            'coxa_joint_0', 'femur_joint_0', 'tibia_joint_0',
            'coxa_joint_1', 'femur_joint_1', 'tibia_joint_1',
            'coxa_joint_2', 'femur_joint_2', 'tibia_joint_2',
            'coxa_joint_3', 'femur_joint_3', 'tibia_joint_3',
            'coxa_joint_4', 'femur_joint_4', 'tibia_joint_4',
            'coxa_joint_5', 'femur_joint_5', 'tibia_joint_5'
        ]

    def get_Q_current_degrees(self):
        return np.degrees(self.__Q_current)

    def joint_state_subscriber_callback(self, msg):
        # Convert from isaac format
        isaac_pos = [a*b for a,b in zip(self.FROM_ISAAC, msg.position)]
        joint_position_dict = dict(zip(msg.name, isaac_pos))
        # Save current joint state
        for i, joint_name in enumerate(self.joint_order):
            if joint_name in joint_position_dict:
                self.__Q_current[i] = joint_position_dict[joint_name]
            else:
                self.get_logger().warn(f"Joint {joint_name} not found in message")

    def __define_joint_limits(self):
        self.COXA_MIN, self.COXA_MAX = -90, 90
        self.FEMUR_MIN, self.FEMUR_MAX = -90, 90
        self.TIBIA_MIN, self.TIBIA_MAX = -180, 180
        self.FROM_ISAAC = np.array([-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, 1,1,1,1,1,1])
        self.TO_ISAAC = np.array([-1,-1,1, -1,-1,1, -1,-1,1, -1,-1,1, -1,-1,1, -1,-1,1])

    def move_leg_command(self, leg_index:int, joint_values:npt.ArrayLike):
        joint_array = np.copy(self.__Q_current)
        start = leg_index * 3
        joint_array[start : start + 3] = np.radians(joint_values)
        

        self.__go_to_pose(joint_array)

    def __go_to_pose(self, joint_pose:npt.ArrayLike):
        self.get_logger().info("Initializing: Moving to Pose safely in joint space...")


        Q_start = np.copy(self.__Q_current)
        Q_end = joint_pose

        steps = 60
        delay = 3.0 / steps 

        for i in range(steps):
            alpha = (i + 1) / steps
            Q_interp = Q_start * (1 - alpha) + Q_end * alpha
            self.publish_joint_setpoint(Q_interp, delay)

        self.get_logger().info("Pose reached!")

    # Create ROS2 JointState Message using Position Array
    def publish_joint_setpoint(self, pos_array, timestep):
        # Safety check
        if len(pos_array) != len(self.joint_order):
            self.get_logger().error(
                f"Failed to convert message: "
                f"The number of joint names ({len(self.joint_order)}) does not match "
                f"the number of received positions ({len(pos_array)})."
            )
            
        # Create and populate Joint State
        joint_state_msg = JointState()
        joint_state_msg.name = self.joint_order
        # Convert to isaac format
        isaac_pos = [a*b for a,b in zip(self.TO_ISAAC, pos_array)]
        joint_state_msg.position = isaac_pos
        joint_state_msg.velocity = []
        joint_state_msg.effort = []

        # Publish and Wait
        self.pid_pos_publisher.publish(joint_state_msg)
        time.sleep(timestep)