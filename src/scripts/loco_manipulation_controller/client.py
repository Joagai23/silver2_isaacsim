import os
import threading
import rclpy
from ros_controller import RosController

class Client():

    def __init__(self):
        self.__ros_controller = RosController()
        self.__ros_thread = threading.Thread(
            target=rclpy.spin, 
            args=(self.__ros_controller,), 
            daemon=True
        )
        self.__ros_thread.start()

    def run_loop(self):
        while True:
            self.__print_menu()
            
            choice = input("Please enter a number to select an option: ")
            self.__clear_screen()
            
            if choice == '1':
                self.__leg_movement_control()

            elif choice == '2':
                self.__show_system_status()
                print("\n-> Displaying current internal state.")
                
            elif choice == '3':
                print("\n-> Exiting the application.")
                break
                
            else:
                print("\n-> Invalid input. Please enter a valid number from the menu.")

    def __print_menu(self):
        print("\n--- Loco-Manipulator Controller ---")
        print("1. Move Leg")
        print("2. Read Joints")
        print("3. Exit")

    def __print_silver2_schema(self):
        print("""
      [ FRONT ]
   (0) \\_   _/ (3)
         | |
   (1) --| |-- (4)
         | |
   (2) /     \\ (5)
      [ BACK ]
        """)

    def __clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def __leg_movement_control(self):
        print("--- Leg Movement Control ---\n")
        self.__print_silver2_schema()
        leg_choice = input("Select a leg to move (0-5): ")

        if leg_choice in ['0', '1', '2', '3', '4', '5']:
            print(f"\n-> Leg {leg_choice} selected. Configuring joints...\n")
                    
            # Use our helper function to safely get the angles
            coxa_val = self.__get_joint_angle("Coxa", self.__ros_controller.COXA_MIN, self.__ros_controller.COXA_MAX)
            femur_val = self.__get_joint_angle("Femur", self.__ros_controller.FEMUR_MIN, self.__ros_controller.FEMUR_MAX)
            tibia_val = self.__get_joint_angle("Tibia", self.__ros_controller.TIBIA_MIN, self.__ros_controller.TIBIA_MAX)
            
            # Summary of the command
            print(f"\n-> SUCCESS: Sending command to Leg {leg_choice}.")
            print(f"   Position: [Coxa: {coxa_val}°, Femur: {femur_val}°, Tibia: {tibia_val}°]")
            
            self.__ros_controller.move_leg_command(int(leg_choice), [coxa_val, femur_val, tibia_val])
        else:
            print("\n-> Error: Invalid leg selection. Please enter a number from 0 to 5.")

    def __get_joint_angle(self, joint_name, min_angle, max_angle):
        while True:
            user_input = input(f"Enter {joint_name} angle ({min_angle}° to {max_angle}°): ")
            
            if not user_input.lstrip('-').isdigit(): 
                print("  -> Error: Please enter a valid whole number.")
                continue
                
            angle = int(user_input)
            
            if min_angle <= angle <= max_angle:
                return angle
            else:
                print(f"  -> Error: To protect the robot, angle must be between {min_angle}° and {max_angle}°.") 

    def __show_system_status(self):
        print("--- System Status: Joint Positions ---")
        leg_status = self.__ros_controller.get_Q_current_degrees()
        for leg_id in range(6):
            # Calculate the starting index for this leg's joints
            idx = leg_id * 3 
            
            # Extract the angles from the flat array
            coxa = leg_status[idx]
            femur = leg_status[idx + 1]
            tibia = leg_status[idx + 2]
            
            # Print a cleanly aligned log line. 
            # The :>5.1f formats the number to 1 decimal place and aligns it.
            print(f"Leg {leg_id} -> Coxa: {coxa:>5.1f}° | Femur: {femur:>5.1f}° | Tibia: {tibia:>5.1f}°")
        
        print("-" * 38)