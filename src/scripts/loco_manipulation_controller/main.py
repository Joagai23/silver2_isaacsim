import rclpy
import sys
from client import Client

def main():
    rclpy.init(args=sys.argv)

    try:
        app = Client()
        app.run_loop()

    except KeyboardInterrupt:
        print("\n\n[INFO] Keyboard interrupt detected. Shutting down...")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
    
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        print("[INFO] System offline.")

if __name__ == "__main__":
    main()         