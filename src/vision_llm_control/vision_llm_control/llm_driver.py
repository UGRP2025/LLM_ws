
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import threading
import json
from ollama import chat

# =================================================================
# Gemma 3 Vision Model (Ollama Implementation)
# =================================================================
class Gemma3VisionModel:
    """
    Uses Ollama to infer steering and speed from an image.
    """
    def __init__(self, model_name='gemma3'):
        self.model_name = model_name
        print(f"Ollama model initialized: {self.model_name}")

    def predict(self, cv_img) -> tuple[float, float]:
        """
        Infers steering angle and speed from the input image using Ollama.
        """
        try:
            resp = chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Predict steering and speed from this image. Respond with a JSON array like [steering, speed].'}],
                images=[cv2.imencode('.png', cv_img)[1].tobytes()]
            )
            content = resp['message']['content']
            steer, speed = json.loads(content)
            return float(steer), float(speed)
        except Exception as e:
            print(f"Error during Ollama inference: {e}")
            return 0.0, 0.0

# =================================================================

class LLMDriverNode(Node):
    """
    ROS2 Node to drive a vehicle using an LLM for vision-based control.
    """
    def __init__(self):
        super().__init__('llm_driver_node')

        # Declare parameters for model name and speed limit
        self.declare_parameter('model_name', 'gemma3')
        self.declare_parameter('speed_limit', 1.5)

        # Get parameters
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.speed_limit = self.get_parameter('speed_limit').get_parameter_value().double_value

        # Initialize CvBridge and the vision model
        self.bridge = CvBridge()
        self.model = Gemma3VisionModel(self.model_name)
        
        # Thread lock and buffer for the latest image
        self.latest_image_msg = None
        self.image_lock = threading.Lock()

        # Subscribers and Publishers
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
            
        # Timer for the main processing loop (20Hz)
        self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

        self.get_logger().info("LLM Driver Node has been started.")
        self.get_logger().info(f"Using Ollama Model: {self.model_name}")
        self.get_logger().info(f"Speed Limit: {self.speed_limit} m/s")

    def image_callback(self, msg: Image):
        """
        Callback for the image subscriber. Stores the latest image message.
        """
        with self.image_lock:
            self.latest_image_msg = msg

    def timer_callback(self):
        """
        Main processing loop. Converts image, runs inference, and publishes drive command.
        """
        with self.image_lock:
            if self.latest_image_msg is None:
                return
            image_msg = self.latest_image_msg
            self.latest_image_msg = None

        try:
            # 1. Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # 2. Run model inference
            steering_angle, speed = self.model.predict(cv_image)

            # 3. Publish the drive command
            self.publish_drive_command(steering_angle, speed)

        except Exception as e:
            self.get_logger().error(f"Failed in timer_callback: {e}")

    def publish_drive_command(self, steering_angle: float, speed: float):
        """
        Creates and publishes an AckermannDriveStamped message.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = min(speed, self.speed_limit)

        self.drive_publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    llm_driver_node = LLMDriverNode()
    rclpy.spin(llm_driver_node)
    llm_driver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
