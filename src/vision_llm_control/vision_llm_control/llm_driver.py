
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import torch
import threading

# =================================================================
# Gemma 3 비전 모델 플레이스홀더
# =================================================================
# 이 클래스는 실제 Gemma 3 모델 로딩 및 추론 로직으로 대체되어야 합니다.
# 현재는 기능 시뮬레이션을 위한 더미(dummy) 구현입니다.

class Gemma3VisionModel:
    """
    Gemma 3 비전-언어 모델의 동작을 시뮬레이션하는 플레이스홀더 클래스입니다.
    실제 모델 로딩(ollama 또는 로컬 체크포인트) 및 추론 로직으로 교체해야 합니다.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        # 실제 모델 초기화 로직 (예: self.model = self.load_model(model_path))
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        print(f"Placeholder model initialized. Path: {self.model_path}")

    def predict(self, image_tensor: torch.Tensor) -> tuple[float, float]:
        """
        입력 이미지 텐서로부터 조향각과 속도를 추론합니다.
        현재는 고정된 값을 반환하는 더미 로직입니다.
        
        Args:
            image_tensor: 전처리된 입력 이미지 (torch.Tensor).

        Returns:
            (조향각, 속도) 튜플.
        """
        # 실제 모델 추론 로직이 여기에 들어갑니다.
        # 예: 
        # with torch.no_grad():
        #     outputs = self.model(image_tensor.to(self.device))
        # steering = outputs['steering'].item()
        # speed = outputs['speed'].item()
        
        # 더미 추론 결과 (직진, 1.0m/s)
        steering_angle = 0.0
        speed = 1.0
        
        return steering_angle, speed

# =================================================================

class LLMDriverNode(Node):
    """
    이미지 데이터를 받아 LLM을 통해 주행 명령을 생성하고 발행하는 노드.
    """
    def __init__(self):
        super().__init__('llm_driver_node')

        # 파라미터 선언 (모델 경로, 최고 속도)
        self.declare_parameter('model_path', '/path/to/your/model')
        self.declare_parameter('speed_limit', 1.5) # m/s

        # 파라미터 값 가져오기
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.speed_limit = self.get_parameter('speed_limit').get_parameter_value().double_value

        # CvBridge 및 모델 초기화
        self.bridge = CvBridge()
        self.model = Gemma3VisionModel(self.model_path)
        
        # 이미지 구독 콜백의 부하를 줄이기 위한 변수 및 잠금
        self.latest_image_msg = None
        self.image_lock = threading.Lock()

        # Subscriber (이미지) & Publisher (주행 명령) 설정
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',  # 구독할 이미지 토픽
            self.image_callback,
            10)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',      # 발행할 주행 명령 토픽
            10)
            
        # 20Hz 타이머 생성 (메인 로직 처리)
        self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

        self.get_logger().info("LLM Driver Node has been started.")
        self.get_logger().info(f"Model Path: {self.model_path}")
        self.get_logger().info(f"Speed Limit: {self.speed_limit} m/s")

    def image_callback(self, msg: Image):
        """
        이미지 토픽을 구독하여 최신 이미지를 버퍼에 저장하는 콜백 함수.
        메인 스레드의 부하를 최소화하기 위해 간단한 작업만 수행합니다.
        """
        with self.image_lock:
            self.latest_image_msg = msg

    def timer_callback(self):
        """
        20Hz 주기로 실행되며, 이미지 처리, 모델 추론, 주행 명령 발행을 수행.
        Heavy work는 이 콜백에서 처리됩니다.
        """
        with self.image_lock:
            if self.latest_image_msg is None:
                return
            # 처리할 이미지를 지역 변수로 복사하고 버퍼를 비웁니다.
            image_msg = self.latest_image_msg
            self.latest_image_msg = None

        try:
            # 1. ROS Image 메시지를 OpenCV BGR 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # 2. 이미지를 PyTorch 텐서로 전처리
            image_tensor = self.preprocess_image(cv_image)

            # 3. 모델 추론 실행
            steering_angle, speed = self.model.predict(image_tensor)

            # 4. 주행 명령 생성 및 발행
            self.publish_drive_command(steering_angle, speed)

        except Exception as e:
            self.get_logger().error(f"Failed to process image and publish drive command: {e}")

    def preprocess_image(self, cv_image) -> torch.Tensor:
        """
        OpenCV 이미지를 모델 입력에 맞는 Torch 텐서로 변환합니다.
        (이 부분은 실제 모델의 요구사항에 맞게 커스터마이zing 되어야 합니다)
        """
        # 예시: 크기 조정, 정규화, 차원 변경 등
        # cv_image = cv2.resize(cv_image, (224, 224))
        # cv_image = cv_image / 255.0
        
        # NumPy 배열을 Torch 텐서로 변환
        image_tensor = torch.from_numpy(cv_image.copy()).float()
        # (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        # 배치 차원 추가 (C, H, W) -> (N, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def publish_drive_command(self, steering_angle: float, speed: float):
        """
        추론된 조향각과 속도로 AckermannDriveStamped 메시지를 발행합니다.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link" # 차량의 기준 프레임
        
        drive_msg.drive.steering_angle = steering_angle
        # 속도 제한 적용
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
