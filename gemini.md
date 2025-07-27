# claude.md — Vision‑LLM Control (ROS 2 Foxy)

## 요구사항
- **ROS 2 package**: `vision_llm_control` (ament_python, format=3)
- **노드**: `vision_llm_node.py`
  * rclpy로 작성
  * Sub: `/camera/camera/color/image_raw` (`sensor_msgs.msg.Image`)
  * Pub: `/drive` (`ackermann_msgs.msg.AckermannDriveStamped`)
- **Launch**: `launch/llm_drive.launch.py` (Python)
- **의존**:
  * apt: `ros-foxy-ackermann-msgs`, `ros-foxy-cv-bridge`
  * pip: `torch`, `ollama` (Gemma 3 inference)
- **테스트**: `tests/test_inference.py` → `pytest` & `launch_testing`
- **빌드**:  
  ```bash
  colcon build --symlink-install
  source install/setup.bash
  ros2 launch vision_llm_control llm_drive.launch.py


## Workspace
LLM_ws/
 ├── src/
 │   ├── realsense2_camera/          # github.com/IntelRealSense/realsense-ros -b foxy
 │   └── vision_llm_control/
 │       ├── package.xml             # format="3", build_type="ament_python"
 │       ├── setup.py
 │       ├── resource/vision_llm_control
 │       ├── vision_llm_control/
 │       │   └── vision_llm_node.py
 │       ├── launch/
 │       │   └── llm_drive.launch.py
 │       ├── config/gemma3.yaml
 │       └── tests/test_inference.py
 └── gemini.md

