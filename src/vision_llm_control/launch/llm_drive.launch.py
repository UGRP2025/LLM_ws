from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # RealSense D455
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='d455',
            parameters=[{
                'enable_color': True,
                'enable_depth': False
            }]
        ),

        # Vision‑LLM Control
        Node(
            package='vision_llm_control',
            executable='vision_llm_node',
            name='vision_llm_node',
            parameters=[{
                'model_path': FindPackageShare('vision_llm_control').find('vision_llm_control') + '/models/gemma3_vision.pt',
                'speed_limit': 2.5
            }],
            output='screen'
        )
    ])

