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
            executable='llm_driver',
            name='llm_driver_node',
            parameters=[{
                'model_name': 'gemma3',
                'speed_limit': 1.5
            }],
            output='screen'
        )
    ])

