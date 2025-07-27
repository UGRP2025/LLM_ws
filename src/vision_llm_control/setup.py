from setuptools import setup
import os
from glob import glob

package_name = 'vision_llm_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'ackermann_msgs', 'cv_bridge', 'image_transport_py', 'torch', 'ollama'],
    zip_safe=True,
    maintainer='DGIST UGRP 2025 F1tenth',
    maintainer_email='ugrpf1@gmain.com',
    description='Gemma 3-driven vision-to-drive control for F1TENTH (ROS2)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_llm_node = vision_llm_control.vision_llm_node:main',
            'llm_driver = vision_llm_control.llm_driver:main',
        ],
    },
)