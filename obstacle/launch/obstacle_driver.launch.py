"""Livox Mid-360 driver launch for the obstacle feature.

Copied from livox_ros_driver2/launch_ROS2/msg_MID360_launch.py with TWO changes:
  1. user_config_path -> resolve the INSTALLED config via ament's package share
     dir, because the original '../config' relative path is wrong once this file
     lives in our repo (outside the livox workspace tree).
  2. node renamed 'livox_obstacle_publisher' so it's distinguishable from the
     mapping driver.

xfer_format = 1 (Livox CustomMsg) -- the SAME format mapping uses. We originally
tried xfer_format = 0 (PointCloud2) for a cheap numpy parse, but THIS driver
build's PointCloud2 path is broken: it publishes a single frame then stalls (the
robot only ever exercises CustomMsg via FAST-LIO). CustomMsg streams reliably at
10 Hz, so the obstacle node consumes CustomMsg (decimated per-point parse) instead.

Everything else matches the upstream defaults: publish_freq=10.0,
frame_id='livox_frame', data_src=0, multi_topic=0.

NOTE: this binds the Mid-360's fixed UDP ports -- it must NOT run at the same
time as the mapping stack (which binds the same ports).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch

################### user configure parameters for ros2 start ###################
xfer_format   = 1    # 1-Livox CustomMsg (reliable here); 0-PointCloud2 path stalls on this build
multi_topic   = 0    # 0-All LiDARs share the same topic, 1-One LiDAR one topic
data_src      = 0    # 0-lidar, others-Invalid data src
publish_freq  = 10.0 # freqency of publish, 5.0, 10.0, 20.0, 50.0, etc.
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = '/home/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'

# Resolve the config that ships in the INSTALLED livox_ros_driver2 package share
# directory (the upstream '../config' relative path does not exist once copied
# into this repo).
user_config_path = os.path.join(
    get_package_share_directory('livox_ros_driver2'),
    'config', 'MID360_config.json')
################### user configure parameters for ros2 end #####################

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code}
]


def generate_launch_description():
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_obstacle_publisher',
        output='screen',
        parameters=livox_ros2_params
        )

    return LaunchDescription([
        livox_driver,
    ])
