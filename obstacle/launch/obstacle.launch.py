"""Top-level obstacle launch: Mid-360 driver + obstacle perception node.

Brings up two processes under one LaunchDescription so a single SIGINT (from
ros2 launch's signal handling, i.e. the dashboard's ObstacleManager.stop())
tears both down together:

  (a) the Livox Mid-360 driver (obstacle_driver.launch.py, PointCloud2 output);
  (b) obstacle_node.py -- the ROS2 perception node that reads /livox/lidar and
      writes /dev/shm/g1_obstacle.json.

All paths are computed relative to THIS file so the package can be moved without
edits.

NOTE: binds the Mid-360's fixed UDP ports -- do NOT run while mapping is active.
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))     # .../obstacle/launch
_OBSTACLE_DIR = os.path.dirname(_THIS_DIR)                   # .../obstacle

_DRIVER_LAUNCH = os.path.join(_THIS_DIR, 'obstacle_driver.launch.py')
_OBSTACLE_NODE = os.path.join(_OBSTACLE_DIR, 'obstacle_node.py')


def generate_launch_description():
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_DRIVER_LAUNCH)
    )

    obstacle_node = ExecuteProcess(
        cmd=['python3', _OBSTACLE_NODE],
        output='screen',
    )

    return LaunchDescription([
        driver,
        obstacle_node,
    ])
