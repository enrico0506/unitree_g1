"""g1_nav DEGRADED STATIONARY Nav2 SMOKE-TEST launch (ROS 2 Foxy).

Purpose: validate the WHOLE plumbing end to end with the robot STANDING STILL --
    Mid-360 (Livox CustomMsg /livox/lidar)
      -> livox_to_pc2            -> /livox/points (sensor_msgs/PointCloud2)
      -> Nav2 local costmap (built-in VoxelLayer, NO STVL) -> DWB -> /cmd_vel
and "see a pole in the local costmap in RViz" -- using ONLY the already-installed
nav2 binaries + the now-built g1_nav. This is a THROWAWAY test rig.

WHAT THIS DELIBERATELY DOES NOT USE (degraded by design):
    NO STVL          -- the local costmap uses the built-in nav2_costmap_2d
                        VoxelLayer (no temporal / blind-spot memory). See the
                        smoke-test params file.
    NO Patchwork++   -- we feed the RAW /livox/points straight into the costmap
                        and drop the floor with a throwaway min_obstacle_height
                        z-gate (0.10) instead of true ground segmentation.
    NO FAST-LIO      -- the robot is STATIONARY, so odom->base_link is a STATIC
                        identity TF published here (NOT FAST-LIO). The costmap
                        therefore does NOT track real motion -- stationary only.
    NO RealSense     -- use_realsense:=false (single-open /dev/video0).
    NO bridge        -- use_bridge:=false: /cmd_vel is NOT forwarded to the
                        Unitree LocoClient. THE ROBOT MUST NOT MOVE.

SINGLE-CONSUMER Mid-360 WARNING:
    Unlike bringup.launch.py (which assumes a Livox driver is ALREADY running),
    THIS launch STARTS the Livox driver itself (msg_MID360_launch.py, CustomMsg
    / xfer_format=1). The Mid-360 is single-consumer at the UDP layer, so the
    mapping stack (g1_mapping_ws), the obstacle/ node, and any other Livox driver
    MUST be STOPPED before you run this -- otherwise the driver publishes one
    frame then stalls. The dashboard (camera only) is fine to leave running.

FRAMES (pinned for this smoke test):
    odom (global, fixed) ; base_link (FLOOR projection, z=0) ; livox_frame.
    odom        -> base_link   : STATIC identity 0 0 0 0 0 0 (here; robot still).
    base_link   -> livox_frame : STATIC, from tf.launch.py (z=1.27, the FULL
                                 floor->LiDAR height; base_link is the floor
                                 projection in this stack).
  base_link is the FLOOR projection (matches tf.launch.py's convention), so the
  whole floor->LiDAR height (~0.47 above pelvis + ~0.8 stance = ~1.27 m) lives in
  base->livox, and odom->base_link is identity for this stationary test. VERIFY
  1.27 with a tape on the standing robot.
  A floor return lands at livox z ~= -1.27 -> odom z ~= 0; a pole maps to odom
  z > 0. The smoke-test params' min_obstacle_height ~0.10 drops the floor and
  keeps the pole.

ENVIRONMENT (must be sourced before `ros2 launch` -- smoketest.sh does this):
    source /home/unitree/g1_mapping_ws/env.sh   (ROS_DOMAIN_ID=99, CycloneDDS)

Run via:  bash /home/unitree/projects/g1/g1_nav/smoketest.sh
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# --- Absolute repo paths (match bringup.launch.py; runs even if g1_nav is not
#     colcon-installed) -------------------------------------------------------
_G1_NAV_DIR = '/home/unitree/projects/g1/g1_nav'
_LAUNCH_DIR = os.path.join(_G1_NAV_DIR, 'launch')
_CONFIG_DIR = os.path.join(_G1_NAV_DIR, 'config')

_BRINGUP_LAUNCH = os.path.join(_LAUNCH_DIR, 'bringup.launch.py')
_SMOKETEST_PARAMS = os.path.join(_CONFIG_DIR, 'nav2_params_smoketest.yaml')


def generate_launch_description():
    # ================================================== 1) Livox MID-360 driver
    # Starts the Livox driver in CustomMsg mode (xfer_format=1) -> /livox/lidar.
    # !! SINGLE-CONSUMER !! Stop mapping / obstacle / any other Livox driver
    # FIRST -- this launch OWNS the sensor for the duration of the test.
    # (bringup.launch.py assumes the driver is already up; this smoke-test
    #  starts it so the rig is self-contained.)
    livox_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('livox_ros_driver2'),
                'launch_ROS2', 'msg_MID360_launch.py')),
    )

    # ============================================ 2) STATIC odom -> base_link
    # The robot is STATIONARY for this smoke test, so we do NOT run FAST-LIO.
    # Instead publish a STATIC identity odom->base_link. This is the ONE TF that
    # bringup/tf.launch.py deliberately omits (FAST-LIO normally owns it).
    # IDENTITY: base_link is the FLOOR projection (z=0), and the full floor->LiDAR
    # height already lives in base->livox z=1.27 (tf.launch.py). So RAW floor
    # returns land near odom z=0 with no extra offset here.
    # Foxy static_transform_publisher arg order: x y z yaw pitch roll parent child.
    # Consequence: the costmap does NOT track motion -- stationary test ONLY.
    tf_odom_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_odom_to_base_smoketest',
        arguments=[
            '0.0', '0.0', '0.0',      # x y z  (m)   -- identity (robot still, base_link=floor)
            '0.0', '0.0', '0.0',      # yaw pitch roll (rad) -- identity
            'odom', 'base_link',
        ],
        output='screen',
    )

    # ============================================ 3) REUSE bringup.launch.py
    # bringup provides (in order):
    #   * tf.launch.py        -> base_link->livox_frame (z=1.27, floor->LiDAR) [+camera, unused]
    #   * livox_to_pc2        -> /livox/lidar (CustomMsg) -> /livox/points (PC2)
    #   * Nav2 servers        -> controller(DWB) / planner(NavFn) / recoveries /
    #                            bt_navigator + lifecycle_manager (autostart)
    # We override ONLY the params (smoke-test VoxelLayer params) and keep the
    # bridge + realsense OFF. bringup does NOT start a Livox driver (we did, in
    # step 1) and does NOT publish odom->base_link (we did, in step 2).
    # params_file: default = degraded VoxelLayer smoke-test params. Override with
    #   params_file:=/home/unitree/projects/g1/g1_nav/config/nav2_params.yaml
    # to run the REAL STVL pipeline (ground_seg -> /patchwork/nonground -> STVL),
    # still stationary. ground_seg is in bringup, so /patchwork/nonground exists.
    params_file = LaunchConfiguration('params_file')
    declare_params_file = DeclareLaunchArgument(
        'params_file', default_value=_SMOKETEST_PARAMS,
        description='Nav2 params YAML. Default = degraded VoxelLayer smoke-test; '
                    'pass config/nav2_params.yaml for the real STVL pipeline (stationary).')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_BRINGUP_LAUNCH),
        launch_arguments={
            'params_file': params_file,        # default smoke-test VoxelLayer; override for STVL
            'use_bridge': 'false',             # SAFETY: robot must NOT move
            'use_realsense': 'false',          # /dev/video0 single-open
        }.items(),
    )

    return LaunchDescription([
        declare_params_file,
        livox_driver,      # 1) Mid-360 driver (CustomMsg) -- SINGLE-CONSUMER
        tf_odom_to_base,   # 2) STATIC odom->base_link (no FAST-LIO; stationary)
        bringup,           # 3) tf + livox_to_pc2 + ground_seg + Nav2 (params arg, bridge OFF)
    ])
