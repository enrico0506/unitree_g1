"""teleop_guard.launch.py -- Nav2-costmap obstacle SLOWDOWN for DASHBOARD TELEOP.

WHAT THIS BRINGS UP (ROS 2 Foxy, domain 99, env.sh sourced)
-----------------------------------------------------------
The operator drives the G1 with W/A/S/D on the dashboard (domain 0). This launch
runs the SENSING + Nav2 local costmap that feeds the EXISTING dashboard obstacle
guard -- it does NOT drive the robot. There is NO /cmd_vel bridge and NO
autonomous motion: the only output is /dev/shm/g1_obstacle.json, the SAME file the
dashboard ObstacleGuard already reads, so the guard / UI / voice / overlay are
reused UNCHANGED with a costmap-derived obstacle source instead of the raw lidar.

PIPELINE (wired end to end here):
    (a) Livox Mid-360 driver (msg_MID360_launch.py) -- OWNS the sensor, CustomMsg
    (b) FAST-LIO (mapping.launch.py, mid360_g1.yaml)
          -> /cloud_registered (PointCloud2, frame camera_init)
          -> /Odometry + TF camera_init->body
    (c) ground_seg  /cloud_registered -> /patchwork/nonground (camera_init, floor removed)
    (d) Nav2 local costmap (STVL) -- controller/planner/bt/recoveries + lifecycle
          (controller_server owns+activates the local_costmap we sample;
           the planner/controller OUTPUT is UNUSED -- no autonomy, no /cmd_vel)
    (e) costmap_to_obstacle  /local_costmap/costmap + TF camera_init->body
          -> per-direction nearest obstacle -> /dev/shm/g1_obstacle.json

# ############################################################################
# # !!! SINGLE-CONSUMER Mid-360 -- this launch OWNS the sensor !!!           #
# # The Mid-360 is single-consumer at the UDP layer. This launch starts its  #
# # OWN Livox driver (a), so DO NOT run the mapping stack (g1_mapping_ws), the#
# # obstacle/ node, or g1_nav bringup.launch.py at the same time -- they all  #
# # bind the same fixed Livox UDP ports and the loser stalls (one frame then  #
# # hangs). Stop all of them before launching the teleop guard.              #
# ############################################################################

ENVIRONMENT (must be sourced before `ros2 launch`):
    source /home/unitree/g1_mapping_ws/env.sh   (ROS_DOMAIN_ID=99, CycloneDDS)
Use teleop_guard.sh, which sources it for you.

FRAMES: FAST-LIO native -- camera_init (world) and body (robot). Used DIRECTLY:
no odom/base_link rename, no livox_frame, no static-TF launch. The costmap's
global_frame=camera_init / robot_base_frame=body, and costmap_to_obstacle reads
the same TF.

OUR-OWN vs INSTALLED resolution (mirrors bringup.launch.py):
  * Our launch/config live at absolute repo paths so this runs even if g1_nav was
    never colcon-installed.
  * Our Python nodes (ground_seg, costmap_to_obstacle) are console_scripts in
    setup.py -> Node(package='g1_nav', executable=...). colcon-build g1_nav first
    (or run from the install space) so they resolve.
  * Third-party pkgs (livox_ros_driver2, fast_lio, nav2_*) resolve via
    get_package_share_directory / by package name (they ARE installed).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# --- Absolute repo paths (so this runs even if g1_nav is not colcon-installed) ---
_G1_NAV_DIR = '/home/unitree/projects/g1/g1_nav'
_CONFIG_DIR = os.path.join(_G1_NAV_DIR, 'config')
_TELEOP_GUARD_PARAMS = os.path.join(_CONFIG_DIR, 'teleop_guard_params.yaml')


def generate_launch_description():
    # ------------------------------------------------------------------ args
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock. Real robot -> false (wall clock).')
    declare_params_file = DeclareLaunchArgument(
        'params_file', default_value=_TELEOP_GUARD_PARAMS,
        description='Nav2 params YAML (teleop-guard costmap variant: '
                    'global_frame camera_init, robot_base_frame body).')
    declare_autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Auto-activate the Nav2 lifecycle nodes via lifecycle_manager.')

    # =================================================== (a) Livox Mid-360 driver
    # OWNS the sensor: publishes /livox/lidar (livox_ros_driver2/CustomMsg, the
    # driver runs xfer_format=1 -- the native PointCloud2 path stalls after one
    # frame on this build) + /livox/imu. SINGLE-CONSUMER -- see the big warning in
    # the module docstring. This is the ONLY Livox driver this launch starts.
    livox_share = get_package_share_directory('livox_ros_driver2')
    livox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(livox_share, 'launch_ROS2', 'msg_MID360_launch.py')))

    # ======================================================== (b) FAST-LIO (SLAM)
    # Consumes /livox/lidar + /livox/imu, publishes /cloud_registered (deskewed
    # current scan in the world frame 'camera_init'), /Odometry[camera_init,body]
    # and the camera_init->body TF. Mirrors g1_mapping/mapping.launch.py's include
    # (same package, same mapping.launch.py), but with the G1-specific config and
    # rviz off. Uses mid360.yaml -- the SAME config the WORKING dashboard mapping
    # (g1_mapping/mapping.launch.py) passes, so it is proven-loadable. The
    # G1-specific mid360_g1.yaml is a FLAT FAST-LIO config (no ros__parameters
    # wrapper) and FAILS as a --params-file ("value before ros__parameters");
    # mid360.yaml's IMU-LiDAR extrinsic is generic but fine for the obstacle costmap.
    fastlio_share = get_package_share_directory('fast_lio')
    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fastlio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={'config_file': 'mid360.yaml',
                          'rviz': 'false'}.items())

    # ===================================== (b2) Livox CustomMsg -> dense PointCloud2
    # STVL needs a DENSE cloud in the SENSOR frame. FAST-LIO's /cloud_registered is
    # downsampled (~200 pts) AND in the world frame (camera_init), which breaks STVL's
    # sensor-origin ray-tracing -> 0 marked cells. So convert the RAW Mid-360 scan
    # (/livox/lidar CustomMsg) to a dense /livox/points (livox_frame, ~20k pts).
    livox_to_pc2 = Node(
        package='g1_nav', executable='livox_to_pc2', name='livox_to_pc2',
        output='screen', parameters=[{'use_sim_time': use_sim_time}])

    # static body->livox_frame: FAST-LIO publishes camera_init->body; the cloud is in
    # livox_frame, so STVL needs livox_frame chained up to camera_init. The Mid-360 is
    # ~at the IMU 'body' (cm-level IMU-LiDAR offset, both ~upright) -> identity is fine
    # for a 5 cm costmap. Refine with the real extrinsic if RViz shows a shift.
    tf_body_to_livox = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_body_to_livox',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'body', 'livox_frame'],
        output='screen')

    # ============================ (c) Ground segmentation (Patchwork++ substitute)
    # Foxy-native plane removal: /cloud_registered -> /patchwork/nonground (+ground),
    # in the SAME frame as the input (camera_init). Removes the floor so LOW
    # obstacles survive into the costmap. sensor_height = LiDAR height above the
    # FLOOR (~1.27 m on the G1 -- see EXTRINSICS_RESEARCH.md; VERIFY on the robot).
    # input_topic is /livox/points (dense raw scan in livox_frame). STVL wants a dense
    # SENSOR-frame cloud; FAST-LIO's /cloud_registered is sparse + world-frame.
    ground_seg = Node(
        package='g1_nav',
        executable='ground_seg',
        name='ground_seg',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/livox/points',
            'nonground_topic': '/patchwork/nonground',
            'ground_topic': '/patchwork/ground',
            'sensor_height': 1.27,
            'ground_clearance_m': 0.08,
        }],
    )

    # ====================================================== (d) Nav2 core (costmap)
    # Brings up the Foxy Nav2 servers directly (nav2_bringup is NOT installed) and
    # hands them to a lifecycle_manager -- IDENTICAL wiring to bringup.launch.py,
    # only params_file differs (teleop_guard_params.yaml: local_costmap in
    # camera_init/body, STVL footprint-clearing on, no height clip).
    #
    # The planner/controller/bt/recoveries OUTPUT is UNUSED here (no autonomy, no
    # /cmd_vel bridge) -- but the controller_server is what instantiates and
    # ACTIVATES the local_costmap we sample, and the lifecycle_manager needs the
    # full server set to reach the 'active' state. So we keep all four servers
    # (DWB controller block present but unused).
    nav2_lifecycle_nodes = [
        'controller_server',
        'planner_server',
        'recoveries_server',   # Foxy name (== behavior_server in Galactic+)
        'bt_navigator',
    ]
    nav2 = GroupAction([
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            # Foxy: recoveries_server (NOT behavior_server).
            package='nav2_recoveries',
            executable='recoveries_server',
            name='recoveries_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'node_names': nav2_lifecycle_nodes,
            }],
        ),
    ])

    # ================================================ (e) costmap -> obstacle JSON
    # Reads /local_costmap/costmap (camera_init, latched QoS) + TF camera_init->body,
    # bins lethal cells into front/back/left/right wedges (robot frame: x fwd, y
    # LEFT), and writes /dev/shm/g1_obstacle.json with the SAME contract + scale
    # logic as the raw-lidar obstacle_node.py -- so the dashboard guard behaves
    # identically. Gap-follow steering is OFF (neutral gap keys). ~10 Hz.
    costmap_to_obstacle = Node(
        package='g1_nav',
        executable='costmap_to_obstacle',
        name='costmap_to_obstacle',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # defaults already camera_init/body /local_costmap/costmap; override here
            # if the costmap frames/topic ever change.
            # 'global_frame': 'camera_init',
            # 'robot_base_frame': 'body',
            # 'costmap_topic': '/local_costmap/costmap',
        }],
    )

    return LaunchDescription([
        # args
        declare_use_sim_time,
        declare_params_file,
        declare_autostart,
        # stack, in dependency order
        livox,                 # (a) Mid-360 driver (OWNS the sensor; CustomMsg)
        livox_to_pc2,          # (b2) CustomMsg -> dense /livox/points (livox_frame)
        fastlio,               # (b) FAST-LIO -> TF camera_init->body
        tf_body_to_livox,      # (b3) static body->livox_frame (STVL chains the cloud up)
        ground_seg,            # (c) floor removal /livox/points -> /patchwork/nonground
        # (d) Delay Nav2 ~14 s so FAST-LIO has published camera_init->body BEFORE the
        # local_costmap activates. Otherwise the costmap activation times out on the
        # missing transform and controller_server never reaches 'active' (no costmap
        # -> the guard reads all-clear "3 m"). The 37deg head tilt slows FAST-LIO's
        # gravity init, so give it generous lead time.
        TimerAction(period=14.0, actions=[nav2]),
        costmap_to_obstacle,   # (e) costmap -> /dev/shm/g1_obstacle.json (fail-open during warmup)
    ])
