"""g1_nav top-level bringup (ROS 2 Foxy).

Orchestrates the whole Nav2-migration stack for the Unitree G1, in dependency
order, with LaunchArguments to toggle the optional pieces. This is a SCAFFOLD:
it is run phase-by-phase with the robot, NOT expected to come up clean as-is
(Patchwork++ and STVL are not installed yet -- see the comment blocks below).

PIPELINE (this file wires it end to end):
    Mid-360 (Livox CustomMsg /livox/lidar)
      -> livox_to_pc2          -> /livox/points (sensor_msgs/PointCloud2)
      -> Patchwork++           -> /patchwork/nonground (+ /patchwork/ground)
      [+ realsense_cloud_pub   -> /camera/points, OPTIONAL]
      -> Nav2 local costmap (STVL-fused) -> planner/controller (DWB holonomic)
      -> /cmd_vel (geometry_msgs/Twist)
      -> g1_cmd_vel_bridge     -> Unitree LocoClient (domain 0)

ENVIRONMENT (must be sourced before `ros2 launch`):
    source /home/unitree/g1_mapping_ws/env.sh
  That puts the Livox/FAST-LIO/Nav2 stack on ROS_DOMAIN_ID=99 (CycloneDDS).
  FAST-LIO must already be running there (it supplies /Odometry and the
  odom->base_link TF -- see tf.launch.py). The g1_cmd_vel_bridge internally
  re-inits a SECOND DDS participant on domain 0 / eth0 to reach LocoClient.

LAUNCH ARGUMENTS (toggle pieces):
    use_realsense   (default 'false') : also publish the RealSense depth cloud
                                        (/camera/points). OFF -- /dev/video0 is
                                        single-open and the dashboard holds it.
    use_sim_time    (default 'false') : real robot -> wall clock.
    params_file     (default our config/nav2_params.yaml) : Nav2 params.
    autostart       (default 'true')  : lifecycle_manager auto-activates Nav2.

  Example:
    ros2 launch /home/unitree/projects/g1/g1_nav/launch/bringup.launch.py \\
        use_realsense:=false

OUR-OWN vs INSTALLED resolution:
  * Our launch/config live at absolute repo paths so this runs even if the
    package was never `colcon build`-installed.
  * Our PYTHON NODES (livox_to_pc2 / realsense_cloud_pub / g1_cmd_vel_bridge)
    are declared as console_scripts in setup.py, so when g1_nav IS installed
    they resolve via Node(package='g1_nav', executable=...). If you have NOT
    colcon-built g1_nav, swap each to the ExecuteProcess fallback marked
    "RUN-FROM-SOURCE FALLBACK" below.
  * Third-party packages (nav2_*, livox_ros_driver2) are resolved via
    get_package_share_directory / by package name (they ARE installed).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# --- Absolute repo paths (so this runs even if g1_nav is not colcon-installed) ---
_G1_NAV_DIR = '/home/unitree/projects/g1/g1_nav'
_LAUNCH_DIR = os.path.join(_G1_NAV_DIR, 'launch')
_CONFIG_DIR = os.path.join(_G1_NAV_DIR, 'config')
_NODES_DIR = os.path.join(_G1_NAV_DIR, 'g1_nav', 'nodes')   # run-from-source fallback

_TF_LAUNCH = os.path.join(_LAUNCH_DIR, 'tf.launch.py')
_NAV2_PARAMS_DEFAULT = os.path.join(_CONFIG_DIR, 'nav2_params.yaml')


def generate_launch_description():
    # ------------------------------------------------------------------ args
    use_realsense = LaunchConfiguration('use_realsense')
    use_bridge = LaunchConfiguration('use_bridge')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')

    declare_use_realsense = DeclareLaunchArgument(
        'use_realsense', default_value='false',
        description="Publish RealSense /camera/points. OFF by default: "
                    "/dev/video0 is single-open and the dashboard holds it.")
    declare_use_bridge = DeclareLaunchArgument(
        'use_bridge', default_value='false',
        description="Start g1_cmd_vel_bridge (/cmd_vel -> Unitree LocoClient). "
                    "OFF by default for SAFETY: enabling it lets Nav2 DRIVE the "
                    "robot. Only set use_bridge:=true once you have validated "
                    "the costmap/DWB output (ros2 topic echo /cmd_vel) and have "
                    "an e-stop in hand.")
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock. Real robot -> false (wall clock).')
    declare_params_file = DeclareLaunchArgument(
        'params_file', default_value=_NAV2_PARAMS_DEFAULT,
        description='Full path to the Nav2 params YAML (DWB holonomic + costmaps).')
    declare_autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Auto-activate the Nav2 lifecycle nodes via lifecycle_manager.')

    # =================================================================== 1) TF
    # Static sensor mounts (base_link->livox_frame, base_link->camera_link).
    # FAST-LIO supplies odom->base_link separately -- NOT duplicated here.
    tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_TF_LAUNCH),
    )

    # ============================================ 2) Livox CustomMsg -> PC2
    # Converts the Mid-360 raw /livox/lidar (livox_ros_driver2/CustomMsg) into
    # sensor_msgs/PointCloud2 on /livox/points. The driver's native PointCloud2
    # path is BROKEN on this build (publishes 1 frame then stalls), so we keep
    # the driver on CustomMsg (xfer_format=1) and convert here.
    # NOTE: the Mid-360 is single-consumer at the UDP layer -- the Livox driver
    # feeding /livox/lidar (FAST-LIO's, or the obstacle driver) must already be
    # running; do NOT also start a second Livox driver from this file.
    livox_to_pc2 = Node(
        package='g1_nav',
        executable='livox_to_pc2',
        name='livox_to_pc2',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # 'input_topic': '/livox/lidar',     # CustomMsg in
            # 'output_topic': '/livox/points',   # PointCloud2 out
            # 'output_frame': 'livox_frame',
        }],
    )
    # RUN-FROM-SOURCE FALLBACK (if g1_nav is NOT colcon-installed) -- replace
    # the Node above with:
    #   from launch.actions import ExecuteProcess
    #   livox_to_pc2 = ExecuteProcess(
    #       cmd=['python3', os.path.join(_NODES_DIR, 'livox_to_pc2.py')],
    #       output='screen')

    # ========================= 3) Ground segmentation (Patchwork++ substitute)
    # Foxy-native plane-removal node: /livox/points -> /patchwork/nonground (+ground).
    # Replaces Patchwork++ (does NOT build on this Foxy/Jetson toolchain: ROS1 wrapper,
    # ROS2 C++ node needs a newer rclcpp, python bindings need a newer cmake). Removes
    # the floor so LOW obstacles survive; the costmap consumes /patchwork/nonground at
    # min_obstacle_height 0.03. sensor_height = LiDAR height above the FLOOR (~1.27 m).
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

    # ============================ 3b) Patchwork++ (GROUND SEG, future upgrade)
    # ------------------------------------------------------------------------
    # NOT INSTALLED YET. Patchwork++ segments /livox/points into ground vs
    # obstacles. Leaving it out of the active LaunchDescription so bringup
    # TOLERATES its absence (the rest of the stack still launches; the costmap
    # obstacle source /patchwork/nonground will simply be empty until this is
    # built and re-enabled).
    #
    # EXPECTED node, once you source-build patchwork-plusplus-ros (Foxy branch):
    #   package  = 'patchworkpp'          (a.k.a. patchwork_pp / patchworkpp_ros)
    #   exe      = 'patchworkpp_node'     (verify with: ros2 pkg executables patchworkpp)
    #   subscribes: /livox/points         (sensor_msgs/PointCloud2)
    #   publishes : /patchwork/nonground  (obstacles -> costmap source 1)
    #               /patchwork/ground     (ground, for viz)
    #
    # patchwork = Node(
    #     package='patchworkpp',
    #     executable='patchworkpp_node',
    #     name='patchworkpp',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': use_sim_time,
    #         'cloud_topic': '/livox/points',
    #         'nonground_topic': '/patchwork/nonground',
    #         'ground_topic': '/patchwork/ground',
    #         # sensor_height = LiDAR height above the GROUND PLANE (metres).
    #         # This is NOT the base->livox_frame z (0.472, above pelvis); it is
    #         # the floor->LiDAR height = 0.472 (URDF) + ~0.79 (pelvis stance)
    #         # = ~1.27 m. Patchwork++ uses it to seed its ground model.
    #         # VERIFY 1.27 with a tape on the standing robot -- the ~0.79 stance
    #         # is posture-dependent and the main uncertainty. (robot.yaml's
    #         # camera_height_m 1.3 is an independent real-world cross-check.)
    #         'sensor_height': 1.27,   # m, floor->Mid-360 (see EXTRINSICS_RESEARCH.md). VERIFY on robot.
    #     }],
    # )
    # TO ENABLE: uncomment the Node above AND add `patchwork` to the returned
    # LaunchDescription list (right after livox_to_pc2). Until then it is omitted.
    # ------------------------------------------------------------------------

    # ============================= 4) RealSense depth -> PC2 (OPTIONAL, gated)
    # Publishes /camera/points from the RealSense Z16 depth via raw V4L2,
    # reusing scripts/lidar_source.py's _depth_to_cloud. OFF by default:
    # /dev/video0 is SINGLE-OPEN -- this node CANNOT run while the dashboard is
    # reading the camera (they will fight over the device). Phase 3 only.
    realsense_cloud_pub = Node(
        package='g1_nav',
        executable='realsense_cloud_pub',
        name='realsense_cloud_pub',
        output='screen',
        condition=IfCondition(use_realsense),
        parameters=[{
            'use_sim_time': use_sim_time,
            # 'device': '/dev/video0',
            # 'output_topic': '/camera/points',
            # 'output_frame': 'camera_link',
        }],
    )
    # RUN-FROM-SOURCE FALLBACK: ExecuteProcess on
    #   os.path.join(_NODES_DIR, 'realsense_cloud_pub.py')  (wrap in GroupAction
    #   with the IfCondition if you switch to ExecuteProcess).

    # ============================================================= 5) Nav2 core
    # nav2_bringup is NOT installed on this robot, so we CANNOT
    # IncludeLaunchDescription(navigation_launch.py). Instead we bring up the
    # Foxy Nav2 servers directly and hand them to a lifecycle_manager.
    #
    # FOXY NOTES baked in here:
    #   * Controller = DWB (nav2_dwb_controller) configured HOLONOMIC in
    #     nav2_params.yaml (the G1 commands vx, vy AND vyaw). Foxy has no
    #     MPPI/RPP -- the controller_server just LOADS the DWB plugin named in
    #     params; there is no separate controller executable to pick.
    #   * Recovery server in Foxy is 'recoveries_server' (pkg nav2_recoveries),
    #     NOT 'behavior_server' (that is Galactic+). Named accordingly below.
    #
    # IF YOU LATER INSTALL nav2_bringup, you may replace this whole GroupAction
    # with:
    #   nav2 = IncludeLaunchDescription(
    #       PythonLaunchDescriptionSource(os.path.join(
    #           get_package_share_directory('nav2_bringup'),
    #           'launch', 'navigation_launch.py')),
    #       launch_arguments={'use_sim_time': use_sim_time,
    #                         'autostart': autostart,
    #                         'params_file': params_file}.items())
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

    # =============================================== 6) /cmd_vel domain bridge
    # Subscribes /cmd_vel (geometry_msgs/Twist, domain 99) and forwards each
    # twist to the Unitree LocoClient on DDS DOMAIN 0 / eth0:
    #   ChannelFactoryInitialize(0, "eth0"); client=LocoClient(); client.Init()
    #   client._CallNoReply(ROBOT_API_ID_LOCO_SET_VELOCITY,
    #       json.dumps({"velocity":[vx,vy,vyaw],"duration":d}))
    # The bridge clamps to the rated caps (robot.yaml): max_vx=1.5, max_vy=0.7,
    # max_vyaw=2.0. Cross-domain handoff lives inside the node, not here.
    # SAFETY: gated behind use_bridge (default 'false'). This node DRIVES the
    # robot -- it does NOT come up unless you explicitly pass use_bridge:=true.
    g1_cmd_vel_bridge = Node(
        package='g1_nav',
        executable='g1_cmd_vel_bridge',
        name='g1_cmd_vel_bridge',
        output='screen',
        condition=IfCondition(use_bridge),
        parameters=[{
            'use_sim_time': use_sim_time,
            # 'cmd_vel_topic': '/cmd_vel',
            # 'interface': 'eth0',
            # 'loco_domain_id': 0,
            # 'max_vx': 1.5, 'max_vy': 0.7, 'max_vyaw': 2.0,
        }],
    )
    # RUN-FROM-SOURCE FALLBACK: ExecuteProcess on
    #   os.path.join(_NODES_DIR, 'g1_cmd_vel_bridge.py').

    return LaunchDescription([
        # args
        declare_use_realsense,
        declare_use_bridge,
        declare_use_sim_time,
        declare_params_file,
        declare_autostart,
        # stack, in order
        tf,                    # 1) static TFs
        livox_to_pc2,          # 2) CustomMsg -> /livox/points
        ground_seg,            # 3) ground removal -> /patchwork/nonground (Patchwork++ substitute)
        # patchwork,           # 3b) Patchwork++  (does not build on Foxy; ground_seg substitutes)
        realsense_cloud_pub,   # 4) RealSense cloud (gated on use_realsense)
        nav2,                  # 5) Nav2 core (servers + lifecycle_manager)
        g1_cmd_vel_bridge,     # 6) /cmd_vel -> LocoClient bridge (gated on use_bridge, OFF by default)
    ])
