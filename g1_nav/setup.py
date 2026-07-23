from glob import glob

from setuptools import setup

package_name = 'g1_nav'

setup(
    name=package_name,
    version='0.0.1',
    # nodes/ is a subpackage; entry points reference g1_nav.nodes.<mod>:main
    packages=[package_name, package_name + '.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch/ and config/ trees so ros2 launch / params resolve
        # from the install space (also works when run from source via run_nav.sh).
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Enrico',
    maintainer_email='fachin.enrico05@gmail.com',
    description='Nav2 migration for the Unitree G1 (ROS 2 Foxy).',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Mid-360 livox_ros_driver2/CustomMsg -> sensor_msgs/PointCloud2.
            'livox_to_pc2 = g1_nav.nodes.livox_to_pc2:main',
            # Foxy-native ground segmentation (Patchwork++ substitute):
            # /livox/points -> /patchwork/nonground (+ /patchwork/ground).
            'ground_seg = g1_nav.nodes.ground_seg:main',
            # RealSense V4L2 depth -> sensor_msgs/PointCloud2 (OPTIONAL, Phase 3).
            'realsense_cloud_pub = g1_nav.nodes.realsense_cloud_pub:main',
            # /cmd_vel (domain 99) -> Unitree LocoClient (domain 0) bridge.
            'g1_cmd_vel_bridge = g1_nav.nodes.g1_cmd_vel_bridge:main',
            # Nav2 local costmap (+ TF camera_init->body) -> per-direction nearest
            # obstacle -> /dev/shm/g1_obstacle.json (the dashboard guard's source).
            'costmap_to_obstacle = g1_nav.nodes.costmap_to_obstacle:main',
            # Drives the fixed circle-patrol via rolling NavigateToPose goals.
            'circle_patrol = g1_nav.nodes.circle_patrol:main',
            # fused_odometry.py health side-car -> /dev/shm/g1_odom_health.json.
            'fused_odom_health = g1_nav.nodes.fused_odom_health:main',
            # DETECT_TRACKS + depth -> bearing/distance -> /dev/shm/g1_person_track.json.
            'person_fusion = g1_nav.nodes.person_fusion:main',
            # Person-interaction state machine (pausing/facing/waving/waiting/resuming).
            'patrol_supervisor = g1_nav.nodes.patrol_supervisor:main',
            # Exhibition mode: roam/gesture/on-request-dance inside the patrol disc
            # (richer sibling to circle_patrol, oracle-vetted via patrol_logic.py).
            'exhibition_conductor = g1_nav.nodes.exhibition_conductor:main',
        ],
    },
)
