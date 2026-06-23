"""Static TF tree for g1_nav (ROS 2 Foxy).

Publishes the FIXED, rigid sensor-mounting transforms that never change while
the robot runs:

    base_link -> livox_frame    (Mid-360 LiDAR mount)
    base_link -> camera_link    (RealSense depth mount)

WHAT THIS FILE DOES *NOT* PUBLISH (and must not):
    odom -> base_link           <-- FAST-LIO already provides this on domain 99.
        FAST-LIO publishes nav_msgs/Odometry on /Odometry *and* broadcasts the
        odom->base_link TF. Do NOT add a static publisher for it here: two
        broadcasters on the same edge make the TF tree flap and Nav2's costmap
        transforms go stale/unusable. Trust FAST-LIO for odom->base_link.
        (If FAST-LIO is configured to publish a different child frame name than
        'base_link', that is the one place to reconcile -- check its launch /
        config, do NOT paper over it with a second static TF.)

FRAME NAMES are PINNED across the whole g1_nav stack (do not rename):
    base_link  = robot base
    livox_frame= Mid-360 LiDAR
    camera_link= RealSense depth
    odom       = FAST-LIO odom frame

------------------------------------------------------------------------------
MOUNTING POSES ARE PHYSICAL MEASUREMENTS -- THEY ARE NOT GUESSED HERE.
Every transform below is a placeholder pending a real measurement on the robot.
Each static_transform_publisher takes:  x y z  yaw pitch roll  parent child
(NOTE Foxy ORDER: translation x y z, THEN rotation YAW PITCH ROLL in radians,
 then parent_frame child_frame -- yaw first, NOT roll first.)

  base_link->livox_frame : from Unitree URDF (see EXTRINSICS_RESEARCH.md)
  base_link->camera_link : from Unitree URDF (see EXTRINSICS_RESEARCH.md)

SOURCE OF THE NUMBERS BELOW (no longer hand-measured):
    Unitree official URDF unitree_ros/robots/g1_description/g1_23dof_rev_1_0.urdf
    composes pelvis->torso_link (xyz -0.0039635 0 0.044, rpy 0 0 0)
    with the sensor joints on torso_link:
       mid360_joint : xyz 0.0002835 0.00003 0.428434  rpy 3.14159265 0.05112069 0
       d435_joint   : xyz 0.0576235  0.01753  0.42987  rpy 0 0.83077672 0
    -> pelvis->mid360 : xyz -0.00368 0.00003 0.47243  rpy 3.14159 0.05112 0
    -> pelvis->d435   : xyz  0.05366 0.01753 0.47387  rpy 0       0.83078 0
    CAVEAT (verify on robot): these are pelvis->sensor. They are correct as
    base_link->sensor ONLY IF base_link is coincident with the URDF `pelvis`
    link. Confirm what FAST-LIO/the robot call base_link before trusting these.
    The Mid-360 roll = pi is REAL (sensor mounted inverted) -- do NOT zero it.
    The D435 pitch ~0.83 rad (downward tilt) is REAL -- do NOT zero it.

    FAST-LIO's mid360_g1.yaml IMU<-LiDAR extrinsic_T [ -0.011,-0.02329,0.04412 ]
    is LiDAR-vs-internal-IMU, NOT base_link, so it is unrelated to these.
------------------------------------------------------------------------------
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ----- base_link -> livox_frame (Mid-360 LiDAR) -----
    # args: x  y  z  yaw  pitch  roll  parent  child   (Foxy order, radians)
    # From Unitree URDF (g1_23dof_rev_1_0): pelvis->mid360 composed.
    #   translation xyz = (-0.00368, 0.00003, 0.47243) m
    #   rotation rpy    = (roll 3.14159, pitch 0.05112, yaw 0.0) rad
    #   -> Foxy CLI order is x y z YAW PITCH ROLL, so: 0.0 0.05112 3.14159
    # VERIFY ON ROBOT: valid only if base_link == URDF pelvis. roll=pi is REAL.
    # Note: the LiDAR z above the FLOOR (Patchwork++ sensor_height) is NOT this
    # 0.472 (that is above the pelvis); add the ~0.79 m stance -> ~1.27 m.
    tf_base_to_livox = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_livox',
        arguments=[
            '-0.00368', '0.00003', '1.27',      # x y z (m): x,y from URDF; z = LiDAR height above the
                                                #   FLOOR (~0.47 above pelvis + ~0.8 stance). base_link is
                                                #   treated as the floor projection for this stack. MEASURE.
            '0.0', '0.05112', '0.0',            # yaw pitch roll (rad). ROLL = 0, *NOT* pi: the Livox driver
                                                #   already applies the 180deg (roll=pi) mount de-roll to the
                                                #   points it publishes (verified: floor reads below, no flip),
                                                #   so the published livox_frame is UPRIGHT. The mount roll=pi
                                                #   and the driver de-roll CANCEL -> TF rotation = mount pitch
                                                #   only. Applying roll=pi here would DOUBLE-flip (floor on
                                                #   ceiling). Verify in RViz.
            'base_link', 'livox_frame',
        ],
        output='screen',
    )

    # ----- base_link -> camera_link (RealSense D435i depth) -----
    # args: x  y  z  yaw  pitch  roll  parent  child   (Foxy order, radians)
    # From Unitree URDF (g1_23dof_rev_1_0): pelvis->d435 composed.
    #   translation xyz = (0.05366, 0.01753, 0.47387) m
    #   rotation rpy    = (roll 0.0, pitch 0.83078, yaw 0.0) rad  (downward tilt, REAL)
    #   -> Foxy CLI order x y z YAW PITCH ROLL, so: 0.0 0.83078 0.0
    # VERIFY ON ROBOT: valid only if base_link == URDF pelvis.
    # NOTE: this is the RealSense MOUNT/BODY frame (d435_link, X forward, Y left,
    # Z up), matching scripts/lidar_source.py's _depth_to_cloud convention. If you
    # later publish in the camera OPTICAL frame, add the standard
    # camera_link->camera_depth_optical_frame rotation on top of this.
    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=[
            '0.05366', '0.01753', '0.47387',   # x y z  (m)   URDF pelvis->d435
            '0.0', '0.83078', '0.0',           # yaw pitch roll (rad)  (pitch=downward tilt, REAL)
            'base_link', 'camera_link',
        ],
        output='screen',
    )

    return LaunchDescription([
        tf_base_to_livox,
        tf_base_to_camera,
    ])
