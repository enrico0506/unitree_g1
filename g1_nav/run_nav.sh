#!/usr/bin/env bash
# =============================================================================
# run_nav.sh -- launch the G1 Nav2 stack (Foxy).
#
# !! WARNING -- SHARED Mid-360 SENSOR !!
# The Mid-360 LiDAR is a SINGLE-CONSUMER source. This Nav2 stack reads
# /livox/lidar (CustomMsg) just like the mapping stack and the obstacle node.
# DO NOT run mapping (g1_mapping_ws) or the obstacle/ node at the same time as
# this -- they will fight over the sensor and one of them will stall (the driver
# publishes one frame then hangs). Stop mapping/obstacle BEFORE running this.
#
# Also note: the RealSense source (realsense_cloud_pub) and the dashboard both
# open /dev/video0, which is single-open. Don't enable the RealSense costmap
# source while the dashboard is reading the camera.
#
# This sources the isolated mapping/Nav2 environment (ROS_DOMAIN_ID=99,
# CycloneDDS). The /cmd_vel bridge crosses to the Unitree control stack on
# domain 0 internally -- see g1_cmd_vel_bridge.
# =============================================================================
set -e

# ROS 2 Foxy + CycloneDDS overlay + domain 99 (isolated from control stack).
source /home/unitree/g1_mapping_ws/env.sh

# Pass any extra args straight through to ros2 launch (e.g. use_sim_time:=false).
exec ros2 launch /home/unitree/projects/g1/g1_nav/launch/bringup.launch.py "$@"
