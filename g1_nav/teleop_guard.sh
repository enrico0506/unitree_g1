#!/usr/bin/env bash
# =============================================================================
# teleop_guard.sh -- Nav2-costmap obstacle SLOWDOWN source for DASHBOARD TELEOP.
#
# The operator drives the G1 with W/A/S/D on the dashboard. This stack does NOT
# drive the robot: it runs the Mid-360 driver + FAST-LIO + ground removal + a
# Nav2 local costmap, then turns that costmap into per-direction obstacle
# distances written to /dev/shm/g1_obstacle.json -- the SAME file the EXISTING
# dashboard ObstacleGuard already reads. The guard / UI / voice / overlay are
# reused UNCHANGED; only the obstacle SOURCE changes (costmap, not raw lidar).
#
# !! NO AUTONOMOUS MOTION !!  There is NO /cmd_vel bridge in this launch. The
# Nav2 planner/controller come up only so the lifecycle activates the local
# costmap we sample; their output is unused. The robot moves ONLY from teleop.
#
# !! SINGLE-CONSUMER Mid-360 !!  This launch starts its OWN Livox driver and
# binds the Mid-360's fixed UDP ports. DO NOT run the mapping stack
# (g1_mapping_ws), the obstacle/ node (run_obstacle.sh), or g1_nav's
# bringup.launch.py / run_nav.sh at the same time -- they all bind the same
# ports and the loser stalls (one frame then hangs). Stop them first.
#
# This sources the isolated mapping/Nav2 environment (ROS_DOMAIN_ID=99,
# CycloneDDS) so it can never disturb robot control (domain 0).
#
# NOTE: g1_nav must be colcon-built (or run from its install space) so the
# ground_seg + costmap_to_obstacle console_scripts resolve. chmod +x this file:
#     chmod +x /home/unitree/projects/g1/g1_nav/teleop_guard.sh
# =============================================================================
set -e

# ROS 2 Foxy + CycloneDDS overlay + domain 99 (isolated from control stack).
source /home/unitree/g1_mapping_ws/env.sh

# Pass any extra args straight through to ros2 launch (e.g. use_sim_time:=false).
exec ros2 launch /home/unitree/projects/g1/g1_nav/launch/teleop_guard.launch.py "$@"
