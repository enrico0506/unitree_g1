#!/usr/bin/env bash
# =============================================================================
# smoketest.sh -- DEGRADED, STATIONARY Nav2 smoke-test for g1_nav (Foxy).
#
# Validates the whole plumbing with the robot STANDING STILL:
#   Mid-360 -> livox_to_pc2 -> Nav2 local costmap (built-in VoxelLayer) -> DWB
#   -> /cmd_vel,  and "see a pole in the local costmap in RViz".
# Uses ONLY the already-installed nav2 binaries + the now-built g1_nav.
# NO STVL, NO Patchwork++, NO FAST-LIO, NO RealSense, NO bridge.
#
# !! WARNING -- SHARED, SINGLE-CONSUMER Mid-360 !!
#   This launch STARTS the Livox driver itself (CustomMsg / xfer_format=1).
#   The Mid-360 is single-consumer at the UDP layer, so you MUST STOP the
#   mapping stack (g1_mapping_ws), the obstacle/ node, and any other Livox
#   driver BEFORE running this -- otherwise the driver publishes one frame
#   then stalls. The dashboard (camera only) is fine to leave running.
#
# !! THE ROBOT WILL NOT MOVE !!
#   The /cmd_vel -> LocoClient bridge stays OFF (use_bridge:=false). DWB will
#   PUBLISH /cmd_vel but nothing forwards it to the robot. odom->base_link is a
#   STATIC identity TF (no FAST-LIO) -- this is a STATIONARY test only; the
#   costmap does NOT track real motion.
#
# Sources the isolated mapping/Nav2 environment (ROS_DOMAIN_ID=99, CycloneDDS).
# See the "SMOKE-TEST RUNBOOK" in MIGRATION_STATUS.md for the full procedure.
# =============================================================================
set -e

# ROS 2 Foxy + CycloneDDS overlay + domain 99 (isolated from the control stack).
source /home/unitree/g1_mapping_ws/env.sh

# Pass any extra args straight through to ros2 launch.
exec ros2 launch /home/unitree/projects/g1/g1_nav/launch/smoketest.launch.py "$@"
