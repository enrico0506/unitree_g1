#!/usr/bin/env bash
# Launch the obstacle perception stack (Mid-360 driver + obstacle_node.py).
# Spawned on-demand by the dashboard's ObstacleManager.start(); SIGINT (sent to
# the process group) tears it all down.
#
# Sources the isolated ROS2 env (domain 99) so it can never disturb robot
# control (domain 0). rclpy + Livox msgs come onto PYTHONPATH from here.
#
# WARNING: this binds the Mid-360's FIXED UDP ports, exactly like the mapping
# stack (run_mapping.sh). The two MUST NOT run at the same time -- starting this
# while mapping is active (or vice versa) will fail to bind the LiDAR. Stop
# mapping before enabling obstacle detection.
set -e
source /home/unitree/g1_mapping_ws/env.sh
# --- TEMP DIAGNOSTIC (remove this line when done) -------------------------------
# Traces the front cone through each filter stage to see why a near wall reads
# clear. Adds a 'front_diag' key to /dev/shm/g1_obstacle.json + a ~1 Hz log line.
# Changes NO perception/safety logic. Delete to turn the diagnostic off.
export G1_OBS_FRONT_DIAG=1
# -------------------------------------------------------------------------------
exec ros2 launch /home/unitree/projects/g1/obstacle/launch/obstacle.launch.py "$@"
