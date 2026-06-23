"""G1 obstacle slow-down, stop & gap-following (Mid-360 LiDAR).

Self-contained feature, two halves bridged by a /dev/shm file:

  obstacle_node.py  -- ROS2 perception node (domain 99). Reads /livox/lidar,
                       computes per-direction nearest distance + a gap profile,
                       writes /dev/shm/g1_obstacle.json.
  guard.py          -- ObstacleGuard. Runs in the dashboard process (domain 0),
                       reads the shm file and scales/steers the commanded
                       velocity right before it is sent to the robot.
  manager.py        -- ObstacleManager. Starts/stops the node subprocess.

Only guard.py and manager.py are imported by the dashboard; obstacle_node.py is
launched as its own process (it needs the sourced ROS2 environment).
"""
