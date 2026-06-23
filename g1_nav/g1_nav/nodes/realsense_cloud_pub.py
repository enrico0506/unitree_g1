#!/usr/bin/env python3
"""realsense_cloud_pub -- RealSense Z16 depth (/dev/video0) -> /camera/points.

============================================================================
!!  /dev/video0 IS SINGLE-OPEN  !!
============================================================================
The RealSense depth node /dev/video0 can be opened by exactly ONE process at a
time. The dashboard (scripts/robot_web_controller.py -> lidar_source.LidarSource)
already grabs it for the browser 3D cloud. THIS NODE MUST NOT RUN while the
dashboard / camera_service is reading the camera -- the second open() fails (or
starves the first). Stop the dashboard's lidar source before launching this node.

Because of that conflict this source is OPTIONAL -- it is a PHASE 3 add-on and is
OFF BY DEFAULT in the Nav2 costmap (the Mid-360 via /patchwork/nonground is the
primary obstacle source). The supported alternative is realsense-ros publishing a
depth/points topic directly, but realsense-ros is NOT installed on this robot, so
this raw-V4L2 republisher is the stopgap.
============================================================================

WHAT IT DOES
  Reuses the raw-V4L2 grabber (_DepthReader) and the depth->cloud unprojection
  (_depth_to_cloud) from scripts/lidar_source.py -- imported by path so the
  intrinsics / mount-height / unprojection math stay single-sourced. Each ~10 Hz
  tick: read a depth frame, unproject to an (N,3) cloud, wrap as PointCloud2, and
  publish on /camera/points with frame_id=camera_link.

  NOTE on frame: lidar_source._depth_to_cloud already returns a world-ish frame
  (x forward, y left, z up) with the camera mount height baked into z. We publish
  that as-is under camera_link; the camera_link->base_link static TF in the
  bringup launch (<ASK ENRICO> for the real mount pose) must account for whatever
  convention this produces. See the costmap notes in g1_nav/config/nav2_params.yaml.

CONTRACT (pinned names)
  out: /camera/points   sensor_msgs/PointCloud2   (fields x,y,z float32)
       frame_id = camera_link
       stamp    = node clock at capture (RealSense V4L2 gives no usable ROS stamp)

Runs under ROS 2 Foxy on DDS domain 99. Source the stack env first:
  source /home/unitree/g1_mapping_ws/env.sh
"""

import importlib.util
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


# --- Import the V4L2 reader + unprojection from scripts/lidar_source.py -------
# Single-source the camera intrinsics / reader by importing the existing module
# by path (it is NOT an installed package). Layout:
#   <repo>/g1_nav/g1_nav/nodes/realsense_cloud_publisher.py   (this file)
#   <repo>/scripts/lidar_source.py                            (reused module)
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
_LIDAR_SRC = os.path.join(_REPO, "scripts", "lidar_source.py")

_spec = importlib.util.spec_from_file_location("g1_lidar_source", _LIDAR_SRC)
_lidar_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lidar_source)

_DepthReader = _lidar_source._DepthReader
_depth_to_cloud = _lidar_source._depth_to_cloud
_cap = _lidar_source._cap
MOUNT_HEIGHT = _lidar_source.MOUNT_HEIGHT      # <ASK ENRICO: confirm RealSense mount height vs base_link>
MAX_POINTS = _lidar_source.MAX_POINTS


_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_POINT_STEP = _POINT_DTYPE.itemsize  # 12


def cloud_to_pc2(cloud, stamp, frame_id):
    """Wrap an (N,3) float32 array as a sensor_msgs/PointCloud2."""
    n = 0 if cloud is None else len(cloud)
    structured = np.zeros(n, dtype=_POINT_DTYPE)
    if n:
        structured["x"] = cloud[:, 0]
        structured["y"] = cloud[:, 1]
        structured["z"] = cloud[:, 2]
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = _FIELDS
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * n
    msg.is_dense = True
    msg.data = structured.tobytes()
    return msg


class RealsenseCloudPub(Node):
    def __init__(self):
        super().__init__("realsense_cloud_pub")
        self.frame_id = "camera_link"
        self.max_points = MAX_POINTS
        self.mount_height = MOUNT_HEIGHT
        self.pub = self.create_publisher(PointCloud2, "/camera/points", qos_profile_sensor_data)

        self.get_logger().warn(
            "realsense_cloud_pub: opening /dev/video0 (SINGLE-OPEN). The dashboard "
            "camera/lidar source MUST be stopped first or this open() will fail."
        )
        self.reader = _DepthReader()
        try:
            self.reader.open()
        except Exception as e:  # noqa: BLE001 -- surface any V4L2/open failure clearly
            self.get_logger().error(
                f"depth camera unavailable ({e}). Is the dashboard holding /dev/video0? "
                "This OPTIONAL node will idle (no frames published)."
            )
            self.reader = None

        self._frames = 0
        # ~10 Hz publish loop.
        self.timer = self.create_timer(0.1, self._on_tick)
        self.get_logger().info(
            f"realsense_cloud_pub -> /camera/points, frame_id={self.frame_id}, ~10 Hz"
        )

    def _on_tick(self):
        if self.reader is None:
            return
        try:
            depth = self.reader.read()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"depth read error: {e}")
            return
        if depth is None:
            return  # select() timeout -- no frame this tick
        cloud = _cap(_depth_to_cloud(depth, self.mount_height), self.max_points)
        msg = cloud_to_pc2(cloud, self.get_clock().now().to_msg(), self.frame_id)
        self.pub.publish(msg)

        self._frames += 1
        if self._frames % 50 == 1:
            self.get_logger().info(f"published frame {self._frames}: {len(cloud)} points")

    def destroy_node(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseCloudPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
