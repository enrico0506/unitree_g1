#!/usr/bin/env python3
"""livox_to_pc2 -- Livox Mid-360 CustomMsg -> sensor_msgs/PointCloud2 converter.

WHY THIS NODE EXISTS
--------------------
The Mid-360 driver build on this robot publishes /livox/lidar as
livox_ros_driver2/msg/CustomMsg. Its native PointCloud2 output path
(xfer_format=0) is broken -- it emits a single frame and then stalls -- so we
CANNOT simply reconfigure the driver to give us a PointCloud2. Nav2 / Patchwork++
/ the costmap obstacle layers all want a standard sensor_msgs/PointCloud2, so this
node bridges the gap: subscribe the CustomMsg, repack the per-point xyz into a
dense PointCloud2, and republish on /livox/points.

CONTRACT (pinned names -- do not change)
  in : /livox/lidar    livox_ros_driver2/msg/CustomMsg  (qos_profile_sensor_data)
  out: /livox/points   sensor_msgs/PointCloud2          (fields x,y,z float32)
       frame_id = livox_frame
       stamp    = msg.header.stamp  (passthrough -- keep the sensor's own stamp)

The CustomMsg cloud is already x-fwd / y-left / z-up at the sensor origin (the
driver applies the 180 deg mount roll), so we DO NOT re-rotate -- xyz are copied
through verbatim. Frame/extrinsics to base_link are handled by a static TF
(livox_frame -> base_link) declared in the bringup launch, NOT here.

PERFORMANCE
  The per-point parse mirrors obstacle/obstacle_node.py::custom_xyz, but instead
  of a Python list comprehension we build a numpy structured array once and
  .tobytes() it straight into the PointCloud2 payload -- O(N) and allocation-light
  so the converter stays well under budget at the Mid-360 ~10 Hz frame rate on the
  Jetson.

Runs under ROS 2 Foxy on DDS domain 99 (CycloneDDS). Source the stack env first:
  source /home/unitree/g1_mapping_ws/env.sh
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from livox_ros_driver2.msg import CustomMsg


# Output PointCloud2 layout: 3 contiguous float32 fields x, y, z (12-byte point).
_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_POINT_STEP = _POINT_DTYPE.itemsize  # 12


def custom_to_xyz(msg):
    """Build an (N, 3) float32 array from a livox CustomMsg's .points list.

    Same per-point parse approach as obstacle/obstacle_node.py::custom_xyz --
    CustomPoint has no zero-copy numpy view, so we read p.x/p.y/p.z per point.
    No decimation here: this node's job is a faithful 1:1 republish; downstream
    consumers (Patchwork++, costmap) do their own downsampling.
    """
    pts = msg.points
    if not pts:
        return np.zeros((0, 3), np.float32)
    return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)


class LivoxToPC2(Node):
    def __init__(self):
        super().__init__("livox_to_pc2")
        # QoS must match the driver's best-effort sensor stream, or no data flows.
        self.pub = self.create_publisher(PointCloud2, "/livox/points", qos_profile_sensor_data)
        self.sub = self.create_subscription(
            CustomMsg, "/livox/lidar", self._on_custom, qos_profile_sensor_data
        )
        self.frame_id = "livox_frame"
        self._frames = 0
        self.get_logger().info(
            "livox_to_pc2: /livox/lidar (CustomMsg) -> /livox/points (PointCloud2), "
            f"frame_id={self.frame_id}"
        )

    def _on_custom(self, msg):
        xyz = custom_to_xyz(msg)
        n = len(xyz)

        # Pack into the structured dtype, then a single .tobytes() into the
        # PointCloud2 payload (no per-point Python work past the parse above).
        structured = np.zeros(n, dtype=_POINT_DTYPE)
        if n:
            structured["x"] = xyz[:, 0]
            structured["y"] = xyz[:, 1]
            structured["z"] = xyz[:, 2]

        out = PointCloud2()
        out.header.stamp = msg.header.stamp  # passthrough: keep the sensor stamp
        out.header.frame_id = self.frame_id
        out.height = 1
        out.width = n
        out.fields = _FIELDS
        out.is_bigendian = False
        out.point_step = _POINT_STEP
        out.row_step = _POINT_STEP * n
        out.is_dense = True  # we copy whatever the driver gives; no NaNs injected
        out.data = structured.tobytes()
        self.pub.publish(out)

        self._frames += 1
        if self._frames % 50 == 1:
            self.get_logger().info(f"converted frame {self._frames}: {n} points")


def main(args=None):
    rclpy.init(args=args)
    node = LivoxToPC2()
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
