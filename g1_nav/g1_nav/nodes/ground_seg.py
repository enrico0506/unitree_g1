#!/usr/bin/env python3
"""ground_seg -- lightweight ground-plane segmentation for the G1 Nav2 stack.

A FOXY-NATIVE substitute for Patchwork++, which does NOT build cleanly on this
ROS 2 Foxy / Jetson toolchain (the official ROS wrapper is ROS1/catkin; the
ROS2 C++ node needs a newer rclcpp than Foxy ships; and the Python bindings need
a newer cmake/scikit-build chain than is available here -- six distinct build
blockers, all confirmed). This node delivers the SAME thing the costmap needs
from Patchwork++: remove the ground so LOW obstacles (curbs, the gantry/harness
feet, fallen items) SURVIVE, instead of being clipped away by a flat height band.

PIPELINE:  /livox/points (PointCloud2, livox_frame)
             -> robust ground-plane fit
             -> /patchwork/nonground  (obstacles -> Nav2 costmap source)
             +  /patchwork/ground     (ground, for RViz)
The Nav2 local costmap consumes /patchwork/nonground at min_obstacle_height 0.03
(the "DO NOT RAISE" floor-object value) -- now that the floor is gone, that low
threshold keeps real low obstacles without the floor flooding the costmap.

METHOD (tilt-aware -- handles the humanoid's sway + the small mount pitch):
  fit a plane  z = a*x + b*y + c  by least squares over the LOW candidate points
  (one robust refit dropping >15 cm residual outliers; sanity-bounded; falls back
  to a level plane at -sensor_height if the fit is degenerate), then keep points
  whose height ABOVE the fitted plane is in (ground_clearance, max_obstacle_above).
  Everything that sticks up from the floor is an obstacle; the floor is removed.

This is flat/indoor-floor ground removal. Patchwork++ is better on slopes /
multi-plane terrain -- revisit it later via a newer ROS distro / cmake / container.
Output is in the SAME frame as the input (livox_frame); the costmap transforms it
via the base_link->livox_frame static TF.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


def pc2_xyz(msg):
    """Parse x,y,z float32 out of a PointCloud2 (offset-driven, no sensor_msgs_py)."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ("x", "y", "z")):
        return np.zeros((0, 3), np.float32)
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

    def col(name):
        o = offs[name]
        return raw[:, o:o + 4].copy().view("<f4").reshape(-1)

    return np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)


def make_xyz_pc2(header, pts):
    """Build a contiguous xyz-float32 PointCloud2 from an (N,3) array."""
    n = int(pts.shape[0])
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name=c, offset=i * 4, datatype=PointField.FLOAT32, count=1)
        for i, c in enumerate(("x", "y", "z"))
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.is_dense = True
    msg.data = np.ascontiguousarray(pts[:, :3], dtype="<f4").tobytes()
    return msg


class GroundSeg(Node):
    def __init__(self):
        super().__init__("ground_seg")
        self.in_topic = self.declare_parameter("input_topic", "/livox/points").value
        self.ng_topic = self.declare_parameter("nonground_topic", "/patchwork/nonground").value
        self.g_topic = self.declare_parameter("ground_topic", "/patchwork/ground").value
        # sensor_height = LiDAR height above the floor (m). From the G1 URDF the
        # Mid-360 sits ~0.47 m above the pelvis; pelvis stance ~0.8 m -> ~1.27 m.
        # MEASURE on the standing robot to nail this (see EXTRINSICS_RESEARCH.md).
        self.sensor_height = float(self.declare_parameter("sensor_height", 1.27).value)
        self.clearance = float(self.declare_parameter("ground_clearance_m", 0.08).value)
        self.max_above = float(self.declare_parameter("max_obstacle_above_m", 2.0).value)
        self.max_range = float(self.declare_parameter("max_range_m", 8.0).value)

        self.pub_ng = self.create_publisher(PointCloud2, self.ng_topic, qos_profile_sensor_data)
        self.pub_g = self.create_publisher(PointCloud2, self.g_topic, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.in_topic, self._cb, qos_profile_sensor_data)
        self._n = 0
        self.get_logger().info(
            f"ground_seg: {self.in_topic} -> {self.ng_topic} (+{self.g_topic}) | "
            f"sensor_height={self.sensor_height} clearance={self.clearance}")

    def _cb(self, msg):
        pts = pc2_xyz(msg)
        if len(pts) == 0:
            return
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            r = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[r <= self.max_range]
        if len(pts) < 30:
            self._publish(msg.header, pts, np.zeros((0, 3), np.float32))
            return

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # --- robust ground-plane fit z = a*x + b*y + c (tilt-aware) ---
        a = b = 0.0
        c = -self.sensor_height                       # level fallback
        cand = z < (-self.sensor_height + 0.4)        # low points = floor candidates
        if int(cand.sum()) >= 50:
            try:
                A = np.column_stack([x[cand], y[cand], np.ones(int(cand.sum()))])
                coef, *_ = np.linalg.lstsq(A, z[cand], rcond=None)
                res = z[cand] - A @ coef               # one robust refit
                keep = np.abs(res) < 0.15
                if int(keep.sum()) >= 50:
                    coef, *_ = np.linalg.lstsq(A[keep], z[cand][keep], rcond=None)
                a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
                # sanity: reject only an IMPLAUSIBLE fit -> fail SAFE to level.
                # Bound raised to ~0.9 (~42 deg) so the G1's tilted-up head (~37 deg
                # -> floor slope a ~= tan(37) ~= 0.75 in the sensor frame) is ACCEPTED,
                # not rejected. floor-height bound widened for the same reason.
                if abs(a) > 0.9 or abs(b) > 0.9 or abs(c + self.sensor_height) > 0.8:
                    a = b = 0.0
                    c = -self.sensor_height
            except Exception:
                a = b = 0.0
                c = -self.sensor_height

        above = z - (a * x + b * y + c)               # height above the fitted floor
        ng = (above > self.clearance) & (above < self.max_above)
        self._publish(msg.header, pts[ng], pts[~ng])

        self._n += 1
        if self._n % 20 == 1:
            self.get_logger().info(
                f"[{self._n}] in={len(pts)} nonground={int(ng.sum())} "
                f"plane c={c:+.2f} tilt=({a:+.3f},{b:+.3f})")

    def _publish(self, header, nonground, ground):
        self.pub_ng.publish(make_xyz_pc2(header, nonground))
        if self.pub_g.get_subscription_count() > 0:
            self.pub_g.publish(make_xyz_pc2(header, ground))


def main(args=None):
    rclpy.init(args=args)
    node = GroundSeg()
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
