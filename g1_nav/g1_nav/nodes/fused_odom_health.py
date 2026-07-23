#!/usr/bin/env python3
"""fused_odom_health.py -- thin ROS2 wrapper around scripts/fused_odometry.py.

Feeds a FusedOdometry complementary filter (leg + IMU + FAST-LIO lidar -- see
that module's docstring for the full design rationale) from the THREE real G1
sources and republishes just its `confidence`/`lidar_healthy` verdict (plus a
couple of cheap diagnostic extras) to /dev/shm/g1_odom_health.json at a low
rate. circle_patrol.py polls that file as a health trip-wire: on an unhealthy
reading it cancels its NavigateToPose goal and holds position rather than
dead-reckoning through a degraded pose estimate (Track A3's stated design --
"wire fused_odometry.py in as a health side-car... rather than promoting it to
the TF source in v1: lower-risk, and this is the module's first real
validation against hardware DDS traffic").

WIRING (follows scripts/fused_odometry.py's own G1FusedOdometryAdapter
docstring recipe -- NOT its .start() method, which is a documented,
intentional NotImplementedError stub for the ROS half of the wiring; the ROS
half is implemented here directly instead of calling that stub):

  * LEG POSE + IMU YAW-RATE both ride Unitree's `rt/odommodestate`
    (SportModeState_) on DDS DOMAIN 0 -- subscribed here exactly like
    scripts/lidar_source.py's OdomReader and g1_cmd_vel_bridge.py's LocoClient
    init (ChannelFactoryInitialize(0, interface)). This is a SEPARATE,
    parallel subscriber from the existing OdomReader -- per the adapter's own
    docstring: "a new, parallel consumer... unitree_sdk2py allows multiple
    subscribers", so OdomReader is left completely untouched.
  * FAST-LIO's ABSOLUTE POSE rides `/Odometry` (nav_msgs/Odometry) on ROS2
    DOMAIN 99 (this node's own rclpy domain, per env.sh / ROS_DOMAIN_ID) -- a
    plain create_subscription.
  * Both a Unitree-SDK domain-0 participant AND an rclpy domain-99 participant
    are initialized in this ONE process, exactly like g1_cmd_vel_bridge.py
    already does for its LocoClient. That file flags this dual-domain
    coexistence as `<ASK ENRICO: verify single-process dual-domain>` and it is
    UNVALIDATED against real DDS traffic; this node inherits the IDENTICAL
    caveat (fused_odometry.py's own docstring says as much: "TODO: validate on
    the G1" -- this is that module's first real hardware wiring, full stop).
  * TIMESTAMPS: both subscriptions feed the filter using time.monotonic() AT
    RECEIPT, not each message's own embedded stamp. This sidesteps the
    cross-domain "do the two clocks share a base" question the adapter's
    docstring flags as unverified, at the cost of ignoring true sensor
    latency/jitter -- an acceptable tradeoff for a coarse health/confidence
    side-car. Revisit if this filter is ever promoted to the real TF source,
    which the plan explicitly says it should NOT be in v1.
  * A failure to init the Unitree-SDK leg/IMU subscription (e.g. the SDK isn't
    reachable, or eth0 is down) does NOT crash this node -- the filter simply
    runs lidar-only (have_pose stays False until FAST-LIO bootstraps it, or
    the leg/IMU path recovers on a later restart). Health reporting degrades
    gracefully rather than the whole side-car going dark.

OUTPUT CONTRACT (/dev/shm/g1_odom_health.json), written at publish_hz (default
5 Hz), atomic write (tmp + os.replace, same idiom as costmap_to_obstacle.py):
    {"t": float, "confidence": float, "lidar_healthy": bool, "have_pose": bool,
     "lidar_age_s": float|null, "healthy": bool}
`healthy` is the single boolean circle_patrol.py actually gates on:
    have_pose AND lidar_healthy AND confidence >= min_confidence
(min_confidence default 0.35 in config/patrol.yaml is a DESIGN-DECISION default,
not specified by the plan -- needs real-robot tuning once FAST-LIO/leg DDS
traffic is actually flowing through this node.)
"""

import importlib.util
import json
import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry


# --- import scripts/fused_odometry.py by path (not an installed package; same
# idiom already used by g1_nav/nodes/realsense_cloud_pub.py for lidar_source.py) --
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
_FUSED_ODOM_SRC = os.path.join(_REPO, "scripts", "fused_odometry.py")

_spec = importlib.util.spec_from_file_location("g1_fused_odometry", _FUSED_ODOM_SRC)
_fused_odometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fused_odometry)

FusedOdometry = _fused_odometry.FusedOdometry
quat_to_yaw = _fused_odometry.quat_to_yaw


SHM_PATH = "/dev/shm/g1_odom_health.json"
YAML_PATH = "/home/unitree/projects/g1/g1_nav/config/patrol.yaml"

DEFAULTS = {
    "publish_hz": 5.0,
    "min_confidence": 0.35,
    "interface": "eth0",
    "lio_topic": "/Odometry",
}


def load_cfg():
    """Load config/patrol.yaml's `fused_odom_health:` block over the baked
    defaults; never raise (missing file / bad YAML / missing keys fall back)."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        sub = data.get("fused_odom_health") if isinstance(data, dict) else None
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except (OSError, yaml.YAMLError):
        pass
    return cfg


class FusedOdomHealth(Node):
    def __init__(self):
        super().__init__("fused_odom_health")
        cfg = load_cfg()
        self.publish_hz = float(
            self.declare_parameter("publish_hz", cfg["publish_hz"]).value)
        self.min_confidence = float(
            self.declare_parameter("min_confidence", cfg["min_confidence"]).value)
        interface = str(self.declare_parameter("interface", cfg["interface"]).value)
        lio_topic = str(self.declare_parameter("lio_topic", cfg["lio_topic"]).value)

        self.fused = FusedOdometry()

        # --- (1)+(2) leg + IMU from rt/odommodestate, Unitree SDK domain 0 ---
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize, ChannelSubscriber)
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            ChannelFactoryInitialize(0, interface)
            self._odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self._odom_sub.Init(self._on_odommodestate, 10)
            self.get_logger().info(
                f"fused_odom_health: subscribed rt/odommodestate (domain 0/{interface})")
        except Exception as e:
            # Leg+IMU are optional inputs to the complementary filter -- it still
            # runs lidar-only (have_pose stays False until FAST-LIO bootstraps it)
            # rather than crashing the whole health side-car over this.
            self.get_logger().warn(
                f"fused_odom_health: leg/IMU subscribe failed ({e}); "
                f"continuing lidar-only")

        # --- (3) FAST-LIO absolute pose, /Odometry, this node's own domain 99 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, lio_topic, self._on_lidar_odometry, qos)

        period = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.2
        self.create_timer(period, self._tick)

        self._write()   # come up writing immediately (healthy=False until real data lands)
        self.get_logger().info(
            f"fused_odom_health: publish_hz={self.publish_hz} "
            f"min_confidence={self.min_confidence} lio_topic={lio_topic} -> {SHM_PATH}")

    # ------------------------------------------------------------- callbacks
    def _on_odommodestate(self, msg):
        """SportModeState_ callback -- runs on the Unitree SDK's own callback
        thread, NOT the rclpy executor thread. add_imu()/add_leg() are cheap,
        lock-free numeric updates on plain attributes (see fused_odometry.py);
        FusedOdometry is not otherwise touched concurrently from the rclpy side
        except via get_state()/get_pose() reads in _tick(), which only read
        already-computed floats -- a benign, best-effort race, not a mutation."""
        t = time.monotonic()
        try:
            gyro_z = float(msg.imu_state.gyroscope[2])
            self.fused.add_imu(gyro_z, t)
            x = float(msg.position[0])
            y = float(msg.position[1])
            yaw = float(msg.imu_state.rpy[2])
            self.fused.add_leg(x, y, yaw, t)
        except Exception:
            pass   # never let a malformed/partial SDK message crash the side-car

    def _on_lidar_odometry(self, msg):
        t = time.monotonic()
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            self.fused.add_lidar(float(p.x), float(p.y), yaw, t)
        except Exception:
            pass

    # ------------------------------------------------------------------ tick
    def _tick(self):
        self._write()

    def _write(self):
        st = self.fused.get_state()
        healthy = bool(st["have_pose"] and st["lidar_healthy"]
                       and st["confidence"] >= self.min_confidence)
        age = st["lidar_age_s"]
        obj = {
            "t": round(time.time(), 3),
            "confidence": round(float(st["confidence"]), 4),
            "lidar_healthy": bool(st["lidar_healthy"]),
            "have_pose": bool(st["have_pose"]),
            "lidar_age_s": (None if not math.isfinite(age) else round(float(age), 3)),
            "healthy": healthy,
        }
        tmp = SHM_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, SHM_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FusedOdomHealth()
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
