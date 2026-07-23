#!/usr/bin/env python3
"""circle_patrol.py -- drives the G1 around a fixed circle via rolling NavigateToPose goals.

WHAT THIS NODE DOES
--------------------
Wraps g1_nav.patrol_logic.CirclePatrolGeometry (pure geometry, no ROS -- see that
module for the carrot-point math and unit tests). On the FIRST pose it receives,
fixes the circle center from the current (x, y, yaw) -- odom-frame-relative,
accepted drift, no persistent map/AMCL exists in this codebase (see
MIGRATION_STATUS.md). Every tick (~goal_resend_hz, default 1 Hz) it computes a
fresh pure-pursuit carrot pose on the circle and sends it as a NavigateToPose
goal, CANCELLING the previous tick's goal first rather than trying to keep one
goal alive indefinitely -- this is the plan's stated design choice: simpler and
more robust against a single aborted/rejected goal than nursing one long-lived
goal along.

POSE SOURCE
-----------
Subscribes FAST-LIO's `/Odometry` (nav_msgs/Odometry) directly, per
g1_nav/config/nav2_params.yaml (`bt_navigator.odom_topic: /Odometry`,
`local_costmap.global_frame: odom`, `robot_base_frame: base_link`) and
g1_nav/launch/tf.launch.py (FAST-LIO owns odom->base_link; this package does
NOT re-publish it). A direct topic subscription is used instead of a TF lookup
(costmap_to_obstacle.py's approach) because /Odometry already IS the pose we
need in the exact frame NavigateToPose goals must be stamped in -- one fewer
moving part. NavigateToPose goals are stamped in `global_frame` (default
"odom") to match nav2_params.yaml.

ODOMETRY-HEALTH GATING (Track A3)
----------------------------------
Polls /dev/shm/g1_odom_health.json (written by fused_odom_health.py, a health
SIDE-CAR around scripts/fused_odometry.py -- see that node for why it is a
side-car and not the TF source in v1). On an unhealthy reading this node
transitions itself to `paused_odom_degraded`: it CANCELS its goal and HOLDS
POSITION -- it does not keep dead-reckoning through a degraded pose estimate.
FAIL-OPEN CAVEAT (documented, not fully hardware-validated): if the health shm
file has NEVER existed (fused_odom_health.py was never started), this node
still drives -- it does not hard-require the side-car to be running at all
(useful for bench-testing circle_patrol.py alone). If the file EXISTED and then
went stale (the side-car died/froze), that is treated as more suspicious and
this node fails CLOSED (paused_odom_degraded). See _odom_healthy().

WRITER OWNERSHIP OF /dev/shm/g1_patrol_state.json (the real coordination detail)
---------------------------------------------------------------------------------
This node is the SOLE writer of g1_patrol_state.json WHILE it is actually
driving the circle (phase == "circling", or its own "paused_odom_degraded" /
"manual_override" excursions). It STOPS writing (goes fully idle, touches
nothing) the instant EITHER of:
  (a) patrol_supervisor.py has taken over -- signalled via a small, separate
      handoff file, /dev/shm/g1_patrol_handoff.json, `{"supervisor_active":
      bool, "t": <written by patrol_supervisor.py>}`. This node treats a
      MISSING or STALE (> handoff_stale_s, default 2 s, judged by file MTIME
      like every other shm freshness check in this codebase) handoff file as
      "supervisor not active" and FAILS OPEN back to driving -- a dead/crashed
      supervisor must not freeze the robot in `pausing`/`facing`/... forever.
      patrol_supervisor.py becomes the writer of g1_patrol_state.json for the
      duration of its takeover and hands writer responsibility back to THIS
      node on its resuming -> circling transition (see that module's matching
      docstring section -- this is the SAME coordination detail described from
      the other side).
  (b) the CURRENT on-disk phase (read fresh every tick, from whoever wrote it
      last) is "manual_override" -- per the plan's cross-workstream decision
      #4, robot_web_controller.py's teleop loop flips phase to
      "manual_override" on any manual joystick input and this node "must
      immediately obey by cancelling its goal and stopping". This node does
      not try to detect the EDGE of that transition (no ownership handshake
      needed for this direction): it simply re-reads the phase field at the
      top of every tick, before deciding whether to write, and complies with
      whatever it currently says.

Everything else (NavigateToPose action wiring, goal cancel/resend, shm I/O) is
plain rclpy; the reusable geometry/FSM math lives in patrol_logic.py so it is
unit-testable without ROS (see scripts/test_circle_patrol.py, a parallel
track's deliverable, and scripts/sim_circle.py).
"""

import json
import math
import os
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

from g1_nav.patrol_logic import CirclePatrolGeometry, PatrolStateMachine


SHM_STATE_PATH = "/dev/shm/g1_patrol_state.json"
HANDOFF_PATH = "/dev/shm/g1_patrol_handoff.json"       # written by patrol_supervisor.py
ODOM_HEALTH_PATH = "/dev/shm/g1_odom_health.json"      # written by fused_odom_health.py
YAML_PATH = "/home/unitree/projects/g1/g1_nav/config/patrol.yaml"

# Baked defaults -- config/patrol.yaml only OVERRIDES; a missing/partial file
# must never crash this node (same idiom as obstacle/obstacle.yaml).
DEFAULTS = {
    "radius_m": 3.0,
    "lookahead_m": 1.25,
    "direction": 1,
    "goal_resend_hz": 1.0,
    "odom_topic": "/Odometry",
    "global_frame": "odom",
    "robot_base_frame": "base_link",
    "action_name": "navigate_to_pose",
    "handoff_stale_s": 2.0,
}
ODOM_HEALTH_STALE_S = 2.0   # health shm considered dead past this age (see _odom_healthy)


def load_cfg():
    """Load config/patrol.yaml's `circle_patrol:` block over the baked defaults;
    never raise (missing file / bad YAML / missing keys all fall back)."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        sub = data.get("circle_patrol") if isinstance(data, dict) else None
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except (OSError, yaml.YAMLError):
        pass
    return cfg


def _quat_to_yaw(qx, qy, qz, qw):
    """Yaw (rad) from a quaternion -- same formula duplicated locally in
    costmap_to_obstacle.py (_yaw_from_quat) and scripts/fused_odometry.py
    (quat_to_yaw); kept as a small local copy rather than a cross-tree import,
    matching this package's existing convention."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class CirclePatrol(Node):
    def __init__(self):
        super().__init__("circle_patrol")
        cfg = load_cfg()

        self.global_frame = str(
            self.declare_parameter("global_frame", cfg["global_frame"]).value)
        self.robot_frame = str(
            self.declare_parameter("robot_base_frame", cfg["robot_base_frame"]).value)
        odom_topic = str(
            self.declare_parameter("odom_topic", cfg["odom_topic"]).value)
        action_name = str(
            self.declare_parameter("action_name", cfg["action_name"]).value)
        radius_m = float(self.declare_parameter("radius_m", cfg["radius_m"]).value)
        lookahead_m = float(self.declare_parameter("lookahead_m", cfg["lookahead_m"]).value)
        direction = int(self.declare_parameter("direction", cfg["direction"]).value)
        goal_hz = float(self.declare_parameter("goal_resend_hz", cfg["goal_resend_hz"]).value)
        self.handoff_stale_s = float(
            self.declare_parameter("handoff_stale_s", cfg["handoff_stale_s"]).value)

        # --- pure logic (no ROS) ---
        self.geometry = CirclePatrolGeometry(
            radius_m=radius_m, lookahead_m=lookahead_m, direction=direction)
        # This node's own FSM instance only ever occupies {circling,
        # paused_odom_degraded, manual_override} -- the interaction states
        # (pausing..resuming) belong exclusively to patrol_supervisor.py's own
        # PatrolStateMachine instance. Both reuse the SAME class/vocabulary
        # from patrol_logic.py; only one process is ever the writer at a time
        # (see module docstring "WRITER OWNERSHIP").
        self.fsm = PatrolStateMachine(state="circling")

        self._pose = None            # (x, y, yaw) latest /Odometry sample, or None
        self._have_pose = False
        self._goal_handle = None
        self._goal_active = False

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos)

        self._action_client = ActionClient(self, NavigateToPose, action_name)

        period = 1.0 / goal_hz if goal_hz > 0 else 1.0
        self.create_timer(period, self._tick)

        # Write an initial 'inactive' frame immediately so a reader (dashboard,
        # patrol_supervisor) never sees a stale/missing file before the first
        # pose arrives -- same "come up writing" idiom as costmap_to_obstacle.py.
        self._write_state(phase="circling", active=False, moving=False)

        self.get_logger().info(
            f"circle_patrol: radius={radius_m}m lookahead={lookahead_m}m "
            f"dir={'CCW' if direction >= 0 else 'CW'} goal_hz={goal_hz} "
            f"odom_topic={odom_topic} action={action_name} frame={self.global_frame}")

    # ------------------------------------------------------------------ pose
    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._pose = (float(p.x), float(p.y), yaw)
        self._have_pose = True

    # ------------------------------------------------------------- shm reads
    @staticmethod
    def _read_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _supervisor_active(self):
        """True iff patrol_supervisor.py currently owns the interaction (this
        node must yield: no goals, no shm writes). See module docstring
        "WRITER OWNERSHIP" for the fail-open staleness rule."""
        try:
            age = time.time() - os.path.getmtime(HANDOFF_PATH)
        except OSError:
            return False                       # supervisor never started -> not active
        if age > self.handoff_stale_s:
            return False                       # stale heartbeat -> FAIL OPEN (see docstring)
        data = self._read_json(HANDOFF_PATH)
        if not isinstance(data, dict):
            return False
        return bool(data.get("supervisor_active"))

    def _odom_healthy(self):
        """True iff fused_odom_health.py reports a healthy fused pose. See
        module docstring for the asymmetric fail-open(missing)/fail-closed
        (stale) design."""
        try:
            age = time.time() - os.path.getmtime(ODOM_HEALTH_PATH)
        except OSError:
            return True                        # side-car never started -> don't block on it
        if age > ODOM_HEALTH_STALE_S:
            return False                       # side-car died/froze -> FAIL CLOSED
        data = self._read_json(ODOM_HEALTH_PATH)
        if not isinstance(data, dict):
            return False
        return bool(data.get("healthy", False))

    # ------------------------------------------------------------------ tick
    def _tick(self):
        # 1) Manual override always wins, from whoever currently owns the file.
        current = self._read_json(SHM_STATE_PATH)
        ext_phase = current.get("phase") if isinstance(current, dict) else None
        if ext_phase == "manual_override":
            self.fsm.try_transition("manual_override")
            self._cancel_goal()
            return                              # do NOT write -- we don't own the file right now
        if self.fsm.state == "manual_override":
            self.fsm.try_transition("circling")  # override cleared externally -> resume

        # 2) Yield to patrol_supervisor.py while it owns the interaction.
        if self._supervisor_active():
            self._cancel_goal()
            return                              # supervisor is the writer right now

        # 3) Odometry-health gate -- pause (don't dead-reckon) on a degraded estimate.
        if not self._odom_healthy():
            self.fsm.try_transition("paused_odom_degraded")
            self._cancel_goal()
            self._write_state(phase="paused_odom_degraded", active=True, moving=False)
            return
        if self.fsm.state == "paused_odom_degraded":
            self.fsm.try_transition("circling")

        # 4) No pose yet -- nothing to anchor on / drive toward.
        if not self._have_pose or self._pose is None:
            self._write_state(phase="circling", active=False, moving=False)
            return

        # 5) Normal driving: anchor once, carrot every tick, cancel+resend goal.
        x, y, yaw = self._pose
        self.geometry.re_anchor_if_needed(x, y, yaw)   # one-time; no-op after the first call
        goal_x, goal_y, goal_yaw = self.geometry.carrot_pose(x, y)
        self._send_goal(goal_x, goal_y, goal_yaw)
        self._write_state(phase="circling", active=True, moving=True)

    # -------------------------------------------------------------- actions
    def _make_pose_stamped(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        return ps

    def _send_goal(self, x, y, yaw):
        """Cancel any goal from the previous tick, then send a fresh one.
        Rolling-goal design (per the plan): simpler and more robust against a
        single aborted/rejected goal than nursing one long-lived goal."""
        if not self._action_client.server_is_ready():
            return                              # Nav2 not up yet this tick; try again next tick
        self._cancel_goal()
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose_stamped(x, y, yaw)
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().warn(f"NavigateToPose send failed: {e}")
            return
        if not handle.accepted:
            return
        self._goal_handle = handle
        self._goal_active = True

    def _cancel_goal(self):
        if self._goal_handle is not None and self._goal_active:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._goal_handle = None
        self._goal_active = False

    # ------------------------------------------------------------- shm write
    def _write_state(self, phase, active, moving):
        """Atomic write: tmp file then os.replace -- a reader never sees a
        partial write. Same idiom as costmap_to_obstacle.py's _write_shm()."""
        obj = {
            "active": bool(active),
            "phase": phase,
            "moving": bool(moving),
            "updated_t": round(time.time(), 3),
        }
        tmp = SHM_STATE_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, SHM_STATE_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CirclePatrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cancel_goal()
            node._write_state(phase="circling", active=False, moving=False)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
