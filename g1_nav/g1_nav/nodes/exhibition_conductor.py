#!/usr/bin/env python3
"""exhibition_conductor.py -- roam + gesture + on-request-dance exhibition
behavior inside the fixed 3 m patrol disc, replacing plain circumference-
walking's CONTENT (not its coordination contract) while `patrol_supervisor.py`
says the robot is not currently greeting a person.

See the plan file's "Track A Extension -- Exhibition Mode" section for the
full design rationale. This node is a SIBLING to circle_patrol.py, not a
modification of it (circle_patrol.py remains available, untouched, as the
simpler launch-time fallback -- decision #6). It copies circle_patrol.py's
manual-override / supervisor_active yield checks, its odometry-health gate,
its one-time anchor-at-start approach, its atomic shm-write idiom, and its
NavigateToPose action-client wiring VERBATIM in spirit (adapted for a single
fixed-point goal per segment instead of a rolling carrot) rather than
refactoring the shared node, per the plan's explicit instruction -- see that
file's own module docstring for the coordination details duplicated below;
nothing about circle_patrol.py's own contract changes.

WHAT THIS NODE ADDS ON TOP OF circle_patrol.py's CONTRACT
----------------------------------------------------------
Instead of always driving the next circle carrot, this node runs its own
small IDLE -> EXECUTING(<segment>) -> IDLE state machine (separate from
PatrolStateMachine, which still only ever occupies {circling,
paused_odom_degraded, manual_override} here -- the interaction states belong
exclusively to patrol_supervisor.py's own instance, unchanged). While idle
(no segment currently executing, and the inter-segment gap has elapsed), it
calls g1_nav.patrol_logic.plan_next_segment() -- the pure lookahead safety
oracle, already written and unit-tested this session -- and begins executing
whatever it returns:
  * roam    -- ONE NavigateToPose goal at the segment's target point (not a
               rolling cancel+resend the way circle_patrol.py's carrot needs,
               since this is a single fixed destination, not a continuously
               advancing pursuit point). Considered done when the action
               reports completion OR a generous timeout
               (duration_s + goal_timeout_slack_s) elapses, whichever first --
               so a rejected/stuck Nav2 goal can never wedge this node.
  * gesture -- fires `{"type":"cmd","name":<gesture>}` over the SAME
               unauthenticated WS pattern patrol_supervisor.py's
               `_send_wave_ws()` already uses (generalized to take the
               gesture name as a parameter; "high_wave" et al. are in
               OPEN_GESTURES and handled before robot_web_controller.py's
               ownership gate, so no control-lock/"hello" dance is needed).
  * idle    -- a brief stationary wait (also the universal safe fallback).
  * dance   -- ONLY EVER constructed from a genuine pending dashboard request
               (see PENDING DASHBOARD DANCE REQUEST below) -- this node never
               asks plan_next_segment for one out of thin air. Fires
               `{"type":"exhibition_dance_fire","fsm_id":...}` over the same
               WS connection pattern -- a NEW message type a PARALLEL task in
               this same effort is adding server-side support for in
               robot_web_controller.py; if that landing hasn't happened yet
               the server simply won't understand the message and will
               ignore it. This is BEST-EFFORT by design: logged clearly, never
               allowed to crash or wedge this node.

PENDING DASHBOARD DANCE REQUEST
--------------------------------
Polls /dev/shm/g1_exhibition_dance_request.json (written by a PARALLEL task's
dashboard-button handler in robot_web_controller.py; expected shape
`{"fsm_id": 503, "name": "Dance Mode", "space_m": 2.0, "requested_t": <epoch>}`).
Freshness-gated by file MTIME against `dance_request_stale_s`, same idiom as
every other shm freshness check in this package (e.g. circle_patrol.py's
_odom_healthy()) -- an old, forgotten request is dropped rather than firing a
dance long after whoever pressed the button walked away.

This node only ever OFFERS the pending request to plan_next_segment() as its
`pending_dance` argument at a genuine decision point (idle, about to pick a
new segment -- never mid-segment). plan_next_segment() itself decides whether
it is safe to fire RIGHT NOW (sufficient clearance for the dance's
`space_m`, robot not mid-interaction, etc. -- see patrol_logic.py); if it
defers (returns something other than "dance"), the request file is left
UNTOUCHED so it gets re-offered at the next decision point. The request file
is only deleted once this node actually DISPATCHES a "dance" segment (not
merely when the oracle was offered-but-declined it).

WRITER OWNERSHIP OF /dev/shm/g1_patrol_state.json
----------------------------------------------------
Identical rule to circle_patrol.py's own docstring section of the same name:
this node is the sole writer while phase == "circling" (or its own
"paused_odom_degraded"/"manual_override" excursions), and stops writing
entirely (goes fully idle) the instant patrol_supervisor.py's handoff claims
active, or the on-disk phase currently reads "manual_override". The ONE
additive change (per the plan's design decision #5: the shared `phase`
vocabulary is never extended) is a new `activity` field riding alongside
`phase` -- "roam", "gesture:<name>", "dance:<fsm_id>", "idle", or null --
describing what THIS node is doing right now. Every existing consumer that
ignores unknown JSON fields (patrol_supervisor.py, sim_circle.py,
sim_runner.py, test_circle_patrol.py, web/nav.js) keeps working unmodified.

Launch-time choice, not a live dashboard toggle in v1: `exhibition.enabled`
in config/patrol.yaml documents whether THIS node or circle_patrol.py should
be the one actually launched; this node does not itself refuse to run if
`enabled: false` (a launch file's job, not this process's), it only logs a
loud warning so a stale config doesn't silently confuse anyone.
"""

import json
import math
import os
import random
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

from g1_nav.patrol_logic import PatrolStateMachine, ExhibitionSegment, plan_next_segment
from g1_nav.exhibition_conductor_logic import (
    build_dance_segment_from_request, activity_label, gap_elapsed,
)

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:                    # pragma: no cover -- degrade gracefully, never crash import
    ws_connect = None


SHM_STATE_PATH = "/dev/shm/g1_patrol_state.json"           # shared writer w/ circle_patrol.py / patrol_supervisor.py
HANDOFF_PATH = "/dev/shm/g1_patrol_handoff.json"            # written by patrol_supervisor.py
ODOM_HEALTH_PATH = "/dev/shm/g1_odom_health.json"           # written by fused_odom_health.py
DANCE_REQUEST_PATH = "/dev/shm/g1_exhibition_dance_request.json"  # written by robot_web_controller.py (parallel task)
YAML_PATH = "/home/unitree/projects/g1/g1_nav/config/patrol.yaml"

# Baked defaults -- config/patrol.yaml only OVERRIDES; a missing/partial file
# must never crash this node (same idiom as every other node in this package).
# `segments` mirrors config/patrol.yaml's shipped `exhibition.segments:` list
# so a missing config still gives a sensible mix.
DEFAULTS = {
    "enabled": True,
    "radius_m": 3.0,                # pulled from circle_patrol: block below if present (see load_cfg)
    "edge_clearance_m": 0.5,
    "roam_speed_mps": 0.5,
    "roam_min_dist_m": 0.8,
    "lookahead_segments": 5,
    "max_reroll_attempts": 8,
    "gesture_duration_s": 3.5,
    "idle_duration_s": 1.0,
    "min_segment_gap_s": 1.5,
    "tick_hz": 10.0,
    "goal_timeout_slack_s": 5.0,
    "dance_request_stale_s": 120.0,
    "dance_default_duration_s": 8.0,
    "segments": [
        {"kind": "roam", "weight": 0.45},
        {"kind": "gesture", "weight": 0.35,
         "choices": ["high_wave", "high_five", "heart", "clap", "kiss"]},
        {"kind": "idle", "weight": 0.20},
    ],
    "odom_topic": "/Odometry",
    "global_frame": "odom",
    "robot_base_frame": "base_link",
    "action_name": "navigate_to_pose",
    "handoff_stale_s": 2.0,
    "ws_url": "ws://127.0.0.1:8080/ws",
    "ws_connect_timeout_s": 2.0,
}
ODOM_HEALTH_STALE_S = 2.0   # health shm considered dead past this age (see _odom_healthy)


def load_cfg():
    """Load config/patrol.yaml's `exhibition:` block over the baked defaults;
    never raise (missing file / bad YAML / missing keys all fall back). Also
    pulls `radius_m` from the `circle_patrol:` block if present there -- same
    cross-block-pull convention patrol_supervisor.py already uses for
    close_distance_m/person_fusion -- so both nodes agree on the same disc
    size if an operator only tunes circle_patrol's radius_m."""
    cfg = dict(DEFAULTS)
    cfg["segments"] = [dict(s) for s in DEFAULTS["segments"]]  # don't share the mutable default
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            sub = data.get("exhibition")
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if k in cfg and v is not None:
                        cfg[k] = v
            cp = data.get("circle_patrol")
            if isinstance(cp, dict) and cp.get("radius_m") is not None:
                cfg["radius_m"] = cp["radius_m"]
    except (OSError, yaml.YAMLError):
        pass
    return cfg


def _quat_to_yaw(qx, qy, qz, qw):
    """Yaw (rad) from a quaternion -- small local copy, matching this
    package's existing convention (circle_patrol.py, costmap_to_obstacle.py,
    scripts/fused_odometry.py each keep their own rather than a cross-tree
    import)."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class ExhibitionConductor(Node):
    def __init__(self):
        super().__init__("exhibition_conductor")
        cfg = load_cfg()
        self.cfg = cfg
        if not cfg.get("enabled", True):
            self.get_logger().warn(
                "exhibition_conductor: config/patrol.yaml's exhibition.enabled is "
                "false, but this node was launched anyway -- that's a launch-file "
                "choice, not something this process can undo; continuing to run.")

        # ROS-wiring parameters, declared exactly like circle_patrol.py's own
        # (same names/semantics, so both nodes are launch-swappable) --
        # richer behavioral knobs (segments, weights, durations, ...) stay as
        # plain cfg dict reads below, matching patrol_supervisor.py's style,
        # since several (e.g. `segments`, a list of dicts) aren't valid ROS
        # parameter types at all.
        self.global_frame = str(
            self.declare_parameter("global_frame", cfg["global_frame"]).value)
        self.robot_frame = str(
            self.declare_parameter("robot_base_frame", cfg["robot_base_frame"]).value)
        odom_topic = str(
            self.declare_parameter("odom_topic", cfg["odom_topic"]).value)
        action_name = str(
            self.declare_parameter("action_name", cfg["action_name"]).value)
        self.radius_m = float(self.declare_parameter("radius_m", cfg["radius_m"]).value)
        self.handoff_stale_s = float(
            self.declare_parameter("handoff_stale_s", cfg["handoff_stale_s"]).value)

        # --- pure logic (no ROS) -- this node's own FSM instance only ever
        # occupies {circling, paused_odom_degraded, manual_override}, exactly
        # like circle_patrol.py's; the interaction states belong exclusively
        # to patrol_supervisor.py's own PatrolStateMachine instance.
        self.fsm = PatrolStateMachine(state="circling")
        self._rng = random.Random()

        self._pose = None            # (x, y, yaw) latest /Odometry sample, or None
        self._have_pose = False
        self._center = None          # one-time disc anchor (x, y), set on first pose

        # --- exhibition segment state machine (separate from self.fsm above:
        # this is "which segment is executing right now", not the person-
        # interaction FSM) ---
        self._segment = None         # current ExhibitionSegment, or None (idle)
        self._segment_start_t = None
        self._segment_deadline_t = None
        self._segment_thread = None  # gesture/dance WS fire-and-forget thread
        self._last_segment_end_t = None

        # --- roam (NavigateToPose) goal tracking ---
        self._goal_handle = None
        self._goal_result_future = None
        self._goal_rejected = False

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos)

        self._action_client = ActionClient(self, NavigateToPose, action_name)

        tick_hz = float(cfg.get("tick_hz", 10.0))
        self.create_timer(1.0 / tick_hz if tick_hz > 0 else 0.1, self._tick)

        # Write an initial 'inactive' frame immediately so a reader never sees
        # a stale/missing file before the first pose arrives -- same
        # "come up writing" idiom as circle_patrol.py / costmap_to_obstacle.py.
        self._write_state(phase="circling", active=False, moving=False, activity=None)

        self.get_logger().info(
            f"exhibition_conductor: radius={self.radius_m}m edge_clearance="
            f"{cfg['edge_clearance_m']}m roam_speed={cfg['roam_speed_mps']}mps "
            f"lookahead={cfg['lookahead_segments']} tick_hz={tick_hz} "
            f"odom_topic={odom_topic} action={action_name} frame={self.global_frame}"
            + ("" if ws_connect is not None else
               " -- WARNING: 'websockets' package not importable, gesture/dance firing DISABLED"))

    # ------------------------------------------------------------------ pose
    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._pose = (float(p.x), float(p.y), yaw)
        self._have_pose = True

    def _anchor_center(self, x, y, yaw):
        """One-time circle-disc center anchor -- SAME formula
        CirclePatrolGeometry.re_anchor_if_needed() uses for direction=+1
        (CCW, that class's and circle_patrol.py's own default): radius_m to
        the robot's LEFT of its current heading, so both nodes would start
        from the exact same place if launch-swapped. Duplicated locally
        (rather than instantiating CirclePatrolGeometry just for its .center)
        since exhibition mode has no "direction" concept of its own -- roam
        targets are omnidirectional within the disc, this is only about
        getting ONE consistent center point."""
        heading = yaw + (math.pi / 2.0)
        cx = x + self.radius_m * math.cos(heading)
        cy = y + self.radius_m * math.sin(heading)
        return (cx, cy)

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
        node must yield: no goals, no shm writes). Identical fail-open-on-
        staleness rule as circle_patrol.py's own method of the same name."""
        try:
            age = time.time() - os.path.getmtime(HANDOFF_PATH)
        except OSError:
            return False                       # supervisor never started -> not active
        if age > self.handoff_stale_s:
            return False                       # stale heartbeat -> FAIL OPEN
        data = self._read_json(HANDOFF_PATH)
        if not isinstance(data, dict):
            return False
        return bool(data.get("supervisor_active"))

    def _odom_healthy(self):
        """True iff fused_odom_health.py reports a healthy fused pose.
        Identical asymmetric fail-open(missing)/fail-closed(stale) design as
        circle_patrol.py's own method of the same name."""
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

    def _poll_pending_dance(self):
        """Returns an ExhibitionSegment(kind="dance", ...) if a fresh
        dashboard request exists at DANCE_REQUEST_PATH, else None. Freshness
        gated by file MTIME against dance_request_stale_s, same idiom as
        _odom_healthy() above. Does NOT clear/consume the file -- see
        _clear_pending_dance(), only called once this node actually
        DISPATCHES the dance (not merely when the oracle was offered-but-
        declined it -- see module docstring)."""
        try:
            age = time.time() - os.path.getmtime(DANCE_REQUEST_PATH)
        except OSError:
            return None                        # no request pending -> normal, not a warning
        if age > self.cfg["dance_request_stale_s"]:
            return None                        # stale -- operator likely forgot / left
        data = self._read_json(DANCE_REQUEST_PATH)
        return build_dance_segment_from_request(data, self.cfg["dance_default_duration_s"])

    def _clear_pending_dance(self):
        try:
            os.remove(DANCE_REQUEST_PATH)
        except OSError:
            pass

    # ------------------------------------------------------------------ tick
    def _tick(self):
        # 1) Manual override always wins, from whoever currently owns the file.
        current = self._read_json(SHM_STATE_PATH)
        ext_phase = current.get("phase") if isinstance(current, dict) else None
        if ext_phase == "manual_override":
            self.fsm.try_transition("manual_override")
            self._abort_segment()
            return                              # do NOT write -- we don't own the file right now
        if self.fsm.state == "manual_override":
            self.fsm.try_transition("circling")  # override cleared externally -> resume

        # 2) Yield to patrol_supervisor.py while it owns the interaction.
        if self._supervisor_active():
            self._abort_segment()
            return                              # supervisor is the writer right now

        # 3) Odometry-health gate -- pause (don't dead-reckon) on a degraded estimate.
        if not self._odom_healthy():
            self.fsm.try_transition("paused_odom_degraded")
            self._abort_segment()
            self._write_state(phase="paused_odom_degraded", active=True, moving=False, activity=None)
            return
        if self.fsm.state == "paused_odom_degraded":
            self.fsm.try_transition("circling")

        # 4) No pose yet -- nothing to anchor on / drive toward.
        if not self._have_pose or self._pose is None:
            self._write_state(phase="circling", active=False, moving=False, activity=None)
            return

        x, y, yaw = self._pose
        if self._center is None:
            self._center = self._anchor_center(x, y, yaw)   # one-time; never re-anchored

        now = time.monotonic()

        # 5) Advance a currently-executing segment, if any.
        if self._segment is not None:
            done = self._advance_segment(now)
            if not done:
                self._write_state(
                    phase="circling", active=True,
                    moving=(self._segment.kind == "roam"),
                    activity=activity_label(self._segment))
                return
            finished = self._segment
            self._segment = None
            self._last_segment_end_t = now
            # The pending-dance request is cleared ONLY once we actually
            # dispatched-and-finished it -- see module docstring.
            if finished.kind == "dance":
                self._clear_pending_dance()

        # 6) Idle: enforce the inter-segment settle gap before deciding again.
        if not gap_elapsed(self._last_segment_end_t, now, self.cfg["min_segment_gap_s"]):
            self._write_state(phase="circling", active=True, moving=False, activity="idle")
            return

        # 7) Decide the next segment (the oracle) and begin executing it.
        cfg = dict(self.cfg)
        cfg["center"] = self._center
        pending = self._poll_pending_dance()   # only ever offered at a genuine decision point
        segment = plan_next_segment((x, y, yaw), cfg, self._rng, pending_dance=pending)
        self._begin_segment(segment, now)
        if pending is not None and segment.kind == "dance":
            self._clear_pending_dance()
        self._write_state(
            phase="circling", active=True, moving=(segment.kind == "roam"),
            activity=activity_label(segment))

    # -------------------------------------------------------- segment exec
    def _begin_segment(self, segment, now):
        self._segment = segment
        self._segment_start_t = now
        self._segment_thread = None
        self._goal_handle = None
        self._goal_result_future = None
        self._goal_rejected = False

        if segment.kind == "roam":
            slack = self.cfg.get("goal_timeout_slack_s", 5.0)
            self._segment_deadline_t = now + segment.duration_s + slack
            self._send_roam_goal(segment.target)
        elif segment.kind == "gesture":
            self._segment_deadline_t = now + segment.duration_s
            self._segment_thread = threading.Thread(
                target=self._send_gesture_ws, args=(segment.gesture,),
                name="exhibition-gesture-ws", daemon=True)
            self._segment_thread.start()
        elif segment.kind == "dance":
            self._segment_deadline_t = now + segment.duration_s
            self._segment_thread = threading.Thread(
                target=self._send_dance_fire, args=(segment.dance_fsm_id,),
                name="exhibition-dance-ws", daemon=True)
            self._segment_thread.start()
        else:   # "idle" (or an unrecognized kind -- fail safe, just wait it out)
            self._segment_deadline_t = now + segment.duration_s

    def _advance_segment(self, now):
        """Returns True once the current segment is considered done."""
        seg = self._segment
        if seg.kind == "roam":
            if self._goal_rejected:
                return True
            if self._goal_result_future is not None and self._goal_result_future.done():
                self._log_roam_result()
                self._cancel_goal()
                return True
            if now >= self._segment_deadline_t:
                self.get_logger().warn(
                    "exhibition_conductor: roam segment timed out (goal_timeout_slack_s "
                    "exceeded) -- abandoning goal, returning to idle")
                self._cancel_goal()
                return True
            return False
        if seg.kind == "gesture":
            thread_done = self._segment_thread is not None and not self._segment_thread.is_alive()
            return thread_done or now >= self._segment_deadline_t
        # "dance" and "idle": fixed wait only (dance deliberately does NOT
        # early-exit on WS-thread completion -- see module docstring/plan:
        # duration_s is the documented placeholder for "how long the routine
        # takes", not merely "how long the WS send took").
        return now >= self._segment_deadline_t

    def _abort_segment(self, *_unused):
        """Cancel/abandon whatever segment is executing (called when yielding
        to manual_override / the supervisor / an odom-health pause). Any
        already-fired gesture/dance WS thread cannot be un-fired -- it simply
        finishes on its own in the background, matching patrol_supervisor.py's
        own _abort_interaction() not trying to un-fire an in-flight wave."""
        self._cancel_goal()
        self._segment = None
        self._segment_thread = None
        self._goal_result_future = None
        self._goal_rejected = False

    # -------------------------------------------------------- NavigateToPose
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

    def _send_roam_goal(self, target):
        """A SINGLE NavigateToPose goal at a fixed point -- not circle_patrol.
        py's rolling cancel+resend carrot, which is unnecessary here since a
        roam segment's destination doesn't move tick-to-tick."""
        if self._pose is None:
            self._goal_rejected = True
            return
        tx, ty = target
        x, y, _yaw = self._pose
        goal_yaw = math.atan2(ty - y, tx - x)
        if not self._action_client.server_is_ready():
            self.get_logger().warn(
                "exhibition_conductor: NavigateToPose server not ready -- abandoning this "
                "roam segment (will be retried via the next oracle decision)")
            self._goal_rejected = True
            return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose_stamped(tx, ty, goal_yaw)
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_roam_goal_response)

    def _on_roam_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().warn(f"exhibition_conductor: NavigateToPose send failed: {e}")
            self._goal_rejected = True
            return
        if not handle.accepted:
            self.get_logger().warn("exhibition_conductor: roam goal rejected by Nav2")
            self._goal_rejected = True
            return
        self._goal_handle = handle
        self._goal_result_future = handle.get_result_async()

    def _log_roam_result(self):
        try:
            result = self._goal_result_future.result()
            self.get_logger().info(f"exhibition_conductor: roam segment finished, status={result.status}")
        except Exception:
            pass

    def _cancel_goal(self):
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._goal_handle = None

    # -------------------------------------------------------------- WS fire
    def _send_gesture_ws(self, gesture_name):
        """Fire the EXISTING safety-gated dashboard gesture over WS -- the
        SAME pattern patrol_supervisor.py's _send_wave_ws() uses, generalized
        to take the gesture name as a parameter (its callers still only ever
        pass an OPEN_GESTURES name from config/patrol.yaml's exhibition.
        segments choices). Runs on a background thread (blocking round trip);
        any failure is logged and swallowed so an unreachable dashboard can
        never crash or wedge this node."""
        if ws_connect is None:
            self.get_logger().warn(
                "exhibition_conductor: 'websockets' package not available -- cannot fire gesture")
            return
        try:
            with ws_connect(self.cfg["ws_url"],
                            open_timeout=self.cfg["ws_connect_timeout_s"]) as ws:
                ws.send(json.dumps({"type": "cmd", "name": gesture_name}))
        except Exception as e:
            self.get_logger().warn(f"exhibition_conductor: gesture WS send failed: {e}")

    def _send_dance_fire(self, fsm_id):
        """Best-effort dance trigger over the same WS connection pattern.
        `exhibition_dance_fire` is a NEW message type a PARALLEL task in this
        same effort is adding server-side support for in
        robot_web_controller.py -- if that hasn't landed yet, the server
        simply won't recognize the message and this is a silent no-op
        server-side; logged clearly here, never allowed to crash this node."""
        if ws_connect is None:
            self.get_logger().warn(
                "exhibition_conductor: 'websockets' package not available -- cannot fire dance")
            return
        try:
            with ws_connect(self.cfg["ws_url"],
                            open_timeout=self.cfg["ws_connect_timeout_s"]) as ws:
                ws.send(json.dumps({"type": "exhibition_dance_fire", "fsm_id": fsm_id}))
            self.get_logger().info(
                f"exhibition_conductor: sent exhibition_dance_fire fsm_id={fsm_id} "
                "(best-effort -- server-side support may not exist yet)")
        except Exception as e:
            self.get_logger().warn(f"exhibition_conductor: dance-fire WS send failed: {e}")

    # ------------------------------------------------------------- shm write
    def _write_state(self, phase, active, moving, activity):
        """Atomic write: tmp file then os.replace -- a reader never sees a
        partial write. Same idiom as circle_patrol.py's _write_state(). The
        ONLY additive field vs. that node is `activity` -- see module
        docstring's "WRITER OWNERSHIP" section (the plan's design decision
        #5: the shared `phase` vocabulary itself is never extended)."""
        obj = {
            "active": bool(active),
            "phase": phase,
            "moving": bool(moving),
            "activity": activity,
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
    node = ExhibitionConductor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._abort_segment()
            node._write_state(phase="circling", active=False, moving=False, activity=None)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
