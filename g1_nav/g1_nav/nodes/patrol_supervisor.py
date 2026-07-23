#!/usr/bin/env python3
"""patrol_supervisor.py -- person-interaction state machine for the G1 circle patrol.

Owns the ONE PatrolStateMachine instance (g1_nav.patrol_logic) that actually
drives the interaction half of the mandated vocabulary:

    circling -> pausing -> facing -> waving -> waiting -> resuming -> circling

manual_override (and, defensively, fault/paused_odom_degraded -- see
_handle_interrupt_state) are honoured from any state, per patrol_logic.py's
transition table.

COORDINATION WITH circle_patrol.py (the real coordination detail)
--------------------------------------------------------------------
Two SEPARATE processes/nodes both care about /dev/shm/g1_patrol_state.json and
about who is allowed to command the robot right now. The rule, symmetric with
circle_patrol.py's own matching docstring section:

  * While this node's FSM is in "circling" (i.e. NOT intervening), it writes
    NOTHING -- neither g1_patrol_state.json nor the handoff file. circle_patrol.py
    is the sole writer/driver in this state.
  * The INSTANT a debounced "person nearby" trigger fires (circling -> pausing),
    this node becomes the writer: every tick it writes BOTH
      (a) /dev/shm/g1_patrol_handoff.json = {"supervisor_active": true, "t": ...}
          -- circle_patrol.py polls this (by file MTIME, fail-OPEN on staleness
          -- see that node) and, once it sees `true`, cancels its own
          NavigateToPose goal and goes idle (does not touch g1_patrol_state.json
          itself while this is true).
      (b) /dev/shm/g1_patrol_state.json's `phase` field, following this node's
          own FSM state (pausing/facing/waving/waiting).
  * On resuming -> circling (the ONLY sanctioned re-entry to the happy path),
    this node writes phase="circling" ONE LAST TIME (so a reader never sees a
    gap), writes supervisor_active=false, and then STOPS writing anything --
    handing writer responsibility back to circle_patrol.py.
  * A KNOWN, ACCEPTED race: there is a brief window (bounded by circle_patrol's
    own tick period, ~1 s at goal_resend_hz=1.0) between this node asserting
    supervisor_active=true and circle_patrol.py actually observing it and
    yielding. This node does not itself drive translation (only in-place yaw,
    during `facing`), so the worst case is circle_patrol.py sends one more
    stale NavigateToPose goal before yielding -- not a safety hazard, just a
    latency footnote, documented rather than engineered away (no hardware to
    validate the real handoff latency against yet).

MANUAL OVERRIDE
-----------------
Every tick, before anything else, this node reads the CURRENT on-disk
g1_patrol_state.json phase (regardless of who wrote it last). If it reads
"manual_override" (per the plan's cross-workstream decision #4:
robot_web_controller.py's teleop loop flips phase on any manual joystick
input), this node immediately: zeroes any /cmd_vel it was publishing, releases
the handoff claim (supervisor_active=false), and transitions its own FSM to
manual_override -- honouring the override exactly like circle_patrol.py does.
It resumes (transitions manual_override -> circling) once the on-disk phase
stops reading "manual_override".

PERSON-NEARBY TRIGGER (circling -> pausing)
-----------------------------------------------
Reads /dev/shm/g1_person_track.json (person_fusion.py). A person counts as
"nearby" via, in priority order:
  1. a real depth reading: distance_m <= close_distance_m (default 1.6 m); else
  2. a box-height/frame-height fallback proxy: (y2-y1)/frame_h >= close_box_frac_h
     (default 0.55) -- used ONLY when distance_m is null, which
     person_fusion.py's docstring documents as the COMMON case until a
     continuous depth-export follow-up lands. This reuses the plan's own
     Track C pattern ("approximate via POSE_TRACKS box heuristic until Track
     A's real bearing/distance fusion lands, then swap") rather than inventing
     a new one.
DEBOUNCED (person_debounce_s, default 1.0 s of CONTINUOUS qualifying presence)
so a single stray frame never fires the interaction -- "don't flip on a single
frame" per the plan. Simplification, documented: a single frame WITHOUT a
qualifying person resets the debounce timer to zero (no partial-credit
hysteresis); and among multiple detected persons, the FIRST qualifying one in
person_fusion's list wins -- there is no multi-person prioritization yet.

FACING (P-controller yaw-only rotation)
-------------------------------------------
Publishes geometry_msgs/Twist on the SAME topic g1_cmd_vel_bridge.py already
subscribes ("/cmd_vel") -- read that file to confirm the exact topic/message
type, not re-invented here. vyaw = clamp(yaw_kp * bearing_err, +-yaw_max_rad_s),
bearing_err re-read from person_fusion's output every tick (closed-loop, not
open-loop), re-acquiring the freshest bearing while the person is still
trackable and holding the last-known bearing (with a lost-timer) if they
briefly drop out. Converges (-> waving) once |bearing_err| stays under
yaw_tolerance_rad for yaw_settle_s, OR after facing_timeout_s regardless (so a
person who steps out of view mid-facing can never dead-end the state machine
in an otherwise-unreachable node -- facing's only legal exit is `waving`).

WAVING (fires the EXISTING, safety-gated dashboard gesture over WS)
------------------------------------------------------------------------
Connects as a plain WebSocket client to robot_web_controller.py's own /ws
(config/robot.yaml server.port, default 8080) and sends EXACTLY
`{"type":"cmd","name":"high_wave"}` -- NOT "wave": scripts/gesture_reactor.py's
own GESTURE_TO_ROBOT comment documents that the native LocoClient.WaveHand()
("wave") is a DEAD no-op on this G1; "high_wave" (arm-action 26) is the
working wave. Confirmed by reading robot_web_controller.py: `{"type":"cmd",
"name":<X>}` with X in OPEN_GESTURES (which includes "high_wave") is handled
BEFORE the drive-lock/ownership gate (see the `mtype == "cmd" and
msg.get("name") in OPEN_GESTURES` branch) -- ANY connected client may fire it,
so this node does NOT need to send "hello"/"take_control" first, matching the
plan's "DO NOT reimplement arm control here, only trigger the existing
safety-gated gesture via WebSocket." The WS call runs on a background thread
(it is a blocking round trip) so it can never stall this node's tick timer;
waving moves on to `waiting` once that thread finishes OR after wave_busy_s
(default 3.5 s, matching robot_web_controller._ARM_BUSY_S["high_wave"]) as a
hard cap, so an unreachable dashboard can never wedge the state machine.

WAITING (the documented voice-command extension point, Track D)
---------------------------------------------------------------------
Exits to `resuming` when EITHER the person leaves view (person_fusion reports
no qualifying nearby person for person_lost_grace_s) OR waiting_timeout_s
(default 8 s) elapses, whichever comes first. Also accepts an OPTIONAL
`resume_hook` constructor callback -- a zero-arg callable returning True when a
LATER, not-yet-built voice-command service wants to force an early resume
(e.g. a "resume"/"come_here" command). No voice code exists yet; this is
purely the documented seam. A concrete future wiring would poll
`/dev/shm/g1_voice_cmd.json` (per the plan's Track C/D shm contract) inside
that callback -- deliberately NOT implemented here since the file doesn't
exist yet and this class must stay callable/testable without it.
"""

import json
import os
import threading
import time

import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import Twist

from g1_nav.patrol_logic import PatrolStateMachine

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:                    # pragma: no cover -- degrade gracefully, never crash import
    ws_connect = None


SHM_STATE_PATH = "/dev/shm/g1_patrol_state.json"       # shared writer w/ circle_patrol.py (see docstring)
HANDOFF_PATH = "/dev/shm/g1_patrol_handoff.json"        # written ONLY by this node
PERSON_TRACK_PATH = "/dev/shm/g1_person_track.json"     # written by person_fusion.py
YAML_PATH = "/home/unitree/projects/g1/g1_nav/config/patrol.yaml"

DEFAULTS = {
    "person_debounce_s": 1.0,
    "person_lost_grace_s": 2.0,
    "pause_settle_s": 0.3,
    "close_distance_m": 1.6,          # overridden below from the person_fusion: block if present
    "close_box_frac_h": 0.55,
    "cmd_vel_topic": "/cmd_vel",
    "tick_hz": 10.0,
    "person_track_stale_s": 1.0,
    "yaw_kp": 1.2,
    "yaw_max_rad_s": 0.6,
    "yaw_tolerance_rad": 0.09,
    "yaw_settle_s": 0.3,
    "facing_timeout_s": 6.0,
    "wave_gesture": "high_wave",
    "wave_busy_s": 3.5,
    "waiting_timeout_s": 8.0,
    "ws_url": "ws://127.0.0.1:8080/ws",
    "ws_connect_timeout_s": 2.0,
}


def load_cfg():
    """Load config/patrol.yaml's `patrol_supervisor:` block over the baked
    defaults (never raise -- missing file / bad YAML / missing keys all fall
    back), ALSO pulling `close_distance_m` from the `person_fusion:` block if
    present there -- it is the SAME "person nearby" distance concept, defined
    once under person_fusion in the yaml (single source of that number)."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            sub = data.get("patrol_supervisor")
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if k in cfg and v is not None:
                        cfg[k] = v
            pf = data.get("person_fusion")
            if isinstance(pf, dict) and pf.get("close_distance_m") is not None:
                cfg["close_distance_m"] = pf["close_distance_m"]
    except (OSError, yaml.YAMLError):
        pass
    return cfg


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PatrolSupervisor(Node):
    def __init__(self, resume_hook=None):
        super().__init__("patrol_supervisor")
        self.cfg = load_cfg()
        self.close_distance_m = float(self.cfg["close_distance_m"])
        self.close_box_frac_h = float(self.cfg["close_box_frac_h"])
        self._resume_hook = resume_hook   # Track D extension point -- see _handle_waiting

        # --- pure logic (no ROS) -- see module docstring for the writer-handoff
        # protocol with circle_patrol.py's OWN separate PatrolStateMachine instance.
        self.fsm = PatrolStateMachine(state="circling")
        self._state_enter_t = time.monotonic()

        # --- debounce / timing state (process-local monotonic clock; no
        # cross-process clock-base concerns here, unlike the shm freshness
        # checks below which use file MTIME / time.time() like the rest of
        # this codebase's idiom) ---
        self._close_since = None
        self._target_bearing = None
        self._facing_settle_since = None
        self._facing_lost_since = None
        self._waiting_lost_since = None
        self._wave_fired = False
        self._wave_thread = None

        self._cmd_pub = self.create_publisher(Twist, self.cfg["cmd_vel_topic"], 10)

        tick_hz = float(self.cfg["tick_hz"])
        self.create_timer(1.0 / tick_hz if tick_hz > 0 else 0.1, self._tick)

        self.get_logger().info(
            f"patrol_supervisor: close_distance_m={self.close_distance_m} "
            f"close_box_frac_h={self.close_box_frac_h} wave_gesture={self.cfg['wave_gesture']!r} "
            f"ws_url={self.cfg['ws_url']} cmd_vel_topic={self.cfg['cmd_vel_topic']}"
            + ("" if ws_connect is not None else
               " -- WARNING: 'websockets' package not importable, waving is DISABLED"))

    # ------------------------------------------------------------- shm reads
    @staticmethod
    def _read_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _read_person_track(self):
        try:
            age = time.time() - os.path.getmtime(PERSON_TRACK_PATH)
        except OSError:
            return None
        if age > self.cfg["person_track_stale_s"]:
            return None
        data = self._read_json(PERSON_TRACK_PATH)
        return data if isinstance(data, dict) else None

    def _closest_close_bearing(self, data):
        """Bearing (rad) of the FIRST person qualifying as "close" this frame,
        or None. See module docstring's PERSON-NEARBY TRIGGER section for the
        real-distance-first / box-proxy-fallback priority and the documented
        "first qualifying person wins" simplification."""
        if data is None:
            return None
        persons = data.get("persons")
        if not isinstance(persons, list):
            return None
        frame_h = data.get("frame_h")
        for p in persons:
            if not isinstance(p, dict):
                continue
            bearing = p.get("bearing_rad")
            if bearing is None:
                continue
            dist = p.get("distance_m")
            if dist is not None:
                if float(dist) <= self.close_distance_m:
                    return float(bearing)
                continue
            box = p.get("box")
            if (isinstance(frame_h, (int, float)) and frame_h > 0
                    and isinstance(box, (list, tuple)) and len(box) == 4):
                y1, y2 = float(box[1]), float(box[3])
                if (y2 - y1) / float(frame_h) >= self.close_box_frac_h:
                    return float(bearing)
        return None

    # -------------------------------------------------------------- shm writes
    def _write_phase(self, phase):
        """Write g1_patrol_state.json -- ONLY called while this node is the
        current writer (pausing/facing/waving/waiting; see module docstring's
        COORDINATION section). `moving` is true only during `facing` (the only
        interaction sub-state that actually commands non-zero velocity)."""
        obj = {
            "active": True,
            "phase": phase,
            "moving": phase == "facing",
            "updated_t": round(time.time(), 3),
        }
        tmp = SHM_STATE_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, SHM_STATE_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")

    def _write_handoff(self, active):
        obj = {"supervisor_active": bool(active), "t": round(time.time(), 3)}
        tmp = HANDOFF_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, HANDOFF_PATH)
        except OSError as e:
            self.get_logger().warn(f"handoff write failed: {e}")

    def _publish_cmd_vel(self, vx, vy, vyaw):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self._cmd_pub.publish(msg)

    def _abort_interaction(self):
        """Zero any velocity we were commanding and release the handoff claim
        -- used on manual_override / a defensive fault path. Does NOT touch
        g1_patrol_state.json (whoever wrote "manual_override" there owns it
        right now, not us)."""
        self._publish_cmd_vel(0.0, 0.0, 0.0)
        self._write_handoff(False)
        self._target_bearing = None

    def _enter_state(self, state, now):
        self.fsm.transition(state)
        self._state_enter_t = now

    # ------------------------------------------------------------------ tick
    def _tick(self):
        now = time.monotonic()

        # 1) Manual override always wins, read fresh from whoever owns the
        # file right now (see module docstring's MANUAL OVERRIDE section).
        current = self._read_json(SHM_STATE_PATH)
        ext_phase = current.get("phase") if isinstance(current, dict) else None
        if ext_phase == "manual_override":
            if self.fsm.state != "manual_override":
                self._abort_interaction()
                self.fsm.try_transition("manual_override")
            return
        if self.fsm.state == "manual_override":
            self._enter_state("circling", now)   # override cleared -> fall through below

        # 2) Defensive: if this node's own FSM ever lands in fault/
        # paused_odom_degraded (not currently driven by any logic below --
        # reserved for a future supervisor-side fault path or an external
        # trigger), behave exactly like manual_override: stop and wait.
        if self.fsm.state in ("fault", "paused_odom_degraded"):
            self._abort_interaction()
            return

        state = self.fsm.state
        if state == "circling":
            self._write_handoff(False)   # not intervening -- release/relax any stale claim
            self._handle_circling(now)
        elif state == "pausing":
            self._write_handoff(True)
            self._write_phase("pausing")
            self._handle_pausing(now)
        elif state == "facing":
            self._write_handoff(True)
            self._write_phase("facing")
            self._handle_facing(now)
        elif state == "waving":
            self._write_handoff(True)
            self._write_phase("waving")
            self._handle_waving(now)
        elif state == "waiting":
            self._write_handoff(True)
            self._write_phase("waiting")
            self._handle_waiting(now)
        elif state == "resuming":
            self._handle_resuming(now)   # writes phase itself, then relinquishes

    # ------------------------------------------------------- per-state logic
    def _handle_circling(self, now):
        data = self._read_person_track()
        bearing = self._closest_close_bearing(data)
        if bearing is None:
            self._close_since = None
            return
        if self._close_since is None:
            self._close_since = now
        if now - self._close_since >= self.cfg["person_debounce_s"]:
            self._target_bearing = bearing
            self._close_since = None
            self._facing_settle_since = None
            self._facing_lost_since = None
            self._enter_state("pausing", now)

    def _handle_pausing(self, now):
        """A short settle window before `facing` starts commanding /cmd_vel --
        gives circle_patrol.py one tick to observe the handoff and cancel its
        own goal first (see module docstring's KNOWN, ACCEPTED race note)."""
        if now - self._state_enter_t >= self.cfg["pause_settle_s"]:
            self._enter_state("facing", now)

    def _handle_facing(self, now):
        data = self._read_person_track()
        bearing = self._closest_close_bearing(data)
        if bearing is not None:
            self._target_bearing = bearing
            self._facing_lost_since = None
        elif self._facing_lost_since is None:
            self._facing_lost_since = now

        err = self._target_bearing if self._target_bearing is not None else 0.0
        vyaw = _clamp(self.cfg["yaw_kp"] * err, -self.cfg["yaw_max_rad_s"], self.cfg["yaw_max_rad_s"])
        self._publish_cmd_vel(0.0, 0.0, vyaw)

        converged = abs(err) <= self.cfg["yaw_tolerance_rad"]
        if converged:
            if self._facing_settle_since is None:
                self._facing_settle_since = now
        else:
            self._facing_settle_since = None
        settled = (self._facing_settle_since is not None
                   and now - self._facing_settle_since >= self.cfg["yaw_settle_s"])
        timed_out = now - self._state_enter_t >= self.cfg["facing_timeout_s"]

        if settled or timed_out:
            self._publish_cmd_vel(0.0, 0.0, 0.0)   # confirm stopped before waving
            self._wave_fired = False
            self._wave_thread = None
            self._enter_state("waving", now)

    def _handle_waving(self, now):
        if not self._wave_fired:
            self._wave_fired = True
            self._wave_thread = threading.Thread(
                target=self._send_wave_ws, name="patrol-wave-ws", daemon=True)
            self._wave_thread.start()

        thread_done = self._wave_thread is not None and not self._wave_thread.is_alive()
        timed_out = now - self._state_enter_t >= self.cfg["wave_busy_s"]
        if thread_done or timed_out:
            self._waiting_lost_since = None
            self._enter_state("waiting", now)

    def _send_wave_ws(self):
        """Fire the EXISTING safety-gated dashboard gesture over WS -- see
        module docstring's WAVING section. Runs on a background thread (a
        blocking round trip); any failure is logged and swallowed so an
        unreachable dashboard can never crash or wedge this node."""
        if ws_connect is None:
            self.get_logger().warn(
                "patrol_supervisor: 'websockets' package not available -- cannot fire wave")
            return
        try:
            with ws_connect(self.cfg["ws_url"],
                            open_timeout=self.cfg["ws_connect_timeout_s"]) as ws:
                ws.send(json.dumps({"type": "cmd", "name": self.cfg["wave_gesture"]}))
        except Exception as e:
            self.get_logger().warn(f"patrol_supervisor: wave WS send failed: {e}")

    def _handle_waiting(self, now):
        # Extension point (Track D voice commands, NOT built here): a future
        # voice-command service could inject an early "resume"/"come_here"
        # signal by calling this constructor with resume_hook=<callable
        # returning True/False>, e.g. one that polls a not-yet-existing
        # /dev/shm/g1_voice_cmd.json. Left wired but inert (no voice code
        # exists yet) so this class stays callable/testable without it.
        if self._resume_hook is not None:
            try:
                if self._resume_hook():
                    self._enter_state("resuming", now)
                    return
            except Exception:
                pass   # a broken hook must never crash the supervisor

        data = self._read_person_track()
        bearing = self._closest_close_bearing(data)
        if bearing is None:
            if self._waiting_lost_since is None:
                self._waiting_lost_since = now
        else:
            self._waiting_lost_since = None

        person_left = (self._waiting_lost_since is not None
                       and now - self._waiting_lost_since >= self.cfg["person_lost_grace_s"])
        timed_out = now - self._state_enter_t >= self.cfg["waiting_timeout_s"]
        if person_left or timed_out:
            self._enter_state("resuming", now)

    def _handle_resuming(self, now):
        """Hand writer responsibility back to circle_patrol.py -- see module
        docstring's COORDINATION section. One last phase="circling" write so a
        reader never sees a gap, THEN go fully quiet."""
        self._write_phase("circling")
        self._write_handoff(False)
        self.fsm.transition("circling")
        self._target_bearing = None


def main(args=None):
    rclpy.init(args=args)
    node = PatrolSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._abort_interaction()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
