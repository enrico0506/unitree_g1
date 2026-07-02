"""Web-based teleop controller for the Unitree G1.

Serves a webpage at http://<robot-ip>:8080.

ARCHITECTURE
    Browser ──HTTP──> serves static files from web/
    Browser <─WS────> robot dispatcher  ─> G1 LocoClient   (commands)
                                          H2 LocoClient    (FSM read-back)

MODES (UI-level)
    zero_torque  Motors off; robot collapses if not on gantry.   FSM 0
    damp         Joints relaxed; safe neutral state.              FSM 1
    stand        Robot is standing but not accepting velocity.    FSM 4
    walk         Standing + velocity commands enabled.            FSM 802

USAGE
    pip3 install fastapi 'uvicorn[standard]' 'websockets>=10'
    python3 scripts/robot_web_controller.py
"""

import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_SET_VELOCITY,
    ROBOT_API_ID_LOCO_SET_SWING_HEIGHT,
)
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient as H2LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
try:
    # Physical-remote button state, for attributing un-commanded FSM changes.
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
except Exception:      # keep the dashboard importable if this IDL is unavailable
    WirelessController_ = None

import yaml

from camera_source import CameraSource
from lidar_source import LidarSource, pack_cloud, OdomReader
from map_builder import MapBuilder
from cmd_shaper import CommandShaper
from step_pacer import StepPacer


# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
CONFIG_PATH = BASE_DIR / "config" / "robot.yaml"

# Obstacle feature lives in the top-level 'obstacle' package. Only scripts/ is on
# sys.path when this file runs directly, so add BASE_DIR for the package import.
sys.path.insert(0, str(BASE_DIR))   # so the top-level 'obstacle' package imports
from obstacle.guard import ObstacleGuard
from obstacle.manager import ObstacleManager
from depth_nearfield import DepthNearField   # D435i near-ground depth fusion (domain 0)

OBSTACLE_CFG_PATH = BASE_DIR / "obstacle" / "obstacle.yaml"

# Which backend feeds the ObstacleGuard. Both write /dev/shm/g1_obstacle.json with
# the same contract, so the guard/UI/voice/overlay are reused unchanged -- only the
# spawned process differs. Selected by obstacle.yaml's 'source' key:
#   lidar       -> obstacle/run_obstacle.sh     (raw-LiDAR obstacle node)
#   nav2costmap -> g1_nav/teleop_guard.sh       (Nav2 local-costmap teleop guard)
# Default to 'lidar' so a missing/unknown key keeps the existing behaviour.
try:
    with open(OBSTACLE_CFG_PATH) as _f:
        OBSTACLE_SOURCE = (yaml.safe_load(_f) or {}).get("source", "lidar")
except FileNotFoundError:
    OBSTACLE_SOURCE = "lidar"
if OBSTACLE_SOURCE == "nav2costmap":
    OBSTACLE_RUN_CMD = BASE_DIR / "g1_nav" / "teleop_guard.sh"
else:
    OBSTACLE_RUN_CMD = BASE_DIR / "obstacle" / "run_obstacle.sh"
print(f"Obstacle source: {OBSTACLE_SOURCE} -> {OBSTACLE_RUN_CMD}", flush=True)
CAMERA_SHM = "/dev/shm/g1_camera.jpg"   # frames written here by camera_service.py

# Pose lane (people skeletons). Produced by the separate pose container
# (~/perception/pose/pose_service.py); we only read/write these shm files.
POSE_SHM = "/dev/shm/g1_pose.jpg"          # annotated JPEG (skeletons + labels)
POSE_TRACKS = "/dev/shm/g1_pose_tracks.json"  # [{id, name, cx, cy}] live track list
POSE_LABELS = "/dev/shm/g1_pose_labels.json"  # {"<id>": "name"} operator labels
POSE_DEMAND = "/dev/shm/g1_pose_demand"    # heartbeat: pose only infers while watched

# Detect lane (object detection). Produced by the separate g1-detect container
# (perception/detect/detect_service.py, under this project); we only read/write these shm files.
DETECT_SHM    = "/dev/shm/g1_detect.jpg"          # annotated JPEG (boxes + labels)
DETECT_TRACKS = "/dev/shm/g1_detect_tracks.json"  # {w,h,items:[{cls,conf,box}]} live detections
DETECT_DEMAND = "/dev/shm/g1_detect_demand"       # heartbeat: detect only infers while watched

# Hands lane (finger landmarks). Produced by the separate g1-hands container
# (perception/hands/hands_service.py); we only read/write these shm files. Hands
# ride the SAME Skeleton toggle as pose -- the browser polls both while it's on.
HANDS_TRACKS = "/dev/shm/g1_hands_tracks.json"  # {w,h,items:[{hand,score,landmarks:[[x,y,z]x21]}]}
HANDS_DEMAND = "/dev/shm/g1_hands_demand"        # heartbeat: hands only infer while watched

# Battery: the G1 publishes BMS state (state-of-charge, pack voltage, current) on a
# low-frequency DDS topic as unitree_hg.BmsState_. The hg LowState_ has no battery
# fields, so this is the source. soc is 0-100.
BMS_TOPIC = "rt/lf/bmsstate"


def load_config(path=CONFIG_PATH):
    """Load config/robot.yaml; fall back to built-in defaults if absent."""
    defaults = {
        "network": {"interface": "eth0", "domain_id": 0},
        "server": {"host": "0.0.0.0", "port": 8080},
        "speeds": {"max_vx": 1.5, "max_vy": 1.0, "max_vyaw": 2.0, "slow_scale": 0.4},
        "control": {"send_hz": 30, "watchdog_timeout": 2.0,
                    "move_duration_s": 1.0, "cmd_resend_hz": 3},
        "motion": dict(CommandShaper.DEFAULTS),   # jerk/accel velocity shaping
        "gait": {"swing_height": None, "stand_height": None},
        "camera": {"stream_hz": 25},
        "lidar": {"stream_hz": 10, "max_points": 25000, "camera_height_m": 1.3},
        "map": {"dir": "maps", "voxel_size_m": 0.05, "max_points": 300000,
                "max_range_m": 4.0, "yaw_sign": 1},
        "mapping": {"run_cmd": "/home/unitree/g1_mapping_ws/run_mapping.sh"},
    }
    try:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        for section, vals in loaded.items():
            if section in defaults and isinstance(vals, dict):
                defaults[section].update(vals)
        print(f"Loaded config from {path}", flush=True)
    except FileNotFoundError:
        print(f"Config {path} not found; using defaults", flush=True)
    return defaults


CFG = load_config()

# --- Config (sourced from config/robot.yaml) ---
INTERFACE = CFG["network"]["interface"]
DOMAIN_ID = CFG["network"]["domain_id"]

HOST = CFG["server"]["host"]
PORT = CFG["server"]["port"]

# Velocity caps -- edit in config/robot.yaml. Pushed to the browser on connect.
MAX_VX = CFG["speeds"]["max_vx"]
MAX_VY = CFG["speeds"]["max_vy"]
MAX_VYAW = CFG["speeds"]["max_vyaw"]
SLOW_SCALE = CFG["speeds"]["slow_scale"]

WATCHDOG_TIMEOUT = CFG["control"]["watchdog_timeout"]
SEND_HZ = CFG["control"]["send_hz"]
DT = 1.0 / SEND_HZ
MOVE_DURATION = CFG["control"]["move_duration_s"]
RESEND_PERIOD = 1.0 / CFG["control"]["cmd_resend_hz"]

# Motion shaping (jerk/accel velocity smoothing) + optional SDK gait tuning.
MOTION_CFG = CFG.get("motion", {})
GAIT_CFG = CFG.get("gait", {})
GAIT_SWING_HEIGHT = GAIT_CFG.get("swing_height")   # None = leave controller default
GAIT_STAND_HEIGHT = GAIT_CFG.get("stand_height")   # None = leave controller default

# Discrete-step pacing (scripts/step_pacer.py): chops the held velocity intent into
# short ON-burst / long OFF-settle pulses so the robot takes genuinely SMALL/MEDIUM
# steps in every direction. STEP_ENABLED is the master kill-switch; default_mode is
# 'continuous' (== today's analog walking) until the operator selects a step size.
STEP_CFG = CFG.get("step", {})
STEP_ENABLED = bool(STEP_CFG.get("enabled", True))
# Per-mode swing height (foot lift): Small lowers it to a shuffle so a smaller step
# completes cleanly; Normal/Medium restore this baseline. None = feature off (never
# touch swing height). Clamped to a safe range when applied.
STEP_SWING_NORMAL = STEP_CFG.get("swing_height_normal")
_last_swing = None   # last value sent to SET_SWING_HEIGHT (dedupe the blocking RPC)

FSM_POLL_HZ = 2.0          # GetFsmId is a blocking RPC; keep it light
FSM_BROADCAST_HZ = 2.0

CAMERA_STREAM_HZ = CFG["camera"]["stream_hz"]
LIDAR_STREAM_HZ = CFG["lidar"]["stream_hz"]
LIDAR_MAX_POINTS = CFG["lidar"]["max_points"]
LIDAR_CAMERA_HEIGHT = CFG["lidar"]["camera_height_m"]

MAP_DIR = str((BASE_DIR / CFG["map"]["dir"]).resolve())
MAP_VOXEL = CFG["map"]["voxel_size_m"]
MAP_MAX_POINTS = CFG["map"]["max_points"]
MAP_MAX_RANGE = CFG["map"]["max_range_m"]
MAP_YAW_SIGN = CFG["map"]["yaw_sign"]
MAPPING_RUN_CMD = CFG["mapping"]["run_cmd"]   # on-demand FAST-LIO launch wrapper

# --- Whole-body mode combos (climb/dance), sourced from config/mapping.yaml ---
MAPPING_PATH = BASE_DIR / "config" / "mapping.yaml"


def load_mode_combos(path=MAPPING_PATH):
    """Read mode_combos {name: fsm_id|None} from config/mapping.yaml.

    Whole-body behaviors (climb, dance) we replay with SetFsmId once the id has
    been captured (scripts/capture_combo_fsm.py). Missing/unparseable file -> {}.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        combos = data.get("mode_combos", {}) or {}
        out = {}
        for name, spec in combos.items():
            fsm = spec.get("fsm_id") if isinstance(spec, dict) else None
            out[str(name).lower()] = int(fsm) if fsm is not None else None
        return out
    except (OSError, ValueError, TypeError):
        return {}


MODE_COMBOS = load_mode_combos()   # {"climb": <id|None>, "dance": <id|None>}
# Reverse lookup for the whole-body FSMs (dance/climb). From these the robot
# ONLY accepts a transition back to main_control (802), not ready_stand -- which
# is why Stand/Walk got stuck mid-dance.
MODE_COMBO_FSM_TO_NAME = {fsm: name for name, fsm in MODE_COMBOS.items()
                          if fsm is not None}
MODE_COMBO_FSMS = set(MODE_COMBO_FSM_TO_NAME)

VALID_MODES = {"zero_torque", "damp", "stand", "walk"}
INITIAL_MODE = "damp"

# --- Verified FSM mapping (Enrico's G1, MyBotShop IEA, 2026-05-28) ---
FSM_ZERO_TORQUE = 0
FSM_DAMPING = 1
FSM_READY_STAND = 4
FSM_MAIN_CONTROL = 802

FSM_NAMES = {
    FSM_ZERO_TORQUE:  "zero_torque",
    FSM_DAMPING:      "damping",
    FSM_READY_STAND:  "ready_stand",
    FSM_MAIN_CONTROL: "main_control",
    # Whole-body combo targets (captured 2026-06-23) -- display only; these are
    # NOT in UI_TO_FSM, so they never become a selectable "mode".
    812: "climb",
    503: "dance",
}

UI_TO_FSM = {
    "zero_torque": FSM_ZERO_TORQUE,
    "damp":        FSM_DAMPING,
    "stand":       FSM_READY_STAND,
    "walk":        FSM_MAIN_CONTROL,
}

# Reverse map: robot FSM id -> UI mode. Lets us reconcile the displayed mode to
# the robot's ACTUAL state (read by the FSM poller) so the dashboard reflects
# reality after a page reload or a dashboard restart -- not just the last command.
FSM_TO_UI = {fsm: ui for ui, fsm in UI_TO_FSM.items()}


# --- Arm-action gestures (G1ArmActionClient, service "arm") ---
# Run through the separate "arm" service, NOT LocoClient, so they only animate
# the arms (full action_map catalog in config/mapping.yaml).
ARM_RELEASE_ID = 99    # "release arm" -> return arms to neutral
ARM_HANDS_UP_ID = 15   # "hands up" (palm up); dashboard toggles it with release

# cmd name (from the browser) -> (action_id, auto_release_after_s | None).
# Hold-poses auto-release to neutral so the robot never gets stuck; self-
# completing gestures use None. Only the UI-exposed gestures are kept here;
# hands_up / release_arm are handled specially (toggle) in apply_cmd.
ARM_GESTURES = {
    "clap":      (17, None),   # remote: double-tap A
    "high_wave": (26, None),   # remote: double-tap Y
    "kiss":      (11, None),   # remote: double-tap X ("blow kiss")
    "high_five": (18, 4.0),
    "hug":       (19, 4.0),
    "heart":     (20, 4.0),
}


# --- Shared state ---
class ControlState:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        # Actually-commanded (shaped + guard-governed) velocity, for the UI readout.
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vyaw = 0.0
        self.last_packet_time = 0.0
        self.is_moving = False
        self.pending_cmd = None
        self.mode = INITIAL_MODE
        # Discrete-step pacing mode: 'continuous' (analog, default), 'small', 'medium'.
        self.step_mode = STEP_CFG.get("default_mode", "continuous")
        self.fsm_id = None
        self.fsm_name = "unknown"
        self.transitioning = False
        # True once we've enabled continuous gait in the current walk session
        self.gait_enabled = False
        # True while an arm gesture is holding the arms raised (hands_up toggle)
        self.arm_raised = False
        # Battery (from rt/lf/bmsstate; None until the first BMS message arrives)
        self.battery_soc = None      # % state-of-charge
        self.battery_v = None        # pack volts
        self.battery_current = None  # amps (negative = discharging)


state = ControlState()
client: LocoClient = None
reader: H2LocoClient = None
arm_client: G1ArmActionClient = None   # gesture/arm-action service ("arm")
remote_watcher = None                  # RemoteWatcher: attributes external FSM changes
clients: set = set()

# --- Single-controller lock (multi-device arbitration) ---
# Exactly one connected device ("the controller") may drive the robot at a time;
# all other clients are read-only until they explicitly tap "Take control". This
# stops two devices from stomping the shared state.vx/vy/vyaw at 30 Hz each.
#   client_meta:      WebSocket -> {"id": <str>, "label": <str>}   (per-connection identity)
#   control_owner_id: the client_id that currently holds the lock, or None (free).
client_meta: dict = {}
control_owner_id = None

camera: CameraSource = None
pose: CameraSource = None
detect: CameraSource = None
lidar: LidarSource = None
odom: OdomReader = None
mapper: MapBuilder = None

guard: ObstacleGuard = None
obstacle_mgr: ObstacleManager = None
depth_nf: DepthNearField = None
shaper: CommandShaper = None
pacer: StepPacer = None
audio = None

_mode_lock = threading.Lock()
_arm_lock = threading.Lock()   # serialize arm-action RPCs (and the hands_up toggle)


def clamp(value, low, high):
    return max(low, min(high, value))


def _finite_vel(v, default=0.0):
    """Coerce an incoming velocity to a FINITE float, or `default`. Guards against a
    non-finite value (nan/inf, a "nan" string, None) ever reaching the robot --
    clamp(nan) fails open to the high limit (full speed), so a bad value must die here."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def send_velocity(vx, vy, vyaw, duration):
    """Fire-and-forget velocity command -- does NOT wait for a robot reply.

    Velocity is sent via the no-reply RPC path so the command thread never blocks
    on a slow robot response (a blocking Move() at high rate freezes the command
    loop -> movement stalls AND the shared DDS bus starves the camera RPC, which
    is why driving made the camera stutter and movement intermittently stop)."""
    p = json.dumps({"velocity": [vx, vy, vyaw], "duration": duration})
    client._CallNoReply(ROBOT_API_ID_LOCO_SET_VELOCITY, p)


def fsm_name(fsm_id):
    if fsm_id is None:
        return "unknown"
    return FSM_NAMES.get(fsm_id, f"fsm_{fsm_id}")


# ---------------------------------------------------------------------------
# FSM read-back (uses H2 client against G1)
# ---------------------------------------------------------------------------

def call_with_timeout(fn, timeout):
    result = {"value": None, "exc": None}

    def runner():
        try:
            result["value"] = fn()
        except Exception as e:
            result["exc"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or result["exc"] is not None:
        return None
    return result["value"]


# ---------------------------------------------------------------------------
# FSM change attribution: tell OUR commands apart from EXTERNAL ones (physical
# remote or the robot's onboard firmware), so a spontaneous mode flip -- e.g. the
# G1's onboard obstacle/climb behaviour switching 802 -> 812 -> 802 on its own --
# is flagged in the log with a best-effort source instead of looking commanded.
# ---------------------------------------------------------------------------

# Remote button bit -> name (matches config/mapping.yaml + capture_combo_fsm.py).
REMOTE_BIT_NAMES = {
    0: "R1", 1: "L1", 2: "start", 3: "select", 4: "R2", 5: "L2",
    6: "F1", 7: "F2", 8: "A", 9: "B", 10: "X", 11: "Y",
    12: "up", 13: "left", 14: "down", 15: "right",
}


def _remote_combo_name(mask):
    names = [REMOTE_BIT_NAMES.get(b, f"bit{b}") for b in range(16) if mask & (1 << b)]
    return "+".join(names) if names else "-"


# FSM ids WE asked the robot to enter -> the monotonic time we asked. A transition
# to an id we requested within the window (or while a mode transition is running) is
# "ours"; anything else is external.
_fsm_intents = {}
_FSM_INTENT_WINDOW_S = 20.0


def note_fsm_intent(fsm_id):
    """Record that the dashboard just commanded fsm_id, so fsm_poll_loop can tell a
    commanded transition from an external (remote / onboard-firmware) one."""
    try:
        _fsm_intents[int(fsm_id)] = time.monotonic()
    except (TypeError, ValueError):
        pass


def fsm_change_is_ours(fsm_id):
    """True if this observed transition was (very likely) commanded by the dashboard."""
    if state.transitioning:                     # inside a run_enter_mode sequence
        return True
    t = _fsm_intents.get(fsm_id)
    return t is not None and (time.monotonic() - t) <= _FSM_INTENT_WINDOW_S


class RemoteWatcher:
    """Caches the physical remote's latest button state (rt/wirelesscontroller) so an
    un-commanded FSM change can be attributed to a remote press -- or, when the remote
    is silent, cleared of it (pointing at the robot's onboard firmware)."""

    def __init__(self):
        self._sub = None
        self._last_keys = 0
        self._press_t = 0.0        # monotonic of the last NON-zero key combo
        self._press_names = "-"
        self._msg_t = 0.0          # monotonic of the last message (is the remote alive?)

    def start(self):
        if WirelessController_ is None:
            print("[REMOTE] WirelessController_ IDL missing; remote attribution off",
                  flush=True)
            return
        try:
            self._sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
            self._sub.Init(self._on_msg, 10)
            print("[REMOTE] watching rt/wirelesscontroller for FSM attribution",
                  flush=True)
        except Exception as e:
            print(f"[REMOTE] subscribe failed ({e}); remote attribution off", flush=True)

    def _on_msg(self, msg):
        # DDS thread: a couple of non-blocking field copies.
        self._msg_t = time.monotonic()
        keys = int(getattr(msg, "keys", 0) or 0)
        if keys:
            self._last_keys = keys
            self._press_t = time.monotonic()
            self._press_names = _remote_combo_name(keys)

    def attribution(self, window_s=2.5):
        """One-line source hint for a just-observed EXTERNAL fsm change."""
        now = time.monotonic()
        if self._press_t > 0 and (now - self._press_t) <= window_s:
            return (f"PHYSICAL REMOTE {self._press_names} "
                    f"(0x{self._last_keys:04x}, {now - self._press_t:.2f}s ago)")
        alive = self._msg_t > 0 and (now - self._msg_t) < 2.0
        if not alive:
            return ("onboard firmware -- remote is OFF/silent AND the dashboard issued "
                    "no command (e.g. the robot's built-in obstacle/climb behaviour)")
        return ("onboard firmware/other -- remote connected but no button pressed AND "
                "the dashboard issued no command")


def fsm_poll_loop():
    period = 1.0 / FSM_POLL_HZ
    while True:
        result = call_with_timeout(reader.GetFsmId, 1.0)
        if result is not None:
            code, fsm = result
            if code == 0:
                if fsm != state.fsm_id:
                    old = state.fsm_id
                    if old is None or fsm_change_is_ours(fsm):
                        # old is None = the first read after startup (state sync), not a
                        # real transition -> plain [FSM], never flagged external.
                        print(f"[FSM] {old} ({fsm_name(old)}) "
                              f"-> {fsm} ({fsm_name(fsm)})", flush=True)
                    else:
                        # No dashboard command matches this transition -> it came from
                        # the physical remote or the robot's onboard firmware. Attribute
                        # the source so a spontaneous flip is obvious in the log.
                        src = (remote_watcher.attribution()
                               if remote_watcher is not None else "unknown source")
                        print(f"[FSM-EXTERNAL] {old} ({fsm_name(old)}) "
                              f"-> {fsm} ({fsm_name(fsm)}) | source: {src}", flush=True)
                    # If we left main_control, we lost continuous gait state
                    if fsm != FSM_MAIN_CONTROL:
                        state.gait_enabled = False
                state.fsm_id = fsm
                state.fsm_name = fsm_name(fsm)

                # Reflect the robot's ACTUAL mode in the UI selection, so a page
                # reload or a dashboard restart shows reality -- not the last
                # commanded mode (which defaults to "damp"). Skip while a commanded
                # transition owns state.mode; and keep "walk" intent when the robot
                # has merely dropped to ready_stand, since command_loop's drive-time
                # auto-rearm steps it back up (downgrading to "stand" here would
                # disable the drive buttons and defeat that recovery).
                if not state.transitioning:
                    ui = FSM_TO_UI.get(fsm)
                    if (ui is not None and ui != state.mode
                            and not (state.mode == "walk"
                                     and fsm == FSM_READY_STAND)):
                        print(f"[MODE] sync '{state.mode}' -> '{ui}' "
                              f"from robot FSM {fsm}", flush=True)
                        state.mode = ui
        time.sleep(period)


# ---------------------------------------------------------------------------
# Locomotion mode setup
# ---------------------------------------------------------------------------

def enable_continuous_gait():
    """Tell the locomotion controller to step continuously and accept
    full 2D + rotational velocity. This is what enables strafe and yaw."""
    if state.gait_enabled:
        return
    print("[GAIT] enabling continuous gait (SetBalanceMode 1)", flush=True)
    try:
        # SetBalanceMode(1) = continuous gait (walk-able); 0 = static stand-only.
        # (BalanceStand() on the G1 SDK just calls SetBalanceMode, so this is all
        # that's needed -- and it takes a required arg, which is why the old
        # bare BalanceStand() call errored.)
        ret = client.SetBalanceMode(1)
        print(f"[GAIT] SetBalanceMode(1) returned {ret}", flush=True)
    except Exception as e:
        print(f"[GAIT] SetBalanceMode failed: {e}", flush=True)

    # Optional SDK gait tuning (both default null in config -> skipped). UNVERIFIED
    # ranges on this robot; only set when the operator opts in via config/robot.yaml.
    if GAIT_STAND_HEIGHT is not None:
        try:
            ret = client.SetStandHeight(float(GAIT_STAND_HEIGHT))
            print(f"[GAIT] SetStandHeight({GAIT_STAND_HEIGHT}) returned {ret}", flush=True)
        except Exception as e:
            print(f"[GAIT] SetStandHeight failed: {e}", flush=True)
    if GAIT_SWING_HEIGHT is not None:
        try:
            # No wrapper on the G1 LocoClient -> call the raw api id. Same {"data": v}
            # payload shape every other SET_* call uses (SetBalanceMode/SetStandHeight).
            p = json.dumps({"data": float(GAIT_SWING_HEIGHT)})
            ret = client._Call(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT, p)
            print(f"[GAIT] SetSwingHeight({GAIT_SWING_HEIGHT}) returned {ret}", flush=True)
        except Exception as e:
            print(f"[GAIT] SetSwingHeight failed: {e}", flush=True)

    state.gait_enabled = True
    # Fresh walk session: re-assert the swing height for the active step mode (the
    # controller may have reset it on exit), forcing past the dedupe.
    apply_step_swing(state.step_mode, force=True)


def apply_step_swing(mode, force=False):
    """Set the locomotion swing height (foot lift) for the active step mode.

    A LOWER swing = a shuffle = less weight transfer per step, so the robot can complete
    a SMALLER clean step than the velocity floor allows. Small's swing_height overrides;
    Normal/Medium restore swing_height_normal. No-op unless swing_height_normal is set.
    Runs the blocking SET_SWING_HEIGHT RPC on a throwaway thread (never blocks the loop)
    and dedupes so it only fires on an actual change. Clamped to a safe range -- the SDK
    range is UNVERIFIED on this robot and too low = toe-scuff/trip."""
    global _last_swing
    if pacer is None or STEP_SWING_NORMAL is None:
        return
    val = pacer.swing_height.get(mode)        # small/medium per-mode override
    if val is None:
        val = STEP_SWING_NORMAL               # Normal (or a null-swing mode) -> baseline
    try:
        val = clamp(float(val), 0.02, 0.15)
    except (TypeError, ValueError):
        return
    if not force and _last_swing is not None and abs(val - _last_swing) < 1e-4:
        return                                 # already at this height -> skip the RPC
    _last_swing = val

    def _worker(v=val, m=mode):
        # Verbose so we can tell on the robot whether SET_SWING_HEIGHT is supported:
        # "calling" then "-> (code,data)" = works; "calling" with nothing after = the
        # RPC hung (unsupported on this firmware); no "calling" = returned early.
        print(f"[STEP] swing height: calling SET_SWING_HEIGHT({v:.3f}) for '{m}'...", flush=True)
        try:
            ret = client._Call(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT, json.dumps({"data": v}))
            print(f"[STEP] swing height SET {v:.3f} m '{m}' -> {ret}", flush=True)
        except Exception as e:
            print(f"[STEP] swing height FAILED: {e}", flush=True)

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Mode commands
# ---------------------------------------------------------------------------

def wait_for_fsm(target_fsm, timeout=15.0, poll_period=0.1):
    note_fsm_intent(target_fsm)     # a dashboard-commanded transition (not external)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if state.fsm_id == target_fsm:
            return True
        time.sleep(poll_period)
    return False


def enter_mode(new_mode):
    new_mode = (new_mode or "").lower()
    if new_mode not in VALID_MODES:
        print(f"[mode error: unknown '{new_mode}']", flush=True)
        return

    try:
        client.StopMove()
    except Exception:
        pass
    state.is_moving = False
    state.vx = state.vy = state.vyaw = 0.0
    state.gait_enabled = False
    if shaper is not None:
        shaper.reset()   # fresh walk session starts the ramp from rest, never a stale value
    if pacer is not None:
        pacer.reset()    # never resume mid-pulse on a fresh walk session

    # Set transitioning BEFORE mode so the FSM poller's mode-sync can't briefly
    # overwrite our just-set intent in the gap between these two assignments.
    state.transitioning = True
    state.mode = new_mode
    print(f"[MODE] requested {new_mode}, current FSM={state.fsm_id}",
          flush=True)

    try:
        ok = False

        # Leaving a routine (dance 503 / climb 812) only works back to
        # main_control (802), never ready_stand -- so exit to 802 first, then let
        # the normal handling below take over (walk = done; stand goes on to damp).
        if state.fsm_id in MODE_COMBO_FSMS and new_mode in ("stand", "walk"):
            routine = MODE_COMBO_FSM_TO_NAME.get(state.fsm_id, "routine")
            print(f"[MODE] leaving {routine} (FSM {state.fsm_id}) -> main_control",
                  flush=True)
            client.SetFsmId(FSM_MAIN_CONTROL)
            if not wait_for_fsm(FSM_MAIN_CONTROL, timeout=15.0):
                print(f"[MODE] {routine} exit failed, FSM stuck at "
                      f"{state.fsm_id}", flush=True)
                state.mode = routine   # reflect reality; don't pretend we're walking
                return

        if new_mode == "zero_torque":
            client.ZeroTorque()
            ok = wait_for_fsm(FSM_ZERO_TORQUE, timeout=5.0)

        elif new_mode == "damp":
            client.Damp()
            ok = wait_for_fsm(FSM_DAMPING, timeout=5.0)

        elif new_mode == "stand":
            # main_control -> ready_stand drops the robot to Damp, so Damp FIRST
            # then ready_stand. (Reaching ready_stand needs the harness -- expected
            # on this robot, and only Stand needs it.)
            if state.fsm_id == FSM_MAIN_CONTROL:
                client.Damp()
                wait_for_fsm(FSM_DAMPING, timeout=5.0)
            client.SetFsmId(FSM_READY_STAND)
            ok = wait_for_fsm(FSM_READY_STAND, timeout=15.0)

        elif new_mode == "walk":
            if state.fsm_id == FSM_MAIN_CONTROL:
                # Already walk-capable (a dance/climb routine just returned here,
                # or walk was re-pressed). Just (re)assert stepping -- do NOT
                # damp / re-sequence, which would needlessly drop the robot down
                # through the unstable states and was the old fall hazard.
                enable_continuous_gait()
                ok = True
            else:
                # Coming from damp/ready_stand: reach ready_stand, settle, then
                # step up to main_control. Walk never routes through Damp, so it
                # stays harness-free (unlike Stand).
                if state.fsm_id != FSM_READY_STAND:
                    client.SetFsmId(FSM_READY_STAND)
                    if not wait_for_fsm(FSM_READY_STAND, timeout=15.0):
                        print(f"[MODE] walk: ready_stand failed, "
                              f"FSM stuck at {state.fsm_id}", flush=True)
                        return

                # Let the stand SETTLE before asking for main_control. Issuing it
                # the instant ready_stand is reached is exactly the transition that
                # silently failed and left the robot stuck at "stand"; a short
                # settle makes 802 take on the first try.
                time.sleep(1.0)

                ret1 = client.SetFsmId(FSM_MAIN_CONTROL)
                print(f"[MODE] SetFsmId({FSM_MAIN_CONTROL}) #1 returned {ret1}",
                      flush=True)

                if wait_for_fsm(FSM_MAIN_CONTROL, timeout=3.0):
                    ok = True
                else:
                    print("[MODE] walk: first attempt didn't transition, "
                          "priming via ready_stand and retrying...", flush=True)
                    client.SetFsmId(FSM_READY_STAND)
                    wait_for_fsm(FSM_READY_STAND, timeout=10.0)
                    time.sleep(1.0)

                    ret2 = client.SetFsmId(FSM_MAIN_CONTROL)
                    print(f"[MODE] SetFsmId({FSM_MAIN_CONTROL}) #2 "
                          f"returned {ret2}", flush=True)
                    ok = wait_for_fsm(FSM_MAIN_CONTROL, timeout=15.0)

                # If we reached main_control, enable continuous gait so that
                # strafe and yaw actually engage stepping (not just leaning).
                if ok:
                    time.sleep(0.5)
                    enable_continuous_gait()

        if ok:
            pass   # walk reached; speed is governed by the velocity we send
        else:
            print(f"[MODE] {new_mode} TIMEOUT, FSM stuck at {state.fsm_id}",
                  flush=True)
    finally:
        state.transitioning = False


def _arm_execute(action_id):
    """Send one arm-action RPC. No-op (logged) if the arm service isn't up."""
    if arm_client is None:
        print(f"[CMD] arm action {action_id} skipped -- arm client not ready",
              flush=True)
        return
    try:
        arm_client.ExecuteAction(action_id)
    except Exception as e:
        print(f"[arm action error: {e}]", flush=True)


def apply_cmd(name):
    name = (name or "").lower()
    if name == "low_stand":
        client.LowStand()
        print("[CMD] low_stand", flush=True)
    elif name == "high_stand":
        client.HighStand()
        print("[CMD] high_stand", flush=True)
    elif name == "wave":
        client.WaveHand()
        print("[CMD] wave", flush=True)
    elif name == "shake":
        client.ShakeHand()
        print("[CMD] shake", flush=True)

    # --- Arm gestures (separate "arm" service) ---
    elif name == "hands_up":
        # Toggle: first press raises the arms (palm up), next press lowers them.
        with _arm_lock:
            if state.arm_raised:
                _arm_execute(ARM_RELEASE_ID)
                state.arm_raised = False
                print("[CMD] hands_up -> lower", flush=True)
            else:
                _arm_execute(ARM_HANDS_UP_ID)
                state.arm_raised = True
                print("[CMD] hands_up -> raise", flush=True)
    elif name == "release_arm":
        with _arm_lock:
            _arm_execute(ARM_RELEASE_ID)
            state.arm_raised = False
        print("[CMD] release_arm", flush=True)
    elif name in ARM_GESTURES:
        action_id, auto_release = ARM_GESTURES[name]
        with _arm_lock:
            _arm_execute(action_id)
        print(f"[CMD] gesture {name} (arm action {action_id})", flush=True)
        if auto_release:
            # This runs in apply_cmd's own daemon thread, so the sleep never
            # stalls the command/velocity loop.
            time.sleep(auto_release)
            with _arm_lock:
                _arm_execute(ARM_RELEASE_ID)
                state.arm_raised = False

    # --- Whole-body mode combos (climb / dance) ---
    elif name in MODE_COMBOS:
        fsm = MODE_COMBOS[name]
        if fsm is None:
            print(f"[CMD] {name}: fsm_id not captured -- run "
                  f"scripts/capture_combo_fsm.py, then set mode_combos.{name}."
                  f"fsm_id in config/mapping.yaml", flush=True)
        elif not _mode_lock.acquire(blocking=False):
            # Serialize with enter_mode (same lock): a combo must not race a mode
            # transition's state.mode write, or command_loop could read mode=="walk"
            # while the robot FSM is a combo and drive velocity into it.
            print(f"[CMD] {name} ignored -- mode transition in progress", flush=True)
        else:
            try:
                # Hand the whole body to this behavior: stop driving so the velocity
                # keepalive can't fight the routine. mode != "walk" makes the command
                # loop go quiet; the FSM poller leaves it (no FSM_TO_UI entry).
                state.vx = state.vy = state.vyaw = 0.0
                state.is_moving = False
                state.gait_enabled = False
                state.mode = name
                print(f"[CMD] {name} -> SetFsmId({fsm})", flush=True)
                note_fsm_intent(fsm)     # dashboard-commanded combo (not external)
                try:
                    client.SetFsmId(fsm)
                except Exception as e:
                    print(f"[mode combo error: {e}]", flush=True)
            finally:
                _mode_lock.release()

    else:
        print(f"[cmd error: unknown '{name}']", flush=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def broadcast(msg):
    if not clients:
        return
    text = json.dumps(msg)
    dead = []
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


# ---------------------------------------------------------------------------
# Single-controller lock helpers
# ---------------------------------------------------------------------------

def make_control_msg():
    """Snapshot of who holds control, sent to every client so each device can tell
    whether it may drive (owner_id === its own client_id) and show the holder."""
    owner_label = None
    roster = []
    for meta in client_meta.values():
        cid = meta.get("id")
        is_owner = cid is not None and cid == control_owner_id
        roster.append({"id": cid, "label": meta.get("label"), "is_owner": is_owner})
        if is_owner:
            owner_label = meta.get("label")
    return {"type": "control", "owner_id": control_owner_id,
            "owner_label": owner_label, "clients": roster}


async def _set_owner(new_id):
    """Transfer the single-controller lock to `new_id`, then tell every client.

    SAFETY: zero the velocity + reset the shaper/pacer on every handoff so the new
    owner starts from a full stop and the previous owner's latched velocity (or a
    mid-ramp/mid-pulse) can never carry over into someone else's session."""
    global control_owner_id
    control_owner_id = new_id
    state.vx = state.vy = state.vyaw = 0.0
    state.is_moving = False
    if shaper is not None:
        shaper.reset()
    if pacer is not None:
        pacer.reset()
    await broadcast(make_control_msg())


def make_state_msg():
    return {
        "type": "fsm_state",
        "fsm_id": state.fsm_id,
        "fsm_name": state.fsm_name,
        "ui_mode": state.mode,
        "transitioning": state.transitioning,
    }


def make_telemetry_msg():
    x, y, yaw = odom.get_pose() if odom else (0.0, 0.0, 0.0)
    msg = {
        "type": "telemetry",
        "x": round(x, 3),
        "y": round(y, 3),
        "yaw_deg": round(math.degrees(yaw), 1),
        "odom_live": bool(odom and odom.is_live()),
        "battery_soc": state.battery_soc,
        "battery_v": state.battery_v,
        "battery_current": state.battery_current,
        # Actually-commanded velocity after shaping + guard (UI shows real speed).
        "cmd_vx": round(state.cmd_vx, 3),
        "cmd_vy": round(state.cmd_vy, 3),
        "cmd_vyaw": round(state.cmd_vyaw, 3),
    }
    # Step-pacing state (authoritative server mode + optional measured/estimated chip)
    # so the dashboard's Step Size selector reflects reality.
    if pacer is not None:
        msg.update(pacer.telemetry())
    return msg


def run_enter_mode(name):
    """Run enter_mode under the mode lock (skips if a transition is already running)."""
    if not _mode_lock.acquire(blocking=False):
        print(f"[MODE] ignored '{name}' -- transition in progress", flush=True)
        return
    try:
        enter_mode(name)
    finally:
        _mode_lock.release()


def command_loop():
    """Send blocking robot RPCs (Move/StopMove/gestures) in a DEDICATED THREAD.

    client.Move() is a blocking DDS RPC that waits for the robot's reply. It must
    NOT run on the asyncio event loop -- doing so freezes the camera MJPEG stream
    and WebSockets every time it's called (i.e. continuously while walking, which
    is exactly when the camera "lost signal"). Running it here keeps the event
    loop free for streaming.
    """
    DEADZONE = 0.01            # m/s; below this a change isn't worth an RPC (small so
                              # the shaper's smooth ramp tail still reaches the robot)
    last_sent = (0.0, 0.0, 0.0)
    last_send_time = 0.0
    last_rearm_time = 0.0
    last_gait_time = 0.0
    last_desired = (0.0, 0.0, 0.0)   # previous tick's shaped velocity -> pacer settle gate

    while True:
        now = time.time()

        # One-shot gesture command -- run in a thread so a blocking gesture RPC
        # (wave/shake) can't stall the velocity loop.
        if state.pending_cmd is not None:
            cmd = state.pending_cmd
            state.pending_cmd = None
            threading.Thread(target=apply_cmd, args=(cmd,), daemon=True).start()

        # Watchdog: browser went silent -> force stop. Comms loss is a safety event,
        # so reset the shaper (instant stop) rather than easing down a stale ramp.
        idle = now - state.last_packet_time
        if (idle > WATCHDOG_TIMEOUT and state.last_packet_time > 0
                and (state.is_moving
                     or (state.vx, state.vy, state.vyaw) != (0.0, 0.0, 0.0))):
            print(f"[WATCHDOG] {idle:.2f}s idle -> stop", flush=True)
            state.vx = state.vy = state.vyaw = 0.0
            if shaper is not None:
                shaper.reset()
            if pacer is not None:
                pacer.reset()   # comms loss: never resume mid-pulse on the next walk

        # Desired velocity (zero unless actively walking).
        #   operator intent (clamped)
        #     -> shaper.normalize()   diagonal -> speed ellipse
        #     -> guard.apply()        obstacle scaling + emergency hard stop (safety)
        #     -> shaper.shape(bypass) jerk/accel-limit the FINAL command, smoothing
        #                             BOTH the operator input AND the guard's gentle
        #                             slow-zone scale-down -- but BYPASS (snap) on any
        #                             axis the guard hard-stopped, so a safety stop
        #                             stays instantaneous. The snap also rebases the
        #                             ramp, so a cleared obstacle re-accelerates from
        #                             rest instead of lurching.
        in_walk = state.mode == "walk" and not state.transitioning
        if in_walk:
            intent = (clamp(state.vx, -MAX_VX, MAX_VX),
                      clamp(state.vy, -MAX_VY, MAX_VY),
                      clamp(state.vyaw, -MAX_VYAW, MAX_VYAW))
            if shaper is not None:
                intent = shaper.normalize(intent)
            held = intent     # operator's true held intent, BEFORE the pacer pulses it
            # Discrete-step pacing: chop the held intent into ON-burst/OFF-settle
            # pulses (small/medium steps). Pure pass-through in 'continuous' mode, so
            # this is a no-op unless the operator selected a step size. UPSTREAM of the
            # guard so the obstacle hard-stop still wins; it never sets the shaper
            # bypass, so each pulse edge stays jerk/accel-limited downstream. Fed a
            # MONOTONIC clock (a wall-clock step must never wedge the pulse gate) and the
            # PREVIOUS tick's shaped velocity so the OFF window holds until the foot
            # actually settles (no merged steps).
            if pacer is not None:
                intent = pacer.modulate(intent, time.monotonic(), vel_fb=last_desired)
            # held -> guard so the blind-zone predictor reads the operator's real forward
            # intent, not the pulsed value (an OFF window is not a forward release).
            governed = guard.apply(intent, held=held) if guard is not None else intent
            hard = guard.hard_stop_flags() if guard is not None else (False, False, False)
            desired = (shaper.shape(governed, DT, bypass=hard)
                       if shaper is not None else governed)
            # FINAL clamp: the jerk-limited shaper can momentarily overshoot the target
            # by a few cm/s; never send a velocity beyond the configured caps.
            desired = (clamp(desired[0], -MAX_VX, MAX_VX),
                       clamp(desired[1], -MAX_VY, MAX_VY),
                       clamp(desired[2], -MAX_VYAW, MAX_VYAW))
        else:
            # Not walking: commanded zero. Keep the guard state machine ticking
            # (FAULT / startup grace) and reset the shaper so the next walk ramps
            # cleanly from rest.
            if guard is not None:
                guard.apply((0.0, 0.0, 0.0))
            if shaper is not None:
                shaper.reset()
            desired = (0.0, 0.0, 0.0)

        state.cmd_vx, state.cmd_vy, state.cmd_vyaw = desired   # for the UI readout
        last_desired = desired   # feeds the pacer's settle gate on the next tick

        moving = desired != (0.0, 0.0, 0.0)
        changed = any(abs(desired[i] - last_sent[i]) > DEADZONE for i in range(3))

        # Send only on CHANGE or as a low-rate refresh -- these are blocking RPCs;
        # flooding them at loop rate saturates the DDS layer (kills movement+camera).
        if in_walk and (changed or (now - last_send_time) >= RESEND_PERIOD):
            try:
                if (moving and state.fsm_id == FSM_READY_STAND
                        and (now - last_rearm_time) > 5.0):
                    # Robot dropped out of walking (idle timeout / safety) but the user
                    # is driving -> auto re-arm (in a thread) so it ALWAYS moves.
                    last_rearm_time = now
                    print("[RECOVER] FSM=ready_stand while driving -> re-arming walk",
                          flush=True)
                    threading.Thread(target=run_enter_mode, args=("walk",),
                                     daemon=True).start()
                else:
                    if moving and not state.gait_enabled and (now - last_gait_time) > 3.0:
                        # re-assert stepping off-thread (blocking RPC) so we don't stall
                        last_gait_time = now
                        threading.Thread(target=enable_continuous_gait, daemon=True).start()
                    # Non-blocking velocity. Zero = Move(0,0,0) keepalive (keeps the gait
                    # hot; NEVER StopMove -> that drops gait and makes the next command
                    # only lean / not respond until walk is re-entered).
                    send_velocity(desired[0], desired[1], desired[2], MOVE_DURATION)
                    state.is_moving = moving
                    last_sent = desired
                    last_send_time = now
            except Exception as e:
                print(f"[move error: {e}]", flush=True)

        time.sleep(DT)


async def broadcast_loop():
    """Async, non-blocking: just pushes FSM + telemetry to clients."""
    last_broadcast = 0.0
    last_broadcast_fsm = None
    last_transitioning = None
    broadcast_period = 1.0 / FSM_BROADCAST_HZ

    while True:
        now = time.time()
        should_broadcast = (
            state.fsm_id != last_broadcast_fsm
            or state.transitioning != last_transitioning
            or (now - last_broadcast) >= broadcast_period
        )
        if should_broadcast:
            await broadcast(make_state_msg())
            await broadcast(make_telemetry_msg())
            last_broadcast = now
            last_broadcast_fsm = state.fsm_id
            last_transitioning = state.transitioning

        # Stream obstacle telemetry at ~10 Hz whenever the perception node is alive
        # (so the 2D/3D views are live even with the motion guard OFF), or while the
        # guard is otherwise active (fault/auto-disabled banner).
        if guard is not None and (
                (obstacle_mgr is not None and obstacle_mgr.is_alive())
                or guard.is_active_ui()):
            tm = guard.telemetry()
            if depth_nf is not None:
                tm["depth"] = depth_nf.telemetry()
            await broadcast(tm)

        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, reader, arm_client, camera, pose, detect, lidar, odom, mapper, depth_nf
    global guard, obstacle_mgr, shaper, pacer, audio, remote_watcher
    print(f"Initializing DDS on interface: {INTERFACE}", flush=True)
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)

    client = LocoClient()
    client.SetTimeout(1.0)   # short: a slow reply can't stall the bus for 10s
    client.Init()
    print("G1 LocoClient ready (commands).", flush=True)

    reader = H2LocoClient()
    reader.SetTimeout(1.0)
    reader.Init()
    print("H2 LocoClient ready (FSM read-back).", flush=True)

    # Arm-action service ("arm") drives the gesture buttons; independent of
    # locomotion. If it isn't running, arm_client stays None so the dashboard
    # still starts and gestures just no-op.
    try:
        arm_client = G1ArmActionClient()
        arm_client.SetTimeout(3.0)
        arm_client.Init()
        print("G1 ArmActionClient ready (gestures).", flush=True)
    except Exception as e:
        arm_client = None
        print(f"ArmActionClient init failed ({e}); gestures disabled.", flush=True)

    # Remote watcher first, so fsm_poll_loop can attribute an external FSM change to
    # the physical remote vs the robot's onboard firmware from the very first flip.
    remote_watcher = RemoteWatcher()
    remote_watcher.start()

    fsm_thread = threading.Thread(target=fsm_poll_loop, daemon=True)
    fsm_thread.start()
    print("FSM poller started.", flush=True)

    # Battery: subscribe to the BMS topic and stash SOC/voltage/current on `state`.
    # The callback runs on the DDS thread (just a few field copies; non-blocking).
    def _on_bms(msg):
        try:
            state.battery_soc = int(msg.soc)
            cells = [v for v in msg.bmsvoltage if v]          # mV per pack, 0 = absent
            state.battery_v = round(max(cells) / 1000.0, 1) if cells else None
            state.battery_current = round(msg.current / 1000.0, 2)  # mA -> A
        except Exception:
            pass
    bms_sub = ChannelSubscriber(BMS_TOPIC, BmsState_)
    bms_sub.Init(_on_bms, 10)
    print(f"BMS subscriber started ({BMS_TOPIC}).", flush=True)

    # Camera runs in its OWN process (own DDS participant + RPC), so a busy/blocked
    # locomotion RPC can't starve it. It writes frames to shared memory; we read them.
    cam_proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "scripts" / "camera_service.py"),
         INTERFACE, str(CAMERA_STREAM_HZ), CAMERA_SHM])
    camera = CameraSource(path=CAMERA_SHM)
    print("Camera service (separate process) started.", flush=True)

    # Pose lane reader. The pose container (g1-pose.service) is OPTIONAL and runs
    # independently; if it's down, pose.is_live() is False and the skeleton view
    # just shows "no signal" -- the raw camera feed is unaffected.
    pose = CameraSource(path=POSE_SHM)
    pose.backend = "pose"

    # Detect lane reader. The g1-detect container is OPTIONAL and runs independently;
    # if it's down, detect.is_live() is False and the object-detection view just shows
    # "no signal" -- the raw camera feed is unaffected.
    detect = CameraSource(path=DETECT_SHM)
    detect.backend = "detect"

    odom = OdomReader()
    odom.start()
    mapper = MapBuilder(MAP_DIR, run_cmd=MAPPING_RUN_CMD, max_points=MAP_MAX_POINTS)

    lidar = LidarSource(max_points=LIDAR_MAX_POINTS, mount_height=LIDAR_CAMERA_HEIGHT,
                        odom=odom, mapper=mapper)
    lidar.start()

    # Obstacle feature: AudioClient for TTS warnings (optional), the perception-node
    # manager (subprocess, started on toggle), and the guard that scales / hard-stops
    # the commanded velocity. Guard starts disabled -- pure pass-through until toggled.
    try:
        audio = AudioClient(); audio.SetTimeout(3.0); audio.Init()
        print("G1 AudioClient ready (TTS).", flush=True)
    except Exception as e:
        audio = None; print(f"AudioClient init failed ({e}); voice disabled.", flush=True)
    obstacle_mgr = ObstacleManager(run_cmd=str(OBSTACLE_RUN_CMD))
    guard = ObstacleGuard(cfg_path=str(OBSTACLE_CFG_PATH), audio=audio, limits=(MAX_VX, MAX_VY, MAX_VYAW))
    guard.start()
    print("Obstacle guard ready (disabled until toggled).", flush=True)

    # Obstacle PERCEPTION runs always (independent of the motion guard) so the 2D/3D
    # views are live from boot. The guard's enable flag governs MOTION only; the node
    # is paused solely for mapping (shared Mid-360) and resumed afterwards.
    if obstacle_mgr is not None and not (mapper is not None and mapper.active):
        obstacle_mgr.start()
        print("Obstacle perception node started (viz always-on).", flush=True)

    # D435i near-ground depth fusion: reuses the dashboard's RealSense cloud (lidar)
    # to fill the Mid-360's near-ground forward blind zone. DEFAULT OFF (validate the
    # frame on the robot first); the guard mixes its distance into the forward stop.
    depth_nf = DepthNearField(lidar_source=lidar, cfg_path=str(OBSTACLE_CFG_PATH),
                              ls_mount_height=LIDAR_CAMERA_HEIGHT)
    depth_nf.start()
    guard.set_depth_source(depth_nf.front_near_m)
    guard.set_depth_ring_source(depth_nf.front_ring)   # per-sector ring -> merged into lidar ring
    print(f"Depth near-field fusion ready ({'ON' if depth_nf.enabled else 'OFF'}; D435i).",
          flush=True)

    # Velocity shaper: jerk/accel-limited smoothing of the teleop command (smooth walk).
    shaper = CommandShaper(cfg=MOTION_CFG, max_speeds=(MAX_VX, MAX_VY, MAX_VYAW))
    print(f"Command shaper ready (motion smoothing {'ON' if shaper.enabled else 'OFF'}).",
          flush=True)

    # Discrete-step pacer: small/medium stutter-steps for tight spaces, all directions.
    # Reads odom (pose/live) for the OPTIONAL distance-quantized refinement (OFF by
    # default). default_mode = continuous = pure pass-through until the operator opts in.
    pacer = StepPacer(cfg=STEP_CFG, max_speeds=(MAX_VX, MAX_VY, MAX_VYAW),
                      pose_fn=(odom.get_pose if odom else None),
                      live_fn=(odom.is_live if odom else None))
    pacer.set_mode(state.step_mode)   # honour the State's startup mode (no-op if continuous)
    print(f"Step pacer ready (step mode '{pacer.mode}', "
          f"{'ON' if pacer.enabled else 'OFF'}).", flush=True)

    # Blocking robot RPCs run in their own thread; the event loop only streams.
    cmd_thread = threading.Thread(target=command_loop, daemon=True)
    cmd_thread.start()
    print("Command loop started (robot RPCs off the event loop).", flush=True)

    task = asyncio.create_task(broadcast_loop())
    print(f"Web controller live at http://<robot-ip>:{PORT}", flush=True)

    yield

    task.cancel()
    try:
        client.StopMove()
    except Exception:
        pass
    if camera is not None:
        camera.stop()
    try:
        cam_proc.terminate()
    except Exception:
        pass
    if lidar is not None:
        lidar.stop()
    if odom is not None:
        odom.stop()
    if guard is not None:
        try:
            guard.stop()
        except Exception:
            pass
    if depth_nf is not None:
        try:
            depth_nf.stop()
        except Exception:
            pass
    if obstacle_mgr is not None:
        try:
            obstacle_mgr.stop()
        except Exception:
            pass
    print("Shutdown -- robot left in current posture.", flush=True)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# The HTML shells reference their JS/CSS with ?v= cache-busting query strings, but
# the shells THEMSELVES had no Cache-Control, so phones/tunnels served a stale copy
# on a soft reload (new ?v= never fetched). "no-cache" = always revalidate the shell
# (still 304-cheap via ETag); the versioned static assets stay fully cacheable.
_HTML_HEADERS = {"Cache-Control": "no-cache"}


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html", headers=_HTML_HEADERS)


@app.get("/phone")
async def phone():
    # Stripped-down touch teleop surface (steering + obstacle toggle + mode ladder).
    # Same /ws protocol as the laptop console; shares the single-controller lock.
    return FileResponse(WEB_DIR / "phone.html", headers=_HTML_HEADERS)


# ---------------------------------------------------------------------------
# Camera — MJPEG over HTTP multipart (consumed by a plain <img>)
# ---------------------------------------------------------------------------

@app.get("/camera/status")
async def camera_status():
    return JSONResponse({
        "live": bool(camera and camera.is_live()),
        "backend": camera.backend if camera else "none",
    })


@app.get("/camera/stream")
async def camera_stream():
    boundary = "frame"
    period = 1.0 / CAMERA_STREAM_HZ

    async def gen():
        while True:
            jpeg = camera.get_jpeg() if camera else None
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


# ---------------------------------------------------------------------------
# Pose — people skeletons (YOLO11-pose). Same MJPEG pattern as the camera, but
# the source JPEG is produced by the pose container. Reading the stream also
# heartbeats POSE_DEMAND so the container only runs the GPU while someone watches.
# ---------------------------------------------------------------------------

def _fresh(path, ttl=2.5):
    """True if `path` was written within `ttl` seconds (overlay JSON liveness)."""
    try:
        return (time.time() - os.path.getmtime(path)) < ttl
    except OSError:
        return False


@app.get("/camera/pose/status")
async def pose_status():
    # Liveness now tracks the geometry JSON (the browser draws it on a canvas; the
    # service no longer bakes an annotated JPEG).
    return JSONResponse({"live": _fresh(POSE_TRACKS), "backend": "pose"})


@app.get("/camera/pose/stream")
async def pose_stream():
    boundary = "frame"
    period = 1.0 / CAMERA_STREAM_HZ

    async def gen():
        while True:
            try:                       # heartbeat -> pose_service runs while watched
                with open(POSE_DEMAND, "wb") as f:
                    f.write(b"1")
            except OSError:
                pass
            jpeg = pose.get_jpeg() if pose else None
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/camera/pose/tracks")
async def pose_tracks():
    """Pose geometry {w, h, items:[{id, name, box, kpts}]} (written by pose_service).

    Polling this also heartbeats POSE_DEMAND, so the pose container only runs the
    GPU while the Skeleton overlay is on (the browser polls this once a second).
    """
    try:
        with open(POSE_DEMAND, "wb") as f:
            f.write(b"1")
    except OSError:
        pass
    try:
        with open(POSE_TRACKS) as f:
            return JSONResponse(json.load(f))
    except (OSError, ValueError):
        return JSONResponse({"w": 0, "h": 0, "items": []})


@app.post("/camera/pose/label")
async def pose_label(req: Request):
    """Map a track id to an operator-chosen name (empty name clears it)."""
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
    tid = str(data.get("id", "")).strip()
    name = str(data.get("name", "")).strip()[:32]
    if not tid:
        return JSONResponse({"ok": False, "error": "missing id"}, status_code=400)
    labels = {}
    try:
        with open(POSE_LABELS) as f:
            labels = json.load(f) or {}
    except (OSError, ValueError):
        pass
    if name:
        labels[tid] = name
    else:
        labels.pop(tid, None)
    tmp = POSE_LABELS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f)
    os.replace(tmp, POSE_LABELS)      # atomic -> pose_service never reads a partial file
    return JSONResponse({"ok": True, "labels": labels})


# ---------------------------------------------------------------------------
# Detect — object detection (YOLO-World). Same MJPEG pattern as the pose lane;
# the source JPEG is produced by the separate g1-detect container. Reading the
# stream heartbeats DETECT_DEMAND so the container only runs the GPU while watched.
# ---------------------------------------------------------------------------

@app.get("/camera/detect/status")
async def detect_status():
    return JSONResponse({"live": _fresh(DETECT_TRACKS), "backend": "detect"})


@app.get("/camera/detect/stream")
async def detect_stream():
    boundary = "frame"
    period = 1.0 / CAMERA_STREAM_HZ

    async def gen():
        while True:
            try:                       # heartbeat -> detect_service runs while watched
                with open(DETECT_DEMAND, "wb") as f:
                    f.write(b"1")
            except OSError:
                pass
            jpeg = detect.get_jpeg() if detect else None
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/camera/detect/objects")
async def detect_objects():
    """Detection geometry {w, h, items:[{cls, conf, box}]} (written by detect_service).

    Polling this also heartbeats DETECT_DEMAND, so the detect container only runs
    the GPU while the Object Detection overlay is on.
    """
    try:
        with open(DETECT_DEMAND, "wb") as f:
            f.write(b"1")
    except OSError:
        pass
    try:
        with open(DETECT_TRACKS) as f:
            return JSONResponse(json.load(f))
    except (OSError, ValueError):
        return JSONResponse({"w": 0, "h": 0, "items": []})


# ---------------------------------------------------------------------------
# Hands — finger landmarks. Same demand-gated geometry pattern as the pose lane;
# the source frame is the shared camera JPEG, the producer is the g1-hands
# container. Polling /camera/hands/tracks heartbeats HANDS_DEMAND so the GPU only
# runs while the Skeleton overlay is on (the browser polls it alongside the pose
# tracks while Skeleton is active).
# ---------------------------------------------------------------------------

@app.get("/camera/hands/status")
async def hands_status():
    return JSONResponse({"live": _fresh(HANDS_TRACKS), "backend": "hands"})


@app.get("/camera/hands/tracks")
async def hands_tracks():
    """Hand geometry {w, h, items:[{hand, score, landmarks:[[x,y,z]x21]}]}."""
    try:
        with open(HANDS_DEMAND, "wb") as f:
            f.write(b"1")
    except OSError:
        pass
    try:
        with open(HANDS_TRACKS) as f:
            return JSONResponse(json.load(f))
    except (OSError, ValueError):
        return JSONResponse({"w": 0, "h": 0, "items": []})


# ---------------------------------------------------------------------------
# LiDAR — 3D point cloud (Z-up) over a dedicated binary WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/lidar")
async def ws_lidar(ws: WebSocket):
    await ws.accept()
    print("[ws/lidar] client connected", flush=True)
    live_period = 1.0 / LIDAR_STREAM_HZ
    map_period = 0.4   # map clouds are larger -> throttle harder
    last_view = None
    try:
        while True:
            # Show the map while mapping or when one is loaded; else the live cloud.
            show_map = mapper is not None and (mapper.active or mapper.has_points())
            cloud = mapper.get_map() if show_map else (lidar.get_cloud() if lidar else None)
            view = "map" if show_map else "live"
            if view != last_view:
                await ws.send_text(json.dumps({"type": "lidar_meta", "view": view}))
                last_view = view
            if cloud is not None and len(cloud):
                await ws.send_bytes(pack_cloud(cloud))
            await asyncio.sleep(map_period if show_map else live_period)
    except WebSocketDisconnect:
        print("[ws/lidar] client disconnected", flush=True)
    except Exception as e:
        print(f"[ws/lidar] error: {e}", flush=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global control_owner_id
    await ws.accept()
    clients.add(ws)
    client_meta[ws] = {"id": None, "label": None}
    print("[ws] client connected", flush=True)

    # Push speed caps so the UI matches config/robot.yaml (single source of truth).
    await ws.send_text(json.dumps({
        "type": "config",
        "max_vx": MAX_VX, "max_vy": MAX_VY, "max_vyaw": MAX_VYAW,
        "slow_scale": SLOW_SCALE,
        # which whole-body combos have a captured FSM id (-> button enabled)
        "mode_combos": {name: (fsm is not None) for name, fsm in MODE_COMBOS.items()},
        # initial obstacle-guard UI flags (enabled [+ depth])
        "obstacle": (dict(guard.ui_config(), depth=depth_nf.telemetry())
                     if guard is not None and depth_nf is not None
                     else (guard.ui_config() if guard is not None else {})),
        # initial discrete-step selector state ('continuous' default -> 'Normal' on load)
        "step": {"mode": state.step_mode, "enabled": STEP_ENABLED,
                 "modes": ["continuous", "medium", "small"]},
    }))
    await ws.send_text(json.dumps(make_state_msg()))
    if mapper is not None:
        await ws.send_text(json.dumps(mapper.status()))
    # Who currently holds the single-controller lock -- the client uses this to know
    # whether it may drive, and to render the control chip / take-control overlay.
    await ws.send_text(json.dumps(make_control_msg()))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            # --- Identity + single-controller arbitration (before any mutation) ---
            if mtype == "hello":
                # A client announces its stable per-load id + display label. If the
                # lock is free, the first announcer grabs it (take-if-free) so a lone
                # laptop/phone always drives; otherwise it starts read-only.
                cid = str(msg.get("client_id") or "")
                label = (str(msg.get("label") or "device"))[:24]
                client_meta[ws] = {"id": cid or None, "label": label}
                if cid and control_owner_id is None:
                    await _set_owner(cid)
                else:
                    await ws.send_text(json.dumps(make_control_msg()))
                continue

            if mtype == "take_control":
                # Explicit "I control the movements now" -- always steals the lock from
                # whoever held it (and zeroes velocity via _set_owner, so no lurch).
                cid = str(msg.get("client_id") or "")
                if cid:
                    meta = client_meta.get(ws) or {"id": None, "label": "device"}
                    meta["id"] = cid
                    client_meta[ws] = meta
                    await _set_owner(cid)
                continue

            # Every other message MUTATES robot state -> only the lock owner may send
            # it. A client that skipped 'hello' but is first to drive grabs a free lock.
            meta = client_meta.get(ws) or {}
            my_id = meta.get("id")
            if my_id is not None and control_owner_id is None:
                await _set_owner(my_id)
            if my_id is None or my_id != control_owner_id:
                continue                      # read-only device -> ignore

            # Owner is alive: ONLY the controller's traffic feeds the motion watchdog,
            # so the robot stops if the controlling device goes silent even while other
            # (read-only) devices keep the socket busy.
            state.last_packet_time = time.time()

            if mtype == "move":
                if state.mode == "walk" and not state.transitioning:
                    # Sanitize to a FINITE float -> 0. A non-finite velocity (e.g. a
                    # client sending {"vx":"nan"}) must never reach the robot: the
                    # downstream clamp(nan) would fail OPEN to full speed.
                    state.vx = _finite_vel(msg.get("vx"))
                    state.vy = _finite_vel(msg.get("vy"))
                    state.vyaw = _finite_vel(msg.get("vyaw"))

            elif mtype == "stop":
                # Zero the velocity; command_loop sends Move(0,0,0), which halts
                # translation while keeping the gait ready (no StopMove -> stays drivable).
                state.vx = state.vy = state.vyaw = 0.0
                state.is_moving = False

            elif mtype == "step_mode":
                # Discrete-step size selector (continuous = analog; small/medium = pulses).
                # set_mode() resets the pacer so a switch never resumes mid-pulse; the
                # next held key starts a clean burst. Selecting 'continuous' is an
                # instant return to pass-through (an effective calm-down control).
                name = msg.get("name", "")
                if name in ("continuous", "small", "medium"):
                    state.step_mode = name
                    if pacer is not None:
                        pacer.set_mode(name)
                    # Apply this mode's foot-lift (Small shuffles lower; Normal/Medium
                    # restore the baseline) -- off-thread, so it never blocks the WS loop.
                    apply_step_swing(name)
                    # Confirm to all clients so connected dashboards stay in sync.
                    await broadcast({"type": "step_mode", "name": state.step_mode})

            elif mtype == "mode":
                requested = msg.get("name", "")
                threading.Thread(target=run_enter_mode, args=(requested,),
                                 daemon=True).start()

            elif mtype == "cmd":
                state.pending_cmd = msg.get("name", "")

            elif mtype == "map" and mapper is not None:
                action = msg.get("action", "")
                if action == "start":
                    # Mapping needs the Mid-360 exclusively -> pause the always-on
                    # obstacle node + motion guard, then start mapping.
                    if guard is not None: guard.set_enabled(False)
                    if obstacle_mgr is not None: obstacle_mgr.stop()
                    mapper.start()
                elif action == "stop":
                    mapper.stop()
                    if obstacle_mgr is not None:
                        obstacle_mgr.start()      # resume the always-on obstacle viz
                elif action == "clear":
                    mapper.clear()
                elif action == "save":
                    mapper.save(msg.get("name", ""))
                elif action == "load":
                    mapper.load(msg.get("name", ""))
                await broadcast(mapper.status())

            elif mtype == "obstacle":
                action = msg.get("action", "")
                if action == "enable":
                    if mapper is not None and mapper.active:
                        await broadcast({"type": "obstacle",
                                         "error": "Stop mapping first (shared LiDAR)."})
                    elif guard is not None:
                        obstacle_mgr.start(); guard.set_enabled(True)
                elif action == "disable":
                    # Motion governing OFF only -- the perception node KEEPS running so
                    # the 2D/3D views stay live (it is paused solely for mapping).
                    if guard is not None: guard.set_enabled(False)
                elif action == "depth_fusion" and depth_nf is not None:
                    depth_nf.set_enabled(bool(msg.get("on")))
                if guard is not None:
                    tm = guard.telemetry()
                    if depth_nf is not None:
                        tm["depth"] = depth_nf.telemetry()
                    await broadcast(tm)

    except WebSocketDisconnect:
        print("[ws] client disconnected", flush=True)
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)
    finally:
        clients.discard(ws)
        meta = client_meta.pop(ws, None)
        # Only halt + release the lock if the DEPARTING client actually held control.
        # A read-only client leaving must NOT stop the robot the owner is still driving.
        if (meta is not None and meta.get("id") is not None
                and meta.get("id") == control_owner_id):
            state.vx = state.vy = state.vyaw = 0.0
            state.is_moving = False
            control_owner_id = None            # lock is free; a present device can grab it
            try:
                await broadcast(make_control_msg())
            except Exception:
                pass


if __name__ == "__main__":
    # The safety prompt only makes sense when a human launched this in a terminal.
    # Under systemd there is no stdin, so input() raised EOFError and the service
    # crash-looped. Gate the prompt on an interactive TTY: manual launches still get
    # the Enter-to-confirm gate; the service starts headless without it.
    if sys.stdin.isatty():
        print("WARNING: Robot must be on its gantry or in a clear open area.")
        print("WARNING: Keep the physical e-stop in reach.")
        input("Press Enter to start, or Ctrl+C to abort... ")
    else:
        print("No TTY (systemd service) -- starting without interactive confirmation.",
              flush=True)
    # timeout_graceful_shutdown: the camera/pose/detect MJPEG streams and the
    # lidar WebSocket are infinite `while True` generators, so on SIGTERM uvicorn
    # would wait forever for them to finish -> systemd's 90s stop-timeout fires
    # -> SIGKILL of the whole cgroup. That hard kill churns the DDS/camera
    # pipeline (spikes videohub_pc4, freezes the feed) on every restart. Capping
    # the graceful wait lets the service stop cleanly in a few seconds instead.
    uvicorn.run(app, host=HOST, port=PORT, log_level="info",
                timeout_graceful_shutdown=5)