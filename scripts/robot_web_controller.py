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
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
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
# Interactive "wave back" demo: the pose->gesture reactor + its safety-gated robot bridge.
# All robot coupling in GreetingService is via callbacks, so it imports nothing back here.
from gesture_reactor import GreetingService, describe as greeting_describe


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

# Depth-for-map handoff: the web process (this file) owns the D435i cloud and, while
# mapping is active, dumps DepthNearField's full leveled cloud here for map_bridge
# (domain 99, g1_mapping_ws) to fuse into the point-cloud map. Contract (both sides
# MUST match): little-endian <double t_epoch><uint32 N><N*3 float32>, BODY frame
# (x-fwd, y-left, z = height above floor). Written atomically (tmp + os.replace)
# ~10 Hz ONLY while mapper.active; removed the moment mapping stops.
DEPTH_MAP_SHM = "/dev/shm/g1_depth_for_map.bin"

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

# --- Auto-climb suppression (flat-ground / presentation mode) ---------------
# The onboard firmware sometimes spontaneously flips 802 (main_control) -> 812
# (climb) -> 802 in ~1s with the remote OFF and no dashboard command. There is
# no SDK call to disable that behaviour, so when the operator arms the flat-
# ground toggle we detect the un-commanded jump into 812 and re-command 802.
FSM_CLIMB = 812            # onboard climb FSM (matches FSM_NAMES[812] / MODE_COMBOS['climb'])
SUPPRESS_POLL_HZ = 8.0     # while armed, sample fast enough to catch the ~1s climb flip
SUPPRESS_WINDOW_S = 10.0   # anti-oscillation observation window
SUPPRESS_MAX_IN_WINDOW = 5 # this many reversals in-window -> auto-disable (likely a REAL climb)

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

# --- Named dance catalog (config/dances.yaml) ---------------------------------
# A list of selectable whole-body routines [{name, fsm_id, space_m, note}]. Kept
# in its OWN file (not mapping.yaml) because the dashboard's Dance Lab writes it
# back on Save, and we don't want to clobber mapping.yaml's hand-written comments.
DANCES_PATH = BASE_DIR / "config" / "dances.yaml"

# FSM ids that are the robot's BASE modes (zero_torque / damp / ready_stand /
# main_control). A dance can never be one of these, and registering one as a
# "routine" would corrupt enter_mode's exit logic (a later Stand/Walk from that
# base id would mis-fire an immediate SetFsmId(802)). So they are rejected
# everywhere a dance/probe id is accepted. Literal (not UI_TO_FSM) because this
# block runs at import BEFORE UI_TO_FSM is defined; asserted equal below.
_BASE_FSM_IDS = {0, 1, 4, 802}
_FSM_ID_MAX = 9999   # sane upper bound for a probe/dance id


def _valid_dance_id(fsm):
    """True if fsm is an int in range and NOT a base mode id -> safe to fire/store."""
    try:
        fsm = int(fsm)
    except (TypeError, ValueError):
        return False
    return 0 <= fsm <= _FSM_ID_MAX and fsm not in _BASE_FSM_IDS


def load_dances(path=DANCES_PATH):
    """Read the named-dance catalog. Returns [{name, fsm_id, space_m, note, verified}].

    Malformed/missing file -> []. Each entry must have a valid (non-base) int
    fsm_id; name/space_m/verified default sensibly. De-duplicated by fsm_id.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("dances", []) or []
    except (OSError, ValueError, TypeError):
        return []
    out, seen = [], set()
    for d in raw:
        if not isinstance(d, dict):
            continue
        fsm = d.get("fsm_id")
        if not _valid_dance_id(fsm):     # skip garbage AND base-mode ids
            continue
        fsm = int(fsm)
        if fsm in seen:
            continue
        seen.add(fsm)
        name = (str(d.get("name") or f"FSM {fsm}").strip() or f"FSM {fsm}")[:40]
        try:
            space = float(d.get("space_m", 2.0))
        except (TypeError, ValueError):
            space = 2.0
        out.append({"name": name, "fsm_id": fsm,
                    "space_m": round(max(0.0, space), 1),
                    "note": str(d.get("note") or "")[:200],
                    "verified": bool(d.get("verified", True))})
    return out


def save_dances(dances, path=DANCES_PATH):
    """Persist the dance catalog back to config/dances.yaml (Dance Lab Save)."""
    payload = {"dances": [{"name": d["name"], "fsm_id": int(d["fsm_id"]),
                           "space_m": float(d["space_m"]), "note": d.get("note", ""),
                           "verified": bool(d.get("verified", True))}
                          for d in dances]}
    tmp = Path(str(path) + ".tmp")
    header = ("# G1 DANCE CATALOG — named whole-body FSM routines (see git history\n"
              "# for the full guide). Auto-written by the dashboard's Dance Lab.\n")
    with open(tmp, "w") as f:
        f.write(header)
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


DANCES = load_dances()   # [{name, fsm_id, space_m, note}], seeded with 503

# Every dance FSM is a "routine" the robot only exits back through main_control
# (802) -- same contract as the climb/dance mode_combos. These runtime-mutable
# maps union the combo FSMs with the catalog's, and the Dance Lab adds a probed
# id here the instant it fires so Stand/Walk can always recover the robot.
ROUTINE_FSM_TO_NAME = dict(MODE_COMBO_FSM_TO_NAME)
for _d in DANCES:
    ROUTINE_FSM_TO_NAME.setdefault(_d["fsm_id"], _d["name"])
ROUTINE_FSMS = set(ROUTINE_FSM_TO_NAME)


def register_routine_fsm(fsm_id, name="routine"):
    """Mark an FSM id as a whole-body routine so enter_mode() exits it via 802.

    Called when firing a catalog dance or a Dance Lab probe -- a just-probed id
    is otherwise unknown to the exit logic, which would leave Stand/Walk unable
    to recover the robot from it. Base-mode ids (0/1/4/802) are NEVER registered
    -- doing so would corrupt enter_mode's ordinary transitions."""
    if not _valid_dance_id(fsm_id):
        return
    fsm_id = int(fsm_id)
    ROUTINE_FSMS.add(fsm_id)
    ROUTINE_FSM_TO_NAME.setdefault(fsm_id, name)


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

# The dance-catalog base-id blocklist (_BASE_FSM_IDS, defined earlier as a literal
# because it runs at import before this point) must exactly match the base modes,
# or a dance could shadow one. Assert it here now that UI_TO_FSM exists.
assert _BASE_FSM_IDS == set(UI_TO_FSM.values()), _BASE_FSM_IDS

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


# --- Persisted UI settings (small writable JSON; survives service restart AND a
# power cycle, unlike /dev/shm). Config proper stays read-only YAML; this tiny
# store only holds operator UI toggles that must persist across sessions
# (currently: suppress_autoclimb). Atomic write, modeled on the POSE_LABELS writer.
SETTINGS_PATH = BASE_DIR / "state" / "ui_settings.json"


def load_ui_settings():
    """Return the persisted UI-settings dict ({} if missing/unreadable)."""
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def save_ui_settings(d):
    """Atomically persist the UI-settings dict. Creates state/ on first write."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(SETTINGS_PATH) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, str(SETTINGS_PATH))   # atomic -> never a partial read
    except OSError as e:
        print(f"[SETTINGS] save failed: {e}", flush=True)


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
        # Wall-clock until which the robot counts as "locomoting" (refreshed on every non-zero
        # velocity command). Latches motion across step-mode settle gaps; see _locomoting().
        self.moving_hold_until = 0.0
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
        # --- Greeting mode (interactive "wave back" demo; see the wiring block in
        # lifespan). Master opt-in toggle: OFF by default, because auto-moving the arms
        # near clients must be deliberately enabled. greeting_status is dashboard feedback.
        self.greeting_mode = False
        # Flat-ground/presentation opt-in: when True, fsm_poll_loop reverses an
        # un-commanded firmware climb (802->812) back to 802. Default OFF; loaded
        # from persisted UI settings just after construction. A real climb/stairs
        # will NOT happen while this is on -- see _suppress_autoclimb.
        self.suppress_autoclimb = False
        self.greeting_status = ""    # e.g. "saw wave -> waving back"
        self.greeting_busy_until = 0.0   # wall-clock until which a greeting gesture is mid-motion
        # Discrete gesture events handed from the greeting daemon thread to broadcast_loop
        # (append/popleft are atomic under the GIL; single-producer/single-consumer, no lock).
        # Each is a ready-to-send {type:"gesture_event", ...} for the feed label + log line.
        self.gesture_events = deque(maxlen=32)
        # Battery (from rt/lf/bmsstate; None until the first BMS message arrives)
        self.battery_soc = None      # % state-of-charge
        self.battery_v = None        # pack volts
        self.battery_current = None  # amps (negative = discharging)


state = ControlState()
# Restore the persisted flat-ground toggle (default OFF -> only ever armed by a
# deliberate operator choice; the value persists across restarts/power cycles).
state.suppress_autoclimb = bool(load_ui_settings().get("suppress_autoclimb", False))
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

# Access model: driving (joysticks), mode changes and whole-body combos (dance/climb)
# stay EXCLUSIVE to the lock owner. These arm "greeting" gestures are SHARED -- any
# connected client may fire them even without the lock, so onlookers can greet while one
# person drives. They're serialized by _arm_lock and gated to an upright robot below.
OPEN_GESTURES = frozenset({"high_wave", "high_five", "clap", "shake", "hug", "heart", "kiss"})

# How long a commanded non-zero velocity keeps the robot considered "locomoting" after the
# last motion command. Bridges the pacer's OFF/settle windows in step mode so the take-over
# gate can't flicker open between steps (review finding #3).
MOVE_HOLD_S = 0.7
# Per-gesture arm-busy windows (s) -- how long an accepted arm gesture keeps the arms moving,
# so overlapping shared/auto gestures can't interleave mid-hold (review finding #4). Shared
# with the auto-greeting path (state.greeting_busy_until).
_ARM_BUSY_S = {"wave": 3.5, "high_five": 5.0, "hug": 5.0, "heart": 5.0,
               "shake": 4.0, "clap": 2.5, "kiss": 2.5, "high_wave": 3.5}


def _locomoting():
    """True iff the robot is actively translating in walk. Mode-guarded (a non-walk robot is
    NEVER locomoting, so it is always take-able -- fixes the is_moving latch after an
    uncommanded FSM drop, review #2) + a short hold window (review #3) so the step-mode settle
    gaps don't momentarily read as 'stopped' and open the take-over gate mid-drive."""
    return (state.mode == "walk" and not state.transitioning
            and time.time() < state.moving_hold_until)


def _arm_busy():
    """True while an arm gesture is still physically in motion (shared with the greeting path
    via state.greeting_busy_until), so a second gesture can't interleave mid-hold (review #4)."""
    return time.time() < state.greeting_busy_until


def _gesture_exec_ok():
    """Safe to run an arm/hero gesture RIGHT NOW: upright, settled, NOT translating, and the
    owner's arms are not raised. Re-checked at execution time because a shared gesture can be
    enqueued by a non-owner and the robot state can change before it drains (review #1,#5,#6)."""
    return (state.mode in ("stand", "walk") and not state.transitioning
            and not _locomoting() and not state.arm_raised)

camera: CameraSource = None
pose: CameraSource = None
detect: CameraSource = None
lidar: LidarSource = None
odom: OdomReader = None
mapper: MapBuilder = None

guard: ObstacleGuard = None
obstacle_mgr: ObstacleManager = None
depth_nf: DepthNearField = None
depth_dumper = None
shaper: CommandShaper = None
pacer: StepPacer = None
audio = None
greeting: GreetingService = None   # safety-gated wave-back bridge (created in lifespan)

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
    return (FSM_NAMES.get(fsm_id)
            or ROUTINE_FSM_TO_NAME.get(fsm_id)
            or f"fsm_{fsm_id}")


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


# Monotonic times of recent forced climb->802 reversals, for the oscillation cap.
_suppress_hist = deque(maxlen=16)


def _suppress_autoclimb(old, fsm):
    """Reverse an un-commanded firmware climb (802->812) back to main_control (802).

    Called EDGE-triggered from fsm_poll_loop's external branch ONLY, so it fires
    once per observed un-commanded jump into 812 -- never on a commanded climb
    (that reads as ours) and never continuously. note_fsm_intent(802) is recorded
    FIRST so our forced return reads as OURS and is not itself re-flagged/re-
    suppressed. SetFsmId(802) is the identical base FSM Walk mode commands -- no
    novel motion, just pinning the normal upright locomotion base.

    Anti-oscillation: if the firmware re-triggers SUPPRESS_MAX_IN_WINDOW times
    within SUPPRESS_WINDOW_S, suppression auto-disables (persisted + broadcast by
    the caller path) on the assumption this may be a REAL climb/stairs.
    """
    note_fsm_intent(FSM_MAIN_CONTROL)     # our forced return is OURS, not external
    try:
        client.SetFsmId(FSM_MAIN_CONTROL)
    except Exception as e:
        print(f"[SUPPRESS] SetFsmId({FSM_MAIN_CONTROL}) failed: {e}", flush=True)
    print(f"[SUPPRESS] external climb {old} ({fsm_name(old)}) -> {fsm} "
          f"({fsm_name(fsm)}) reversed -> {FSM_MAIN_CONTROL} (main_control); "
          f"flat-ground mode", flush=True)
    now = time.monotonic()
    _suppress_hist.append(now)
    recent = sum(1 for t in _suppress_hist if now - t <= SUPPRESS_WINDOW_S)
    if recent >= SUPPRESS_MAX_IN_WINDOW:
        state.suppress_autoclimb = False
        d = load_ui_settings()
        d["suppress_autoclimb"] = False
        save_ui_settings(d)
        print("[SUPPRESS] firmware re-triggering repeatedly -- disabling "
              "(possible REAL climb/stairs); check the robot", flush=True)


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


class DepthMapDumper(threading.Thread):
    """While mapping is active, dumps DepthNearField's full leveled D435i cloud to
    DEPTH_MAP_SHM at ~hz so map_bridge (domain 99) can fuse it into the point-cloud
    map -- fills the Mid-360's near-ground blind zone in the saved map too. Self-gates
    on mapper.active: writes nothing (and removes any stale file) while not mapping,
    so a consumer that opens the path mid-mapping never sees a leftover cloud from a
    previous session. Runs continuously from startup; cheap no-op while not mapping."""

    def __init__(self, depth_nf, mapper, shm_path=DEPTH_MAP_SHM, hz=10.0):
        super().__init__(name="depth-map-dumper", daemon=True)
        self._depth_nf = depth_nf
        self._mapper = mapper
        self._shm_path = shm_path
        self._tmp_path = shm_path + ".tmp"
        self.period = 1.0 / max(1.0, float(hz))
        self._stop = threading.Event()
        self._had_file = False

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)
        self._remove()

    def _remove(self):
        try:
            os.remove(self._shm_path)
        except OSError:
            pass
        self._had_file = False

    def run(self):
        while not self._stop.wait(self.period):
            if not (self._mapper is not None and self._mapper.active):
                if self._had_file:
                    self._remove()
                continue
            try:
                cloud = self._depth_nf.map_cloud() if self._depth_nf is not None else None
                if cloud is None or len(cloud) == 0:
                    continue
                t = time.time()
                n = len(cloud)
                data = (struct.pack("<dI", float(t), int(n))
                        + np.asarray(cloud, dtype="<f4").tobytes())
                with open(self._tmp_path, "wb") as f:
                    f.write(data)
                os.replace(self._tmp_path, self._shm_path)
                self._had_file = True
            except Exception as e:
                print(f"[DEPTH-MAP] dump error: {e}", flush=True)


def fsm_poll_loop():
    while True:
        # While suppression is armed, sample faster (8 Hz) so the ~1s firmware
        # climb flip is caught ~4x before it self-reverts. The blocking RPC stays
        # wrapped in a 1.0s timeout (unchanged); only the trailing sleep shortens,
        # so a slow RPC degrades gracefully -- no busy-spin, no stacked calls.
        hz = SUPPRESS_POLL_HZ if state.suppress_autoclimb else FSM_POLL_HZ
        period = 1.0 / hz
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
                        # Flat-ground mode: reverse an un-commanded climb. Reached
                        # ONLY from this external branch, so a commanded climb (which
                        # reads as ours) is never touched.
                        if state.suppress_autoclimb and fsm == FSM_CLIMB:
                            _suppress_autoclimb(old, fsm)
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

    # Capture the mode we're LEAVING before overwriting it -- a routine name here
    # (dance/probe/climb, not a VALID_MODE) is how we know to exit via 802 even if
    # a probe landed in an fsm id we never registered.
    prev_mode = state.mode

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
        # Exit to main_control first when the robot is in a known routine FSM OR
        # when we believe we're in a routine "mode" (state.mode is a dance/probe
        # name, not a base mode) -- the latter recovers the robot even if a probe
        # landed in an fsm id we never registered.
        if new_mode in ("stand", "walk") and (state.fsm_id in ROUTINE_FSMS
                                              or prev_mode not in VALID_MODES):
            routine = ROUTINE_FSM_TO_NAME.get(state.fsm_id, prev_mode or "routine")
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
        else:
            fire_whole_body(name, fsm)

    else:
        print(f"[cmd error: unknown '{name}']", flush=True)


def fire_whole_body(name, fsm, require_upright=False):
    """Hand the whole body to a routine FSM (climb / dance / a Dance Lab probe).

    Serialized with enter_mode via _mode_lock: a routine must not race a mode
    transition's state.mode write, or command_loop could read mode=="walk" while
    the robot is in a routine FSM and drive velocity into it. Setting state.mode
    to `name` (not a VALID_MODE) makes command_loop go quiet; the FSM poller
    leaves it (no FSM_TO_UI entry). Registers the id as a routine so Stand/Walk
    exit it via main_control. Returns True if the SetFsmId was issued.

    require_upright: refuse unless the robot is settled + upright (stand/walk) and
    not locomoting -- the safety gate for operator-fired dances and probes.
    """
    if not _valid_dance_id(fsm):
        print(f"[CMD] {name} refused -- {fsm!r} is not a valid dance id "
              f"(base modes 0/1/4/802 and out-of-range are rejected)", flush=True)
        return False
    if require_upright and (state.mode not in ("stand", "walk")
                            or state.transitioning or _locomoting()):
        print(f"[CMD] {name} refused -- robot not upright/settled "
              f"(mode={state.mode} moving={_locomoting()})", flush=True)
        return False
    if not _mode_lock.acquire(blocking=False):
        print(f"[CMD] {name} ignored -- mode transition in progress", flush=True)
        return False
    try:
        register_routine_fsm(fsm, name)   # so Stand/Walk can exit it via 802
        state.vx = state.vy = state.vyaw = 0.0
        state.is_moving = False
        state.gait_enabled = False
        # Mark transitioning BEFORE the mode write so fsm_poll_loop's mode-sync
        # can't revert state.mode back to "walk" while the robot is still reading
        # the OLD fsm (802) mid-handoff -- which would let the velocity keepalive
        # fight the routine. Cleared once the handoff settles (below).
        state.transitioning = True
        state.mode = name
        print(f"[CMD] {name} -> SetFsmId({fsm})", flush=True)
        note_fsm_intent(fsm)     # dashboard-commanded routine (not external)
        try:
            client.SetFsmId(fsm)
        except Exception as e:
            print(f"[mode combo error: {e}]", flush=True)
    finally:
        # Release the lock BEFORE waiting so an abort (operator hits Stand on a bad
        # probe) is never blocked for the settle window.
        _mode_lock.release()
    # Wait for the robot to actually enter the routine, then drop the transitioning
    # flag -- but only if WE still own this handoff (a concurrent Stand/Walk that
    # grabbed the lock now owns state.mode + its own transitioning; don't stomp it).
    try:
        wait_for_fsm(fsm, timeout=5.0)
    finally:
        if state.mode == name:
            state.transitioning = False
    return True


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
        # True while the robot is actively locomoting (walk + moving, latched across step
        # gaps). Clients gate the "take over" button on it -- exactly the server's take_control
        # rule -- so a non-walk/stopped robot is always take-able.
        "moving": _locomoting(),
        # Greeting-mode toggle + last feedback line ("saw wave -> waving back"), so the
        # dashboard button reflects reality and shows what the robot just reacted to.
        "greeting_mode": state.greeting_mode,
        "greeting_status": state.greeting_status,
        # Flat-ground auto-climb-suppression toggle. Carried on every broadcast +
        # the on-connect send, so the UI reflects the current (possibly auto-
        # disabled) value on all clients.
        "suppress_autoclimb": state.suppress_autoclimb,
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
            # Execution-time safety re-check for SHARED arm gestures: one can be enqueued by a
            # non-owner and then the robot starts moving / leaves upright before it drains
            # (review #1 exec gap, #6 TOCTOU). Never run an arm/hero gesture on a translating
            # or non-upright robot; owner-only commands (modes/combos/hands_up) are unaffected.
            if cmd in OPEN_GESTURES and not _gesture_exec_ok():
                print(f"[GESTURE] dropped shared '{cmd}' -- unsafe at execution "
                      f"(mode={state.mode} moving={_locomoting()})", flush=True)
            else:
                if cmd in OPEN_GESTURES:
                    # Start the arm-busy window when the gesture actually RUNS (not at
                    # enqueue) so a gesture dropped by the re-check above never blocks the arms.
                    state.greeting_busy_until = now + _ARM_BUSY_S.get(cmd, 3.5)
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
        if moving:
            # Refresh the locomotion hold every tick a non-zero velocity is commanded, so the
            # take-over gate stays closed across step-mode OFF windows and for MOVE_HOLD_S
            # after the operator releases (review #3). Runs regardless of in_walk; _locomoting()
            # additionally requires mode == walk so a non-walk robot is never "locomoting".
            state.moving_hold_until = now + MOVE_HOLD_S
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
                # near-ground DEPTH detections (cables/low objects the Mid-360 can't see) so
                # the 2D ring + 3D sphere show EXACTLY what the fused guard reacts to.
                tm["depth_points"] = depth_nf.front_points()   # flat [x,y,z] (viz frame) or None
                tm["depth_ring"] = depth_nf.front_ring()       # per-sector depth ring or None
            await broadcast(tm)

        # Flush any discrete gesture events (feed label + log line) the greeting thread
        # queued -- independent of the telemetry throttle so a burst isn't coalesced.
        while state.gesture_events:
            await broadcast(state.gesture_events.popleft())

        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, reader, arm_client, camera, pose, detect, lidar, odom, mapper, depth_nf, depth_dumper
    global guard, obstacle_mgr, shaper, pacer, audio, remote_watcher, greeting
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
    # to fill the Mid-360's near-ground forward blind zone. Always on (no UI toggle) +
    # self-gated by the frame-sanity check; the guard mixes its per-sector ring into
    # the fused ring, min-only (never clears a reading).
    depth_nf = DepthNearField(lidar_source=lidar, cfg_path=str(OBSTACLE_CFG_PATH),
                              ls_mount_height=LIDAR_CAMERA_HEIGHT)
    depth_nf.set_enabled(True)   # always on -- no UI toggle to disable it
    depth_nf.start()
    guard.set_depth_source(depth_nf.front_near_m)
    guard.set_depth_ring_source(depth_nf.front_ring)   # per-sector ring -> merged into lidar ring
    print(f"Depth near-field fusion ready ({'ON' if depth_nf.enabled else 'OFF'}; D435i).",
          flush=True)

    # Depth-for-map dumper: while mapping, writes DepthNearField's full leveled cloud to
    # DEPTH_MAP_SHM for map_bridge (domain 99) to fuse into the saved/live map. Self-gated
    # on mapper.active, so it's safe to just run it from startup -- no-op until mapping starts.
    depth_dumper = DepthMapDumper(depth_nf, mapper)
    depth_dumper.start()
    print(f"Depth-for-map dumper ready (writes {DEPTH_MAP_SHM} only while mapping).",
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

    # =========================================================================
    # Interactive "wave back" greeting demo (gesture_reactor.GreetingService).
    # OPTIONAL and OFF by default. The service runs its own daemon thread that reads the
    # pose container's per-person skeletons from POSE_TRACKS -- and, when available, the hand
    # landmarks from HANDS_TRACKS (palm fusion, so a wave still reads when the short robot
    # can't see the head/shoulders) -- classifies human gestures,
    # and -- ONLY while greeting mode is ON *and* it is safe right now -- fires the mapped
    # robot gesture through the EXISTING serialized command path (it sets state.pending_cmd,
    # which command_loop drains into apply_cmd under _arm_lock). No deep surgery: every
    # robot coupling below is a small callback, so gesture_reactor stays controller-agnostic
    # and unit-testable (scripts/test_gesture_reactor.py).
    #
    # SAFETY GATE (_greeting_safe) -- an arm gesture auto-fires near a client ONLY when ALL:
    #   * the arm + loco clients are up;
    #   * the robot is in normal balance-standing ("walk" FSM) -- upright and controllable
    #     (never while damped/zero-torque/stand-up transitioning or in a whole-body combo);
    #   * it is NOT actively driving (is_moving), NOT mid mode-change (transitioning), and
    #     NOT already holding an arm up (hands_up);
    #   * no command is already queued (don't stomp an operator's button press);
    #   * a previous greeting gesture has physically FINISHED (greeting_busy_until) -- the
    #     arm actions auto-release after ~4 s but don't hold _arm_lock across that hold, so
    #     we track their motion window explicitly to honour "never fire mid-gesture".
    # The reactor adds its own debounce on top (per-gesture cooldown + a global refractory),
    # so a continuous wave fires exactly once and gestures can never machine-gun.
    #
    def _greeting_enabled():
        return state.greeting_mode

    def _greeting_safe():
        return (arm_client is not None and client is not None
                and state.mode == "walk"          # upright, balancing, ready -- not damped
                and not _locomoting()              # not translating (latched across step gaps)
                and not state.transitioning        # not mid mode-change
                and not state.arm_raised            # arms not already raised (hands_up)
                and state.pending_cmd is None       # nothing already queued to run
                and time.time() >= state.greeting_busy_until)   # last greeting finished

    def _greeting_fire(robot_gesture):
        # Reuse the operator path: command_loop picks up pending_cmd and runs apply_cmd in
        # its own thread under _arm_lock. Never clobber a command already queued, and mark
        # the arm busy for the gesture's duration so nothing else fires mid-motion.
        if robot_gesture and state.pending_cmd is None:
            state.pending_cmd = robot_gesture
            state.greeting_busy_until = time.time() + _ARM_BUSY_S.get(robot_gesture, 4.0)

    def _greeting_on_event(ev):
        # Every classified gesture updates the dashboard feedback line (surfaced by
        # make_state_msg on the next broadcast tick), fired-or-not.
        state.greeting_status = greeting_describe(ev)
        print(f"[GREETING] {state.greeting_status}", flush=True)
        # Hand a discrete gesture_event to broadcast_loop so the camera feed can label the
        # person + the dashboard logs a line -- on EVERY classified gesture, fired or not.
        # fired = would the robot actually move (same safety gate, a pure read here); box/w/h
        # anchor the on-feed label even when the skeleton overlay is off.
        state.gesture_events.append({
            "type": "gesture_event",
            "track_id": ev.get("track_id"),
            "human": ev.get("human", ev.get("gesture")),
            "robot": ev.get("robot_gesture"),
            "source": ev.get("source"),   # "skeleton" or "skeleton+palm" -- tuning visibility
            "fired": _greeting_safe(),
            "ts": ev.get("t", time.time()),
            "box": ev.get("box"), "w": ev.get("w"), "h": ev.get("h"),
        })

    def _greeting_on_skip(ev):
        # Classified but gated out by _greeting_safe -> tell the operator we saw it and why
        # nothing moved, rather than silently swallowing it.
        state.greeting_status = f"saw {ev.get('human', ev['gesture'])} -- holding (not safe)"
        print(f"[GREETING] skipped {ev['gesture']}: not safe right now", flush=True)

    greeting = GreetingService(
        tracks_path=POSE_TRACKS, demand_path=POSE_DEMAND,
        hands_path=HANDS_TRACKS, hands_demand_path=HANDS_DEMAND,
        enabled_fn=_greeting_enabled, safe_fn=_greeting_safe, fire_fn=_greeting_fire,
        on_event=_greeting_on_event, on_skip=_greeting_on_skip)
    greeting.start()
    print("Greeting service ready (wave-back demo; OFF until toggled).", flush=True)
    # Dashboard toggle: the "🤖 Greet" button (web/index.html, controller.js) sends
    # {type:"greeting", on:<bool>} -> the "greeting" WS handler below sets state.greeting_mode;
    # every make_state_msg carries {greeting_mode, greeting_status} back so the button lights
    # and shows the last feedback line (e.g. "saw wave -> waving back").

    task = asyncio.create_task(broadcast_loop())
    print(f"Web controller live at http://<robot-ip>:{PORT}", flush=True)

    yield

    task.cancel()
    if greeting is not None:
        greeting.stop()
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
    if depth_dumper is not None:
        try:
            depth_dumper.stop()
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

# --- Motion tab (record -> recreate; motion/PLAN.md Phase 1). Routes under /motion/*.
# Wrapped defensively: a motion import error prints one line and never takes the dashboard
# down. get_control_owner reads the LIVE control_owner_id each request (lambda, not a value),
# so record/recreate are gated to whoever currently holds the single-controller lock. ---
try:
    from motion.app.jobs import JobStore
    from motion.app.replay import StubProvider, SonicProvider
    from motion.app.routes import build_motion_router
    _motion_store = JobStore(BASE_DIR / "motion" / "data" / "clips")
    # StubProvider (copies the clip) by default; the real ROMP->GMR->SONIC pipeline is
    # opt-in with MOTION_PROVIDER=sonic (ON-DEVICE only -- needs the Orin's ML stack).
    _use_sonic = os.environ.get("MOTION_PROVIDER", "").lower() == "sonic"
    _motion_provider = SonicProvider() if _use_sonic else StubProvider()
    app.include_router(build_motion_router(
        get_control_owner=lambda: control_owner_id,
        store=_motion_store,
        provider=_motion_provider,
        frame_source=CAMERA_SHM,
        record_hz=30.0, max_seconds=30.0,
    ))
    print(f"Motion tab mounted at /motion/* (provider={_motion_provider.name})", flush=True)
except Exception as e:
    print(f"[MOTION] disabled (mount failed): {e}", flush=True)


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
    map_period = 0.4   # map clouds are larger -> throttle harder
    idle_period = 1.0  # nothing to stream -> just watch for the map appearing
    last_view = None
    try:
        while True:
            # Stream the FAST-LIO map while mapping is active or a map is loaded /
            # has accumulated points. Otherwise send NOTHING here -- the raw D435i
            # cloud (lidar.get_cloud()) is no longer streamed over this socket; the
            # browser falls back to the always-on 'obstacle' telemetry (leveled
            # sphere cloud) for its idle view instead. lidar.get_cloud() itself is
            # untouched and still used by DepthNearField for depth-fusion guard logic.
            show_map = mapper is not None and (mapper.active or mapper.has_points())
            view = "map" if show_map else "idle"
            if view != last_view:
                await ws.send_text(json.dumps({"type": "lidar_meta", "view": view}))
                last_view = view
            if show_map:
                cloud = mapper.get_map()
                if cloud is not None and len(cloud):
                    await ws.send_bytes(pack_cloud(cloud))
                await asyncio.sleep(map_period)
            else:
                await asyncio.sleep(idle_period)
    except WebSocketDisconnect:
        print("[ws/lidar] client disconnected", flush=True)
    except Exception as e:
        print(f"[ws/lidar] error: {e}", flush=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global control_owner_id, DANCES
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
        # named dance catalog [{name, fsm_id, space_m, note}] for the Dance chooser
        "dances": DANCES,
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
                # Explicit "I control the movements now". Safety handoff: you may seize the
                # lock when it is FREE or when the robot is NOT actively moving. Never rip
                # control from a device whose robot is translating -- wait until it stops.
                # _set_owner zeroes velocity on handoff, so there's no lurch.
                cid = str(msg.get("client_id") or "")
                if cid:
                    held_by_other = control_owner_id is not None and control_owner_id != cid
                    if held_by_other and _locomoting():
                        # Refuse only while the robot is actually locomoting (walk + moving).
                        # A non-walk / damped / faulted robot is ALWAYS take-able so a backup
                        # operator can recover it (review #2); the hold timer keeps this closed
                        # across step-mode settle gaps (#3). Tell the asker so its button waits.
                        await ws.send_text(json.dumps(
                            {"type": "takeover_denied", "reason": "moving"}))
                    else:
                        meta = client_meta.get(ws) or {"id": None, "label": "device"}
                        meta["id"] = cid
                        client_meta[ws] = meta
                        await _set_owner(cid)
                continue

            if mtype == "cmd" and msg.get("name") in OPEN_GESTURES:
                # SHARED arm gesture -- any connected client may fire it, with or without the
                # drive lock. Mirrors the auto-greeting safety gate (_greeting_safe): upright
                # and settled (stand/walk), NOT translating (never move the arms mid-stride --
                # review #1), the owner's hands_up-raised arms are left alone (#5), and no arm
                # gesture is already queued or physically in motion (#4). Sets the shared
                # arm-busy window on accept so overlapping gestures can't interleave.
                # Deliberately does NOT touch last_packet_time: a shared gesture must never
                # feed the driver's motion watchdog (a silent driver still halts the robot).
                gname = msg.get("name", "")
                if (state.mode in ("stand", "walk") and not state.transitioning
                        and not _locomoting() and not state.arm_raised
                        and state.pending_cmd is None and not _arm_busy()):
                    state.pending_cmd = gname   # arm-busy window starts at execution (drain)
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

            elif mtype == "dance":
                # Play a named catalog dance by its FSM id. Owner-only (reached only
                # past the lock gate above). Fired in a thread so the blocking SetFsmId
                # RPC never stalls the event loop; require_upright is the safety gate.
                try:
                    fsm = int(msg.get("fsm_id"))
                except (TypeError, ValueError):
                    fsm = None
                entry = next((d for d in DANCES if d["fsm_id"] == fsm), None) if fsm is not None else None
                if entry is not None:
                    threading.Thread(target=fire_whole_body,
                                     args=(entry["name"], entry["fsm_id"]),
                                     kwargs={"require_upright": True},
                                     daemon=True).start()
                else:
                    print(f"[DANCE] rejected unknown fsm_id {msg.get('fsm_id')!r} "
                          f"(not in catalog)", flush=True)

            elif mtype == "dance_probe":
                # Dance Lab: SUPERVISED probe of a candidate FSM id. Same safety gate +
                # threading as a named dance, but the id is arbitrary (unknown routine)
                # -> register_routine_fsm (inside fire_whole_body) lets Stand/Walk recover
                # it. _valid_dance_id rejects out-of-range AND base-mode ids (0/1/4/802).
                fsm = msg.get("fsm_id")
                if not _valid_dance_id(fsm):
                    print(f"[DANCE-LAB] rejected probe id {fsm!r} "
                          f"(base modes 0/1/4/802 and out-of-range not allowed)", flush=True)
                else:
                    fsm = int(fsm)
                    print(f"[DANCE-LAB] supervised probe -> SetFsmId({fsm})", flush=True)
                    threading.Thread(target=fire_whole_body,
                                     args=(f"probe {fsm}", fsm),
                                     kwargs={"require_upright": True},
                                     daemon=True).start()

            elif mtype == "dance_save":
                # Dance Lab: persist a just-verified dance to the catalog (append or
                # replace by fsm_id, verified=True), then rebroadcast so every client's
                # chooser updates. Rejects base-mode ids via _valid_dance_id.
                fsm = msg.get("fsm_id")
                dname = (str(msg.get("name") or "").strip())[:40]
                try:
                    space = round(max(0.0, float(msg.get("space_m", 2.0))), 1)
                except (TypeError, ValueError):
                    space = 2.0
                if not _valid_dance_id(fsm) or not dname:
                    print(f"[DANCE-LAB] save rejected (fsm={fsm!r} name={dname!r})", flush=True)
                else:
                    fsm = int(fsm)
                    cat = [d for d in DANCES if d["fsm_id"] != fsm]
                    cat.append({"name": dname, "fsm_id": fsm, "space_m": space,
                                "note": str(msg.get("note") or "")[:200], "verified": True})
                    cat.sort(key=lambda d: d["fsm_id"])
                    try:
                        save_dances(cat)
                        DANCES = cat
                        register_routine_fsm(fsm, dname)
                        print(f"[DANCE-LAB] saved '{dname}' (FSM {fsm}, ~{space} m)", flush=True)
                        await broadcast({"type": "dances", "dances": DANCES})
                    except OSError as e:
                        print(f"[DANCE-LAB] save failed: {e}", flush=True)

            elif mtype == "dance_delete":
                # Dance Lab: remove a catalog entry (a dud candidate, or any tile) by
                # fsm_id, persist, and rebroadcast. Does NOT unregister the routine FSM
                # (harmless to leave; only affects log naming + exit recovery).
                fsm = msg.get("fsm_id")
                try:
                    fsm = int(fsm)
                except (TypeError, ValueError):
                    fsm = None
                if fsm is None or not any(d["fsm_id"] == fsm for d in DANCES):
                    print(f"[DANCE-LAB] delete: no catalog entry for {msg.get('fsm_id')!r}",
                          flush=True)
                else:
                    cat = [d for d in DANCES if d["fsm_id"] != fsm]
                    try:
                        save_dances(cat)
                        DANCES = cat
                        print(f"[DANCE-LAB] deleted dance FSM {fsm}", flush=True)
                        await broadcast({"type": "dances", "dances": DANCES})
                    except OSError as e:
                        print(f"[DANCE-LAB] delete failed: {e}", flush=True)

            elif mtype == "greeting":
                # Master toggle for the interactive wave-back demo. Turning it OFF resets
                # the reactor so a pose held during the OFF window can't fire the instant it
                # flips back ON. The service thread itself does the shm reading + gating.
                on = bool(msg.get("on"))
                state.greeting_mode = on
                if not on:
                    state.greeting_status = ""
                    if greeting is not None:
                        greeting.reactor.reset()
                await broadcast(make_state_msg())

            elif mtype == "suppress_climb":
                # Flat-ground/presentation toggle: arm/disarm dashboard-side
                # reversal of the firmware's spontaneous climb. Owner-only (past the
                # ownership gate above). Persists so the choice survives a restart,
                # but ships/defaults OFF. A REAL climb will NOT happen while ON.
                on = bool(msg.get("on"))
                state.suppress_autoclimb = on
                d = load_ui_settings()
                d["suppress_autoclimb"] = on
                save_ui_settings(d)
                print(f"[SUPPRESS] auto-climb suppression "
                      f"{'ON (flat-ground mode)' if on else 'off'}", flush=True)
                await broadcast(make_state_msg())

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