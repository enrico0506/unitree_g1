#!/usr/bin/env python3
"""sim_runner.py -- headless mujoco G1 circle-patrol sim. NO ROBOT, NO ROS, NO DDS.

Track B (B2) of the Nav-dashboard plan. Runs the SAME high-level circle-patrol /
person-interaction DECISION logic the real robot uses (imported from
`g1_nav.patrol_logic`, a plain-Python module with no ROS dependency -- see the
import block below) against a fake, purely KINEMATIC mujoco backend instead of
`LocoClient`/Nav2. The point is a genuine dry-run of the decision logic, not a
reimplementation of it.

WHAT `g1_nav.patrol_logic` PROVIDES VS. WHAT THIS SCRIPT ADDS
    `CirclePatrolGeometry` is PURE GEOMETRY -- it anchors a circle center once
    and hands back a carrot point/pose on the circle. It does NOT compute a
    velocity. `PatrolStateMachine` is a PURE TRANSITION-TABLE FSM -- it validates
    and performs state changes (`try_transition()`), it does NOT decide WHEN to
    change state (no timers, no distance thresholds, no sensing). On the real
    robot, `g1_nav/g1_nav/nodes/circle_patrol.py` wraps the geometry (turning a
    carrot pose into NavigateToPose goals for Nav2/DWB to actually drive), and
    the not-yet-written `patrol_supervisor.py` will wrap the FSM (deciding when
    a person is "close enough, long enough" to fire circling->pausing, when a
    facing/wave/wait timer has elapued, etc.) -- see g1_nav/config/patrol.yaml's
    `patrol_supervisor:`/`person_fusion:` blocks, which exist FOR that future
    node. This script is the sim-side equivalent of both wrappers combined: a
    small local `_pursuit_command()` (steering law, mirrors the established
    pattern in scripts/sim_circle.py's `pursuit_command()`/scripts/sim_walk.py's
    `policy_goal()`) plus `InteractionSupervisor` (the timer/threshold decision
    logic), reading the SAME `g1_nav/config/patrol.yaml` tuning values so sim and
    the eventual real supervisor share not just code, but numbers.

DESIGN DECISION -- kinematic "puppeted" sim, not dynamic balance (per the plan's
"Cross-workstream decisions" #2): the floating base pose is integrated directly
from the commanded (vx, vy, vyaw), exactly like scripts/sim_walk.py already does
("commanded velocity == achieved velocity"). Full RL-trained balanced walking in
mujoco is explicitly out of scope -- a separate research-scale effort.

DESIGN DECISION -- mj_forward, not mj_step: a naive `mj_step` on this model (real
gravity + no active balance controller) just free-falls the floating base -- there
is no PD/balance controller driving the legs to hold up 29 DOF (verified: a
bare-loop mj_step test on this model free-falls to qpos[2] ~ -490 in 5000 steps
with no ground contact resolving it). Since the whole point here is a PUPPETED
pose (we already know x,y,yaw and want a walk-cycle animation, not physics), we
set qpos directly every tick (base pose from the integrated patrol command, leg
joints from a procedural walk-cycle keyed to |v|) and call `mujoco.mj_forward()`
-- this recomputes kinematics/contacts from the given qpos without integrating
any dynamics. Simpler, matches the plan's explicitly-offered v1 fallback ("use
mujoco purely as a kinematic visualizer/model-poser"), and sidesteps needing a
balance controller entirely.

DESIGN DECISION -- synthetic person is DATA, not a second mujoco body: adding a
second free body to the vendored G1 MJCF would mean either hand-editing the
fetched (gitignored, re-fetchable) upstream XML -- which would silently diverge
from upstream on the next `sim/fetch_model.sh` -- or using mujoco's mjSpec model-
surgery API to splice one in at runtime, which is more moving parts than this
v1 needs. The interaction logic only needs bearing/distance/appearance timing to
react (exactly what a real `g1_person_track.json` reader would hand it), so the
person is modeled as a plain (x, y) point with an appear/disappear window,
exposed as {"track_id", "x", "y", "bearing_rad", "distance_m"} -- structurally
identical to the real per-person schema, just synthetic.

SAFETY PROPERTY (load-bearing, double-checked): this script NEVER opens any
connection to the real robot. No `unitree_sdk2py`/`LocoClient` import, no rclpy/
ROS2 node, no DDS traffic (domain 0 or 99). It reads/writes ONLY local files
(the mujoco model on disk, config/patrol.yaml, `/dev/shm/g1_sim_state.json`).
Grep this file for "^import\|^from" and confirm every import is either stdlib,
mujoco, yaml, or `g1_nav.patrol_logic` (a plain-Python, no-ROS-dependency
module, confirmed by reading it: no rclpy/ROS import anywhere in that file) if
you are re-verifying this claim.

Run (always via the isolated venv -- it has the pinned mujoco, not system python):
    sim/.venv/bin/python3 scripts/sim_runner.py                # live, writes shm @ ~15 Hz
    sim/.venv/bin/python3 scripts/sim_runner.py --demo-person   # + a repeating synthetic person
    sim/.venv/bin/python3 scripts/sim_runner.py --demo-dance-request  # + a periodic dance request
    sim/.venv/bin/python3 scripts/sim_runner.py --no-exhibition  # plain circumference pursuit only
    sim/.venv/bin/python3 scripts/sim_runner.py --selftest      # bounded, asserts + exits

Exhibition mode (Track A Extension, on by default in live mode): while
PatrolStateMachine says "circling", drives the SAME roam/gesture/(on-request)
dance repertoire as the real g1_nav/g1_nav/nodes/exhibition_conductor.py node,
using its identical g1_nav.patrol_logic.plan_next_segment/predict_segment_end
oracle (see ExhibitionDriver below) -- not a reimplementation. `activity` in
the shm state dict ("roam" / "gesture:<name>" / "dance:<fsm_id>" / "idle" /
null) mirrors the real conductor's own vocabulary so web/nav.js renders both
identically.
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SHM_PATH = "/dev/shm/g1_sim_state.json"
MODEL_DIR = os.path.join(ROOT, "sim", "models", "g1")
MODEL_PATHS = [os.path.join(MODEL_DIR, "scene.xml"), os.path.join(MODEL_DIR, "g1.xml")]
PATROL_YAML = os.path.join(ROOT, "g1_nav", "config", "patrol.yaml")

STAND_HEIGHT_M = 0.79          # from the model's "stand" keyframe (qpos[2])
WRITE_HZ_DEFAULT = 15.0        # within the requested ~10-20 Hz shm write rate
SPEED_MPS_DEFAULT = 0.35       # sim-only: no equivalent tunable in patrol.yaml
                                # (real robot's circling speed is DWB/Nav2's job,
                                # not this repo's geometry layer)


def _wrap(a):
    """Wrap an angle to (-pi, pi]. Deliberately a local duplicate, not an
    import -- matches this codebase's established convention of NOT cross-
    importing this one-liner (see circle_patrol.py's `_quat_to_yaw` comment
    and patrol_logic.py's own `wrap()` docstring, both making the same
    choice)."""
    v = math.fmod(a + math.pi, 2.0 * math.pi)
    if v <= 0.0:
        v += 2.0 * math.pi
    return v - math.pi


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# -----------------------------------------------------------------------------
# Real dependency: g1_nav.patrol_logic (CirclePatrolGeometry, PatrolStateMachine).
#
# This is a PARALLEL, independently-developed module (Track A, per the plan) that
# may not exist yet, or may land with a slightly different call surface than
# assumed here. Imported for real when present -- the fallback below is NOT a
# permanent reimplementation. It exists only so this script doesn't hard-crash if
# g1_nav/g1_nav/patrol_logic.py is missing/renamed, and to pin down (in code) the
# exact interface this script needs.
#
# Import path note: g1_nav is a ROS2 ament_python package. Its source layout is
# <repo>/g1_nav/setup.py + <repo>/g1_nav/g1_nav/__init__.py (outer dir = ROS
# package source root, inner same-named dir = the actual importable Python
# package -- confirmed via g1_nav/setup.py's `packages=['g1_nav', 'g1_nav.nodes']`
# and its entry_points referencing `g1_nav.nodes.<mod>:main`). So we add the
# OUTER g1_nav/ directory to sys.path and import plain `g1_nav.patrol_logic`.
#
# EXPECTED INTERFACE (as actually landed 2026-07-23, confirmed by reading
# g1_nav/g1_nav/patrol_logic.py -- the fallback below mirrors it exactly so
# everything downstream in this file works identically regardless of which
# is active):
#
#   @dataclass
#   class CirclePatrolGeometry:
#       radius_m: float = 3.0
#       lookahead_m: float = 1.25
#       direction: int = 1                     # +1 CCW, -1 CW
#       center: Optional[Tuple[float, float]] = None
#       anchored: bool = False
#       def re_anchor_if_needed(self, x, y, yaw, force=False) -> bool:
#           """Fix the circle center from the CURRENT pose, ONCE. No-op after
#           the first call unless force=True."""
#       def carrot_point(self, x, y, lookahead_m=None) -> (float, float):
#           """Pure-pursuit carrot point ON the circle."""
#       def carrot_pose(self, x, y, lookahead_m=None) -> (float, float, float):
#           """carrot_point() + a tangent heading (rad)."""
#       def radial_drift_m(self, x, y) -> float:
#           """|distance-from-center - radius_m| -- diagnostic only."""
#       def reset(self) -> None: ...
#
#   STATES: circling, pausing, facing, waving, waiting, resuming, fault,
#           manual_override, paused_odom_degraded
#   @dataclass
#   class PatrolStateMachine:
#       """PURE transition-table FSM -- no timers, no sensing. The CALLER
#       decides WHEN to call try_transition(); this class only validates
#       WHETHER that move is legal."""
#       state: str = "circling"
#       history: List[str] = field(default_factory=list)
#       def can_transition(self, to_state) -> bool: ...
#       def transition(self, to_state) -> str:
#           """Raises InvalidTransition if illegal."""
#       def try_transition(self, to_state) -> bool:
#           """Non-raising: True on success, False (unchanged) if illegal."""
#       def reset(self, state="circling") -> None: ...
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "g1_nav"))
try:
    from g1_nav.patrol_logic import (          # noqa: E402
        CirclePatrolGeometry, PatrolStateMachine,
        ExhibitionSegment, predict_segment_end, plan_next_segment,
    )
    PATROL_LOGIC_SOURCE = "g1_nav.patrol_logic (real, imported)"
except Exception as _import_err:  # noqa: BLE001 -- any import-time failure -> fallback
    PATROL_LOGIC_SOURCE = "FALLBACK STUB -- g1_nav.patrol_logic not importable (%r)" % (_import_err,)

    from dataclasses import dataclass, field
    from typing import List, Optional, Tuple

    @dataclass
    class CirclePatrolGeometry:
        """FALLBACK stub mirroring g1_nav.patrol_logic.CirclePatrolGeometry."""
        radius_m: float = 3.0
        lookahead_m: float = 1.25
        direction: int = 1
        center: Optional[Tuple[float, float]] = None
        anchored: bool = False

        def __post_init__(self):
            if self.radius_m <= 0.0:
                raise ValueError("radius_m must be > 0")
            if self.direction not in (1, -1):
                raise ValueError("direction must be +1 (CCW) or -1 (CW)")

        def re_anchor_if_needed(self, x, y, yaw, force=False):
            if self.anchored and not force:
                return False
            side = 1.0 if self.direction >= 0 else -1.0
            heading = yaw + side * (math.pi / 2.0)
            self.center = (x + self.radius_m * math.cos(heading),
                           y + self.radius_m * math.sin(heading))
            self.anchored = True
            return True

        def _require_anchor(self):
            if not self.anchored or self.center is None:
                raise RuntimeError("CirclePatrolGeometry: call re_anchor_if_needed() first")

        def phase_angle(self, x, y):
            self._require_anchor()
            cx, cy = self.center
            return math.atan2(y - cy, x - cx)

        def carrot_point(self, x, y, lookahead_m=None):
            self._require_anchor()
            la = self.lookahead_m if lookahead_m is None else lookahead_m
            theta2 = self.phase_angle(x, y) + self.direction * (la / self.radius_m)
            cx, cy = self.center
            return (cx + self.radius_m * math.cos(theta2), cy + self.radius_m * math.sin(theta2))

        def carrot_pose(self, x, y, lookahead_m=None):
            self._require_anchor()
            la = self.lookahead_m if lookahead_m is None else lookahead_m
            theta = self.phase_angle(x, y)
            theta2 = theta + self.direction * (la / self.radius_m)
            cx, cy = self.center
            px = cx + self.radius_m * math.cos(theta2)
            py = cy + self.radius_m * math.sin(theta2)
            heading = theta2 + self.direction * (math.pi / 2.0)
            return (px, py, _wrap(heading))

        def radial_drift_m(self, x, y):
            self._require_anchor()
            cx, cy = self.center
            return abs(math.hypot(x - cx, y - cy) - self.radius_m)

        def reset(self):
            self.center = None
            self.anchored = False

    STATES = ("circling", "pausing", "facing", "waving", "waiting", "resuming",
              "fault", "manual_override", "paused_odom_degraded")
    STATES_SET = frozenset(STATES)
    _HAPPY_PATH = {"circling": ("pausing",), "pausing": ("facing",), "facing": ("waving",),
                   "waving": ("waiting",), "waiting": ("resuming",), "resuming": ("circling",)}
    _INTERRUPT_STATES = ("manual_override", "paused_odom_degraded", "fault")
    _INTERRUPT_EXITS = {"manual_override": ("circling",), "paused_odom_degraded": ("circling",),
                        "fault": ("circling",)}

    class InvalidTransition(RuntimeError):
        pass

    @dataclass
    class PatrolStateMachine:
        """FALLBACK stub mirroring g1_nav.patrol_logic.PatrolStateMachine."""
        state: str = "circling"
        history: List[str] = field(default_factory=list)

        def __post_init__(self):
            if self.state not in STATES_SET:
                raise ValueError(f"unknown initial state {self.state!r}")
            if not self.history:
                self.history = [self.state]

        def allowed_next(self):
            nxt = set(_INTERRUPT_STATES)
            nxt.discard(self.state)
            if self.state in _HAPPY_PATH:
                nxt.update(_HAPPY_PATH[self.state])
            if self.state in _INTERRUPT_EXITS:
                nxt.update(_INTERRUPT_EXITS[self.state])
            return nxt

        def can_transition(self, to_state):
            if to_state not in STATES_SET:
                return False
            return to_state == self.state or to_state in self.allowed_next()

        def transition(self, to_state):
            if not self.can_transition(to_state):
                raise InvalidTransition(f"illegal patrol transition: {self.state!r} -> {to_state!r}")
            self.state = to_state
            self.history.append(to_state)
            return self.state

        def try_transition(self, to_state):
            try:
                self.transition(to_state)
                return True
            except InvalidTransition:
                return False

        def reset(self, state="circling"):
            self.state = state
            self.history = [state]

    # -------------------------------------------------------------------
    # FALLBACK STUBS -- Exhibition mode (ExhibitionSegment, predict_segment_end,
    # plan_next_segment). Mirrors g1_nav.patrol_logic's "Exhibition mode" section
    # (as landed 2026-07-23) exactly enough that everything downstream in this
    # file works identically regardless of which branch is active. See that
    # module for the authoritative docstrings/design rationale -- these are
    # deliberately terse copies, not a divergent reimplementation.
    # -------------------------------------------------------------------
    @dataclass
    class ExhibitionSegment:
        kind: str
        target: Optional[Tuple[float, float]] = None
        gesture: Optional[str] = None
        dance_fsm_id: Optional[int] = None
        dance_name: Optional[str] = None
        dance_space_m: Optional[float] = None
        duration_s: float = 0.0

    def _stub_in_safe_disc(pt, center, radius_m, margin_m):
        cx, cy = center
        return math.hypot(pt[0] - cx, pt[1] - cy) <= max(radius_m - margin_m, 0.0)

    def _stub_random_point_in_disc(center, radius_m, edge_clearance_m, rng,
                                    min_dist_from=None, min_dist_m=0.0, max_attempts=20):
        cx, cy = center
        safe_r = max(radius_m - edge_clearance_m, 0.0)
        for _ in range(max_attempts):
            r = safe_r * math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            if (min_dist_from is not None
                    and math.hypot(x - min_dist_from[0], y - min_dist_from[1]) < min_dist_m):
                continue
            return (x, y)
        return (cx, cy)

    def predict_segment_end(pose, segment, center, radius_m, edge_clearance_m,
                             roam_speed_mps=0.5):
        x, y, yaw = pose
        if segment.kind == "roam":
            tx, ty = segment.target
            if not _stub_in_safe_disc((tx, ty), center, radius_m, edge_clearance_m):
                return pose, False, "roam target outside safety margin"
            return (tx, ty, math.atan2(ty - y, tx - x)), True, "ok"
        if segment.kind == "dance":
            margin = max(edge_clearance_m, segment.dance_space_m or edge_clearance_m)
            if not _stub_in_safe_disc((x, y), center, radius_m, margin):
                return pose, False, "insufficient clearance for dance space_m"
            return pose, True, "ok"
        if segment.kind in ("gesture", "idle"):
            if not _stub_in_safe_disc((x, y), center, radius_m, edge_clearance_m):
                return pose, False, "current position outside safety margin"
            return pose, True, "ok"
        return pose, False, f"unknown segment kind {segment.kind!r}"

    def _stub_draw_weighted_kind(cfg_segments, rng):
        total = sum(max(0.0, s.get("weight", 0.0)) for s in cfg_segments)
        if total <= 0.0:
            return None
        r = rng.uniform(0.0, total)
        upto = 0.0
        for s in cfg_segments:
            upto += max(0.0, s.get("weight", 0.0))
            if r <= upto:
                return s
        return cfg_segments[-1]

    def _stub_draw_candidate_segment(pose, cfg, rng, pending_dance=None):
        if pending_dance is not None:
            return pending_dance
        picked = _stub_draw_weighted_kind(cfg["segments"], rng)
        if picked is None:
            return ExhibitionSegment(kind="idle", duration_s=cfg.get("idle_duration_s", 1.0))
        kind = picked.get("kind")
        if kind == "roam":
            target = _stub_random_point_in_disc(
                cfg["center"], cfg["radius_m"], cfg["edge_clearance_m"], rng,
                min_dist_from=(pose[0], pose[1]), min_dist_m=cfg.get("roam_min_dist_m", 0.0))
            speed = max(cfg.get("roam_speed_mps", 0.5), 1e-3)
            duration = math.hypot(target[0] - pose[0], target[1] - pose[1]) / speed
            return ExhibitionSegment(kind="roam", target=target, duration_s=duration)
        if kind == "gesture":
            choices = picked.get("choices") or ["high_wave"]
            gesture = choices[rng.randrange(len(choices))]
            return ExhibitionSegment(kind="gesture", gesture=gesture,
                                      duration_s=cfg.get("gesture_duration_s", 3.5))
        if kind == "idle":
            return ExhibitionSegment(kind="idle", duration_s=cfg.get("idle_duration_s", 1.0))
        return ExhibitionSegment(kind="idle", duration_s=cfg.get("idle_duration_s", 1.0))

    def plan_next_segment(pose, cfg, rng, pending_dance=None):
        lookahead = max(1, int(cfg.get("lookahead_segments", 5)))
        max_attempts = max(1, int(cfg.get("max_reroll_attempts", 8)))

        for attempt in range(max_attempts):
            first = _stub_draw_candidate_segment(
                pose, cfg, rng, pending_dance if attempt == 0 else None)
            chain_pose, ok, _reason = predict_segment_end(
                pose, first, cfg["center"], cfg["radius_m"], cfg["edge_clearance_m"],
                cfg.get("roam_speed_mps", 0.5))
            if not ok:
                continue
            ok_chain = True
            for _ in range(lookahead - 1):
                nxt = _stub_draw_candidate_segment(chain_pose, cfg, rng, None)
                chain_pose, ok, _reason = predict_segment_end(
                    chain_pose, nxt, cfg["center"], cfg["radius_m"], cfg["edge_clearance_m"],
                    cfg.get("roam_speed_mps", 0.5))
                if not ok:
                    ok_chain = False
                    break
            if ok_chain:
                return first

        cx, cy = cfg["center"]
        speed = max(cfg.get("roam_speed_mps", 0.5), 1e-3)
        return ExhibitionSegment(kind="roam", target=(cx, cy),
                                  duration_s=math.hypot(pose[0] - cx, pose[1] - cy) / speed)


# -----------------------------------------------------------------------------
# mujoco (isolated venv only -- import mujoco alone creates NO GL/EGL/display
# context; verified headless with no DISPLAY set, see sim/models/README.md).
# -----------------------------------------------------------------------------
try:
    import mujoco
except ImportError:
    print("ERROR: `mujoco` is not importable. This script must be run via the "
          "isolated sim venv, not system python3:\n"
          "    bash sim/setup_venv.sh   # one-time setup\n"
          "    sim/.venv/bin/python3 scripts/sim_runner.py\n", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None   # config/patrol.yaml becomes optional -> baked defaults only


def load_model():
    """Loads the G1 mujoco model, with a clear guard if it hasn't been fetched."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return mujoco.MjModel.from_xml_path(path), path
    print("ERROR: no G1 mujoco model found under %s\n"
          "  Fetch it with:  bash sim/fetch_model.sh\n"
          "  (see sim/models/README.md for provenance + license notes)\n"
          % MODEL_DIR, file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# config/patrol.yaml -- SAME tuning file the (not-yet-written) real
# patrol_supervisor.py is designed to read (see that file's docstring-in-waiting
# via g1_nav/config/patrol.yaml's own header comment). Baked defaults so a
# missing/partial file never crashes this script, mirroring circle_patrol.py's
# load_cfg() idiom exactly.
# -----------------------------------------------------------------------------
_CIRCLE_DEFAULTS = {"radius_m": 3.0, "lookahead_m": 1.25, "direction": 1}
_SUPERVISOR_DEFAULTS = {
    "person_debounce_s": 1.0, "person_lost_grace_s": 2.0, "yaw_kp": 1.2,
    "yaw_max_rad_s": 0.6, "yaw_tolerance_rad": 0.09, "yaw_settle_s": 0.3,
    "facing_timeout_s": 6.0, "wave_busy_s": 3.5, "waiting_timeout_s": 8.0,
}
_PERSON_FUSION_DEFAULTS = {"close_distance_m": 1.6}
PAUSE_SETTLE_S = 1.0   # not in patrol.yaml (that file has no pausing-dwell key);
                       # matches scripts/sim_circle.py's PatrolConfig.pause_settle_s
                       # default for consistency with that sibling Track A0 harness.

# Track A Extension -- Exhibition mode defaults. These MATCH g1_nav/config/
# patrol.yaml's real `exhibition:` block verbatim (confirmed present in this repo
# as of 2026-07-23) -- baked here only so a missing/partial yaml file never
# crashes this script, exactly like the three dicts above.
_EXHIBITION_DEFAULTS = {
    "enabled": True,
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
}
_EXHIBITION_SEGMENTS_DEFAULT = [
    {"kind": "roam", "weight": 0.45},
    {"kind": "gesture", "weight": 0.35,
     "choices": ["high_wave", "high_five", "heart", "clap", "kiss"]},
    {"kind": "idle", "weight": 0.20},
]


def load_patrol_cfg():
    circle = dict(_CIRCLE_DEFAULTS)
    supervisor = dict(_SUPERVISOR_DEFAULTS)
    person_fusion = dict(_PERSON_FUSION_DEFAULTS)
    exhibition = dict(_EXHIBITION_DEFAULTS)
    exhibition["segments"] = [dict(s) for s in _EXHIBITION_SEGMENTS_DEFAULT]
    if yaml is not None:
        try:
            with open(PATROL_YAML) as f:
                data = yaml.safe_load(f) or {}
            for key, dst in (("circle_patrol", circle), ("patrol_supervisor", supervisor),
                              ("person_fusion", person_fusion), ("exhibition", exhibition)):
                sub = data.get(key) if isinstance(data, dict) else None
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        if k in dst and v is not None:
                            dst[k] = v
        except (OSError, Exception):  # noqa: BLE001 -- never let a bad yaml crash the sim
            pass
    return circle, supervisor, person_fusion, exhibition


# -----------------------------------------------------------------------------
# Steering law -- NOT part of g1_nav.patrol_logic (that module is pure geometry/
# FSM, see module docstring). Mirrors scripts/sim_circle.py's pursuit_command()
# / scripts/sim_walk.py's policy_goal(): face the target with yaw, drive forward
# into it, ease vx off while badly misaligned.
# -----------------------------------------------------------------------------
def pursuit_command(x, y, yaw, target_xy, max_vx, max_vyaw, k_yaw=1.6):
    tx, ty = target_xy
    bearing = _wrap(math.atan2(ty - y, tx - x) - yaw)
    vyaw = _clamp(k_yaw * bearing, -max_vyaw, max_vyaw)
    vx = max_vx * max(0.0, math.cos(bearing))
    return (vx, 0.0, vyaw)


# -----------------------------------------------------------------------------
# InteractionSupervisor -- the sim-side equivalent of the not-yet-written
# patrol_supervisor.py: decides WHEN to fire PatrolStateMachine transitions
# (the FSM itself only validates WHETHER a move is legal). Reads the SAME
# g1_nav/config/patrol.yaml tuning the real supervisor is designed around.
# -----------------------------------------------------------------------------
class InteractionSupervisor:
    def __init__(self, patrol, supervisor_cfg, person_fusion_cfg):
        self.patrol = patrol
        self.close_distance_m = person_fusion_cfg["close_distance_m"]
        self.person_debounce_s = supervisor_cfg["person_debounce_s"]
        self.person_lost_grace_s = supervisor_cfg["person_lost_grace_s"]
        self.yaw_kp = supervisor_cfg["yaw_kp"]
        self.yaw_max_rad_s = supervisor_cfg["yaw_max_rad_s"]
        self.yaw_tolerance_rad = supervisor_cfg["yaw_tolerance_rad"]
        self.yaw_settle_s = supervisor_cfg["yaw_settle_s"]
        self.facing_timeout_s = supervisor_cfg["facing_timeout_s"]
        self.wave_busy_s = supervisor_cfg["wave_busy_s"]
        self.waiting_timeout_s = supervisor_cfg["waiting_timeout_s"]

        self._phase_t = 0.0
        self._close_since = None
        self._settle_since = None
        self._lost_since = None
        self.face_target = 0.0
        self.pre_pause_yaw = 0.0

    def _enter(self, to_state):
        if self.patrol.try_transition(to_state):
            self._phase_t = 0.0
            self._settle_since = None
            self._lost_since = None
            return True
        return False

    def update(self, dt, x, y, yaw, nearest_person):
        """nearest_person: {"bearing_rad","distance_m",...} or None. Returns
        nothing -- inspect self.patrol.state afterward."""
        self._phase_t += dt
        state = self.patrol.state

        if state == "circling":
            close = nearest_person is not None and nearest_person["distance_m"] <= self.close_distance_m
            if close:
                if self._close_since is None:
                    self._close_since = self._phase_t
                if self._phase_t - self._close_since >= self.person_debounce_s:
                    self.pre_pause_yaw = yaw
                    self.face_target = _wrap(yaw + nearest_person["bearing_rad"])
                    self._enter("pausing")
            else:
                self._close_since = None

        elif state == "pausing":
            if self._phase_t >= PAUSE_SETTLE_S:
                self._enter("facing")

        elif state == "facing":
            if nearest_person is not None:
                self.face_target = _wrap(yaw + nearest_person["bearing_rad"])
            err = abs(_wrap(self.face_target - yaw))
            if err <= self.yaw_tolerance_rad:
                if self._settle_since is None:
                    self._settle_since = self._phase_t
                if self._phase_t - self._settle_since >= self.yaw_settle_s:
                    self._enter("waving")
            else:
                self._settle_since = None
            if self.patrol.state == "facing" and self._phase_t >= self.facing_timeout_s:
                self._enter("waving")   # give up converging, wave anyway

        elif state == "waving":
            if self._phase_t >= self.wave_busy_s:
                self._enter("waiting")

        elif state == "waiting":
            if nearest_person is None:
                if self._lost_since is None:
                    self._lost_since = self._phase_t
                if self._phase_t - self._lost_since >= self.person_lost_grace_s:
                    self.face_target = self.pre_pause_yaw
                    self._enter("resuming")
            else:
                self._lost_since = None
            if self.patrol.state == "waiting" and self._phase_t >= self.waiting_timeout_s:
                self.face_target = self.pre_pause_yaw
                self._enter("resuming")

        elif state == "resuming":
            if abs(_wrap(self.face_target - yaw)) <= self.yaw_tolerance_rad:
                self._enter("circling")

    def command(self, x, y, yaw, geometry, speed_mps):
        state = self.patrol.state
        if state == "circling":
            geometry.re_anchor_if_needed(x, y, yaw)   # one-time; no-op after the first call
            carrot = geometry.carrot_point(x, y)
            return pursuit_command(x, y, yaw, carrot, speed_mps, max_vyaw=1.5, k_yaw=1.6)
        if state in ("facing", "resuming"):
            err = _wrap(self.face_target - yaw)
            vyaw = _clamp(self.yaw_kp * err, -self.yaw_max_rad_s, self.yaw_max_rad_s)
            return (0.0, 0.0, vyaw)
        return (0.0, 0.0, 0.0)   # pausing, waving, waiting, fault, manual_override, paused_odom_degraded


# -----------------------------------------------------------------------------
# ExhibitionDriver -- Track A Extension (Exhibition Mode). Sim-side equivalent of
# the not-yet-landed g1_nav/g1_nav/nodes/exhibition_conductor.py: while
# PatrolStateMachine says "circling" (i.e. NOT mid-person-interaction), this
# drives roam/gesture/idle/(on-request)dance segments using the SAME
# `plan_next_segment`/`predict_segment_end` oracle the real conductor uses,
# instead of plain circumference pursuit. It runs ALONGSIDE, not in place of,
# InteractionSupervisor: `run()`'s main loop only ever consults this driver
# while state == "circling" and defers to `supervisor.command()` otherwise --
# the exact same gate plain circling already yields through (see
# InteractionSupervisor.command()'s own "if state == 'circling'" branch). A
# segment in progress is abandoned (not paused/resumed) the moment circling is
# yielded -- a fresh decision is made from wherever the robot ends up once
# circling resumes, which mirrors how plain carrot-pursuit itself has no
# persisted state to resume either (it just recomputes the carrot each tick).
# -----------------------------------------------------------------------------
class ExhibitionDriver:
    def __init__(self, cfg, rng):
        """cfg must already carry "center" (x,y) and "radius_m" alongside the
        g1_nav/config/patrol.yaml `exhibition:` block's own keys (edge_clearance_m,
        roam_speed_mps, roam_min_dist_m, lookahead_segments, max_reroll_attempts,
        segments, gesture_duration_s, idle_duration_s, goal_timeout_slack_s,
        dance_default_duration_s) -- see run()'s construction of this dict."""
        self.cfg = cfg
        self.rng = rng
        self.segment = None
        self.seg_elapsed = 0.0
        self.activity = None          # "roam" / "gesture:<name>" / "idle" / "dance:<id>" / None
        self.pending_dance = None     # an ExhibitionSegment(kind="dance", ...) or None

    def request_dance(self, fsm_id=503, name="Dance Mode", space_m=2.0, duration_s=None):
        """Queues an operator-requested dance, mirroring the real conductor's
        poll of /dev/shm/g1_exhibition_dance_request.json (sim has no dashboard
        WS request to actually read, so --demo-dance-request calls this
        directly on a timer -- see main()). Overwrites any not-yet-consumed
        prior request, matching "the operator pressed the button again"."""
        self.pending_dance = ExhibitionSegment(
            kind="dance", dance_fsm_id=fsm_id, dance_name=name, dance_space_m=space_m,
            duration_s=(duration_s if duration_s is not None
                        else self.cfg.get("dance_default_duration_s", 8.0)))

    def reset(self):
        """Abandon any in-progress segment -- called by run() the instant
        circling is yielded to a person interaction."""
        self.segment = None
        self.seg_elapsed = 0.0
        self.activity = None

    @staticmethod
    def _activity_str(seg):
        if seg is None:
            return None
        if seg.kind == "gesture":
            return f"gesture:{seg.gesture}"
        if seg.kind == "dance":
            return f"dance:{seg.dance_fsm_id}"
        return seg.kind   # "roam" / "idle"

    def update(self, dt, x, y, yaw):
        """Called only while patrol.state == 'circling'. Returns (vx, vy, vyaw)
        in the same body frame convention as pursuit_command()/supervisor.command()."""
        if self.segment is None:
            self.segment = plan_next_segment((x, y, yaw), self.cfg, self.rng,
                                              pending_dance=self.pending_dance)
            if self.pending_dance is not None and self.segment.kind == "dance":
                self.pending_dance = None   # request consumed -- see plan_next_segment's
                                            # own docstring: an unsafe-right-now request is
                                            # NOT cleared here (it falls through to a normal
                                            # draw and stays queued for the caller to re-offer)
            self.seg_elapsed = 0.0
            self.activity = self._activity_str(self.segment)

        seg = self.segment
        self.seg_elapsed += dt

        if seg.kind == "roam":
            tx, ty = seg.target
            arrived = math.hypot(tx - x, ty - y) < 0.1
            stuck = self.seg_elapsed >= seg.duration_s + self.cfg.get("goal_timeout_slack_s", 5.0)
            if arrived or stuck:
                self.segment = None
                return (0.0, 0.0, 0.0)
            speed = self.cfg.get("roam_speed_mps", 0.5)
            return pursuit_command(x, y, yaw, (tx, ty), speed, max_vyaw=1.5, k_yaw=1.6)

        # gesture / idle / dance: stationary hold for duration_s.
        if self.seg_elapsed >= seg.duration_s:
            self.segment = None
        return (0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Procedural walk-cycle animation -- VISUAL PLAUSIBILITY ONLY, not a balance
# controller (explicitly out of scope, see module docstring). Simple sinusoidal
# hip/knee/ankle gait, amplitude scaled by |v|, frozen (amplitude 0) at rest. A
# small decorative arm-raise plays during the "waving" phase.
# -----------------------------------------------------------------------------
LEG_JOINTS = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
              "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"]
ARM_WAVE_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"]

WALK_CADENCE_HZ = 1.4          # stride frequency at full patrol speed
HIP_AMP_RAD = 0.35
KNEE_AMP_RAD = 0.55
ANKLE_COUNTER = 0.5            # ankle counters hip pitch to keep the foot roughly flat


class Puppet:
    """Owns the mujoco model/data and pushes a kinematic pose into it each tick
    via mj_forward (NOT mj_step -- see module docstring for why)."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        # base qpos addr is always 0 for a freejoint declared first in the tree
        # (confirmed: model.joint("floating_base_joint").qposadr == [0]).
        self.base_adr = int(model.joint("floating_base_joint").qposadr[0])
        self.leg_adr = {n: int(model.joint(n).qposadr[0]) for n in LEG_JOINTS}
        all_names = [model.joint(i).name for i in range(model.njnt)]
        self.arm_adr = {n: int(model.joint(n).qposadr[0]) for n in ARM_WAVE_JOINTS if n in all_names}
        self._gait_phase = 0.0

    def step(self, x, y, yaw, speed_mps, dt, waving=False):
        qpos = self.data.qpos
        b = self.base_adr
        qpos[b + 0] = x
        qpos[b + 1] = y
        qpos[b + 2] = STAND_HEIGHT_M
        half = yaw * 0.5
        qpos[b + 3] = math.cos(half)   # quat w
        qpos[b + 4] = 0.0              # x
        qpos[b + 5] = 0.0              # y
        qpos[b + 6] = math.sin(half)   # z

        amp_scale = min(1.0, abs(speed_mps) / max(1e-6, SPEED_MPS_DEFAULT))
        self._gait_phase += 2.0 * math.pi * WALK_CADENCE_HZ * amp_scale * dt
        ph = self._gait_phase

        def leg(phase_offset):
            hip = amp_scale * HIP_AMP_RAD * math.sin(ph + phase_offset)
            knee = amp_scale * KNEE_AMP_RAD * max(0.0, math.sin(ph + phase_offset))
            ankle = -ANKLE_COUNTER * hip
            return hip, knee, ankle

        lh, lk, la = leg(0.0)
        rh, rk, ra = leg(math.pi)
        qpos[self.leg_adr["left_hip_pitch_joint"]] = lh
        qpos[self.leg_adr["left_knee_joint"]] = lk
        qpos[self.leg_adr["left_ankle_pitch_joint"]] = la
        qpos[self.leg_adr["right_hip_pitch_joint"]] = rh
        qpos[self.leg_adr["right_knee_joint"]] = rk
        qpos[self.leg_adr["right_ankle_pitch_joint"]] = ra

        if waving and self.arm_adr:
            wave = math.sin(6.0 * time.monotonic())
            if "right_shoulder_pitch_joint" in self.arm_adr:
                qpos[self.arm_adr["right_shoulder_pitch_joint"]] = -1.2
            if "right_shoulder_roll_joint" in self.arm_adr:
                qpos[self.arm_adr["right_shoulder_roll_joint"]] = -0.3
            if "right_elbow_joint" in self.arm_adr:
                qpos[self.arm_adr["right_elbow_joint"]] = 1.0 + 0.3 * wave

        mujoco.mj_forward(self.model, self.data)


# -----------------------------------------------------------------------------
# Synthetic person (DATA only -- see module docstring for why not a mujoco body).
# -----------------------------------------------------------------------------
class SyntheticPerson:
    def __init__(self, px, py, appear_at, disappear_at, track_id=1):
        self.px, self.py = px, py
        self.appear_at = appear_at
        self.disappear_at = disappear_at
        self.track_id = track_id

    def active(self, t):
        return self.appear_at <= t < self.disappear_at

    def observation(self, t, x, y, yaw):
        if not self.active(t):
            return None
        dx, dy = self.px - x, self.py - y
        return {"track_id": self.track_id, "x": self.px, "y": self.py,
                "bearing_rad": _wrap(math.atan2(dy, dx) - yaw), "distance_m": math.hypot(dx, dy)}


# -----------------------------------------------------------------------------
# shm write -- atomic tmp-file + os.replace, mirroring obstacle/obstacle_node.py's
# _write_shm idiom exactly (a reader must never see a partial file).
# -----------------------------------------------------------------------------
def write_shm(state):
    tmp = SHM_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, separators=(",", ":"))
        os.replace(tmp, SHM_PATH)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# main sim loop (shared by live mode and --selftest)
# -----------------------------------------------------------------------------
def run(duration_s, dt, radius_m, speed_mps, write_hz, person_schedule=None,
        realtime=False, record=False, exhibition=True, dance_schedule=None,
        exhibition_seed=None, use_viewer=False):
    """Drives the puppeted sim for `duration_s` sim-seconds (or forever if None,
    live mode only). `person_schedule(t, x, y, yaw) -> SyntheticPerson-or-None`
    lets callers script a synthetic person appearance (used by --demo-person and
    --selftest); returns a log dict if record=True.

    `exhibition`: while PatrolStateMachine says "circling" (i.e. not mid-person-
    interaction), drive the Track A Extension roam/gesture/(on-request)dance
    repertoire via g1_nav.patrol_logic's plan_next_segment/predict_segment_end
    oracle instead of plain circumference pursuit (see ExhibitionDriver above).
    True by default (this is what makes Sim mode preview the exhibition
    behavior); the first --selftest run passes False to keep exercising the
    original plain-circling radius-tracking check unmodified. Also gated by
    g1_nav/config/patrol.yaml's own `exhibition.enabled` (a config-level kill
    switch wins over a True argument here, mirroring circle_patrol.py's
    fallback-node launch-time choice -- see that file's own docstring).

    `dance_schedule(t) -> ExhibitionSegment-or-None` lets callers script a
    periodic operator dance request (used by --demo-dance-request and
    --selftest), analogous to `person_schedule`. Only polled while exhibition
    is active and no request is already queued (mirrors "the conductor holds
    one pending request at a time").

    `exhibition_seed`: seeds the exhibition oracle's own RNG (kept SEPARATE
    from any RNG person_schedule/dance_schedule callers might use) so
    --selftest runs are reproducible.

    `use_viewer`: open mujoco's OWN real interactive 3D viewer (the actual
    G1 model, actual mujoco rendering/lighting -- not a hand-built stand-in)
    via `mujoco.viewer.launch_passive`, and push this function's kinematic
    pose into it every tick with `viewer.sync()`. This needs a real display
    (GLFW/a window system) -- it will NOT work over a bare SSH session with no
    X forwarding, which is why this is meant to be run on a normal computer
    with a screen attached, not the (headless, over SSH) Jetson. Forces
    `realtime` semantics regardless of the `realtime` argument (an interactive
    viewer must run at wall-clock pace to look like real motion, not a shm-
    write-rate throttle) and stops cleanly the moment the viewer window is
    closed, in addition to the usual Ctrl-C."""
    if use_viewer:
        realtime = True
    model, model_path = load_model()
    puppet = Puppet(model)

    circle_cfg, supervisor_cfg, person_fusion_cfg, exhibition_cfg_raw = load_patrol_cfg()
    lookahead_m = circle_cfg["lookahead_m"]
    direction = circle_cfg["direction"]

    geometry = CirclePatrolGeometry(radius_m=radius_m, lookahead_m=lookahead_m, direction=direction)
    patrol = PatrolStateMachine()
    supervisor = InteractionSupervisor(patrol, supervisor_cfg, person_fusion_cfg)

    x = y = yaw = 0.0
    geometry.re_anchor_if_needed(x, y, yaw)

    exhibition_driver = None
    if exhibition and exhibition_cfg_raw.get("enabled", True):
        exhibition_cfg = dict(exhibition_cfg_raw)
        exhibition_cfg["center"] = geometry.center
        exhibition_cfg["radius_m"] = radius_m
        exhibition_driver = ExhibitionDriver(exhibition_cfg, random.Random(exhibition_seed))

    person = None
    was_circling = patrol.state == "circling"
    write_period = 1.0 / write_hz
    t_since_write = 0.0
    t = 0.0
    log = {"t": [], "x": [], "y": [], "phase": [], "activity": []} if record else None

    print("[sim_runner] patrol logic source: %s" % PATROL_LOGIC_SOURCE)
    print("[sim_runner] model: %s" % model_path)
    print("[sim_runner] exhibition mode: %s" % ("ON" if exhibition_driver is not None else "off"))

    # Real mujoco viewer (the actual G1 model, actual mujoco rendering) -- see
    # this function's docstring. contextlib.nullcontext() when not requested,
    # so the loop below has exactly ONE code path regardless of use_viewer.
    viewer_ctx = contextlib.nullcontext()
    if use_viewer:
        try:
            import mujoco.viewer   # lazy: only needed (and only importable with a
                                    # real display) when a viewer is actually requested
            viewer_ctx = mujoco.viewer.launch_passive(model, puppet.data)
            print("[sim_runner] viewer window opening -- close it (or Ctrl-C here) to stop.")
        except Exception as e:
            print(f"[sim_runner] could not open the mujoco viewer ({e}).\n"
                  f"  This needs a real display (GLFW/a window system) -- it will not work\n"
                  f"  over a bare SSH session. Run this on a normal computer with a screen\n"
                  f"  attached, not headless.", flush=True)
            return {"log": log, "geometry": geometry, "patrol": patrol,
                    "exhibition_driver": exhibition_driver, "viewer_failed": True}

    with viewer_ctx as viewer:
        try:
            while duration_s is None or t < duration_s:
                loop_t0 = time.monotonic()

                if person_schedule is not None:
                    candidate = person_schedule(t, x, y, yaw)
                    if candidate is not None:
                        person = candidate

                persons_obs = []
                nearest = None
                if person is not None:
                    obs = person.observation(t, x, y, yaw)
                    if obs is not None:
                        persons_obs.append(obs)
                        nearest = obs

                supervisor.update(dt, x, y, yaw, nearest)
                now_circling = patrol.state == "circling"

                activity = None
                if exhibition_driver is not None:
                    if was_circling and not now_circling:
                        exhibition_driver.reset()   # yield: abandon any in-progress segment
                    if now_circling:
                        if (dance_schedule is not None
                                and exhibition_driver.pending_dance is None):
                            req = dance_schedule(t)
                            if req is not None:
                                exhibition_driver.pending_dance = req
                        vx, vy, vyaw = exhibition_driver.update(dt, x, y, yaw)
                        activity = exhibition_driver.activity
                    else:
                        vx, vy, vyaw = supervisor.command(x, y, yaw, geometry, speed_mps)
                else:
                    vx, vy, vyaw = supervisor.command(x, y, yaw, geometry, speed_mps)
                was_circling = now_circling

                # integrate the pose -- commanded velocity == achieved velocity,
                # identical math to scripts/sim_walk.py.
                x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
                y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
                yaw = _wrap(yaw + vyaw * dt)

                puppet.step(x, y, yaw, math.hypot(vx, vy), dt, waving=(patrol.state == "waving"))

                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        print("\n[sim_runner] viewer window closed -- stopping.")
                        break

                if record:
                    log["t"].append(t)
                    log["x"].append(x)
                    log["y"].append(y)
                    log["phase"].append(patrol.state)
                    log["activity"].append(activity)

                t_since_write += dt
                if t_since_write >= write_period:
                    t_since_write = 0.0
                    write_shm({
                        "x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4),
                        "phase": patrol.state, "activity": activity,
                        "persons": persons_obs, "ts": time.time(),
                    })

                t += dt
                if realtime:
                    elapsed = time.monotonic() - loop_t0
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n[sim_runner] stopped.")

    return {"log": log, "geometry": geometry, "patrol": patrol,
            "exhibition_driver": exhibition_driver}


# -----------------------------------------------------------------------------
# --selftest
# -----------------------------------------------------------------------------
def selftest():
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and bool(cond)

    radius_m = 3.0
    dt = 1.0 / WRITE_HZ_DEFAULT

    # --- run 1: plain circling, no person -- radius-tracking check ---
    # exhibition=False here: this run's whole point is checking pure circumference
    # pursuit stays on the circle, which exhibition mode deliberately replaces
    # while circling (see runs 3/4 below for exhibition's OWN safety checks).
    result = run(duration_s=40.0, dt=dt, radius_m=radius_m, speed_mps=SPEED_MPS_DEFAULT,
                 write_hz=WRITE_HZ_DEFAULT, person_schedule=None, realtime=False, record=True,
                 exhibition=False)
    log = result["log"]
    geometry = result["geometry"]
    max_dev = max((geometry.radial_drift_m(px, py) for px, py, ph in
                   zip(log["x"], log["y"], log["phase"]) if ph == "circling"), default=1e9)
    n_circling = sum(1 for ph in log["phase"] if ph == "circling")
    print("  circling samples=%d  max radial drift (geometry.radial_drift_m)=%.3fm" % (n_circling, max_dev))
    c("circling stays within 0.20m of target radius (%.1fm)" % radius_m, max_dev < 0.20)
    c("all sim ticks produced a valid phase", n_circling > 0 and all(p is not None for p in log["phase"]))

    # --- run 2: forced person appearance -- interaction transition-table check ---
    print("\n[sim_runner] re-running with a synthetic person appearance...")
    appear_at = 5.0

    def schedule(t, x, y, yaw):
        if t < appear_at:
            return None
        ahead = 0.6   # comfortably inside close_distance_m (1.6m default) -- guaranteed trigger
        return SyntheticPerson(x + ahead * math.cos(yaw), y + ahead * math.sin(yaw),
                                appear_at=appear_at, disappear_at=appear_at + 20.0)

    result2 = run(duration_s=45.0, dt=dt, radius_m=radius_m, speed_mps=SPEED_MPS_DEFAULT,
                  write_hz=WRITE_HZ_DEFAULT, person_schedule=schedule, realtime=False, record=True,
                  exhibition=False)
    log2 = result2["log"]
    patrol2 = result2["patrol"]
    seen_phases = set(log2["phase"])
    print("  phases observed over the run: %s" % sorted(seen_phases))
    required = ("circling", "pausing", "facing", "waving", "waiting", "resuming")
    for ph in required:
        c("interaction cycle visits '%s'" % ph, ph in seen_phases)
    c("ends back in 'circling' after the interaction", patrol2.state == "circling")
    c("history shows the happy-path order (no illegal jumps)",
      _contains_subsequence(patrol2.history, list(required) + ["circling"]))

    # --- run 3: Track A Extension -- exhibition mode, no person interaction ---
    # Asserts the oracle-driven roam/gesture/idle mix actually varies (not stuck
    # in one segment forever) and that every single pose sampled during the run
    # stays inside the disc's safety margin -- the exhibition driver's own core
    # safety property, independent of run 1's plain-circumference check above.
    print("\n[sim_runner] re-running with exhibition mode (roam/gesture/idle)...")
    result3 = run(duration_s=90.0, dt=dt, radius_m=radius_m, speed_mps=SPEED_MPS_DEFAULT,
                  write_hz=WRITE_HZ_DEFAULT, person_schedule=None, realtime=False, record=True,
                  exhibition=True, exhibition_seed=1234)
    log3 = result3["log"]
    geometry3 = result3["geometry"]
    cx, cy = geometry3.center
    max_from_center = max(math.hypot(px - cx, py - cy) for px, py in zip(log3["x"], log3["y"]))
    kinds_seen = {a.split(":")[0] for a in log3["activity"] if a is not None}
    print("  activity kinds observed: %s" % sorted(kinds_seen))
    print("  max distance from disc center over the run: %.3fm (radius=%.1fm)" % (max_from_center, radius_m))
    c("exhibition mix includes 'roam'", "roam" in kinds_seen)
    c("exhibition mix includes 'gesture'", "gesture" in kinds_seen)
    c("exhibition mix is not stuck in a single kind the whole run", len(kinds_seen) >= 2)
    c("every exhibition pose stays within the disc (radius_m=%.1fm)" % radius_m,
      max_from_center <= radius_m + 1e-6)

    # --- run 4: Track A Extension -- --demo-dance-request eventually fires ---
    # A dance request is queued at t=0 (kept queued -- never cleared here -- until
    # plan_next_segment finds it safe, exactly like the real conductor re-offering
    # an unconsumed dashboard request every tick); asserts it is eventually
    # accepted and shows up in the activity stream as "dance:503".
    print("\n[sim_runner] re-running with a demo dance request queued at t=0...")

    def dance_schedule(t):
        return (ExhibitionSegment(kind="dance", dance_fsm_id=503, dance_name="Dance Mode",
                                   dance_space_m=2.0, duration_s=8.0)
                if t < dt else None)   # queue exactly once, at the very first tick

    result4 = run(duration_s=150.0, dt=dt, radius_m=radius_m, speed_mps=SPEED_MPS_DEFAULT,
                  write_hz=WRITE_HZ_DEFAULT, person_schedule=None, realtime=False, record=True,
                  exhibition=True, exhibition_seed=99, dance_schedule=dance_schedule)
    log4 = result4["log"]
    danced = any(a == "dance:503" for a in log4["activity"] if a is not None)
    print("  activity kinds observed: %s"
          % sorted({a.split(":")[0] for a in log4["activity"] if a is not None}))
    c("a queued --demo-dance-request-style request eventually fires ('dance:503')", danced)

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def _contains_subsequence(seq, sub):
    """True iff `sub` appears as a CONTIGUOUS run inside `seq`."""
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return True
    return False


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="bounded run, asserts, exit 0/1")
    ap.add_argument("--radius", type=float, default=3.0, help="patrol circle radius (m)")
    ap.add_argument("--speed", type=float, default=SPEED_MPS_DEFAULT, help="patrol linear speed (m/s)")
    ap.add_argument("--hz", type=float, default=WRITE_HZ_DEFAULT, help="shm write rate (Hz)")
    ap.add_argument("--dt", type=float, default=1.0 / WRITE_HZ_DEFAULT, help="sim tick (s)")
    ap.add_argument("--duration", type=float, default=None,
                    help="live mode: stop after N sim-seconds (default: run until Ctrl-C)")
    ap.add_argument("--demo-person", action="store_true",
                    help="live mode: a synthetic person repeatedly appears near the path")
    ap.add_argument("--no-exhibition", action="store_true",
                    help="live mode: fall back to plain circumference pursuit while "
                         "circling instead of the Track A Extension roam/gesture/dance "
                         "repertoire (mirrors circle_patrol.py's launch-time fallback choice)")
    ap.add_argument("--demo-dance-request", action="store_true",
                    help="live mode: periodically synthesize an operator dance request "
                         "(fsm 503, 'Dance Mode'), exactly mirroring --demo-person -- "
                         "the real dashboard button instead writes "
                         "/dev/shm/g1_exhibition_dance_request.json for the real conductor")
    ap.add_argument("--viewer", action="store_true",
                    help="open mujoco's OWN real interactive 3D viewer (the actual G1 "
                         "model) and watch this run live. NEEDS A REAL DISPLAY -- will "
                         "not work over a bare SSH session with no X forwarding; run this "
                         "on a normal computer with a screen attached. Runs until you close "
                         "the viewer window or Ctrl-C (or --duration elapses, if set).")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    print("[sim_runner] SAFETY: standalone simulation only -- no DDS, no ROS2, "
          "no LocoClient, no real-robot connection of any kind.")

    person_schedule = None
    if args.demo_person:
        demo_cycle_s = 40.0
        state = {"person": None}

        def person_schedule(t, x, y, yaw):  # noqa: F811
            cyc = t % demo_cycle_s
            if 5.0 <= cyc < 25.0:
                if state["person"] is None or not state["person"].active(t):
                    ahead = 1.2
                    state["person"] = SyntheticPerson(
                        x + ahead * math.cos(yaw), y + ahead * math.sin(yaw),
                        appear_at=t, disappear_at=t + 20.0)
            return state["person"]

    dance_schedule = None
    if args.demo_dance_request:
        demo_dance_cycle_s = 60.0   # per the plan: "every ~60 sim-seconds"
        dance_state = {"last_cycle": -1}

        def dance_schedule(t):  # noqa: F811
            cyc = int(t // demo_dance_cycle_s)
            if cyc != dance_state["last_cycle"]:
                dance_state["last_cycle"] = cyc
                # duration_s=8.0: documented PLACEHOLDER, matches
                # g1_nav/config/patrol.yaml exhibition.dance_default_duration_s
                # (no authoritative "how long does dance 503 take" value exists
                # anywhere else in this codebase yet -- see that key's own comment).
                return ExhibitionSegment(kind="dance", dance_fsm_id=503, dance_name="Dance Mode",
                                         dance_space_m=2.0, duration_s=8.0)
            return None

    run(duration_s=args.duration, dt=args.dt, radius_m=args.radius, speed_mps=args.speed,
        write_hz=args.hz, person_schedule=person_schedule, realtime=True,
        exhibition=not args.no_exhibition, dance_schedule=dance_schedule,
        use_viewer=args.viewer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
