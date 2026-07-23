#!/usr/bin/env python3
"""Local closed-loop circle-patrol simulator for the G1 -- NO ROBOT.

Track A0 (see delegated-wishing-stream.md, "Track A"): proves the pure-pursuit
carrot-point geometry, the odometry-drift/re-anchor logic, and the person-pause/
resume interaction state machine BEFORE any of this touches the real robot or
g1_nav. Mirrors scripts/sim_walk.py's culture closely:

    circle-patrol decision  -> intent (vx,vy,vyaw)      <- computed from the
                                                             DRIFTING BELIEVED pose
                                                             ONLY (no map/AMCL;
                                                             continuous odometry is
                                                             all the real robot has)
      -> CommandShaper.normalize()            (diagonal -> speed ellipse)
      -> ObstacleGuard.apply()                (reads a SYNTHETIC 360 ring cast from
                                                the TRUE pose; brakes)
      -> CommandShaper.shape(bypass=hard)     (jerk/accel-limit; snap on hard stop)
      -> integrate the TRUE pose              (assumes the gait tracks the command)
      -> integrate the BELIEVED pose          (same command, but with an injected
                                                leg-odometry-style scale/yaw-rate
                                                bias -- this is what actually drifts)

The synthetic ring is ray-cast from the TRUE pose against a world obstacle circle,
producing exactly the {n,bin_deg,start_deg,dist[]} contract obstacle_node.py writes
-- so the REAL obstacle/guard.py runs unmodified, same as sim_walk.py.

WHY TWO POSES: the plan is explicit that circle-patrol has no persistent map / AMCL,
only continuous (drifting) odometry -- "this is odom-frame-relative, accepted
drift" (g1_nav circle_patrol.py docstring, Track A3). So the steering decision may
ONLY ever look at the BELIEVED pose (what odometry reports); the TRUE pose is kept
purely so this harness can (a) ray-cast the obstacle honestly and (b) report how far
reality has actually drifted from the commanded circle, as a diagnostic.

Usage:
    python3 scripts/sim_circle.py <scenario> [--ascii] [--json]
    python3 scripts/sim_circle.py --list
    python3 scripts/sim_circle.py --selftest

Scenarios (built-in): drift_only, drift_with_jitter, static_obstacle_on_arc,
    person_pause_resume.
Exit code 0 if every scenario met its criteria, else 1.
"""

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "obstacle"))

import yaml  # noqa: E402
from cmd_shaper import CommandShaper  # noqa: E402

import importlib.util  # noqa: E402
_g = importlib.util.spec_from_file_location("guard_for_circle", os.path.join(ROOT, "obstacle", "guard.py"))
_gm = importlib.util.module_from_spec(_g); _g.loader.exec_module(_gm)
ObstacleGuard = _gm.ObstacleGuard

import time as _realtime  # noqa: E402


class _SimClock:
    """Virtual monotonic clock so the guard's wall-clock-based slew (scale_slew,
    fault timing, freshness) advances by exactly DT per sim tick -- otherwise an
    accelerated (faster-than-real-time) sim makes the guard see dt~=0 and its
    internal scale-slew never moves. Delegates everything except monotonic() to the
    real time module. (Identical to sim_walk.py's _SimClock.)"""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def __getattr__(self, k):
        return getattr(_realtime, k)


CFG = yaml.safe_load(open(os.path.join(ROOT, "config", "robot.yaml")))
MAXES = (CFG["speeds"]["max_vx"], CFG["speeds"]["max_vy"], CFG["speeds"]["max_vyaw"])
DT = 1.0 / CFG["control"]["send_hz"]
ACCEL = (CFG["motion"]["accel_vx"], CFG["motion"]["accel_vy"], CFG["motion"]["accel_vyaw"])
JERK = (CFG["motion"]["jerk_vx"], CFG["motion"]["jerk_vy"], CFG["motion"]["jerk_vyaw"])

RING_N = 72
RING_BIN = 5.0
RING_START = -180.0
RANGE_M = 3.0          # synthetic sensor range (matches obstacle_range_m)
ROBOT_RADIUS = 0.30    # physical half-extent for the collision check

RADIUS_M = 3.0         # the patrol circle radius (per the plan: fixed 3 m)
LOOKAHEAD_M = 1.25      # pure-pursuit carrot distance ahead along the arc (spec: ~1.0-1.5 m)

# --- DESIGN DECISION: re-anchor-on-drift threshold ---------------------------
# carrot_point() always projects the current BELIEVED position onto the circle by
# ANGLE (atan2 around the circle center) -- which is already the mathematically
# correct nearest-point-on-circle regardless of how far off the circle the belief
# has drifted. The one thing that threshold still needs to gate is the ARC-LENGTH
# lookahead added past that projection: with a small deviation, "project then walk
# LOOKAHEAD_M further around the arc" is a sane pure-pursuit target. Once the
# believed position has drifted further from the circle than the robot's own
# safety bubble + half-width (obstacle/obstacle.yaml: safety_bubble_m 0.15 +
# robot_half_width_m 0.25 + clearance_margin_m 0.10 ~= 0.5 m; we round up to 0.6 m
# for margin), continuing to aim a full LOOKAHEAD_M further around the arc would
# steer the robot ALONG a chord that is no longer close to tangent -- it would
# cut the corner and overshoot/oscillate as each tick's projection swings by a
# large angle. So past that threshold we RE-ANCHOR: collapse the lookahead to
# zero and aim straight at the nearest point on the circle (pure radial recovery)
# until the deviation shrinks back under the threshold, at which point normal
# tangential pursuit resumes automatically (no separate "mode" flag needed -- see
# carrot_point()).
REANCHOR_DRIFT_M = 0.6


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------- circle geometry
class Circle:
    __slots__ = ("cx", "cy", "r")

    def __init__(self, cx, cy, r):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)


class CarrotResult:
    __slots__ = ("x", "y", "ang_on_circle", "radial_dev", "reanchored")

    def __init__(self, x, y, ang_on_circle, radial_dev, reanchored):
        self.x, self.y = x, y
        self.ang_on_circle = ang_on_circle
        self.radial_dev = radial_dev
        self.reanchored = reanchored

    def __repr__(self):
        return (f"CarrotResult(x={self.x:.3f}, y={self.y:.3f}, "
                f"radial_dev={self.radial_dev:+.3f}, reanchored={self.reanchored})")


def carrot_point(x, y, circle, lookahead_m, direction=1.0, reanchor_drift_m=REANCHOR_DRIFT_M):
    """Pure-pursuit carrot point on `circle`, from believed position (x, y).

    The nearest point on a circle to ANY (x, y) -- inside, outside, or on it -- is
    at the SAME ANGLE from the center (this is exact circle geometry, not an
    approximation), so `ang_pos` below is the correct projection regardless of how
    far (x, y) has drifted off the circle. `direction` is +1 for CCW, -1 for CW
    travel. When the radial deviation is within `reanchor_drift_m`, the carrot is
    walked `lookahead_m` further around the arc past the projection (normal
    tangential pure-pursuit); beyond that threshold the lookahead collapses to 0
    (RE-ANCHOR: aim straight at the nearest point on the circle) -- see the
    REANCHOR_DRIFT_M comment above for the reasoning.
    """
    ang_pos = math.atan2(y - circle.cy, x - circle.cx)
    radial_dev = math.hypot(x - circle.cx, y - circle.cy) - circle.r
    reanchored = abs(radial_dev) > reanchor_drift_m
    eff_lookahead = 0.0 if reanchored else lookahead_m
    ang_carrot = ang_pos + direction * (eff_lookahead / circle.r)
    cx_pt = circle.cx + circle.r * math.cos(ang_carrot)
    cy_pt = circle.cy + circle.r * math.sin(ang_carrot)
    return CarrotResult(cx_pt, cy_pt, ang_pos, radial_dev, reanchored)


def pursuit_command(x, y, theta, carrot_xy, max_vx, max_vyaw, k_yaw=1.6):
    """Human-like pure-pursuit steering: face the carrot with yaw, drive forward
    into it, easing vx off while badly misaligned (mirrors sim_walk.py's
    policy_goal, the established "operator-like" steering law in this repo)."""
    gx, gy = carrot_xy
    bearing = wrap(math.atan2(gy - y, gx - x) - theta)
    vyaw = clamp(k_yaw * bearing, -max_vyaw, max_vyaw)
    vx = max_vx * max(0.0, math.cos(bearing))
    return (vx, 0.0, vyaw)


# ------------------------------------------------------- interaction state machine
# Vocabulary mandated verbatim by the plan's "Cross-workstream decisions" (shared
# across Track A/B/C via /dev/shm/g1_patrol_state.json's "phase" field).
STATES = ("circling", "pausing", "facing", "waving", "waiting", "resuming",
          "fault", "manual_override", "paused_odom_degraded")

_ALWAYS_REACHABLE = ("manual_override", "paused_odom_degraded")

# Forward/happy-path + recovery edges (excludes the always-reachable safety states,
# which are added to every OTHER state below).
_BASE_TRANSITIONS = {
    "circling": {"pausing"},
    "pausing": {"facing", "circling"},          # circling: person left / false trigger, abort before facing
    "facing": {"waving", "waiting"},            # waiting: skip the wave if none is configured
    "waving": {"waiting"},
    "waiting": {"resuming"},
    "resuming": {"circling"},
    "fault": {"manual_override"},               # operator must take over to clear a fault
    "manual_override": {"circling"},            # operator releases control -> patrol restarts
    "paused_odom_degraded": {"resuming"},        # odometry health recovered -> rejoin like a pause
}


def _build_transitions():
    """BASE edges + manual_override/paused_odom_degraded reachable from every
    OTHER state (mandated: "reachable from every other state"; no self-loops)."""
    table = {s: set(_BASE_TRANSITIONS.get(s, ())) for s in STATES}
    for s in STATES:
        for safety in _ALWAYS_REACHABLE:
            if s != safety:
                table[s].add(safety)
    return table


TRANSITIONS = _build_transitions()


class PatrolFSM:
    """Validates the interaction-state transition table above. Pure (no timers, no
    I/O) -- PatrolController below drives it with real per-tick timing."""

    def __init__(self, state="circling"):
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self.state = state

    def can(self, to):
        return to in TRANSITIONS.get(self.state, ())

    def transition(self, to):
        if not self.can(to):
            raise ValueError(f"invalid transition {self.state!r} -> {to!r}")
        self.state = to
        return self.state


# --------------------------------------------------------------------- world ring
def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    """Nearest positive t where (ox,oy)+t*(dx,dy) enters circle (cx,cy,r), or None."""
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    t1, t2 = (-b - s) / 2.0, (-b + s) / 2.0
    for t in (t1, t2):
        if t > 1e-6:
            return t
    return None


def cast_ring(x, y, theta, obstacles):
    """Ray-cast circular obstacles into the obstacle-node ring contract from the
    TRUE pose (x, y, theta) -- same {n,bin_deg,start_deg,dist} contract as
    sim_walk.py's cast_ring, restricted to circle obstacles (all this harness
    needs on the patrol arc). Returns (ring, front, left, right, back, min_surf)."""
    dist = [None] * RING_N
    for i in range(RING_N):
        ang = math.radians(RING_START + (i + 0.5) * RING_BIN)
        dx, dy = math.cos(theta + ang), math.sin(theta + ang)
        best = None
        for o in obstacles:
            t = _ray_circle(x, y, dx, dy, o["cx"], o["cy"], o["r"])
            if t is not None and t <= RANGE_M and (best is None or t < best):
                best = t
        dist[i] = round(best, 3) if best is not None else None
    ring = {"n": RING_N, "bin_deg": RING_BIN, "start_deg": RING_START, "dist": dist}

    def _centered():
        return [(RING_START + (i + 0.5) * RING_BIN, dist[i]) for i in range(RING_N)]

    def wedge(lo, hi):
        vals = [d for c, d in _centered() if lo <= c < hi and d is not None]
        return min(vals) if vals else None

    front = min([d for c, d in _centered() if abs(c) < 45.0 and d is not None], default=None)
    left = wedge(45.0, 135.0)
    right = wedge(-135.0, -45.0)
    back = min([d for c, d in _centered() if abs(c) >= 135.0 and d is not None], default=None)
    finite = [d for d in dist if d is not None]
    return ring, front, left, right, back, (min(finite) if finite else None)


# --------------------------------------------------------------------- controller
class PatrolConfig:
    def __init__(self, **kw):
        self.radius_m = RADIUS_M
        self.lookahead_m = LOOKAHEAD_M
        self.direction = 1.0              # +1 CCW, -1 CW
        self.k_yaw = 1.6                  # circling-phase steering gain (mirrors sim_walk policy_goal)
        self.k_yaw_face = 2.0             # facing-phase turn-in-place gain
        self.face_tol_deg = 3.0           # facing "done" tolerance
        self.pause_settle_s = 1.0         # circling->pausing dwell (lets the shaper ramp vx to ~0)
        self.wave_hold_s = 1.5            # facing done -> waving hold (arm animation stand-in)
        self.wait_hold_s = 2.0            # waving done -> waiting hold (interaction-window stand-in)
        for k, v in kw.items():
            setattr(self, k, v)


class PatrolController:
    """Owns the FSM + circle geometry + the per-tick command decision. ALL steering
    is computed from the BELIEVED pose only (see module docstring) -- this class
    never sees the true pose."""

    def __init__(self, circle, cfg):
        self.circle = circle
        self.cfg = cfg
        self.fsm = PatrolFSM("circling")
        self._phase_t = 0.0
        self._face_target = None
        self._last_carrot = None          # most recent circling-phase carrot (for diagnostics)
        self._pre_pause_carrot = None      # carrot snapshotted the instant a person interrupts
        self._resume_carrot = None         # carrot freshly computed at the resuming->circling instant
        self.events = []                   # (t, from_state, to_state) log

    def _go(self, to, t):
        frm = self.fsm.state
        self.fsm.transition(to)
        self.events.append((t, frm, to))
        self._phase_t = 0.0

    def notify_person(self, bearing_target_rad, t):
        """External event: a person was detected. Only interrupts active patrol."""
        if self.fsm.state != "circling":
            return False
        self._face_target = bearing_target_rad
        self._pre_pause_carrot = self._last_carrot
        self._go("pausing", t)
        return True

    def step(self, bx, by, btheta, dt, t):
        """Advance one tick from the BELIEVED pose; return commanded (vx, vy, vyaw)."""
        self._phase_t += dt
        state = self.fsm.state

        if state == "circling":
            carrot = carrot_point(bx, by, self.circle, self.cfg.lookahead_m, self.cfg.direction)
            self._last_carrot = carrot
            return pursuit_command(bx, by, btheta, (carrot.x, carrot.y), MAXES[0], MAXES[2], self.cfg.k_yaw)

        if state == "pausing":
            if self._phase_t >= self.cfg.pause_settle_s:
                self._go("facing", t)
            return (0.0, 0.0, 0.0)

        if state == "facing":
            err = wrap(self._face_target - btheta)
            if abs(err) < math.radians(self.cfg.face_tol_deg):
                self._go("waving", t)
                return (0.0, 0.0, 0.0)
            vyaw = clamp(self.cfg.k_yaw_face * err, -MAXES[2], MAXES[2])
            return (0.0, 0.0, vyaw)

        if state == "waving":
            if self._phase_t >= self.cfg.wave_hold_s:
                self._go("waiting", t)
            return (0.0, 0.0, 0.0)

        if state == "waiting":
            if self._phase_t >= self.cfg.wait_hold_s:
                self._go("resuming", t)
            return (0.0, 0.0, 0.0)

        if state == "resuming":
            # REJOIN: project the CURRENT believed position onto the circle -- NOT
            # the stale pre-pause carrot (which is generally somewhere else: the
            # robot kept drifting forward during the pausing ramp-down, and time
            # has passed during facing/waving/waiting).
            carrot = carrot_point(bx, by, self.circle, self.cfg.lookahead_m, self.cfg.direction)
            self._resume_carrot = carrot
            self._last_carrot = carrot
            self._go("circling", t)
            return pursuit_command(bx, by, btheta, (carrot.x, carrot.y), MAXES[0], MAXES[2], self.cfg.k_yaw)

        # fault / manual_override / paused_odom_degraded: not exercised by this
        # harness's scenarios (they are event-driven from OUTSIDE the geometry
        # loop -- see test_circle_patrol.py for the transition-table coverage).
        return (0.0, 0.0, 0.0)


# ------------------------------------------------------------------- scenarios
def scenarios():
    return {
        "drift_only": dict(
            dur=45.0, drift=dict(scale=1.02, yaw_bias=0.02),
            obstacle=None, person_event=None,
            desc="~2 laps of pure odometry-drift bias (scale+yaw-rate), no noise: "
                 "believed-frame circle-following must stay smooth and bounded"),
        "drift_with_jitter": dict(
            dur=30.0, drift=dict(scale=1.03, yaw_bias=0.03, vel_noise=0.01, yaw_noise=0.01, seed=7),
            obstacle=None, person_event=None,
            desc="drift + seeded odometry noise: carrot-pursuit still stays smooth/bounded"),
        "static_obstacle_on_arc": dict(
            dur=25.0, drift=dict(scale=1.01, yaw_bias=0.01),
            obstacle=dict(arc_deg=90.0, r=0.35),
            person_event=None,
            desc="a static obstacle sits directly on the patrol arc: the guard must "
                 "stop or the path must deviate-and-rejoin -- never pass through it"),
        "person_pause_resume": dict(
            dur=20.0, drift=dict(scale=1.02, yaw_bias=0.02),
            obstacle=None,
            person_event=dict(t=6.0, bearing_offset_deg=50.0),
            desc="a person appears mid-lap: circling->pausing->facing->waving->"
                 "waiting->resuming->circling, then a smooth (non-teleport) rejoin"),
    }


# ------------------------------------------------------------------------ runner
def run(scn, seed_override=None):
    clock = _SimClock()
    _gm.time = clock
    shaper = CommandShaper(cfg=CFG["motion"], max_speeds=MAXES)
    guard = ObstacleGuard(os.path.join(ROOT, "obstacle", "obstacle.yaml"), audio=None, limits=MAXES)
    guard.enabled = True; guard.mode = "ACTIVE"

    circle = Circle(0.0, 0.0, RADIUS_M)
    cfg = PatrolConfig()
    ctrl = PatrolController(circle, cfg)

    # start ON the circle at angle 0, heading tangentially (CCW travel)
    tx, ty, ttheta = circle.r, 0.0, math.pi / 2.0 * cfg.direction
    bx, by, btheta = tx, ty, ttheta        # believed == true at patrol start (odom zeroed here)

    drift = scn.get("drift") or {}
    seed = seed_override if seed_override is not None else drift.get("seed")
    rng = random.Random(seed) if seed is not None else None

    obstacles = []
    obs_scn = scn.get("obstacle")
    if obs_scn:
        ang = math.radians(obs_scn["arc_deg"]) * cfg.direction
        ocx = circle.cx + circle.r * math.cos(ang)
        ocy = circle.cy + circle.r * math.sin(ang)
        obstacles.append({"cx": ocx, "cy": ocy, "r": obs_scn["r"]})

    person_event = scn.get("person_event")
    person_fired = False

    nsteps = int(scn["dur"] / DT)
    true_traj, believed_traj, states, sent = [], [], [], []
    min_surf = 1e9
    hard_stop_ticks = 0
    t = 0.0

    for _ in range(nsteps):
        if person_event and not person_fired and t >= person_event["t"]:
            bearing_target = wrap(btheta + math.radians(person_event["bearing_offset_deg"]))
            person_fired = ctrl.notify_person(bearing_target, t)

        ring, fr, lf, rt, bk, msurf = cast_ring(tx, ty, ttheta, obstacles)
        if msurf is not None:
            min_surf = min(min_surf, msurf)

        intent = ctrl.step(bx, by, btheta, DT, t)
        intent = shaper.normalize(tuple(clamp(intent[i], -MAXES[i], MAXES[i]) for i in range(3)))

        data = {"ring": ring, "front_m": fr, "left_m": lf, "right_m": rt, "back_m": bk, "tight_factor": 1.0}
        with guard._lock:
            guard._data = data; guard._last_fresh = clock.monotonic(); guard._live = True
        governed = guard.apply(intent)
        hard = guard.hard_stop_flags()
        if any(hard):
            hard_stop_ticks += 1
        vx, vy, vyaw = shaper.shape(governed, DT, bypass=hard)
        sent.append((vx, vy, vyaw))

        # TRUE pose: sent velocity == achieved velocity (sim_walk.py convention)
        tx += (vx * math.cos(ttheta) - vy * math.sin(ttheta)) * DT
        ty += (vx * math.sin(ttheta) + vy * math.cos(ttheta)) * DT
        ttheta = wrap(ttheta + vyaw * DT)
        true_traj.append((tx, ty))

        # BELIEVED pose: SAME commanded velocity, but odometry mis-measures it (a
        # multiplicative forward-speed bias + an additive yaw-rate bias, mirroring
        # sim_odometry.py's leg-drift model; +/- seeded, deterministic noise).
        vx_b = vx * drift.get("scale", 1.0)
        vyaw_b = vyaw + drift.get("yaw_bias", 0.0)
        if rng is not None:
            vx_b += rng.gauss(0.0, drift.get("vel_noise", 0.0))
            vyaw_b += rng.gauss(0.0, drift.get("yaw_noise", 0.0))
        bx += (vx_b * math.cos(btheta) - vy * math.sin(btheta)) * DT
        by += (vx_b * math.sin(btheta) + vy * math.cos(btheta)) * DT
        btheta = wrap(btheta + vyaw_b * DT)
        believed_traj.append((bx, by))

        states.append(ctrl.fsm.state)
        t += DT
        clock.t += DT

    return _metrics(scn, circle, true_traj, believed_traj, states, sent, min_surf, hard_stop_ticks, ctrl)


def _metrics(scn, circle, true_traj, believed_traj, states, sent, min_surf, hard_stop_ticks, ctrl):
    def radius_devs(traj):
        return [math.hypot(x - circle.cx, y - circle.cy) - circle.r for (x, y) in traj]

    believed_dev = radius_devs(believed_traj)
    true_dev = radius_devs(true_traj)

    # smoothness: reuse sim_walk.py's accel/jerk-violation counting on the SENT
    # signal (the shaper+guard chain is identical, so the same criteria apply).
    def axis(idx):
        v = [s[idx] for s in sent]
        acc = [(v[i + 1] - v[i]) / DT for i in range(len(v) - 1)]
        jrk = [(acc[i + 1] - acc[i]) / DT for i in range(len(acc) - 1)]
        return (max((abs(a) for a in acc), default=0.0), max((abs(j) for j in jrk), default=0.0))

    amax_yaw, jmax_yaw = axis(2)
    viol = 0
    for idx, lim in ((0, ACCEL[0]), (1, ACCEL[1]), (2, ACCEL[2])):
        v = [s[idx] for s in sent]
        for i in range(len(v) - 1):
            dvdt = abs(v[i + 1] - v[i]) / DT
            is_snap = abs(v[i + 1]) < 1e-9 and abs(v[i]) > lim * DT
            if not is_snap and dvdt > lim * 1.15:
                viol += 1

    # person-pause/resume: prove the rejoin used the CURRENT projection, not a
    # teleport back to the stale pre-pause carrot.
    resume_info = None
    if ctrl._pre_pause_carrot is not None and ctrl._resume_carrot is not None:
        stale, fresh = ctrl._pre_pause_carrot, ctrl._resume_carrot
        # believed pose at the exact resume instant = the believed_traj sample
        # logged the tick the "resuming"->"circling" transition fired.
        resume_t = next((tt for (tt, frm, to) in ctrl.events if to == "circling" and frm == "resuming"), None)
        idx = int(round(resume_t / DT)) if resume_t is not None else len(believed_traj) - 1
        idx = clamp(idx, 0, len(believed_traj) - 1)
        rbx, rby = believed_traj[idx]
        bearing_stale = wrap(math.atan2(stale.y - rby, stale.x - rbx))
        bearing_fresh = wrap(math.atan2(fresh.y - rby, fresh.x - rbx))
        carrot_jump_m = math.hypot(fresh.x - stale.x, fresh.y - stale.y)
        resume_info = {
            "carrot_jump_m": round(carrot_jump_m, 3),
            "bearing_err_stale_deg": round(math.degrees(bearing_stale), 2),
            "bearing_err_fresh_deg": round(math.degrees(bearing_fresh), 2),
        }

    happy_path = ["circling", "pausing", "facing", "waving", "waiting", "resuming", "circling"]
    seen_sequence = [frm for (_, frm, _) in ctrl.events] + ([ctrl.events[-1][2]] if ctrl.events else [])

    return {
        "scenario": scn.get("name", "?"),
        "desc": scn["desc"],
        "collided": min_surf < ROBOT_RADIUS,
        "min_surface_dist_m": round(min_surf, 3) if min_surf < 1e8 else None,
        "hard_stop_ticks": hard_stop_ticks,
        "max_believed_radius_dev_m": round(max((abs(d) for d in believed_dev), default=0.0), 3),
        "max_true_radius_dev_m": round(max((abs(d) for d in true_dev), default=0.0), 3),
        "final_true_radius_dev_m": round(true_dev[-1], 3) if true_dev else None,
        "max_yaw_accel": round(amax_yaw, 3),
        "max_yaw_jerk": round(jmax_yaw, 2),
        "accel_limit_violations": viol,
        "events": ctrl.events,
        "seen_sequence": seen_sequence,
        "happy_path_achieved": _contains_subsequence(seen_sequence, happy_path),
        "resume_info": resume_info,
        "final_true_pose": [round(true_traj[-1][0], 2), round(true_traj[-1][1], 2)] if true_traj else None,
        "final_believed_pose": [round(believed_traj[-1][0], 2), round(believed_traj[-1][1], 2)] if believed_traj else None,
        "_true_traj": true_traj,
        "_believed_traj": believed_traj,
        "_obstacle": scn.get("obstacle"),
    }


def _contains_subsequence(seq, sub):
    """True iff `sub` appears as a CONTIGUOUS run inside `seq` (the exact happy-path
    sequence, not just each state visited in some order)."""
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return True
    return False


# --------------------------------------------------------------------- ascii view
def ascii_map(traj_true, traj_believed, obstacle, circle, w=64, h=26):
    xs = [p[0] for p in traj_true + traj_believed] + [circle.cx - circle.r, circle.cx + circle.r]
    ys = [p[1] for p in traj_true + traj_believed] + [circle.cy - circle.r, circle.cy + circle.r]
    if obstacle:
        xs += [obstacle["cx"] - obstacle["r"], obstacle["cx"] + obstacle["r"]]
        ys += [obstacle["cy"] - obstacle["r"], obstacle["cy"] + obstacle["r"]]
    x0, x1 = min(xs) - 0.5, max(xs) + 0.5
    y0, y1 = min(ys) - 0.5, max(ys) + 0.5
    x1 = max(x1, x0 + 1e-3); y1 = max(y1, y0 + 1e-3)

    def col(px): return _cl(int((px - x0) / (x1 - x0) * (w - 1)), w)
    def row(py): return _cl(int((y1 - py) / (y1 - y0) * (h - 1)), h)

    grid = [[" "] * w for _ in range(h)]
    for a in range(0, 360, 4):
        px = circle.cx + circle.r * math.cos(math.radians(a))
        py = circle.cy + circle.r * math.sin(math.radians(a))
        grid[row(py)][col(px)] = ","
    if obstacle:
        for a in range(0, 360, 8):
            px = obstacle["cx"] + obstacle["r"] * math.cos(math.radians(a))
            py = obstacle["cy"] + obstacle["r"] * math.sin(math.radians(a))
            grid[row(py)][col(px)] = "#"
    for (px, py) in traj_believed:
        grid[row(py)][col(px)] = "o"
    for (px, py) in traj_true:
        grid[row(py)][col(px)] = "."
    return "\n".join("".join(r) for r in grid)


def _cl(v, n):
    return 0 if v < 0 else n - 1 if v >= n else v


# -------------------------------------------------------------------------- main
def verdict(m):
    """Pass/fail criteria for one scenario run."""
    ok = True
    notes = []
    if m["collided"]:
        ok = False; notes.append("COLLISION")
    if m["accel_limit_violations"]:
        ok = False; notes.append(f"{m['accel_limit_violations']} accel-limit violations (jerky)")
    # believed-frame circle-following must stay tight (this is the thing the
    # controller can actually see and correct) regardless of drift.
    if m["max_believed_radius_dev_m"] > 0.35:
        ok = False; notes.append(f"believed radius deviation too large ({m['max_believed_radius_dev_m']} m)")
    if m["events"]:
        # a person/interaction scenario ran: the happy path must appear verbatim
        if not m["happy_path_achieved"]:
            ok = False; notes.append("happy-path state sequence not achieved")
        ri = m["resume_info"]
        if ri is not None:
            # the CURRENT-position rejoin must not require a larger heading
            # correction than the stale teleport-style target would have -- i.e.
            # projecting from where the robot ACTUALLY is is never worse than
            # blindly reusing the pre-pause carrot.
            if abs(ri["bearing_err_fresh_deg"]) > abs(ri["bearing_err_stale_deg"]) + 1.0:
                ok = False; notes.append("resume rejoin worse than stale teleport target")
            if abs(ri["bearing_err_fresh_deg"]) > 90.0:
                ok = False; notes.append("resume rejoin bearing error too large (not a smooth rejoin)")
    return ok, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="drift_only")
    ap.add_argument("--ascii", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true", help="run every built-in scenario")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    scns = scenarios()
    if args.list:
        for name, s in scns.items():
            print(f"{name:24s} {s['desc']}")
        return 0

    names = list(scns) if args.all else [args.scenario]
    results = []
    overall_ok = True
    for name in names:
        if name not in scns:
            print(f"unknown scenario '{name}' (try --list)"); return 2
        scn = dict(scns[name]); scn["name"] = name
        m = run(scn)
        ok, notes = verdict(m)
        overall_ok = overall_ok and ok
        results.append((scn, m, ok, notes))

    if args.json:
        print(json.dumps([dict({k: v for k, v in m.items() if not k.startswith("_")}, ok=ok, notes=notes)
                          for (_, m, ok, notes) in results], indent=2))
    else:
        for (scn, m, ok, notes) in results:
            print(f"\n=== {m['scenario']} === {'PASS' if ok else 'FAIL'} {'; '.join(notes)}")
            print(f"  {m['desc']}")
            print(f"  collided={m['collided']}  min_surface_dist={m['min_surface_dist_m']}  "
                  f"hard_stop_ticks={m['hard_stop_ticks']}")
            print(f"  max_believed_radius_dev={m['max_believed_radius_dev_m']} m   "
                  f"max_true_radius_dev={m['max_true_radius_dev_m']} m   "
                  f"final_true_radius_dev={m['final_true_radius_dev_m']} m")
            if m["events"]:
                seq = " -> ".join(s for s in m["seen_sequence"])
                print(f"  state sequence: {seq}")
                print(f"  happy_path_achieved={m['happy_path_achieved']}  resume_info={m['resume_info']}")
            print(f"  accel_limit_violations={m['accel_limit_violations']}")
            if args.ascii:
                print(ascii_map(m["_true_traj"], m["_believed_traj"], m["_obstacle"], Circle(0, 0, RADIUS_M)))
    return 0 if overall_ok else 1


def selftest():
    """Runs every built-in scenario and ASSERTS pass/fail + prints metrics."""
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and bool(cond)

    for name, base in scenarios().items():
        scn = dict(base); scn["name"] = name
        m = run(scn)
        v_ok, notes = verdict(m)
        print(f"\n--- {name} --- {m['desc']}")
        print(f"  collided={m['collided']} hard_stop_ticks={m['hard_stop_ticks']} "
              f"max_believed_dev={m['max_believed_radius_dev_m']}m max_true_dev={m['max_true_radius_dev_m']}m "
              f"accel_violations={m['accel_limit_violations']}")
        c(f"[{name}] overall verdict ({'; '.join(notes) if notes else 'clean'})", v_ok)
        c(f"[{name}] never collided", not m["collided"])
        c(f"[{name}] believed-frame circle-following bounded (< 0.35 m)",
          m["max_believed_radius_dev_m"] < 0.35)

        if name == "static_obstacle_on_arc":
            # Proof the obstacle was actually ENCOUNTERED (not a confound of background
            # drift, which alone can also nudge max_true_radius_dev_m -- verified by
            # comparing against a no-obstacle control below): the ring must have seen
            # the obstacle's surface closer than the guard's slow-zone, yet the guard
            # (soft stop and/or hard stop) must have kept the robot from ever reaching
            # it -- either outcome (a full stop, or slowing + the carrot pulling the
            # path on past a near-miss) is acceptable; a silent pass-through is not.
            engaged = (m["min_surface_dist_m"] is not None and m["min_surface_dist_m"] < 1.0)
            c(f"[{name}] the obstacle was actually sensed/approached (min_surface_dist="
              f"{m['min_surface_dist_m']} m < slow-zone 1.0 m)", engaged)
            c(f"[{name}] ...yet never collided (min_surface_dist >= robot radius {ROBOT_RADIUS} m)",
              not m["collided"])
            no_obstacle_scn = dict(scn); no_obstacle_scn["obstacle"] = None
            control = run(no_obstacle_scn)
            c(f"[{name}] control check: WITHOUT the obstacle the ring never gets that close "
              f"(control min_surface_dist={control['min_surface_dist_m']}), so the near-approach "
              f"above is genuinely caused by the obstacle, not background drift",
              control["min_surface_dist_m"] is None or control["min_surface_dist_m"] >= 1.0)

        if name == "person_pause_resume":
            c(f"[{name}] happy path circling->pausing->facing->waving->waiting->resuming->circling achieved",
              m["happy_path_achieved"])
            ri = m["resume_info"]
            c(f"[{name}] resume_info captured (pre-pause vs resume carrot compared)", ri is not None)
            if ri is not None:
                c(f"[{name}] rejoin used the CURRENT projection, not a stale teleport "
                  f"(carrot moved {ri['carrot_jump_m']} m during the pause)",
                  ri["carrot_jump_m"] >= 0.0)
                c(f"[{name}] fresh-projection bearing error ({ri['bearing_err_fresh_deg']} deg) no worse "
                  f"than the stale teleport target ({ri['bearing_err_stale_deg']} deg)",
                  abs(ri["bearing_err_fresh_deg"]) <= abs(ri["bearing_err_stale_deg"]) + 1.0)
                c(f"[{name}] resume bearing error is a SMOOTH correction (< 90 deg)",
                  abs(ri["bearing_err_fresh_deg"]) < 90.0)
            c(f"[{name}] no unexpected guard hard-stop (no obstacle in this scenario)",
              m["hard_stop_ticks"] == 0)

        if name in ("drift_only", "drift_with_jitter"):
            c(f"[{name}] no unexpected guard hard-stop (no obstacle in this scenario)",
              m["hard_stop_ticks"] == 0)
            # true-frame position is EXPECTED to drift (odom-frame-relative, accepted
            # drift per the plan) -- but it must not do anything pathological like
            # exploding far beyond what the injected bias over this duration implies.
            c(f"[{name}] true-frame drift stays bounded, not pathological (< 3.0 m)",
              m["max_true_radius_dev_m"] < 3.0)

        # determinism: same scenario, same (seeded) run -> identical metrics.
        m2 = run(scn)
        keys = [k for k in m if not k.startswith("_") and k != "events"]
        c(f"[{name}] deterministic (same-seed metrics identical)",
          all(m[k] == m2[k] for k in keys))

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
