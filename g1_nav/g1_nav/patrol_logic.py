#!/usr/bin/env python3
"""patrol_logic.py -- pure-Python circle-patrol geometry + interaction state machine.

NO rclpy / ROS import of ANY kind, and no shm/file I/O -- this module is plain
Python + math/dataclasses so it is importable and unit-testable on a plain
workstation, per Track A's design mandate. It is the SHARED logic core for:

  * g1_nav/g1_nav/nodes/circle_patrol.py      -- the real ROS2 node (this module's
    primary consumer): wraps CirclePatrolGeometry, drives NavigateToPose goals.
  * g1_nav/g1_nav/nodes/patrol_supervisor.py  -- wraps PatrolStateMachine to run
    the person-interaction sequence (pausing -> facing -> waving -> waiting ->
    resuming).
  * scripts/sim_circle.py / scripts/test_circle_patrol.py (Track A0, a PARALLEL
    agent's sim/unit-test harness) -- imports this module directly so the sim
    proves the SAME geometry/FSM code the real node runs, not a re-implementation.
  * scripts/sim_runner.py (Track B, mujoco dashboard sim) -- same reason: "Sim
    mode" must run the identical high-level decision logic as the real robot.

STATE VOCABULARY (mandated, verbatim, shared across every track/consumer of
/dev/shm/g1_patrol_state.json -- see the plan's "Cross-workstream decisions"):

    circling, pausing, facing, waving, waiting, resuming,
    fault, manual_override, paused_odom_degraded

Do not rename or add to this vocabulary without updating every consumer above.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def wrap(angle_rad):
    """Wrap an angle to (-pi, pi]. Same convention as scripts/fused_odometry.wrap
    (duplicated here, not imported, to keep this module dependency-free of
    anything outside the stdlib)."""
    a = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


# =============================================================================
# Circle-patrol pure-pursuit geometry
# =============================================================================

@dataclass
class CirclePatrolGeometry:
    """Fixed-circle pure-pursuit carrot-point geometry.

    The circle CENTER is fixed exactly ONCE, from the robot's pose at patrol
    start (see re_anchor_if_needed()) -- there is no persistent map / AMCL in
    this codebase (g1_nav is a map-free, rolling-costmap stack; see
    MIGRATION_STATUS.md), so the center is expressed in whatever frame the pose
    was given in (odom, in the real node) and DRIFTS along with it for the rest
    of the patrol. This is the plan's explicitly accepted design: "fixes circle
    center once from current pose at patrol start ... this is odom-frame-
    relative, accepted drift."

    We deliberately do NOT re-anchor the CENTER on radial drift mid-patrol --
    that would fight the odom drift instead of accepting it, and would make the
    "circle" itself jump around underneath the robot. carrot_point() instead
    re-projects the robot's CURRENT angular position onto the fixed circle on
    every call, which self-corrects small radial drift for free: a robot that
    has drifted off the circle's radius still gets a carrot exactly ON the
    circle, gently pulling it back in. Only the CENTER is a one-time anchor;
    the tracking geometry is recomputed fresh every tick.
    """

    radius_m: float = 3.0
    lookahead_m: float = 1.25        # 1.0-1.5 m per the plan
    direction: int = 1                # +1 = CCW (left-turning), -1 = CW
    center: Optional[Tuple[float, float]] = None
    anchored: bool = False

    def __post_init__(self):
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be > 0")
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 (CCW) or -1 (CW)")

    def re_anchor_if_needed(self, x, y, yaw, force=False):
        """Fix the circle center from the CURRENT pose -- ONCE.

        No-op (returns False) if already anchored and force=False: this is the
        "fixed once at patrol start" design, not a per-tick re-anchor. Pass
        force=True only on a deliberate fresh-restart (e.g. the supervisor
        starting a brand-new lap after a long manual_override where the old
        circle is no longer relevant).

        Places the center `radius_m` to the robot's LEFT (direction=+1, CCW) or
        RIGHT (direction=-1, CW) of its current heading, so the CURRENT pose
        lands exactly ON the circle and the robot can start circling
        immediately without first driving to a distant center.

        Returns True iff the center was (re)anchored this call.
        """
        if self.anchored and not force:
            return False
        side = 1.0 if self.direction >= 0 else -1.0
        heading = yaw + side * (math.pi / 2.0)
        cx = x + self.radius_m * math.cos(heading)
        cy = y + self.radius_m * math.sin(heading)
        self.center = (cx, cy)
        self.anchored = True
        return True

    def _require_anchor(self):
        if not self.anchored or self.center is None:
            raise RuntimeError(
                "CirclePatrolGeometry: call re_anchor_if_needed() before "
                "using carrot_point()/carrot_pose() -- no circle center fixed yet.")

    def phase_angle(self, x, y):
        """Angle (rad) of the robot's CURRENT position about the fixed center."""
        self._require_anchor()
        cx, cy = self.center
        return math.atan2(y - cy, x - cx)

    def carrot_point(self, x, y, lookahead_m=None):
        """Pure-pursuit carrot point ON the circle, `lookahead_m` of arc length
        ahead of the robot's current angular position, in the patrol direction.

        Self-corrects radial drift (see class docstring): a robot that is
        currently off the circle (closer to or farther from the center than
        radius_m) still gets a carrot exactly on the circle.

        Returns (x, y).
        """
        self._require_anchor()
        la = self.lookahead_m if lookahead_m is None else lookahead_m
        theta = self.phase_angle(x, y)
        dtheta = self.direction * (la / self.radius_m)
        theta2 = theta + dtheta
        cx, cy = self.center
        return (cx + self.radius_m * math.cos(theta2),
                cy + self.radius_m * math.sin(theta2))

    def carrot_pose(self, x, y, lookahead_m=None):
        """carrot_point() plus a tangent heading (rad, wrapped to (-pi, pi])
        facing the patrol direction of travel -- what a NavigateToPose goal
        needs for its orientation, not just its position.

        Returns (x, y, yaw_rad).
        """
        self._require_anchor()
        la = self.lookahead_m if lookahead_m is None else lookahead_m
        theta = self.phase_angle(x, y)
        dtheta = self.direction * (la / self.radius_m)
        theta2 = theta + dtheta
        cx, cy = self.center
        px = cx + self.radius_m * math.cos(theta2)
        py = cy + self.radius_m * math.sin(theta2)
        heading = theta2 + self.direction * (math.pi / 2.0)
        return (px, py, wrap(heading))

    def radial_drift_m(self, x, y):
        """|distance-from-center - radius_m| -- a DIAGNOSTIC/telemetry value
        only. The geometry does not correct this mid-patrol (see class
        docstring); this is just for logging/dashboards to see how far odom
        drift has pulled the robot off its nominal circle."""
        self._require_anchor()
        cx, cy = self.center
        return abs(math.hypot(x - cx, y - cy) - self.radius_m)

    def reset(self):
        """Drop the anchor so the NEXT re_anchor_if_needed() call takes effect
        (e.g. after the robot has been carried/manually driven far from the old
        center and a fresh lap should pick a brand-new one)."""
        self.center = None
        self.anchored = False


# =============================================================================
# Interaction state machine
# =============================================================================

STATES = (
    "circling", "pausing", "facing", "waving", "waiting", "resuming",
    "fault", "manual_override", "paused_odom_degraded",
)
STATES_SET = frozenset(STATES)

# Happy-path transition table (state -> allowed next states along the normal
# cycle):  circling -> pausing -> facing -> waving -> waiting -> resuming -> circling
_HAPPY_PATH = {
    "circling": ("pausing",),
    "pausing": ("facing",),
    "facing": ("waving",),
    "waving": ("waiting",),
    "waiting": ("resuming",),
    "resuming": ("circling",),
}

# "Interrupt" states reachable from ANY other state, per the plan: manual_override
# and paused_odom_degraded are explicitly called out as reachable from anywhere.
# DESIGN DECISION (not fully specified by the plan): `fault` is treated the same
# way. The plan lists `fault` in the mandated vocabulary without giving it a
# specific transition rule; a catch-all safety state that could NOT actually be
# reached from anywhere would be useless as a safety net, so it is wired
# identically to the other two interrupts.
_INTERRUPT_STATES = ("manual_override", "paused_odom_degraded", "fault")

# Where an interrupt state may go once it clears. Kept deliberately narrow: the
# only sanctioned re-entry point is the TOP of the happy path, `circling` --
# never resume mid-interaction (facing/waving/waiting) after an override/fault,
# since the paused-mid-gesture context cannot be trusted any more (the person
# may have moved, the odom estimate may have jumped, an operator may have
# physically relocated the robot, etc). Restarting the lap from `circling` is
# the simple, safe recovery; the node layer (circle_patrol.py) separately
# decides whether to re-anchor a NEW circle center or keep the old one.
_INTERRUPT_EXITS = {
    "manual_override": ("circling",),
    "paused_odom_degraded": ("circling",),
    "fault": ("circling",),
}


class InvalidTransition(RuntimeError):
    """Raised by PatrolStateMachine.transition() on a disallowed state change."""


@dataclass
class PatrolStateMachine:
    """Explicit transition-table FSM over the mandated patrol/interaction state
    vocabulary (see module docstring). Pure Python: no ROS, no I/O, no
    threading, no clock -- callers own timing/debounce and just call
    transition()/try_transition() when THEY decide a move should happen.

    Self-transitions (state -> the SAME state) are always allowed and are a
    no-op: callers may re-affirm/tick the current phase every cycle without the
    FSM treating that as an error (a deliberate ergonomic choice -- the
    alternative would force every driving loop to special-case "am I already
    here?" before every call).
    """

    state: str = "circling"
    history: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.state not in STATES_SET:
            raise ValueError(f"unknown initial state {self.state!r}")
        if not self.history:
            self.history = [self.state]

    def allowed_next(self):
        """Set of states reachable from the CURRENT state in one transition
        (excluding the trivial self-loop, which can_transition() always allows
        separately)."""
        nxt = set(_INTERRUPT_STATES)          # reachable from anywhere...
        nxt.discard(self.state)               # ...except back into itself (that's the self-loop)
        if self.state in _HAPPY_PATH:
            nxt.update(_HAPPY_PATH[self.state])
        if self.state in _INTERRUPT_EXITS:
            nxt.update(_INTERRUPT_EXITS[self.state])
        return nxt

    def can_transition(self, to_state):
        """True iff `to_state` is a legal next state (or the current state)."""
        if to_state not in STATES_SET:
            return False
        if to_state == self.state:
            return True                       # self-loop always allowed
        return to_state in self.allowed_next()

    def transition(self, to_state):
        """Move to `to_state`. Raises InvalidTransition if not allowed."""
        if not self.can_transition(to_state):
            raise InvalidTransition(
                f"illegal patrol transition: {self.state!r} -> {to_state!r} "
                f"(allowed: {sorted(self.allowed_next() | {self.state})})")
        self.state = to_state
        self.history.append(to_state)
        return self.state

    def try_transition(self, to_state):
        """Non-raising variant of transition(): returns True on success, False
        (state unchanged) on an illegal move."""
        try:
            self.transition(to_state)
            return True
        except InvalidTransition:
            return False

    def reset(self, state="circling"):
        """Hard-reset the FSM to `state` (default "circling"), clearing
        history. Use for a genuine fresh start (e.g. node restart), not for
        normal operation -- normal recovery from an interrupt goes through
        transition(), not reset()."""
        if state not in STATES_SET:
            raise ValueError(f"unknown reset state {state!r}")
        self.state = state
        self.history = [state]


# =============================================================================
# Camera pixel-bearing mapping (optional shared helper for person_fusion.py)
# =============================================================================
#
# scripts/depth_nearfield.py fuses the D435i's own 3D point cloud (already
# unprojected; it never does pixel-column -> bearing arithmetic directly) and
# consistently uses the D435i's datasheet HORIZONTAL FOV -- 87 deg / 1.5184 rad
# -- as its angular reference (see its "ring_fov_half_deg": 43.0 default, and
# the identical 1.5184 rad constant in g1_nav/config/nav2_params.yaml's
# `horizontal_fov_angle` for the realsense STVL source). D435I_HFOV_RAD below
# is that SAME constant, reused verbatim rather than re-measured, so a person's
# detected pixel box maps to a bearing using the one FOV value already trusted
# elsewhere in this codebase for this exact camera.

D435I_HFOV_RAD = 1.5184     # rad; 87 deg, Intel D435i depth datasheet (matches
                            # scripts/depth_nearfield.py's ring_fov_half_deg*2
                            # and g1_nav/config/nav2_params.yaml's realsense
                            # horizontal_fov_angle -- same camera, same constant)


# =============================================================================
# Exhibition mode -- roam-inside-the-disc segments + a lookahead safety oracle
# =============================================================================
#
# Added on top of the already-tested circle-walking core above WITHOUT touching
# it: CirclePatrolGeometry/PatrolStateMachine/STATES are all unmodified (see
# their own docstrings' "mandated, do not rename" warning -- extending that
# vocabulary would ripple into every consumer: circle_patrol.py,
# patrol_supervisor.py, sim_circle.py, sim_runner.py, web/nav.js). Exhibition
# mode instead answers a DIFFERENT question -- "what should the robot be doing
# while PatrolStateMachine says circling (i.e. NOT currently greeting a
# person)?" -- with a richer, randomized-but-vetted repertoire instead of pure
# circumference-following. g1_nav/g1_nav/nodes/exhibition_conductor.py is the
# sole real-node consumer; scripts/sim_runner.py is the sim-side consumer
# (same "sim runs the identical decision logic" property as the rest of this
# module).
#
# DESIGN: dances are NEVER drawn by the weighted random pool below -- an
# operator requests one from the dashboard, and the conductor decides *when*
# it's safe to actually perform it (see plan_next_segment's `pending_dance`
# param). This is a deliberate, explicit product decision (not a technical
# limitation): random selection of a whole-body FSM routine at an arbitrary
# moment was rejected as unsafe/unwanted; see the plan file's "Track A
# Extension" section, decision #3.

@dataclass
class ExhibitionSegment:
    """One unit of exhibition behavior. `kind` is one of:
      "roam"    -- walk in a straight line to `target` (x, y) inside the disc.
      "gesture" -- fire `gesture` (an OPEN_GESTURES name, e.g. "high_wave") in
                   place via the existing safety-gated dashboard WS path.
      "idle"    -- a brief stationary pause (segment-gap breathing room, and the
                   universal safe fallback element when nothing else fits).
      "dance"   -- fire `dance_fsm_id` in place. ONLY ever constructed from an
                   operator's pending dashboard request (see module docstring
                   above) -- never drawn by _draw_candidate_segment's weighted
                   pool. `dance_space_m` is the clearance radius the dances.yaml
                   catalog entry says this routine needs, carried here so
                   predict_segment_end() doesn't need its own YAML/catalog
                   access (this module stays pure Python / zero I/O).
    """
    kind: str
    target: Optional[Tuple[float, float]] = None
    gesture: Optional[str] = None
    dance_fsm_id: Optional[int] = None
    dance_name: Optional[str] = None
    dance_space_m: Optional[float] = None
    duration_s: float = 0.0


def random_point_in_disc(center, radius_m, edge_clearance_m, rng,
                          min_dist_from=None, min_dist_m=0.0, max_attempts=20):
    """A random point strictly inside a disc of `radius_m` around `center`,
    shrunk by `edge_clearance_m` for safety margin, optionally re-rolled until
    it is at least `min_dist_m` from `min_dist_from` (e.g. the robot's current
    position, so a "roam" segment is never a pointless few-cm nudge).

    Uniform over the disc (not just uniformly-random radius, which would over-
    sample the center): r = safe_radius * sqrt(u), theta = uniform(0, 2*pi),
    the standard disc-sampling identity. Bounded retries for the min-distance
    constraint; if every attempt fails (e.g. min_dist_m is larger than the
    disc itself), falls back to `center` -- always safe by construction, since
    `center` trivially satisfies the edge-clearance margin whenever
    edge_clearance_m < radius_m.
    """
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


def _in_safe_disc(pt, center, radius_m, margin_m):
    cx, cy = center
    return math.hypot(pt[0] - cx, pt[1] - cy) <= max(radius_m - margin_m, 0.0)


def predict_segment_end(pose, segment, center, radius_m, edge_clearance_m,
                         roam_speed_mps=0.5):
    """The oracle's single-segment rollout + safety check. Pure kinematics, no
    I/O, no ROS -- callable thousands of times a second if needed.

    Returns (end_pose, ok, reason).

    roam: only the ENDPOINT needs checking against the shrunk disc -- a chord
    between two points that are both inside a convex disc never exits it, so
    the STARTING pose is assumed already-valid (it came from a prior accepted
    segment, or the real robot's actual current position) and no per-step
    path sampling is needed. New heading points toward the target.

    dance: stationary; checked against max(edge_clearance_m, the segment's own
    dance_space_m) -- a dance typically needs MORE clearance than the generic
    edge margin, never less, so the stricter of the two always wins.

    gesture / idle: stationary; checked against the generic edge_clearance_m.
    """
    x, y, yaw = pose
    if segment.kind == "roam":
        tx, ty = segment.target
        if not _in_safe_disc((tx, ty), center, radius_m, edge_clearance_m):
            return pose, False, "roam target outside safety margin"
        return (tx, ty, math.atan2(ty - y, tx - x)), True, "ok"
    if segment.kind == "dance":
        margin = max(edge_clearance_m, segment.dance_space_m or edge_clearance_m)
        if not _in_safe_disc((x, y), center, radius_m, margin):
            return pose, False, "insufficient clearance for dance space_m"
        return pose, True, "ok"
    if segment.kind in ("gesture", "idle"):
        if not _in_safe_disc((x, y), center, radius_m, edge_clearance_m):
            return pose, False, "current position outside safety margin"
        return pose, True, "ok"
    return pose, False, f"unknown segment kind {segment.kind!r}"


def _draw_weighted_kind(cfg_segments, rng):
    """Weighted random pick from cfg["segments"] (each a dict with at least
    "kind" and "weight"). Entries with weight <= 0 are never drawn (e.g. the
    shipped config's "idle" entry has weight 0 -- it is a fallback-only
    element, reached through the oracle's guaranteed-safe path below, not
    through this random draw). Returns None if every weight is <= 0."""
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


def _draw_candidate_segment(pose, cfg, rng, pending_dance=None):
    """One candidate segment: the pending operator-requested dance if one was
    passed in (checked by the caller only on the FIRST slot of a chain -- see
    plan_next_segment), otherwise a weighted-random draw from cfg["segments"].

    SAFETY NET: if a "dance" entry somehow ends up in cfg["segments"]'s
    weighted pool (a config mistake -- dances must only ever arrive via
    pending_dance, per this module's design), it is refused here and downgraded
    to "idle" rather than firing an unrequested whole-body routine.
    """
    if pending_dance is not None:
        return pending_dance
    picked = _draw_weighted_kind(cfg["segments"], rng)
    if picked is None:
        return ExhibitionSegment(kind="idle", duration_s=cfg.get("idle_duration_s", 1.0))
    kind = picked.get("kind")
    if kind == "roam":
        target = random_point_in_disc(
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
    # "dance" (config mistake -- see docstring) or anything unrecognized: fail safe.
    return ExhibitionSegment(kind="idle", duration_s=cfg.get("idle_duration_s", 1.0))


def plan_next_segment(pose, cfg, rng, pending_dance=None):
    """THE ORACLE. Decides the next segment to actually execute, right now.

    cfg (a plain dict) must provide: center=(cx,cy), radius_m, edge_clearance_m,
    roam_speed_mps, roam_min_dist_m, segments (weighted list, see
    _draw_candidate_segment), lookahead_segments (default 5, per the plan),
    max_reroll_attempts (default 8).

    For each of up to max_reroll_attempts tries: draw a first candidate (the
    pending dance request on the FIRST try only -- if it's unsafe THIS tick,
    fall through to an ordinary random draw rather than hammering the same
    currently-unsafe request; the request itself stays queued and the CALLER
    re-offers it next tick), then roll (lookahead_segments - 1) MORE
    hypothetical random draws forward from wherever the first would leave the
    robot, checking every segment in the chain with predict_segment_end(). If
    the whole chain checks out, return the FIRST segment -- only it is ever
    actually executed; the rest of the chain was purely a preview so a pick
    that LOOKS fine in isolation but leads somewhere bad a few moves later
    gets rejected too. This is re-run fresh at the next decision point
    (receding-horizon), not a 5-segment commitment.

    If every attempt fails (should be rare given the geometry, but is a hard
    safety backstop, not a "shouldn't happen"), fall back to the one segment
    that is ALWAYS safe by construction: roam straight to the disc's own
    center.
    """
    lookahead = max(1, int(cfg.get("lookahead_segments", 5)))
    max_attempts = max(1, int(cfg.get("max_reroll_attempts", 8)))

    for attempt in range(max_attempts):
        first = _draw_candidate_segment(
            pose, cfg, rng, pending_dance if attempt == 0 else None)
        chain_pose, ok, _reason = predict_segment_end(
            pose, first, cfg["center"], cfg["radius_m"], cfg["edge_clearance_m"],
            cfg.get("roam_speed_mps", 0.5))
        if not ok:
            continue
        ok_chain = True
        for _ in range(lookahead - 1):
            nxt = _draw_candidate_segment(chain_pose, cfg, rng, None)
            chain_pose, ok, _reason = predict_segment_end(
                chain_pose, nxt, cfg["center"], cfg["radius_m"], cfg["edge_clearance_m"],
                cfg.get("roam_speed_mps", 0.5))
            if not ok:
                ok_chain = False
                break
        if ok_chain:
            return first

    # Guaranteed-safe fallback: the center is always inside (radius_m -
    # edge_clearance_m) of itself, so this can never fail predict_segment_end.
    cx, cy = cfg["center"]
    speed = max(cfg.get("roam_speed_mps", 0.5), 1e-3)
    return ExhibitionSegment(kind="roam", target=(cx, cy),
                              duration_s=math.hypot(pose[0] - cx, pose[1] - cy) / speed)


def pixel_x_to_bearing_rad(px, frame_w, hfov_rad=D435I_HFOV_RAD):
    """Map a detection-box pixel x-coordinate to a bearing (rad) from the
    camera boresight, using a pinhole model (exact across the whole frame, not
    a small-angle linear approximation): bearing = atan2(half_w - px, focal_px),
    focal_px derived from the camera's horizontal FOV so a box exactly at the
    image edge maps to +-hfov/2.

    Sign convention matches the rest of this codebase's ring/guard math (e.g.
    obstacle/guard.py's ring angle, scripts/depth_nearfield.py's
    `ang = atan2(left, fwd)`): 0 = dead ahead, POSITIVE = LEFT (px increases
    rightward, so a smaller px than half_w is to the left -> positive bearing).
    """
    if frame_w <= 0:
        raise ValueError("frame_w must be > 0")
    half_w = frame_w / 2.0
    focal_px = half_w / math.tan(hfov_rad / 2.0)
    return math.atan2(half_w - px, focal_px)
