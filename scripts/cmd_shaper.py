"""CommandShaper -- jerk- and acceleration-limited velocity shaping for the G1.

WHY
    The web teleop produces a BINARY velocity target: pressing W snaps the forward
    command from 0 to max in one step, releasing snaps it back to 0. Feeding that
    step straight to the locomotion controller makes the gait lurch -- a hard
    weight shift on start, an abrupt plant on stop, jerky direction changes. Real
    machines (and humans) ramp velocity with bounded acceleration and bounded
    jerk (the rate of change of acceleration), which is what produces the smooth
    "ease-in / ease-out" S-curve. This module turns the step target into that
    S-curve on the server, the single point every input source (keyboard, touch,
    a future gamepad) flows through.

WHERE IT SITS (ordering -- this matters for safety)
    operator intent (clamped)            <- state.vx/vy/vyaw, per-axis caps
      -> normalize_xy()                  <- diagonal never exceeds the speed ellipse
      -> shaper.shape(target, dt)        <- THIS: accel + jerk limited ramp
      -> guard.apply(shaped)             <- obstacle governor; INSTANT hard stops
      -> shaper.sync(governed)           <- baseline = what we actually commanded
      -> send to robot
    The shaper limits how fast the OPERATOR INTENT changes. The obstacle guard's
    hard stop still zeroes the output instantly (the shaper is upstream of it), so
    safety is never delayed. sync() then pulls the shaper's state down to whatever
    the guard actually allowed, so when a guard stop clears the robot re-accelerates
    smoothly FROM REST instead of lurching to the speed the ramp drifted to while
    the robot was held stopped.

MODEL (per axis, independent)
    State is (v, a): current velocity and current acceleration. Each tick we pick a
    jerk (+/- j_max) that drives v toward the target while guaranteeing we can bleed
    the acceleration back to zero before overshooting -- the standard jerk-limited
    "brake in time" rule -- then clamp the acceleration to +/- a_max and integrate.
    The result: jerk ramps the accel up, it cruises at a_max, then jerk ramps it
    back down so v eases onto the target with no overshoot. A symmetric a_max gives
    matching ease-in / ease-out; per-axis a_max/j_max are read from config.

This module has NO robot / SDK / rclpy dependency, so it is unit-testable in
isolation (see scripts/test_cmd_shaper.py)."""

import math


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def normalize_xy(vx, vy, max_vx, max_vy):
    """Scale (vx, vy) so it never leaves the speed ELLIPSE (vx/max_vx)^2 +
    (vy/max_vy)^2 <= 1. A pure diagonal (W+A) otherwise has magnitude larger than
    either axis cap, so the robot would translate faster diagonally than straight.
    The axes have different caps (forward != strafe), hence an ellipse not a circle.
    Direction is preserved; only the magnitude is reduced when outside the ellipse."""
    if max_vx <= 0.0 or max_vy <= 0.0:
        return vx, vy
    m = (vx / max_vx) ** 2 + (vy / max_vy) ** 2
    if m > 1.0:
        s = 1.0 / math.sqrt(m)
        return vx * s, vy * s
    return vx, vy


class _Axis:
    """One jerk-limited single-axis velocity tracker. State: v (velocity), a
    (acceleration). step() advances one tick toward `target`."""

    __slots__ = ("v", "a", "a_max", "j_max")

    def __init__(self, a_max, j_max):
        self.v = 0.0
        self.a = 0.0
        self.a_max = max(1e-6, float(a_max))
        self.j_max = max(1e-6, float(j_max))

    def reset(self, v=0.0):
        self.v = float(v)
        self.a = 0.0

    def sync(self, v_actual, tol=1e-3):
        """Force the velocity state to v_actual (what was really commanded after the
        guard). On a meaningful change (the guard scaled or hard-stopped us) also zero
        the acceleration so the next ramp restarts cleanly from the real speed."""
        if abs(v_actual - self.v) > tol:
            self.a = 0.0
        self.v = float(v_actual)

    def step(self, target, dt):
        """Advance one tick toward `target` under accel + jerk limits; return v.

        Sliding-mode jerk-limited law: a_ref is the acceleration whose constant-jerk
        ramp-down to zero would brake velocity exactly onto the target,
        a_ref = sign(err) * sqrt(2 * j_max * |err|), capped at a_max. We slew the
        current acceleration toward a_ref by at most j_max*dt (so jerk <= j_max by
        construction) and integrate. As v -> target, err -> 0, a_ref -> 0 and a -> 0
        together, so velocity eases onto the target with NO snap and NO limit-cycle
        chatter -- the S-curve falls out naturally."""
        if dt <= 0.0:
            return self.v
        v, a = self.v, self.a
        a_max, j_max = self.a_max, self.j_max
        err = target - v

        # Steady state: settle exactly and stop (kills sub-tick discretization drift).
        if abs(err) < 1e-5 and abs(a) < 1e-3:
            self.v, self.a = target, 0.0
            return self.v

        jdt = j_max * dt
        # a_brake: constant-jerk ramp-down to zero brakes exactly onto the target
        # (-> 0 as err -> 0). a_cap: the acceleration that lands on the target in ONE
        # tick (never overshoot). Desire whichever demands LESS |a| -- brake when far,
        # don't-overshoot when close -- then jerk-limit the slew toward it so |Δa| is
        # bounded EVERY tick (the landing included; the old hard v-clamp set
        # a=(target-v)/dt which spiked jerk to 5-7x on small-dt / moving targets).
        a_brake = math.copysign(math.sqrt(2.0 * j_max * abs(err)), err)
        a_cap = err / dt
        a_des = min(a_brake, a_cap) if err >= 0.0 else max(a_brake, a_cap)
        a_des = clamp(a_des, -a_max, a_max)
        a = a + clamp(a_des - a, -jdt, jdt)
        a = clamp(a, -a_max, a_max)

        v_new = v + a * dt
        # Clean, jerk-safe settle: snap to the target only when within a hair AND the
        # acceleration is already small (<= one jerk step), so the snap can never spike
        # jerk. a_cap keeps v from crossing the target, so there is no overshoot.
        if abs(target - v_new) < 1e-4 and abs(a) <= jdt:
            v_new, a = target, 0.0
        self.v, self.a = v_new, a
        return self.v


class CommandShaper:
    """Three-axis (vx, vy, vyaw) jerk/accel-limited velocity shaper.

    cfg is the `motion` section of config/robot.yaml (with baked defaults so a
    missing file/section never crashes). When enabled is False, shape() is a pure
    pass-through (returns the target unchanged) so the feature can be turned off
    without removing the call site."""

    DEFAULTS = {
        "enabled": True,
        "accel_vx": 1.2,       # m/s^2   forward/back acceleration limit
        "accel_vy": 1.0,       # m/s^2   strafe
        "accel_vyaw": 4.0,     # rad/s^2 yaw
        "jerk_vx": 4.0,        # m/s^3   forward/back jerk limit (S-curve sharpness)
        "jerk_vy": 4.0,        # m/s^3   strafe
        "jerk_vyaw": 16.0,     # rad/s^3 yaw
        "normalize_diagonal": True,
    }

    def __init__(self, cfg=None, max_speeds=(1.5, 0.7, 2.0)):
        c = dict(self.DEFAULTS)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if k in c and v is not None:
                    c[k] = v
        self.cfg = c
        self.enabled = bool(c["enabled"])
        self.normalize_diagonal = bool(c["normalize_diagonal"])
        self.max_vx, self.max_vy, self.max_vyaw = (float(max_speeds[0]),
                                                   float(max_speeds[1]),
                                                   float(max_speeds[2]))
        self.ax = _Axis(c["accel_vx"], c["jerk_vx"])
        self.ay = _Axis(c["accel_vy"], c["jerk_vy"])
        self.ayaw = _Axis(c["accel_vyaw"], c["jerk_vyaw"])

    def reset(self):
        """Zero all axis state -- call on mode change / (re)entering walk so a fresh
        walk session never resumes from a stale ramp."""
        self.ax.reset()
        self.ay.reset()
        self.ayaw.reset()

    def normalize(self, target):
        """Ellipse-normalize the (vx, vy) of a target when normalize_diagonal is on,
        so a diagonal never exceeds the speed ellipse. Applied to the operator INTENT
        before the obstacle guard. Independent of `enabled` (over-speed protection
        always applies). yaw is passed through."""
        if not self.normalize_diagonal:
            return (float(target[0]), float(target[1]), float(target[2]))
        vx, vy = normalize_xy(float(target[0]), float(target[1]), self.max_vx, self.max_vy)
        return (vx, vy, float(target[2]))

    def shape(self, target, dt, bypass=(False, False, False)):
        """Jerk/accel-limit each axis toward `target`; return the smoothed
        (vx, vy, vyaw). Applied to the FINAL (post-guard) command, so it smooths BOTH
        the operator's input AND the guard's gentle slow-zone scale-down.

        bypass[i] True snaps axis i straight to target[i] and rebases its ramp -- the
        guard sets these for an EMERGENCY stop (hard stop / FAULT / recovery) so a
        safety stop is INSTANT, never rate-limited, while a non-emergency slow-down on
        the same axis is still eased. Pure pass-through when the shaper is disabled."""
        tvx, tvy, tvyaw = float(target[0]), float(target[1]), float(target[2])
        if not self.enabled:
            self.ax.reset(tvx); self.ay.reset(tvy); self.ayaw.reset(tvyaw)
            return tvx, tvy, tvyaw

        def axis(ax, t, bp):
            if bp:
                ax.reset(t)          # snap + rebase: emergency stop is immediate
                return t
            return ax.step(t, dt)

        return (axis(self.ax, tvx, bypass[0]),
                axis(self.ay, tvy, bypass[1]),
                axis(self.ayaw, tvyaw, bypass[2]))

    def sync(self, actual):
        """Pull axis velocity state to the ACTUAL commanded velocity (post-guard),
        so a guard scale/hard-stop rebases the ramp and there is no lurch on clear."""
        self.ax.sync(float(actual[0]))
        self.ay.sync(float(actual[1]))
        self.ayaw.sync(float(actual[2]))

    def state(self):
        """Current shaped (v, a) per axis -- for telemetry / debugging."""
        return {
            "vx": round(self.ax.v, 3), "vy": round(self.ay.v, 3),
            "vyaw": round(self.ayaw.v, 3),
            "ax": round(self.ax.a, 3), "ay": round(self.ay.a, 3),
            "ayaw": round(self.ayaw.a, 3),
        }
