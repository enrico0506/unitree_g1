"""StepPacer -- discrete-step (stutter-step) pacing for the G1 teleop.

WHY
    The Unitree onboard locomotion controller maps a commanded VELOCITY to a gait
    internally; the SDK exposes NO step-length / stride / cadence knob (only
    SET_VELOCITY vx/vy/vyaw, SET_SWING_HEIGHT, SET_STAND_HEIGHT -- GET_PHASE is
    deprecated with no G1 client wrapper). So lowering vx mostly lowers the gait
    CADENCE, not the stride: "press the arrow longer at low speed" yields the
    SAME-size step, just less often. The operator wants a genuinely SHORTER stride
    to maneuver in tight spaces -- forward, back, strafe L/R, rotate L/R.

WHAT
    The only server-side way to force a short stride is to feed the controller a
    short BURST of velocity then a commanded ZERO long enough for the swing foot to
    plant and the robot to reach double-stance before the next burst. StepPacer is a
    PURE transform `modulate(intent, now) -> gated_intent` that chops the held
    continuous operator intent into ON-burst / OFF-settle pulses, so the controller
    is forced to take one short, fully-settled discrete step per burst, in EVERY
    direction. It defaults to mode "continuous" = pure pass-through, so behaviour is
    byte-for-byte identical to today until the operator selects "small" / "medium".

WHERE IT SITS (ordering -- this matters for safety)
    operator intent (clamped)            <- state.vx/vy/vyaw, per-axis caps
      -> shaper.normalize()              <- diagonal never exceeds the speed ellipse
      -> pacer.modulate(intent, now)     <- THIS: chop the held intent into pulses
      -> guard.apply(pulsed)             <- obstacle governor; INSTANT hard stops
      -> shaper.shape(governed, bypass)  <- accel + jerk limited ramp (square -> trapezoid)
      -> send to robot
    The pacer is UPSTREAM of the guard, so the ObstacleGuard remains the LAST safety
    authority over the pulsed signal: in an OFF window the intent is already
    (0,0,0), so the guard is a no-op there; in an ON burst the guard can still
    slow-zone-scale OR hard-stop it, and because the pacer is upstream it can never
    re-inject velocity into an axis the guard hard-stopped. The pacer NEVER sets the
    shaper bypass -- both pulse edges stay jerk/accel-limited (bypass is reserved
    exclusively for the guard's emergency stop), so each square ON/OFF edge becomes
    an executable trapezoid and the actual stride is the AREA under the shaped pulse.

ONE COUPLED PHASE CLOCK (for safe diagonals)
    The three axes share ONE ON/OFF phase clock, not three independent gates, so a
    diagonal (W+A) or a translate+turn pulses as ONE coherent stride. The active
    window lengths are the MAX over the currently-demanded axes, so a combined burst
    always gets the MOST conservative (longest) settle of its contributing DOFs --
    the least-stable axis (strafe) is never short-changed. Axis membership and
    direction SIGN are LATCHED at each ON edge, so a key pressed or flipped mid-burst
    cannot inject a fresh lateral lurch into an in-flight step; it joins at the next
    burst.

OPTIONAL DISTANCE-QUANTIZED REFINEMENT (default OFF via distance_quantize=false)
    When enabled AND odom is live, the ON window stays open until the MEASURED odom
    displacement for a demanded axis reaches a per-mode target (meters / radians),
    bounded by a drive_timeout_s backstop, then forces the OFF settle -- closing the
    loop on distance for repeatable stride regardless of shaper attenuation / battery
    / posture. If odom goes stale it transparently degrades to the pure time-based
    pulse (never a free-running velocity); a telemetry flag surfaces estimated-vs-
    measured stride. Layered ON TOP of the proven time base; ships OFF until
    gantry-validated.

This module has NO robot / SDK / rclpy dependency (only `import math`), so it is
unit-testable in isolation (see scripts/test_step_pacer.py), mirroring cmd_shaper.py.
"""

import math


IDLE_EPS_DEFAULT = 1e-3   # |intent| above this on an axis == the operator is holding it


def ang_wrap(d):
    """Wrap an angle difference to [-pi, pi] (shortest-angle), for yaw distance mode."""
    return (d + math.pi) % (2.0 * math.pi) - math.pi


class _AxisCfg:
    """Per-axis, per-mode pacing params.

    mag_frac : burst magnitude as a fraction of the axis velocity cap (0..1). The
               SHAPER attenuates a short pulse, so the realized peak is lower -- tune
               by odom displacement per pulse, NOT by this commanded peak.
    on_s     : ON (burst) window length, seconds. Just long enough for the shaper to
               jerk the velocity up and induce one step.
    off_s    : OFF (settle) window length, seconds. THE single most safety-critical
               param: >= one full foot-plant/settle so each induced step COMPLETES
               and the robot reaches double-stance before the next burst.
    target   : distance-mode stride target (meters for vx/vy, radians for vyaw).
               Used ONLY when distance_quantize is on and odom is live.
    """

    __slots__ = ("mag_frac", "on_s", "off_s", "target")

    def __init__(self, mag_frac, on_s, off_s, target):
        self.mag_frac = float(mag_frac)
        self.on_s = max(1e-3, float(on_s))
        self.off_s = max(1e-3, float(off_s))
        self.target = max(1e-4, float(target))


# Baked defaults for the whole 'step' section so a missing file / section / key
# never crashes the pacer (same pattern as CommandShaper.DEFAULTS). CONSERVATIVE:
# long off_s settle, small magnitudes; tune on the gantry by odom displacement.
# off_s is now only the MINIMUM settle (the SETTLE GATE extends it until the shaper's
# realized velocity is ~0). mag_frac is intentionally CONSERVATIVE -- the shaper inflates
# a short pulse's integral, so realized strides ran 1.5-2.25x over target in sim; strafe
# (vy, least laterally stable) is deliberately the SMALLEST burst, not the largest. Final
# strides MUST be tuned on the gantry by measured odom displacement (or enable
# distance_quantize). These are safe starting points, not calibrated values.
# CRITICAL: the burst must reach a velocity ABOVE the robot's minimum stepping speed or
# the controller just sways in place (a low plateau = a STALL, not a small step). on_s is
# kept long enough for the jerk-limited shaper to ramp INTO the stepping range AND give a
# fairly uniform pulse; mag_frac sets the burst velocity (the "will it step" knob -- raise
# if it stalls). The smallest reliable velocity-pulsed step is bounded BELOW by
# V_min_step * one step period (~0.2 m), so "small" is ~0.2 m, not 0.1. off_s is only the
# settle MINIMUM. ALL of these need gantry verification -- raise on stall, lower on_s to shrink.
_MODE_DEFAULTS = {
    "small": {
        # The realized burst peak is ACCEL-limited by on_s (mag is above the cap), so on_s
        # is what sets the step speed. ~0.34 m/s (on_s 0.44) was BELOW the clean-step
        # threshold -> half-step + rock back. ~0.44 m/s (on_s 0.50) COMPLETES the weight
        # transfer cleanly (operator-confirmed). This is about the floor for a clean
        # velocity-pulsed step; longer off_s keeps the deliberate stomp cadence.
        "vx":   {"mag_frac": 0.48, "on_s": 0.54, "off_s": 0.45, "target": 0.22},  # ~0.50 m/s (faster+snappier)
        "vy":   {"mag_frac": 0.62, "on_s": 0.55, "off_s": 0.50, "target": 0.16},  # strafe (slower axis)
        "vyaw": {"mag_frac": 0.65, "on_s": 0.42, "off_s": 0.44, "target": 0.22},  # yaw: ~1.1 rad/s rate
        "swing_height": 0.06,   # SHUFFLE: lower foot lift -> a smaller step completes cleanly
    },
    "medium": {
        "vx":   {"mag_frac": 0.55, "on_s": 0.60, "off_s": 0.45, "target": 0.35},  # realized ~0.6 m/s
        "vy":   {"mag_frac": 0.75, "on_s": 0.60, "off_s": 0.50, "target": 0.28},
        "vyaw": {"mag_frac": 0.95, "on_s": 0.46, "off_s": 0.42, "target": 0.35},  # yaw: ~1.5 rad/s rate
        "swing_height": None,
    },
}

VALID_MODES = ("continuous", "small", "medium")


def _build_modes(cfg):
    """Parse step.small / step.medium from the config dict into per-axis _AxisCfg,
    falling back to _MODE_DEFAULTS for any missing mode / axis / key. Returns
    {"small": [ax_vx, ax_vy, ax_vyaw], "medium": [...]} -- the same axis order as the
    velocity tuple (0=vx, 1=vy, 2=vyaw)."""
    out = {}
    for mode, dflt in _MODE_DEFAULTS.items():
        src = cfg.get(mode) if isinstance(cfg.get(mode), dict) else {}
        axes = []
        for axis_key in ("vx", "vy", "vyaw"):
            d = dict(dflt[axis_key])
            s = src.get(axis_key) if isinstance(src.get(axis_key), dict) else {}
            for k in ("mag_frac", "on_s", "off_s", "target"):
                if s.get(k) is not None:
                    d[k] = s[k]
            axes.append(_AxisCfg(d["mag_frac"], d["on_s"], d["off_s"], d["target"]))
        out[mode] = axes
    return out


class StepPacer:
    """Three-axis (vx, vy, vyaw) discrete-step pacer.

    cfg is the `step` section of config/robot.yaml (with baked defaults so a missing
    file / section never crashes). max_speeds = (MAX_VX, MAX_VY, MAX_VYAW) for the
    per-axis burst magnitude. pose_fn / live_fn are OdomReader.get_pose / .is_live
    (or None) for the optional distance-quantized refinement.

    modulate(intent, now) is a PURE transform: it never calls the robot, never
    blocks, and holds only its own pulse state plus a cheap odom read. When disabled
    OR in mode "continuous" it returns the intent UNCHANGED, so behaviour is
    byte-for-byte identical to today until the operator opts in."""

    AXES = 3   # 0 = vx, 1 = vy, 2 = vyaw

    def __init__(self, cfg=None, max_speeds=(1.5, 0.7, 2.0),
                 pose_fn=None, live_fn=None):
        c = cfg if isinstance(cfg, dict) else {}
        self.max = (float(max_speeds[0]), float(max_speeds[1]), float(max_speeds[2]))
        self.enabled = bool(c.get("enabled", True))            # master kill-switch
        self.mode = c.get("default_mode", "continuous")        # MUST default continuous
        if self.mode not in VALID_MODES:
            self.mode = "continuous"
        self.modes = _build_modes(c)
        self.idle_eps = float(c.get("idle_eps", IDLE_EPS_DEFAULT))
        self.dq = bool(c.get("distance_quantize", False))      # OFF until gantry-validated
        self.drive_timeout = float(c.get("drive_timeout_s", 1.5))
        # SETTLE GATE: the OFF window does not end until the SHAPER's realized velocity
        # (fed back as vel_fb) on every demanded axis is below this epsilon -- proof the
        # induced step has finished and the foot planted. off_s is only the MINIMUM; this
        # closes the loop on the shaper so steps never merge regardless of its accel/jerk
        # gains, and keeps a diagonal's axes phase-locked (both restart from rest together).
        self.settle_eps = (
            float(c.get("settle_eps_lin", 0.03)),    # vx (m/s)
            float(c.get("settle_eps_lin", 0.03)),    # vy (m/s)
            float(c.get("settle_eps_ang", 0.08)),    # vyaw (rad/s)
        )
        self.off_max_s = float(c.get("off_max_s", 2.0))        # backstop: never wait forever to settle
        # Odom hooks for the optional distance refinement; may be None (-> time mode).
        self.pose = pose_fn if callable(pose_fn) else None
        self.live = live_fn if callable(live_fn) else None
        # Optional per-mode SET_SWING_HEIGHT (m) for a lower shuffle on small steps;
        # None = leave the controller default. Read by the caller, not used here.
        self.swing_height = {
            m: (c.get(m, {}) or {}).get("swing_height")
            if isinstance(c.get(m), dict) else None
            for m in ("small", "medium")
        }
        self.estimated = False   # telemetry: True if distance mode fell back to time
        self.reset()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_mode(self, name):
        """Switch step mode (WS 'step_mode' handler). Resets the pulse state so the
        pacer never resumes mid-pulse on a mode switch -- the next held key starts a
        clean burst, and selecting 'continuous' instantly returns to pass-through."""
        if name in VALID_MODES and name != self.mode:
            self.mode = name
            self.reset()

    def reset(self):
        """Clear all pulse state -- call on (re)entering walk, the watchdog stop, and
        a mode change, so a fresh / recovered walk session never resumes mid-pulse and
        can't emit a stale surprise burst (mirrors CommandShaper.reset())."""
        self.phase = "OFF"                    # start in settle/neutral, never mid-burst
        self.t_start = 0.0
        self.locked = (False, False, False)   # axis membership latched at the ON edge
        self.sign = (0.0, 0.0, 0.0)           # direction latched at the ON edge
        self.anchor = (0.0, 0.0, 0.0)         # odom anchor for distance mode
        self.estimated = False

    def telemetry(self):
        """Server->browser dict merged into the telemetry message so the UI reflects
        the authoritative server mode + the optional measured/estimated chip."""
        return {
            "step_mode": self.mode,
            "step_enabled": bool(self.enabled),
            "step_phase": self.phase,
            "step_estimated": bool(self.estimated),
        }

    def modulate(self, intent, now, vel_fb=(0.0, 0.0, 0.0)):
        """Chop the held continuous intent into ON-burst / OFF-settle pulses.

        intent = (vx, vy, vyaw), already ellipse-clamped and finite. vel_fb is the
        REALIZED (shaper-output) velocity from the previous tick -- the OFF window stays
        open until it has settled to ~0 on every demanded axis (foot planted) so steps
        cannot merge; callers that pass no feedback get pure off_s timing. Returns a
        gated (vx, vy, vyaw): the calibrated burst on the latched axes during an ON
        window, else (0, 0, 0). In mode 'continuous' (or when disabled) the intent is
        returned UNCHANGED -- the SAFE DEFAULT that nothing changes until the operator
        opts in.
        """
        if not self.enabled or self.mode == "continuous":
            return intent                     # SAFE DEFAULT: byte-for-byte unchanged

        P = self.modes[self.mode]             # [ax_vx, ax_vy, ax_vyaw]
        demand = tuple(abs(intent[i]) > self.idle_eps for i in range(self.AXES))
        if not any(demand):
            # Operator released everything -> clean neutral; restart on the next press.
            self.reset()
            return (0.0, 0.0, 0.0)

        # Coupled phase clock: ON/OFF window lengths are the MAX over the demanded
        # axes, so a combined burst always gets the MOST conservative settle.
        on_s = max(P[i].on_s for i in range(self.AXES) if demand[i])
        off_s = max(P[i].off_s for i in range(self.AXES) if demand[i])
        elapsed = now - self.t_start
        if elapsed < 0.0:                     # wall clock stepped backwards -> resync,
            self.t_start = now                # never let a negative elapsed WEDGE the gate
            elapsed = 0.0                     # (low-grade runaway). monotonic clock + this.

        if self.phase == "OFF":
            # SETTLE-GATED: off_s is the MINIMUM; the burst only (re)starts once the
            # shaper's realized velocity has returned to ~0 on every demanded axis (foot
            # planted, double-stance), bounded by off_max_s so it can never hang. This is
            # what makes the steps DISCRETE instead of merging into a glide, and keeps a
            # diagonal's axes phase-locked (both restart from rest in the same burst).
            if elapsed >= off_s and (self._settled(demand, vel_fb)
                                     or elapsed >= self.off_max_s):
                self._begin_burst(intent, demand, now)
                return self._burst(P)
            return (0.0, 0.0, 0.0)           # actively command 0 -> step ends, foot plants

        # phase == "ON"
        if self._on_done(P, on_s, elapsed):  # time- OR distance-terminated
            self.phase = "OFF"
            self.t_start = now
            return (0.0, 0.0, 0.0)           # begin the settle
        return self._burst(P)                # emit the calibrated pulse on latched axes

    def _settled(self, demand, vel_fb):
        """True iff every demanded axis's REALIZED (shaper-output) speed is below its
        settle epsilon -- the induced step has completed and the foot has planted. vel_fb
        defaults to zeros, so a caller that passes no feedback gets pure off_s timing
        (and the unit tests stay deterministic)."""
        try:
            for i in range(self.AXES):
                if demand[i] and abs(vel_fb[i]) > self.settle_eps[i]:
                    return False
        except (TypeError, IndexError):
            return True                      # malformed feedback -> fall back to off_s
        return True

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _begin_burst(self, intent, demand, now):
        """Open an ON window: latch axis membership + direction sign + (distance mode)
        the odom anchor. Latching at the edge means a key pressed or flipped mid-burst
        can't inject a lateral lurch or reverse a swing foot -- it joins next burst."""
        self.phase = "ON"
        self.t_start = now
        self.locked = demand
        self.sign = tuple(math.copysign(1.0, intent[i]) if demand[i] else 0.0
                          for i in range(self.AXES))
        if self.dq and self.pose is not None and self.live is not None and self.live():
            p = self.pose()
            if p is not None:
                self.anchor = (float(p[0]), float(p[1]), float(p[2]))
                self.estimated = False
                return
        # distance mode wanted but no usable odom -> fall back to time (flag it).
        self.anchor = (0.0, 0.0, 0.0)
        self.estimated = bool(self.dq)

    def _on_done(self, P, on_s, elapsed):
        """Decide whether the ON window should close. Time-based termination is always
        available as the floor / fallback; distance-quantize closes on measured odom
        displacement when enabled, live, and not already degraded to time."""
        # Time-based termination (the deterministic floor / fallback).
        if not self.dq or self.estimated or self.pose is None:
            return elapsed >= on_s
        if self.live is None or not self.live():   # odom died mid-burst -> degrade to time
            self.estimated = True
            return elapsed >= on_s
        if elapsed >= self.drive_timeout:          # backstop: never hang the gate open
            return True
        p = self.pose()
        if p is None:                              # transient read miss -> degrade to time
            self.estimated = True
            return elapsed >= on_s
        x, y, yaw = float(p[0]), float(p[1]), float(p[2])
        for i in range(self.AXES):
            if not self.locked[i]:
                continue
            if i == 2:
                d = abs(ang_wrap(yaw - self.anchor[2]))
            else:
                d = abs((x if i == 0 else y) - self.anchor[i])
            if d >= P[i].target:                   # a demanded axis hit its target stride
                return True
        return False

    def _burst(self, P):
        """The calibrated pulse: sign * mag_frac * cap on each latched axis, 0 else."""
        return tuple(self.sign[i] * P[i].mag_frac * self.max[i] if self.locked[i] else 0.0
                     for i in range(self.AXES))
