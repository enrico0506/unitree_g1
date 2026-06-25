"""ObstacleGuard -- dashboard-side (domain 0) velocity governor for the G1.

The obstacle perception node (obstacle_node.py, ROS2 / domain 99) writes a small
JSON snapshot to /dev/shm/g1_obstacle.json each frame. This guard runs INSIDE the
dashboard process (domain 0, Unitree SDK, NO rclpy): it polls that shm file in a
background thread and, on every commanded velocity, scales / hard-stops / steers
the command right before it is handed to the robot's Move() RPC.

Split of concerns:
  - obstacle_node.py  -> perception, writes shm        (CONTRACT 1)
  - guard.py (this)   -> reads shm, governs velocity   (CONTRACT 2 + 3 + 5)
  - manager.py        -> starts/stops the node process  (CONTRACT 4)

apply() is on the hot path (called ~30 Hz from the command thread). It does NO
file IO and NO blocking RPCs: it only reads a cached dict under a lock. The guard
speaks exactly ONE word -- "on" when enabled, "off" when disabled / auto-disabled
-- fired on a short throwaway daemon thread (never inline). No other narration.

shm contract recap (null distance == clear / infinitely far):
  seq, stamp, front_m, front_filt_m, left_m, right_m, back_m, zone, speed_scale,
  side{left,right}, gap{state,center_deg,passable,yaw_cmd,vy_cmd,turn_factor},
  profile[...].
"""

import json
import math
import os
import threading
import time

import yaml


# --- Defaults baked in so a missing yaml key never crashes the guard ----------
# (Mirror obstacle.yaml; the file is the source of truth, these are the fallback.)
_DEFAULTS = {
    "sensor_height_m": 1.10,
    "obstacle_range_m": 3.0,
    "slow_at_m": 1.6,          # only slow within this (keeps speed when it has space)
    "stop_at_m": 0.6,
    "safety_bubble_m": 0.35,   # arm-clearance bubble: slow/stop this much EARLIER, all directions
    "slow_min_scale": 0.5,     # node: min speed in the slow zone (keep moving, never freeze)
    "front_blind_zone_m": 1.0, # MEASURED: blind to obstacles closer than this straight ahead
    "predict_blind_zone": True,  # dead-reckon a front wall through the blind cone (see apply())
    "predict_timeout_s": 4.0,  # max seconds to trust the dead-reckoned estimate while creeping forward
    "filter_alpha": 0.25,      # smoother distance EMA
    # directional + tight-space governing (node reads ease_scale/tight_*; guard
    # reads scale_slew_per_s + stop_at_m; both bake the same defaults)
    "ease_scale": True,        # smoothstep easing for distance->scale (node)
    "tight_open_m": 1.2,       # a direction starts "closing in" below this (node)
    "tight_min": 0.7,          # floor of the global tight-space multiplier (node)
    "scale_slew_per_s": 1.8,   # GUARD: max change in applied speed-scale per second
    # announcements
    "announce_cooldown_s": 3.0,
    "speaker_id": 0,
    "led_warnings": True,
    # fail-safe
    "stale_timeout_s": 1.0,
    "startup_grace_s": 8.0,
    "fault_stop_s": 3.0,
    # gap-follow / steering caps (only the ones the guard itself reads)
    "max_yaw_rate": 0.6,
    "max_lateral_speed": 0.2,
    "turn_factor_min": 0.3,
    "search_yaw_rate": 0.4,
    # 360 ring directional gating (guard consumes the node's "ring"; falls back
    # to the 4-wedge per-axis logic whenever a "ring" key is absent).
    "ring_gating": True,
    "direction_cone_deg": 25.0,
    "direction_cone_max_deg": 70.0,
    "front_blind_compose_deg": 30.0,
    "ring_min_motion": 0.03,
    "robot_half_width_m": 0.25,
    "clearance_margin_m": 0.10,
    # initial flags
    "enabled_default": False,
    "gap_follow_default": False,
    "recovery_default": False,
}

SHM_PATH = "/dev/shm/g1_obstacle.json"


def clamp(value, low, high):
    """Clamp value into [low, high]."""
    return low if value < low else high if value > high else value


class ObstacleGuard:
    """Governs commanded (vx, vy, vyaw) using shared-memory obstacle data.

    See CONTRACT 2 for the public API and CONTRACT 3 for apply() behaviour.
    Mode is one of: DISABLED, STARTING, ACTIVE, FAULT.
    """

    def __init__(self, cfg_path, audio=None, limits=(1.5, 0.7, 2.0),
                 shm_path=SHM_PATH):
        self.cfg_path = str(cfg_path)
        self.audio = audio
        # limits kept verbatim for the FINAL re-clamp of apply()'s output.
        self.max_vx, self.max_vy, self.max_vyaw = (float(limits[0]),
                                                   float(limits[1]),
                                                   float(limits[2]))
        self.shm_path = shm_path

        # --- load tuning (bake defaults) ---
        cfg = dict(_DEFAULTS)
        try:
            with open(self.cfg_path, "r") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception as e:                       # missing / unreadable yaml
            print(f"[GUARD] could not load {self.cfg_path}: {e}; using defaults",
                  flush=True)
        self.cfg = cfg

        # frequently-read tuning, pulled out as attributes.
        # Safety bubble (arm clearance): slow/stop this much EARLIER in every direction.
        self.bubble = float(cfg.get("safety_bubble_m", 0.0))
        self.slow_at = float(cfg["slow_at_m"]) + self.bubble
        self.stop_at = float(cfg["stop_at_m"]) + self.bubble
        self.slow_min = float(cfg.get("slow_min_scale", 0.5))
        # Predictive blind-zone memory: dead-reckon a front wall through the ~blind cone.
        self.blind_zone = float(cfg.get("front_blind_zone_m", 1.0))
        self.predict_blind = bool(cfg.get("predict_blind_zone", True))
        self.predict_timeout_s = float(cfg.get("predict_timeout_s", 4.0))
        self._pred_front = None        # dead-reckoned front distance while in the blind zone (None = inactive)
        self._pred_age = 0.0           # seconds we've been creeping forward on the estimate
        self._last_vx_cmd = 0.0        # forward speed actually applied last cycle (for dead-reckoning)
        # Optional D435i near-ground depth fusion: a callable returning the nearest
        # forward near-ground obstacle distance (m) or None. Set by the dashboard from
        # DepthNearField; mixed into front_eff so low things the Mid-360 missed brake too.
        self._depth_fn = None
        # After the guard auto-disables itself (perception fault), latch forward at 0
        # until the operator RELEASES the forward command -- so a held W doesn't lurch
        # the robot blind through whatever stopped it the instant control hands back.
        self._autodis_hold = False
        self.stale_timeout_s = float(cfg["stale_timeout_s"])
        self.startup_grace_s = float(cfg["startup_grace_s"])
        self.fault_stop_s = float(cfg["fault_stop_s"])
        self.max_yaw_rate = float(cfg["max_yaw_rate"])
        self.max_lateral_speed = float(cfg["max_lateral_speed"])
        self.search_yaw_rate = float(cfg["search_yaw_rate"])
        self.scale_slew_per_s = float(cfg["scale_slew_per_s"])
        self.announce_cooldown_s = float(cfg["announce_cooldown_s"])
        self.speaker_id = int(cfg["speaker_id"])
        self.led_warnings = bool(cfg["led_warnings"])

        # --- 360 ring directional gating tuning ---
        self.ring_gating = bool(cfg.get("ring_gating", True))
        self.direction_cone_deg = float(cfg.get("direction_cone_deg", 25.0))
        self.direction_cone_max_deg = float(cfg.get("direction_cone_max_deg", 70.0))
        self.front_blind_compose_deg = float(cfg.get("front_blind_compose_deg", 30.0))
        self.ring_min_motion = float(cfg.get("ring_min_motion", 0.03))
        self.robot_half_w = float(cfg.get("robot_half_width_m", 0.25))
        self.clearance_margin = float(cfg.get("clearance_margin_m", 0.10))

        # --- operator toggles (initial values from *_default) ---
        self.enabled = bool(cfg["enabled_default"])
        self.gap_follow = bool(cfg["gap_follow_default"])
        self.recovery = bool(cfg["recovery_default"])

        # --- fault state machine ---
        # If enabled at construction, behave as if just enabled -> STARTING.
        self.mode = "STARTING" if self.enabled else "DISABLED"
        self.auto_disabled = False          # sticky: guard turned ITSELF off
        self._enable_time = time.monotonic() if self.enabled else 0.0
        self._fault_time = 0.0              # monotonic time we entered FAULT
        self._fault_spoken = False         # the fault line speaks once per fault

        # --- slew-rate state for the applied per-axis speed scales ---------
        # These ramp continuously at scale_slew_per_s toward the target scale so
        # the real speed is smooth despite the ~6 Hz perception update rate. They
        # are ONLY touched on the ACTIVE+fresh path and are reset to 1.0 whenever
        # the guard is (re-)enabled or disabled so re-enabling never starts from a
        # stale low scale.
        self._ax = 1.0                     # applied forward/back scale
        self._ay = 1.0                     # applied left/right (strafe) scale
        self._ayaw = 1.0                   # applied yaw scale (tight-space only)
        self._at = 1.0                     # applied translational scale (ring path)
        # Per-axis EMERGENCY hard-stop flags set during apply() (vx, vy, vyaw). A
        # downstream output smoother reads these to bypass its ramp on a safety stop
        # (so a hard stop / FAULT is instantaneous) while still smoothing the gentle
        # slow-zone scale-down. Same thread as apply(), so no lock needed.
        self._hard = [False, False, False]
        self._last_apply_t = 0.0           # monotonic time of last slew step

        # --- cached shm state (under lock; None until first good frame) ---
        self._lock = threading.Lock()
        self._data = None                  # last good parsed dict, or None
        self._mtime = 0.0                  # mtime of last successfully read file
        self._last_fresh = 0.0             # monotonic clock of last fresh read
        self._live = False                 # did we ever read a good frame

        # --- announcement cooldowns (per-key) ---
        self._speak_lock = threading.Lock()
        self._last_spoke = {}              # key -> monotonic time

        # --- reader thread ---
        self._stop = threading.Event()
        self._thread = None

    # -----------------------------------------------------------------
    # Background reader thread (CONTRACT 2: poll shm by mtime ~20 Hz)
    # -----------------------------------------------------------------

    def start(self):
        """Start the background shm reader (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="obstacle-guard",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background reader."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Poll the shm file by mtime; cache the parsed dict under the lock.

        Pattern mirrors OdomReader/LidarSource in scripts/lidar_source.py: a tiny
        daemon loop that keeps the latest-good snapshot behind a lock. Tolerates a
        missing / partial / non-JSON file by keeping the last good frame and just
        not advancing 'last_fresh' (so the fault machine notices staleness).
        """
        period = 0.05                       # ~20 Hz
        while not self._stop.is_set():
            try:
                mtime = os.path.getmtime(self.shm_path)
            except OSError:
                mtime = None                # file not present yet
            if mtime is not None and mtime != self._mtime:
                try:
                    with open(self.shm_path, "r") as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict):
                        with self._lock:
                            self._data = parsed
                            self._mtime = mtime
                            self._last_fresh = time.monotonic()
                            self._live = True
                except (OSError, ValueError):
                    # partial write / mid-os.replace / not-yet-JSON: keep last good
                    pass
            time.sleep(period)

    # -----------------------------------------------------------------
    # Snapshot helpers
    # -----------------------------------------------------------------

    def _snapshot(self):
        """Return (data_dict_or_None, age_seconds, live_bool) atomically."""
        with self._lock:
            data = self._data
            last_fresh = self._last_fresh
            live = self._live
        age = (time.monotonic() - last_fresh) if last_fresh > 0 else float("inf")
        return data, age, live

    @staticmethod
    def _num(value):
        """Coerce a JSON value to a FINITE float, treating null/missing/non-finite
        as None (clear). Rejecting NaN/inf here stops a corrupt shm value (e.g. a
        NaN tight_factor or gap.vy_cmd) from poisoning the velocity downstream."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    @staticmethod
    def _ring_usable(ring):
        """True only if `ring` is a dict with a non-empty dist list AND FINITE,
        positive bin geometry. Corrupt metadata (bin_deg 0/NaN/inf, start_deg
        NaN/inf) would make _ring_cone_min silently detect nothing -> fail OPEN and
        also skip the legacy wedge stops; rejecting it here falls back to the robust
        4-wedge path instead."""
        if not isinstance(ring, dict):
            return False
        dist = ring.get("dist")
        if not isinstance(dist, (list, tuple)) or len(dist) == 0:
            return False
        try:
            bd = float(ring.get("bin_deg"))
            sd = float(ring.get("start_deg"))
        except (TypeError, ValueError):
            return False
        return math.isfinite(bd) and bd > 0.0 and math.isfinite(sd)

    # -----------------------------------------------------------------
    # Operator toggles (CONTRACT 2)
    # -----------------------------------------------------------------

    def set_enabled(self, on):
        """Enable/disable the guard. Enabling resets to STARTING + clears the
        sticky auto-disabled flag and records the enable time (grace window)."""
        on = bool(on)
        if on:
            self.enabled = True
            self.mode = "STARTING"
            self.auto_disabled = False
            self._autodis_hold = False
            self._enable_time = time.monotonic()
            self._fault_time = 0.0
            self._fault_spoken = False
            self._reset_slew()                            # never start from a stale low scale
            self._speak("on", "toggle", cooldown=0.0)     # one word, normal voice
        else:
            self.enabled = False
            self.mode = "DISABLED"
            self._autodis_hold = False                    # manual disable: hand back control cleanly
            self._reset_slew()                            # leave pass-through clean for next enable
            self._speak("off", "toggle", cooldown=0.0)    # one word, normal voice

    def _reset_slew(self):
        """Reset the slew-limited applied scales to full (1.0) AND the blind-zone
        predictor.

        Called on enable/disable so neither the per-axis ramp nor the dead-reckoned
        front estimate resumes from a stale value -- a re-enable must not carry a
        phantom wall that freezes the robot in a clear area.
        """
        self._ax = 1.0
        self._ay = 1.0
        self._ayaw = 1.0
        self._at = 1.0
        self._last_apply_t = 0.0
        self._pred_front = None        # drop any stale dead-reckoned wall
        self._pred_age = 0.0
        self._last_vx_cmd = 0.0

    def set_depth_source(self, fn):
        """Register a callable -> nearest forward near-ground depth distance (m) or
        None (D435i fusion). apply() mixes it into front_eff. Pass None to clear."""
        self._depth_fn = fn if callable(fn) else None

    def set_gap_follow(self, on):
        self.gap_follow = bool(on)

    def set_recovery(self, on):
        self.recovery = bool(on)

    # -----------------------------------------------------------------
    # Speech / LED feedback (always off-thread, never blocks apply())
    # -----------------------------------------------------------------

    def _speak(self, text, key, cooldown=None):
        """Speak `text` off-thread, honouring a per-key cooldown.

        audio.TtsMaker() is a blocking RPC, so it MUST run on a throwaway daemon
        thread -- never inline in apply(). If there's no audio client, silently
        skip. The cooldown dict prevents spamming the same announcement.
        """
        if self.audio is None:
            return
        cooldown = self.announce_cooldown_s if cooldown is None else cooldown
        now = time.monotonic()
        with self._speak_lock:
            last = self._last_spoke.get(key, 0.0)
            if now - last < cooldown:
                return
            self._last_spoke[key] = now

        def _worker():
            try:
                self.audio.TtsMaker(text, self.speaker_id)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    # -----------------------------------------------------------------
    # CONTRACT 3: apply()
    # -----------------------------------------------------------------

    def apply(self, desired, held=None):
        """Govern a commanded velocity. Non-blocking, thread-safe, ~30 Hz.

        desired = (vx, vy, vyaw): vx>0 forward, vy>0 left, vyaw>0 left/CCW.
        held = the OPERATOR's pre-pacer held intent (or None == use desired). The
        blind-zone predictor keys its "is the operator still pushing forward?" release
        on held[0], NOT the (possibly step-pacer-pulsed) desired[0] -- otherwise a
        step mode's OFF window looks like a forward release and the predictor drops the
        phantom wall every pulse, defeating the blind-zone stop. Returns a possibly-
        modified (vx, vy, vyaw), re-clamped to the __init__ limits.
        """
        vx, vy, vyaw = float(desired[0]), float(desired[1]), float(desired[2])
        # Operator's true held-forward intent (pre-pacer); falls back to desired so a
        # caller that does not pass held behaves exactly as before.
        try:
            held_fwd = float(held[0]) if held is not None else vx
        except (TypeError, IndexError, ValueError):
            held_fwd = vx
        self._hard = [False, False, False]   # reset; set True only on an emergency zero

        # --- Pass-through when disabled (but still keep the cache warm) ---
        if not self.enabled:
            # If WE auto-disabled (perception fault), do NOT hand back full forward
            # speed while forward is still held -- that lurches the robot blind through
            # whatever stopped it. Hold forward at 0 until the operator releases
            # (acknowledges manual control). Reverse / strafe / yaw stay free to escape.
            if self._autodis_hold:
                if vx > 0.0:
                    vx = 0.0
                    self._hard[0] = True
                else:
                    self._autodis_hold = False   # operator released -> manual control resumes
            return self._clamp_out(vx, vy, vyaw)

        data, age, _live = self._snapshot()
        fresh = age < self.stale_timeout_s

        # --- Fault state machine -------------------------------------------
        now = time.monotonic()

        if self.mode == "STARTING":
            if fresh:
                self.mode = "ACTIVE"               # first fresh frame -> ACTIVE
            elif now - self._enable_time > self.startup_grace_s:
                self._enter_fault(now)             # never got a frame in time
                # fall through into FAULT handling below

        elif self.mode == "ACTIVE":
            if not fresh:
                self._enter_fault(now)             # frames went stale

        if self.mode == "FAULT":
            # Zero FORWARD vx only; reverse / yaw / strafe still allowed.
            if vx > 0.0:
                vx = 0.0
                self._hard[0] = True
            if now - self._fault_time >= self.fault_stop_s:
                # Auto-disable: turn ourselves off and stick the UI banner. Speak
                # the same one-word "off" so the operator knows it shut itself down.
                # Latch forward-zero until the operator releases, so handing control
                # back doesn't lurch the robot blind forward (see the disabled path).
                self.enabled = False
                self.auto_disabled = True
                self._autodis_hold = True
                self.mode = "DISABLED"
                self._speak("off", "toggle", cooldown=0.0)
            return self._clamp_out(vx, vy, vyaw)

        if self.mode == "STARTING":
            # Still waiting for the first frame, within grace: no braking yet.
            return self._clamp_out(vx, vy, vyaw)

        # --- ACTIVE + fresh: normal governing ------------------------------
        if data is None:
            # Defensive: marked ACTIVE but no parsed dict yet -> don't brake.
            return self._clamp_out(vx, vy, vyaw)

        # --- per-direction smooth scales + global tight factor (defaults 1.0
        #     so an old / short JSON without these keys never crashes). -------
        def _scale(key):
            s = self._num(data.get(key))
            return 1.0 if s is None else clamp(s, 0.0, 1.0)

        scale_front = _scale("scale_front")
        scale_back = _scale("scale_back")
        scale_left = _scale("scale_left")
        scale_right = _scale("scale_right")
        tight_factor = _scale("tight_factor")

        # RAW distances drive the per-direction HARD STOPS (no smoothing/slew).
        front_raw = self._num(data.get("front_m"))
        back_raw = self._num(data.get("back_m"))
        left_raw = self._num(data.get("left_m"))
        right_raw = self._num(data.get("right_m"))

        # --- predictive blind-zone memory ---------------------------------
        # The front blind zone (~1 m) is LARGER than stop_at, so a head-on wall
        # VANISHES before the normal stop can fire. front_eff is the raw distance
        # when the sensor sees the front, else the dead-reckoned estimate carried
        # from when the wall slipped into the blind cone (the estimate is updated
        # below, once the forward speed applied this cycle is known).
        # front_eff is the CLOSER of the live front reading and the dead-reckoned
        # estimate -- not just the estimate when the reading vanishes. A low/short
        # obstacle grazed at its top reports a PLATEAUED (too-far) front_raw while the
        # body closes in; taking the min lets the dead-reckoned closing distance drive
        # the stop instead of the misleadingly-far measurement.
        front_eff = front_raw
        if self.predict_blind and self._pred_front is not None:
            front_eff = (self._pred_front if front_eff is None
                         else min(front_eff, self._pred_front))
        # D435i near-ground depth fusion: nearest forward obstacle the up-looking
        # Mid-360 missed (low things ~0.5-2 m ahead). Mixed in as another min so it
        # eases + hard-stops exactly like a lidar return. None when OFF / too few pts.
        if self._depth_fn is not None:
            try:
                dn = self._depth_fn()
            except Exception:
                dn = None
            if dn is not None:
                front_eff = dn if front_eff is None else min(front_eff, dn)
        # Ease the forward scale on front_eff whenever it is closer than the live wedge
        # reading drove scale_front -- i.e. the blind-zone predictor OR the depth fusion
        # sees something nearer. (A no-op when front_eff == the live front reading.)
        if front_eff is not None:
            scale_front = min(scale_front, self._front_scale(front_eff))

        # --- choose gating path: ring (360 travel-direction) or legacy 4-wedge ---
        ring = data.get("ring") if self.ring_gating else None
        use_ring = self._ring_usable(ring)

        # slew timing (shared by both paths)
        now_t = time.monotonic()
        dt = (now_t - self._last_apply_t) if self._last_apply_t > 0.0 else 0.0
        dt = clamp(dt, 0.0, 0.1)
        self._last_apply_t = now_t
        rate = self.scale_slew_per_s

        if use_ring:
            try:
                trans_mag = math.hypot(vx, vy)
                gate_d = None
                if trans_mag >= self.ring_min_motion:
                    heading = math.atan2(vy, vx)
                    gate_d = self._ring_cone_min(ring, heading)
                    # front-blind compose: near-forward travel honours the predictor
                    if abs(math.degrees(heading)) < self.front_blind_compose_deg \
                            and front_eff is not None:
                        gate_d = (front_eff if gate_d is None
                                  else min(gate_d, front_eff))
                    scale_trans = 1.0 if gate_d is None else self._front_scale(gate_d)
                else:
                    scale_trans = 1.0    # no travel direction -> no ring gating
                tgt_trans = scale_trans * tight_factor
                tgt_yaw = tight_factor
                self._at = self._ramp(self._at, tgt_trans, dt, rate)
                self._ayaw = self._ramp(self._ayaw, tgt_yaw, dt, rate)
                vx *= self._at
                vy *= self._at
                vyaw *= self._ayaw
                # hard stop in the travel direction (RAW ring distance, no slew)
                if trans_mag >= self.ring_min_motion and gate_d is not None \
                        and gate_d < self.stop_at:
                    vx = 0.0
                    vy = 0.0
                    self._hard[0] = self._hard[1] = True
            except Exception:
                use_ring = False        # any ring error -> fall through to legacy

        if not use_ring:
            # 1) Pick each axis's TARGET scale by the sign of motion, times the
            #    global tight-space multiplier. (vyaw is only governed by tight.)
            tgt_vx = (scale_front if vx > 0.0 else
                      scale_back if vx < 0.0 else 1.0) * tight_factor
            tgt_vy = (scale_left if vy > 0.0 else
                      scale_right if vy < 0.0 else 1.0) * tight_factor
            tgt_yaw = tight_factor

            # 2) SLEW-LIMIT the applied scales toward their targets so the real
            #    speed ramps continuously at this ~30 Hz rate (smoothness fix).
            self._ax = self._ramp(self._ax, tgt_vx, dt, rate)
            self._ay = self._ramp(self._ay, tgt_vy, dt, rate)
            self._ayaw = self._ramp(self._ayaw, tgt_yaw, dt, rate)

            vx *= self._ax
            vy *= self._ay
            vyaw *= self._ayaw

            # 3) PER-DIRECTION HARD STOP -- independent of EMA/slew, RAW distances.
            # Front uses front_eff (raw, or the dead-reckoned estimate) so a wall
            # that vanished into the blind zone still stops the robot.
            if vx > 0.0 and front_eff is not None and front_eff < self.stop_at:
                vx = 0.0
                self._hard[0] = True
            if vx < 0.0 and back_raw is not None and back_raw < self.stop_at:
                vx = 0.0
                self._hard[0] = True
            if vy > 0.0 and left_raw is not None and left_raw < self.stop_at:
                vy = 0.0
                self._hard[1] = True
            if vy < 0.0 and right_raw is not None and right_raw < self.stop_at:
                vy = 0.0
                self._hard[1] = True

        # --- update the blind-zone predictor for the NEXT cycle -----------
        # Dead-reckons a forward obstacle through the ~1 m front blind cone. Seeded and
        # corrected ONLY from the RING's forward-CORRIDOR minimum (ring_fwd) -- which is
        # path-accurate (low false-positive: an off-path obstacle in the broad +/-45 deg
        # wedge does NOT seed it) and reads sparse/grazing returns the 10-point wedge
        # overestimates. Key: the estimate is dead-reckoned CONTINUOUSLY by the distance
        # travelled and a measurement may only CORRECT IT LOWER -- so a low/short obstacle
        # grazed only at its top edge (whose measured slant range PLATEAUS while the body
        # closes in) still converges to a stop instead of being trusted as "not closing".
        if self.predict_blind:
            turning = abs(vyaw) > 0.15
            ring_fwd = self._ring_cone_min(ring, 0.0) if isinstance(ring, dict) else None
            seed_window = max(self.blind_zone + 0.5, self.slow_at)
            visible = ring_fwd is not None and ring_fwd < seed_window
            if turning or self._last_vx_cmd < -0.02:
                self._pred_front = None        # turned / backed up -> dead-ahead estimate invalid
                self._pred_age = 0.0
            elif visible:
                # In view in the forward corridor -> TRACK the live distance (up AND down,
                # so a receding/static obstacle is followed, not falsely driven to a stop).
                self._pred_front = ring_fwd
                self._pred_age = 0.0
            elif self._pred_front is not None:
                # Dropped out of view (a low/short obstacle grazed at its top edge falls
                # below the upward-tilted FOV in the near field) -> DEAD-RECKON it closer by
                # the distance travelled so it still converges to a stop, then KEEP the stop
                # while the operator pushes forward (it may still be there, below the FOV --
                # safety over liveness). Release only when it closes in, or when it goes
                # stale AND the operator is no longer driving into it (then re-press resumes).
                if self._last_vx_cmd > 0.02:
                    self._pred_front -= self._last_vx_cmd * dt
                self._pred_age += dt
                # Release when it closes in (pred<=0), OR as soon as the operator stops
                # pushing FORWARD into it (desired[0] small). While they hold forward we
                # KEEP the stop (a grazing below-FOV obstacle may still be there -- safety
                # over liveness). The instant they release, the phantom clears, so a
                # crossed/departed obstacle resumes immediately on the next press -- no
                # 4 s wait. A long-stale estimate is also dropped as a backstop:
                # predict_timeout_s caps how long a phantom may freeze the robot even
                # while forward is HELD. Without it, once the phantom dead-reckons below
                # stop_at the robot stops, last_vx -> 0 so the estimate stops decreasing
                # and never reaches 0 -- a permanent forward latch in an already-clear
                # area. After the timeout we drop it and let the ring re-seed if the
                # obstacle is genuinely still there.
                if (self._pred_front <= 0.0 or held_fwd <= 0.05
                        or self._pred_age > self.predict_timeout_s):
                    self._pred_front = None
                    self._pred_age = 0.0

        # 4) Stage 8: gap-follow auto steering (only when the operator is
        #    actively commanding forward AND gap_follow is on). Applies AFTER the
        #    directional scaling above, on the already-scaled vx.
        gap = data.get("gap")
        if not isinstance(gap, dict):     # a non-dict truthy (string/int/list) in shm
            gap = {}                       # would crash gap.get() and kill the command thread
        gap_state = gap.get("state", "SEARCH")
        if self.gap_follow and desired[0] > 0.05:
            yaw_cmd = self._num(gap.get("yaw_cmd")) or 0.0
            vy_cmd = self._num(gap.get("vy_cmd")) or 0.0
            turn_factor = self._num(gap.get("turn_factor"))
            if turn_factor is None:
                turn_factor = 1.0
            # Add the steering ONTO the already-governed (slew-scaled, post-hard-stop)
            # vy/vyaw -- NOT onto the raw operator command -- so we keep the directional
            # + tight-space scaling instead of discarding it. The hard-stop re-enforce
            # below then guarantees a safety-stopped axis stays zeroed.
            vyaw = vyaw + yaw_cmd              # additive auto-steer toward gap
            vy = vy + vy_cmd                   # additive corridor centering
            vx *= turn_factor                  # slow down when turning hard

        # 5) Recovery: boxed in with no gap -> turn in place toward open side.
        #    Only when recovery is on, the gap is BLOCKED, and the operator is
        #    commanding forward.
        if self.recovery and gap_state == "BLOCKED" and desired[0] > 0.05:
            vx = 0.0
            self._hard[0] = True               # boxed-in stop is immediate
            left_m = self._num(data.get("left_m"))
            right_m = self._num(data.get("right_m"))
            # null == infinitely far == most open. Turn toward the more-open side
            # (+ = left). If right is more open, turn right (negative).
            l = float("inf") if left_m is None else left_m
            r = float("inf") if right_m is None else right_m
            vyaw = self.search_yaw_rate if l >= r else -self.search_yaw_rate

        # SAFETY re-enforce: gap-follow may INJECT a vy_cmd strafe (or vyaw) AFTER the
        # hard-stop checks, into an axis the operator wasn't commanding -- so re-run the
        # per-direction hard stops on the FINAL velocity. An injected strafe/forward into
        # anything within stop_at is zeroed (and flagged), and any prior hard-stop is
        # kept. The recovery turn-in-place yaw survives (yaw has no distance gate here).
        if vx > 0.0 and front_eff is not None and front_eff < self.stop_at:
            vx = 0.0; self._hard[0] = True
        if vx < 0.0 and back_raw is not None and back_raw < self.stop_at:
            vx = 0.0; self._hard[0] = True
        if vy > 0.0 and left_raw is not None and left_raw < self.stop_at:
            vy = 0.0; self._hard[1] = True
        if vy < 0.0 and right_raw is not None and right_raw < self.stop_at:
            vy = 0.0; self._hard[1] = True
        if self._hard[0]:
            vx = 0.0
        if self._hard[1]:
            vy = 0.0
        if self._hard[2]:
            vyaw = 0.0

        # Remember the forward speed actually applied, for next cycle's dead-reckoning.
        self._last_vx_cmd = vx
        return self._clamp_out(vx, vy, vyaw)

    def _enter_fault(self, now):
        """Transition into FAULT: brake forward. No speech here -- the one-word
        "off" is spoken when the guard auto-disables after the stop."""
        if self.mode != "FAULT":
            self.mode = "FAULT"
            self._fault_time = now

    @staticmethod
    def _ramp(cur, target, dt, rate):
        """Move `cur` toward `target` by at most `rate` per second over `dt`.

        Used to slew-rate-limit each applied per-axis speed scale so the real
        speed ramps continuously at 30 Hz instead of stepping with the ~6 Hz
        perception update. dt is expected pre-clamped to [0, 0.1].
        """
        step = rate * dt
        delta = target - cur
        if delta > step:
            return cur + step
        if delta < -step:
            return cur - step
        return target

    def _ring_cone_min(self, ring, heading_rad):
        """Nearest ring obstacle in the robot's swept CORRIDOR along the travel
        heading, or None (clear).

        Per-bin corridor test (no two-pass probe -- that missed a close obstacle
        just outside the narrow floor cone). The body sweeps a corridor of
        half-width (robot_half_width + clearance_margin) along the heading, so a
        bin at angle a, distance d, is "in the way" iff |a - heading| is within
        the angle that corridor subtends AT THAT d: asin(corridor / d). Each bin
        uses its OWN distance, so a near obstacle is caught at a wide angle and a
        far one only when nearly dead-ahead -- the geometrically correct test, and
        strictly less over-conservative than the old fixed +/-45 deg front wedge.
        A direction_cone_deg FLOOR keeps a minimum cone (heading/sensor noise);
        direction_cone_max_deg CAPS it so a ~0.05 m return can't make the corridor
        omnidirectional and freeze all motion.
        """
        try:
            dist = ring.get("dist")
            if not isinstance(dist, (list, tuple)) or not dist:  # a dict passes len()
                return None                                       # but dist[i] would KeyError
            n = len(dist)
            bin_deg = float(ring.get("bin_deg", 5.0))
            start_deg = float(ring.get("start_deg", -180.0))
        except (AttributeError, TypeError, ValueError):
            return None
        heading_deg = math.degrees(heading_rad)
        corridor = self.robot_half_w + self.clearance_margin
        best = None
        for i in range(n):
            d = self._num(dist[i])
            if d is None:
                continue
            center = start_deg + (i + 0.5) * bin_deg
            diff = abs(((center - heading_deg + 180.0) % 360.0) - 180.0)
            # half-angle the corridor subtends at this obstacle's distance.
            if d <= corridor:
                half = self.direction_cone_max_deg      # basically on top of us
            else:
                half = math.degrees(math.asin(corridor / d))
                half = max(self.direction_cone_deg, min(half, self.direction_cone_max_deg))
            if diff <= half and (best is None or d < best):
                best = d
        return best

    def _front_scale(self, d):
        """Forward speed scale for a (predicted) front distance -- mirrors the
        node's _dist_scale: smoothstep over [stop_at, slow_at] with the slow_min
        floor, 0 at/under stop_at. Used to ease down on the dead-reckoned wall."""
        if d >= self.slow_at:
            return 1.0
        if d <= self.stop_at:
            return 0.0
        span = self.slow_at - self.stop_at
        t = clamp((d - self.stop_at) / span, 0.0, 1.0) if span > 0 else 1.0
        e = t * t * (3.0 - 2.0 * t)
        return self.slow_min + (1.0 - self.slow_min) * e

    def _clamp_out(self, vx, vy, vyaw):
        """Final re-clamp of the output to the __init__ limits. Any non-finite value
        (a NaN that slipped through from corrupt shm) is forced to 0 -- a safe stop,
        never a clamp(NaN)=NaN that would poison the shaper and the robot command."""
        vx = vx if math.isfinite(vx) else 0.0
        vy = vy if math.isfinite(vy) else 0.0
        vyaw = vyaw if math.isfinite(vyaw) else 0.0
        return (clamp(vx, -self.max_vx, self.max_vx),
                clamp(vy, -self.max_vy, self.max_vy),
                clamp(vyaw, -self.max_vyaw, self.max_vyaw))

    # -----------------------------------------------------------------
    # CONTRACT 2 / CONTRACT 5: introspection
    # -----------------------------------------------------------------

    def hard_stop_flags(self):
        """Per-axis (vx, vy, vyaw) emergency-stop flags from the LAST apply() call.
        True means the guard forced that axis to zero for safety (hard stop / FAULT /
        recovery), as opposed to a gentle slow-zone scale-down. A downstream output
        smoother bypasses its ramp on these axes so a safety stop stays instantaneous.
        Read on the same thread immediately after apply()."""
        return tuple(self._hard)

    def forward_speed_scale(self):
        """Cached forward speed scale (1.0 when there is no usable data).

        Front-based: prefers scale_front, falls back to the legacy speed_scale
        key (the node keeps speed_scale == scale_front for backward-compat)."""
        data, _age, _live = self._snapshot()
        if data is None:
            return 1.0
        scale = self._num(data.get("scale_front"))
        if scale is None:
            scale = self._num(data.get("speed_scale"))
        if scale is None:
            return 1.0
        return clamp(scale, 0.0, 1.0)

    def telemetry(self):
        """CONTRACT 5 server->browser dict (type 'obstacle')."""
        data, age, _live = self._snapshot()
        live = (data is not None) and (age < self.stale_timeout_s)

        if data is None:
            front_m = front_filt_m = None
            left_m = right_m = back_m = None
            back_filt_m = left_filt_m = right_filt_m = None
            zone = "CLEAR"
            speed_scale = 1.0
            scale_front = scale_back = scale_left = scale_right = 1.0
            tight_factor = 1.0
            side = {"left": False, "right": False}
            gap = {
                "state": "SEARCH", "center_deg": 0.0, "passable": False,
                "yaw_cmd": 0.0, "vy_cmd": 0.0, "turn_factor": 1.0,
            }
            profile = []
            ring = None
            points = []
        else:
            front_m = self._num(data.get("front_m"))
            front_filt_m = self._num(data.get("front_filt_m"))
            left_m = self._num(data.get("left_m"))
            right_m = self._num(data.get("right_m"))
            back_m = self._num(data.get("back_m"))
            back_filt_m = self._num(data.get("back_filt_m"))
            left_filt_m = self._num(data.get("left_filt_m"))
            right_filt_m = self._num(data.get("right_filt_m"))
            profile = data.get("profile") or []
            ring = data.get("ring")
            pts = data.get("points")
            points = pts if isinstance(pts, list) else []
            zone = data.get("zone", "CLEAR")

            def _tscale(key, fallback=None):
                s = self._num(data.get(key))
                if s is None:
                    return fallback
                return clamp(s, 0.0, 1.0)

            # scale_front is the canonical forward scale; speed_scale stays equal
            # to it (the node writes both; fall back to the legacy speed_scale key
            # if an old JSON lacks scale_front).
            legacy_ss = _tscale("speed_scale", 1.0)
            scale_front = _tscale("scale_front", legacy_ss)
            scale_back = _tscale("scale_back", 1.0)
            scale_left = _tscale("scale_left", 1.0)
            scale_right = _tscale("scale_right", 1.0)
            tight_factor = _tscale("tight_factor", 1.0)
            speed_scale = scale_front          # speed_scale == scale_front
            raw_side = data.get("side")
            if not isinstance(raw_side, dict):
                raw_side = {}
            side = {"left": bool(raw_side.get("left")),
                    "right": bool(raw_side.get("right"))}
            raw_gap = data.get("gap")
            if not isinstance(raw_gap, dict):
                raw_gap = {}
            gap = {
                "state": raw_gap.get("state", "SEARCH"),
                "center_deg": self._num(raw_gap.get("center_deg")) or 0.0,
                "passable": bool(raw_gap.get("passable")),
                "yaw_cmd": self._num(raw_gap.get("yaw_cmd")) or 0.0,
                "vy_cmd": self._num(raw_gap.get("vy_cmd")) or 0.0,
                "turn_factor": (self._num(raw_gap.get("turn_factor"))
                                if raw_gap.get("turn_factor") is not None else 1.0),
            }

        return {
            "type": "obstacle",
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "front_m": front_m,
            "front_filt_m": front_filt_m,
            "left_m": left_m,
            "right_m": right_m,
            "back_m": back_m,
            "back_filt_m": back_filt_m,
            "left_filt_m": left_filt_m,
            "right_filt_m": right_filt_m,
            "zone": zone,
            "speed_scale": speed_scale,
            "scale_front": scale_front,
            "scale_back": scale_back,
            "scale_left": scale_left,
            "scale_right": scale_right,
            "tight_factor": tight_factor,
            "side": side,
            "gap": gap,
            "profile": profile,
            "ring": ring,
            "points": points,
            "fault": self.mode == "FAULT",
            "auto_disabled": bool(self.auto_disabled),
            "live": bool(live),
            "gap_follow": bool(self.gap_follow),
            "recovery": bool(self.recovery),
        }

    def ui_config(self):
        """Initial flags for the connect 'config' message (CONTRACT 5/6)."""
        return {
            "enabled": bool(self.enabled),
            "gap_follow": bool(self.gap_follow),
            "recovery": bool(self.recovery),
        }

    def is_active_ui(self):
        """True if the UI should show the guard (enabled, faulting, or it
        auto-disabled itself and the banner is still sticky)."""
        return bool(self.enabled) or bool(self.auto_disabled) or self.mode == "FAULT"
