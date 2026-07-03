#!/usr/bin/env python3
"""Human-gesture -> robot-gesture REACTOR (the interactive "wave back" demo).

A person waves at the robot; the robot waves back. This module is the PERCEPTION +
DECISION half: it consumes the per-person skeletons that pose_service.py publishes to
/dev/shm/g1_pose_tracks.json ({w,h,items:[{id,name,box,kpts}]}, COCO-17 keypoints in
IMAGE pixels, y DOWN, conf 0-1) and classifies a small, DELIBERATELY SMALL set of
robust human gestures:

    WAVE          a wrist held ABOVE its shoulder AND oscillating horizontally ~1 s
    RAISE_HAND    a wrist held ABOVE the head (still, not oscillating)
    BOTH_ARMS_UP  both wrists held above both shoulders

Design priorities (in order): ROBUSTNESS, FEW FALSE POSITIVES, simplicity. A reliable
wave-back in front of a client beats six flaky gestures. Everything here is temporal
(a single frame never fires a gesture) and SCALE-INVARIANT (thresholds are expressed in
shoulder-widths, so a person at 2 m and a person at 6 m behave the same).

Split into two layers:
  * GestureClassifier / classify() -- PURE (no robot, no I/O, no shm). Deterministic.
    Fully unit-testable from synthesized keypoint sequences (see sim_gestures.py).
  * GreetingService -- the thin, OPTIONAL bridge that reads shm, runs a reactor, and
    fires the mapped robot gesture through a caller-supplied callback ONLY when a
    "greeting mode" gate + a safety gate both say OK. All robot coupling is via
    callbacks, so this file imports nothing from the controller and stays testable.

Image-coords reminder: y increases DOWNWARD, so "ABOVE" means a SMALLER y.

    python3 scripts/gesture_reactor.py --selftest
"""
import json
import math
import os
import threading
import time
from collections import deque

# --- COCO-17 keypoint indices (the order pose_service.py emits) ---
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE, L_ANK, R_ANK = 13, 14, 15, 16
HEAD_KPTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)   # topmost of these = head top

KP_MIN_CONF = 0.3   # matches pose_service.KP_MIN_CONF: below this a joint is "missing"


# --- human gesture -> robot response ---------------------------------------
# The mapping is intentionally "impressive but calm": the hero is wave->wave (the G1's
# native LocoClient.WaveHand). raise_hand -> high_five reads as "up top!"; both_arms_up
# -> hug is the big celebratory one. Swap both_arms_up to "heart" for a softer look.
# Values are cmd names the controller's apply_cmd() already knows (see ARM_GESTURES +
# the wave/shake branches), so firing reuses the EXISTING serialized execution path.
GESTURE_TO_ROBOT = {
    "wave":         "wave",       # LocoClient.WaveHand() -- native two-way wave (the hero)
    "raise_hand":   "high_five",  # arm action 18 (auto-releases) -- "up top!"
    "both_arms_up": "hug",        # arm action 19 (auto-releases); alt: "heart" (20)
}

# Short human-readable feedback for the dashboard ("saw wave -> waving back").
GESTURE_HUMAN = {
    "wave":         "wave",
    "raise_hand":   "raised hand",
    "both_arms_up": "both arms up",
}


class ReactorConfig:
    """All tunables in one place (keyword-overridable so tests can tighten/loosen).

    Times are seconds; distances are in SHOULDER-WIDTHS (scale-invariant). Defaults are
    tuned for ~10-15 fps pose input and favour missing a marginal gesture over firing a
    false one."""

    def __init__(self, **kw):
        # history / target
        self.window_s = 1.3            # temporal window classified each update
        self.kp_min_conf = KP_MIN_CONF
        self.track_ttl_s = 0.8         # drop a track's history if unseen this long
        self.min_track_age_s = 0.4     # target must be tracked this long before it can fire
        self.min_conf_frac = 0.15      # a target must have >= this fraction of its 17 kpts

        # WAVE: wrist above shoulder + horizontal oscillation
        self.wave_min_frames = 5       # need several samples to see an oscillation
        self.wave_min_span_s = 0.5     # ...spanning at least this long (~1 s wave)
        self.wave_up_frac = 0.6        # >= this fraction of frames: wrist above shoulder
        self.wave_amp_frac = 0.25      # peak-to-peak wrist-x sweep, in shoulder-widths
        self.wave_min_reversals = 2    # direction changes -> genuine back-and-forth (not a sweep)

        # RAISE_HAND: wrist above the head, held STILL
        self.hold_min_frames = 4
        self.raise_min_span_s = 0.4
        self.raise_up_frac = 0.7       # fraction of frames: wrist above head-top
        self.raise_max_reversals = 1   # must be still-ish (else it's a wave)

        # BOTH_ARMS_UP: both wrists above both shoulders, held
        self.both_min_span_s = 0.35
        self.both_up_frac = 0.6

        # oscillation deadband: wrist-x moves smaller than this (shoulder-widths) are jitter
        self.deadband_frac = 0.06

        # debounce
        self.cooldown_s = 4.0          # per-gesture: a CONTINUOUS wave fires exactly once
        self.refractory_s = 2.0        # global: min gap between ANY two fired gestures

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown ReactorConfig field: {k}")
            setattr(self, k, v)


DEFAULT_CFG = ReactorConfig()


# --------------------------------------------------------------------- helpers
def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _kp(kpts, idx, min_conf):
    """(x, y) for keypoint idx if present AND confident enough, else None.

    Robust to short/ragged kpts lists and to a joint reported with low confidence
    (pose_service still EMITS low-conf joints; here we treat them as MISSING)."""
    if idx >= len(kpts):
        return None
    k = kpts[idx]
    if len(k) < 3:
        return None
    x, y, c = k[0], k[1], k[2]
    if c is None or c < min_conf:
        return None
    return (float(x), float(y))


def _frame_scale(joints, box):
    """A per-frame body-size unit (pixels) to normalize distances -> scale invariance.

    Prefer shoulder width (most reliable, gesture-relevant); fall back to torso height
    (shoulder->hip), then to a fraction of box width. Never returns 0 (guards divides)."""
    ls, rs = joints.get(L_SHO), joints.get(R_SHO)
    if ls and rs:
        w = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
        if w > 1e-3:
            return w
    # torso: mean shoulder to mean hip vertical span
    sho = [p for p in (ls, rs) if p]
    hips = [p for p in (joints.get(L_HIP), joints.get(R_HIP)) if p]
    if sho and hips:
        sy = sum(p[1] for p in sho) / len(sho)
        hy = sum(p[1] for p in hips) / len(hips)
        t = abs(hy - sy)
        if t > 1e-3:
            return t
    if box:
        bw = abs(box[2] - box[0])
        if bw > 1e-3:
            return 0.30 * bw
    return 1.0


def _head_top_y(joints):
    """Smallest y (highest point) among available head keypoints, or None."""
    ys = [joints[i][1] for i in HEAD_KPTS if i in joints]
    return min(ys) if ys else None


def _reversals(xs, deadband):
    """Count horizontal direction changes in the wrist-x series, ignoring sub-`deadband`
    jitter. A held/still hand -> ~0; a single raise-then-lower sweep -> 0-1; a genuine
    back-and-forth WAVE -> >= 2. This is what separates a wave from a raised hand."""
    dirn = 0
    rev = 0
    last = xs[0]
    for x in xs[1:]:
        d = x - last
        if abs(d) < deadband:
            continue                 # too small to be intentional motion
        s = 1 if d > 0 else -1
        if dirn != 0 and s != dirn:
            rev += 1
        dirn = s
        last = x
    return rev


def _arm_metrics(history, wri_idx, sho_idx, cfg):
    """Temporal metrics for ONE arm over the window, or None if too few good samples.

    history: list of (t, joints, scale). Only frames where BOTH the wrist and its
    shoulder are confident contribute. All distances normalized by the median scale."""
    ts, xs, ups, head_ups, scales = [], [], [], [], []
    for (t, joints, scale) in history:
        w = joints.get(wri_idx)
        s = joints.get(sho_idx)
        if not (w and s):
            continue
        ts.append(t)
        xs.append(w[0])
        ups.append(1 if w[1] < s[1] else 0)          # wrist ABOVE shoulder (smaller y)
        ht = _head_top_y(joints)
        if ht is not None:
            head_ups.append(1 if w[1] < ht else 0)    # wrist ABOVE head top
        scales.append(scale)
    if len(ts) < 2:
        return None
    scale = _median(scales) or 1.0
    amp = (max(xs) - min(xs)) / scale
    rev = _reversals(xs, cfg.deadband_frac * scale)
    return {
        "n": len(ts),
        "span": ts[-1] - ts[0],
        "up_frac": sum(ups) / len(ups),
        "head_up_frac": (sum(head_ups) / len(head_ups)) if head_ups else 0.0,
        "amp": amp,
        "rev": rev,
    }


def _both_up(history, cfg):
    """True if BOTH wrists are above their shoulders for a sustained fraction of frames."""
    ts, ups = [], []
    for (t, joints, _scale) in history:
        lw, ls = joints.get(L_WRI), joints.get(L_SHO)
        rw, rs = joints.get(R_WRI), joints.get(R_SHO)
        if not (lw and ls and rw and rs):
            continue
        ts.append(t)
        ups.append(1 if (lw[1] < ls[1] and rw[1] < rs[1]) else 0)
    if len(ts) < cfg.hold_min_frames:
        return False
    if (ts[-1] - ts[0]) < cfg.both_min_span_s:
        return False
    return (sum(ups) / len(ups)) >= cfg.both_up_frac


def classify(history, cfg=DEFAULT_CFG):
    """PURE classifier: a track's recent (t, joints, scale) history -> gesture name or None.

    Order matters and encodes precedence: BOTH_ARMS_UP (needs both wrists up) is checked
    first; then WAVE (one wrist up AND oscillating); then RAISE_HAND (one wrist above the
    HEAD and STILL). A single-arm wave can't trip both_arms_up (the other wrist is down),
    and an oscillating hand is a wave, not a raise -- so the three are mutually exclusive
    in practice."""
    if len(history) < 2:
        return None

    # 1) BOTH ARMS UP -- the big celebratory pose.
    if _both_up(history, cfg):
        return "both_arms_up"

    la = _arm_metrics(history, L_WRI, L_SHO, cfg)
    ra = _arm_metrics(history, R_WRI, R_SHO, cfg)

    # 2) WAVE -- pick whichever arm is more raised; require oscillation + amplitude.
    for m in _best_first(la, ra, key="up_frac"):
        if (m["n"] >= cfg.wave_min_frames
                and m["span"] >= cfg.wave_min_span_s
                and m["up_frac"] >= cfg.wave_up_frac
                and m["amp"] >= cfg.wave_amp_frac
                and m["rev"] >= cfg.wave_min_reversals):
            return "wave"

    # 3) RAISE HAND -- above the HEAD and held STILL (few reversals).
    for m in _best_first(la, ra, key="head_up_frac"):
        if (m["n"] >= cfg.hold_min_frames
                and m["span"] >= cfg.raise_min_span_s
                and m["head_up_frac"] >= cfg.raise_up_frac
                and m["rev"] <= cfg.raise_max_reversals):
            return "raise_hand"

    return None


def _best_first(la, ra, key):
    """The two arm-metric dicts (dropping Nones), most-`key` first."""
    ms = [m for m in (la, ra) if m is not None]
    ms.sort(key=lambda m: m[key], reverse=True)
    return ms


# ------------------------------------------------------------------- reactor
def parse_frame(frame):
    """Accept either the raw shm dict {w,h,items:[...]} or a bare items list; return
    (items, w, h). Tolerant of missing w/h (centrality just gets skipped)."""
    if isinstance(frame, dict):
        return frame.get("items", []) or [], frame.get("w"), frame.get("h")
    return list(frame or []), None, None


def _target_score(item, w, h):
    """Rank persons for "who is the robot greeting": biggest (nearest) box wins, with a
    mild penalty for being off-centre and a small bonus for keypoint confidence. Returns
    -inf for a person with no usable box so they're never picked."""
    box = item.get("box")
    if not box or len(box) < 4:
        return float("-inf")
    bw, bh = abs(box[2] - box[0]), abs(box[3] - box[1])
    area = bw * bh
    if area <= 0:
        return float("-inf")
    score = area
    if w and h:
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        off = math.hypot(cx - w / 2.0, cy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
        score *= (1.0 - 0.3 * min(1.0, off))     # up to 30% penalty for edge-of-frame
    kpts = item.get("kpts") or []
    confs = [k[2] for k in kpts if len(k) >= 3 and k[2] is not None]
    if confs:
        score *= (0.85 + 0.15 * (sum(confs) / len(confs)))
    return score


class GestureReactor:
    """Stateful wrapper around classify(): keeps per-track keypoint history, picks ONE
    target person per frame, and DEBOUNCES so a continuous wave fires exactly once.

    Clean API:  reactor.update(frame, t) -> None | {gesture, track_id, robot_gesture, ...}
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or DEFAULT_CFG
        self._hist = {}          # track_id -> deque[(t, joints, scale)]
        self._first_seen = {}    # track_id -> t first observed
        self._cooldown = {}      # gesture name -> t until which it may not re-fire
        self._last_fire_t = -1e9

    def reset(self):
        """Forget all history/debounce state (call when greeting mode toggles OFF so a
        stale held pose can't fire the instant it comes back ON)."""
        self._hist.clear()
        self._first_seen.clear()
        self._cooldown.clear()
        self._last_fire_t = -1e9

    def _ingest(self, items, w, h, t):
        """Fold this frame's persons into per-track history; age out stale tracks."""
        seen = set()
        for it in items:
            tid = it.get("id")
            if tid is None:
                continue
            kpts = it.get("kpts") or []
            joints = {}
            for idx in range(len(kpts)):
                p = _kp(kpts, idx, self.cfg.kp_min_conf)
                if p is not None:
                    joints[idx] = p
            scale = _frame_scale(joints, it.get("box"))
            dq = self._hist.get(tid)
            if dq is None:
                dq = deque()
                self._hist[tid] = dq
                self._first_seen[tid] = t
            dq.append((t, joints, scale))
            # trim to the temporal window
            cutoff = t - self.cfg.window_s
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            seen.add(tid)
        # drop tracks not seen for track_ttl_s
        for tid in list(self._hist):
            dq = self._hist[tid]
            if not dq or (t - dq[-1][0]) > self.cfg.track_ttl_s:
                self._hist.pop(tid, None)
                self._first_seen.pop(tid, None)

    def update(self, frame, t=None):
        """Feed one pose-track frame; return a gesture event or None.

        frame: {w,h,items:[{id,box,kpts},...]} (shm dict) OR a bare items list.
        t:     monotonic-ish seconds (defaults to wall clock). Passing your own clock
               makes the reactor fully deterministic (the sim/tests rely on this)."""
        if t is None:
            t = time.time()
        items, w, h = parse_frame(frame)
        self._ingest(items, w, h, t)

        # pick the ONE target person present THIS frame
        target = None
        best = float("-inf")
        for it in items:
            if it.get("id") is None:
                continue
            s = _target_score(it, w, h)
            if s > best:
                best, target = s, it
        if target is None:
            return None
        tid = target["id"]

        # require the target to have enough visible keypoints (ignore a near-empty box)
        kpts = target.get("kpts") or []
        n_conf = sum(1 for idx in range(len(kpts))
                     if _kp(kpts, idx, self.cfg.kp_min_conf) is not None)
        if n_conf < self.cfg.min_conf_frac * 17:
            return None

        # stability: don't react to a track we've only just started seeing
        if (t - self._first_seen.get(tid, t)) < self.cfg.min_track_age_s:
            return None

        dq = self._hist.get(tid)
        if not dq:
            return None
        gesture = classify(list(dq), self.cfg)
        if gesture is None:
            return None

        # debounce: global refractory + per-gesture cooldown (continuous wave -> once)
        if (t - self._last_fire_t) < self.cfg.refractory_s:
            return None
        if t < self._cooldown.get(gesture, -1e9):
            return None
        self._last_fire_t = t
        self._cooldown[gesture] = t + self.cfg.cooldown_s

        return {
            "gesture": gesture,
            "track_id": tid,
            "robot_gesture": GESTURE_TO_ROBOT.get(gesture),
            "human": GESTURE_HUMAN.get(gesture, gesture),
            "t": t,
        }


def describe(event):
    """Dashboard feedback string, e.g. 'saw wave -> waving back'."""
    if not event:
        return ""
    robot = event.get("robot_gesture") or "?"
    verb = {"wave": "waving back", "high_five": "high-fiving",
            "hug": "hugging", "heart": "sending a heart"}.get(robot, robot)
    return f"saw {event.get('human', event.get('gesture'))} -> {verb}"


# =====================================================================================
# GreetingService -- the SAFETY-GATED bridge to the robot (shm -> reactor -> callback).
# =====================================================================================
class GreetingService:
    """Runs a GestureReactor in a daemon thread against /dev/shm/g1_pose_tracks.json and
    fires the mapped robot gesture through `fire_fn` -- but ONLY when BOTH gates pass:

        enabled_fn()  -- the "greeting mode" master toggle is ON (default: caller keeps
                         it OFF; auto-firing arms near clients must be opt-in).
        safe_fn()     -- it is safe RIGHT NOW: robot not walking/moving, not mid-gesture
                         or mid-transition, no arm currently raised. (Supplied by the
                         controller; see the wiring block in robot_web_controller.py.)

    Every piece of robot coupling is a callback, so this class imports nothing from the
    controller and is exercised in tests with plain lambdas. Firing goes through the
    caller's EXISTING serialized path (the controller sets state.pending_cmd), so the
    _arm_lock, auto-release, and single-flight guarantees are reused unchanged.

    Additional safety layers beyond the reactor's own debounce:
      * stale-feed guard: only advances on a NEW pose frame (mtime change), so a frozen
        or dead pose feed can never fire.
      * demand heartbeat: while enabled, touches POSE_DEMAND so the (demand-gated) pose
        container keeps inferring -- greeting mode is one of its "viewers".
      * on OFF: resets the reactor so a pose held during the OFF window can't fire the
        instant it flips back ON.
    """

    def __init__(self, tracks_path, enabled_fn, safe_fn, fire_fn,
                 demand_path=None, on_event=None, on_skip=None,
                 cfg=None, period_s=0.1, idle_period_s=0.3):
        self.tracks_path = tracks_path
        self.demand_path = demand_path
        self.enabled_fn = enabled_fn
        self.safe_fn = safe_fn
        self.fire_fn = fire_fn
        self.on_event = on_event        # called on EVERY classified event (for dashboard feedback)
        self.on_skip = on_skip          # called when an event was gated out by safe_fn
        self.reactor = GestureReactor(cfg)
        self.period_s = period_s
        self.idle_period_s = idle_period_s
        self._stop = threading.Event()
        self._thread = None
        self._last_mtime = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="greeting", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _read_frame(self):
        """Return the parsed pose frame ONLY if the file changed since last read, else
        None (so a static/frozen feed never advances the classifier)."""
        try:
            mt = os.path.getmtime(self.tracks_path)
        except OSError:
            return None
        if mt == self._last_mtime:
            return None
        self._last_mtime = mt
        try:
            with open(self.tracks_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _touch_demand(self):
        if not self.demand_path:
            return
        try:
            with open(self.demand_path, "wb") as f:
                f.write(b"1")
        except OSError:
            pass

    def _handle_event(self, event):
        """Apply the SAFETY gate to one classified event and fire (or skip) accordingly.

        Split out of the loop so the gating is unit-testable with plain lambdas, no thread
        or shm needed (see test_gesture_reactor.py). on_event always fires (dashboard sees
        every gesture); the robot only moves when safe_fn() says it's safe RIGHT NOW, else
        on_skip is notified. Returns True iff the robot gesture was actually fired."""
        if event is None:
            return False
        if self.on_event:
            self.on_event(event)
        if self.safe_fn():
            self.fire_fn(event["robot_gesture"])
            return True
        if self.on_skip:
            self.on_skip(event)
        return False

    def poll_once(self, frame, t):
        """Run the reactor on ONE pose frame and dispatch any event through the safety gate.
        Assumes greeting mode is already enabled (the loop gates that upstream). Called by
        _loop with a freshly-read shm frame, and directly by tests with synthetic frames."""
        return self._handle_event(self.reactor.update(frame, t))

    def _loop(self):
        while not self._stop.is_set():
            if not self.enabled_fn():
                self.reactor.reset()          # clean slate when greeting mode is OFF
                self._stop.wait(self.idle_period_s)
                continue
            self._touch_demand()               # keep the pose GPU alive while greeting
            frame = self._read_frame()
            if frame is not None:
                self.poll_once(frame, time.time())
            self._stop.wait(self.period_s)


# ------------------------------------------------------------------------- selftest
def selftest():
    """Fast smoke test of the PURE classifier + debounce, with no camera/robot/shm.

    Synthesizes a minimal wave / raise / idle history directly (sim_gestures.py has the
    full harness) and checks the core fires correctly and debounces."""
    cfg = ReactorConfig()
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and cond

    def person(lwri, rwri, cx=320.0, cy=240.0, sw=120.0):
        """One skeleton item; wrists given as (x,y). Shoulders sw apart at y=cy."""
        def k(x, y):
            return [x, y, 0.9]
        kpts = [[0, 0, 0.0]] * 17
        kpts[NOSE] = k(cx, cy - 0.7 * sw)
        kpts[L_EYE] = k(cx - 0.12 * sw, cy - 0.78 * sw)
        kpts[R_EYE] = k(cx + 0.12 * sw, cy - 0.78 * sw)
        kpts[L_EAR] = k(cx - 0.22 * sw, cy - 0.72 * sw)
        kpts[R_EAR] = k(cx + 0.22 * sw, cy - 0.72 * sw)
        kpts[L_SHO] = k(cx - 0.5 * sw, cy)
        kpts[R_SHO] = k(cx + 0.5 * sw, cy)
        kpts[L_HIP] = k(cx - 0.35 * sw, cy + 1.2 * sw)
        kpts[R_HIP] = k(cx + 0.35 * sw, cy + 1.2 * sw)
        kpts[L_WRI] = k(*lwri)
        kpts[R_WRI] = k(*rwri)
        box = [cx - sw, cy - 0.9 * sw, cx + sw, cy + 2.0 * sw]
        return {"id": 1, "name": "", "box": box, "kpts": kpts}

    def run(seq, dt=0.08):
        r = GestureReactor(cfg)
        fired = []
        t = 100.0
        for it in seq:
            ev = r.update({"w": 640, "h": 480, "items": [it]}, t)
            if ev:
                fired.append(ev["gesture"])
            t += dt
        return fired

    cx, cy, sw = 320.0, 240.0, 120.0
    down = (cx + 0.5 * sw, cy + 1.15 * sw)   # right wrist hanging at the hip
    ldown = (cx - 0.5 * sw, cy + 1.15 * sw)

    # WAVE: right wrist above shoulder, x oscillating ~1.3 Hz for ~1.4 s
    wave = []
    for i in range(20):
        t = i * 0.08
        wx = cx + 0.5 * sw + 0.25 * sw * math.sin(2 * math.pi * 1.3 * t)
        wave.append(person(ldown, (wx, cy - 0.5 * sw)))
    f = run(wave)
    c("wave fires", "wave" in f)
    c("wave fires exactly once (cooldown)", f.count("wave") == 1)

    # RAISE_HAND: right wrist above the head, held still
    raise_seq = [person(ldown, (cx + 0.5 * sw, cy - 0.95 * sw)) for _ in range(16)]
    f = run(raise_seq)
    c("raise_hand fires", "raise_hand" in f and f.count("raise_hand") == 1)

    # BOTH_ARMS_UP: both wrists above both shoulders, held
    both = [person((cx - 0.5 * sw, cy - 0.5 * sw), (cx + 0.5 * sw, cy - 0.5 * sw))
            for _ in range(14)]
    f = run(both)
    c("both_arms_up fires", "both_arms_up" in f and f.count("both_arms_up") == 1)

    # IDLE: both wrists hanging at the hips -> NOTHING
    idle = [person(ldown, down) for _ in range(20)]
    f = run(idle)
    c("idle fires nothing (no false positive)", f == [])

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
