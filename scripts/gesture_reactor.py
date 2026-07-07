#!/usr/bin/env python3
"""Human-gesture -> robot-gesture REACTOR (the interactive "wave back" demo).

A person waves at the robot; the robot waves back. This module is the PERCEPTION +
DECISION half: it consumes the per-person skeletons that pose_service.py publishes to
/dev/shm/g1_pose_tracks.json ({w,h,items:[{id,name,box,kpts}]}, COCO-17 keypoints in
IMAGE pixels, y DOWN, conf 0-1) and -- OPTIONALLY -- the MediaPipe hand landmarks that
hands_service.py publishes to /dev/shm/g1_hands_tracks.json (same camera, same pixels),
and classifies one DELIBERATELY SIMPLE gesture:

    WAVE          a hand held UP AND oscillating horizontally ~1 s

Two sources, FUSED to be both more sensitive and harder to fool -- important because the
G1 is short, so a normal wave is often at SHOULDER LEVEL (hand not raised high) and a close
person's head + shoulders drop out of frame entirely:
  * SKELETON wave: forearm RAISED (wrist above the ELBOW -- true for an overhead OR a
    shoulder-level wave; a walking arm-swing has the wrist BELOW the elbow, so it is rejected)
    with a horizontal oscillation. A CLEAR oscillation (strict amplitude + >=2 reversals)
    fires on its own, at any height.
  * PALM corroboration: a SUBTLE wave (small-amplitude oscillation) fires ONLY if the hand
    feed also shows an OPEN palm oscillating at the SAME wrist. The subtle signal alone never
    fires -- the two must agree, which keeps false waves down.
  * Two raised, oscillating arms at once are VETOED as a clap / two-handed gesture.

Design priorities (in order): ROBUSTNESS, FEW FALSE POSITIVES, simplicity. A reliable
wave-back in front of a client beats six flaky gestures. Everything here is temporal
(a single frame never fires a gesture) and SCALE-INVARIANT (skeleton thresholds are in
shoulder-widths, hand thresholds in palm-lengths, so distance doesn't matter).

Split into two layers:
  * classify() / _skeleton_wave() / _hand_corroborates() -- PURE (no robot, no I/O, no
    shm). Deterministic, fully unit-testable from synthesized sequences (sim_gestures.py).
  * GreetingService -- the thin, OPTIONAL bridge that reads shm, runs a reactor, and fires
    the mapped robot gesture through a caller-supplied callback ONLY when a "greeting mode"
    gate + a safety gate both say OK. All robot coupling is via callbacks, so this file
    imports nothing from the controller and stays testable.

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

KP_MIN_CONF = 0.3   # matches pose_service.KP_MIN_CONF: below this a joint is "missing"

# --- MediaPipe hand-landmark indices (hands_service.py emits {landmarks:[[x,y,z]x21]} in
# the SAME source-frame pixels as the skeleton, so hand + body line up pixel-for-pixel). ---
H_WRIST = 0
H_INDEX_MCP, H_INDEX_PIP, H_INDEX_TIP = 5, 6, 8
H_MIDDLE_MCP, H_MIDDLE_PIP, H_MIDDLE_TIP = 9, 10, 12
H_RING_MCP, H_RING_PIP, H_RING_TIP = 13, 14, 16
H_PINKY_MCP, H_PINKY_PIP, H_PINKY_TIP = 17, 18, 20
# (tip, pip) for the four reliable fingers (thumb excluded -- its geometry is too noisy).
_FINGERS = ((H_INDEX_TIP, H_INDEX_PIP), (H_MIDDLE_TIP, H_MIDDLE_PIP),
            (H_RING_TIP, H_RING_PIP), (H_PINKY_TIP, H_PINKY_PIP))


# --- human gesture -> robot response ---------------------------------------
# One gesture, one response: a person waves -> the robot waves back. The value is a cmd
# name the controller's apply_cmd() already knows (see ARM_GESTURES), so firing reuses the
# EXISTING serialized execution path. NB: the native LocoClient.WaveHand() ("wave") is a
# DEAD no-op on this G1 (see web/controller.js) -- the working wave is arm-action 26
# ("high_wave"), the same one the manual wave button uses.
GESTURE_TO_ROBOT = {
    "wave":         "high_wave",  # arm action 26 (self-completing) -- the working wave-back
}

# Short human-readable feedback for the dashboard ("saw wave -> waving back").
GESTURE_HUMAN = {
    "wave":         "wave",
}


class ReactorConfig:
    """All tunables in one place (keyword-overridable so tests can tighten/loosen).

    Times are seconds; skeleton distances are in SHOULDER-WIDTHS, hand distances in
    PALM-LENGTHS (both scale-invariant). Defaults are tuned for ~10-15 fps input and favour
    missing a marginal gesture over firing a false one."""

    def __init__(self, **kw):
        # history / target
        self.window_s = 1.3            # temporal window classified each update
        self.kp_min_conf = KP_MIN_CONF
        self.track_ttl_s = 0.8         # drop a track's history if unseen this long
        self.min_track_age_s = 0.4     # target must be tracked this long before it can fire
        self.min_conf_frac = 0.15      # a target must have >= this fraction of its 17 kpts

        # SKELETON WAVE: forearm RAISED (wrist above the ELBOW) + horizontal oscillation.
        # Admission is elbow-based so a natural SHOULDER-LEVEL wave (hand not raised high) is
        # seen too -- a walking arm-swing keeps the wrist BELOW the elbow and is rejected.
        self.wave_min_frames = 5       # need several samples to see an oscillation
        self.wave_min_span_s = 0.5     # ...spanning at least this long (~1 s wave)
        self.wave_up_frac = 0.6        # >= this fraction of frames: wrist above the ELBOW (raised)
        self.wave_elbow_margin_frac = 0.10  # wrist must clear the elbow by this (shoulder-widths)
                                       #   to count as raised -- absorbs jitter / mild elbow flex

        # STRICT oscillation -> a "strong" wave that fires on the skeleton ALONE, at ANY height
        # (as long as the forearm is up): a clear side-to-side sweep IS a wave.
        self.wave_amp_frac = 0.25      # peak-to-peak wrist-x sweep, in shoulder-widths
        self.wave_min_reversals = 2    # direction changes -> genuine back-and-forth (not a sweep)

        # RELAXED oscillation -> a WEAK/subtle wave (small sweep) that fires ONLY with palm
        # corroboration. Same reversal floor as strict (only the amplitude is relaxed), so a
        # one-shot reach or a single-burst gesticulation (rev < 2) can't sneak through.
        self.wave_amp_frac_lo = 0.15   # a smaller sweep still counts if the palm agrees
        self.wave_min_reversals_lo = 2

        # BIMANUAL veto: two raised, oscillating arms at once = clap / two-handed gesticulation,
        # not a one-arm wave. A LOWER frame/span bar than admission so a hand briefly occluded
        # mid-clap (the two hands meeting) still counts toward the veto.
        self.wave_bimanual_veto = True
        self.wave_veto_min_frames = 2
        self.wave_veto_min_span_s = 0.15

        # OSCILLATION SHAPE (jitter-robust). Reversals + amplitude come from a hysteresis
        # zig-zag (_swings) that only counts a swing once the wrist RETRACES by >= swing_min
        # from the running extreme, where swing_min = max(wave_swing_frac*scale, wave_swing_min_px).
        # The ABSOLUTE px floor is the key: fixed pose jitter (~4-6 px) can never manufacture a
        # reversal or amplitude even at distance (small scale), where a scale-only deadband would
        # collapse below the noise and read a HELD hand as a wave (the on-robot false positive).
        self.wave_swing_frac = 0.10
        self.wave_swing_min_px = 9.0   # (was 14) low enough to catch a small/far wave; safe
                                       #   because the wrist series is SMOOTHED first (_smooth3)
        self.wave_smooth = True        # 3-point moving average before swing counting
        # HORIZONTAL DOMINANCE: a wave sweeps sideways at ~constant forearm height; require the
        # x-swing to beat the y-swing by this factor -> kills a circular stir / vertical nod / bob.
        self.wave_horiz_dom = 1.3      # x-swing must beat y-swing by this -> rejects a circular
                                       #   stir; smoothing already shrinks a real wave's y-wobble
        # RATE CAP: a deliberate wave is ~1-2.5 Hz; reject faster motion (tremor / fidget), in rev/s.
        self.wave_max_rev_rate = 6.0

        # PALM feed (MediaPipe hand landmarks) -- corroborates a WEAK skeleton wave.
        self.use_hands = True          # fuse the hand feed at all
        self.hand_ttl_s = 0.5          # forget the tracked hand if unseen this long
        self.hand_assoc_frac = 0.9     # hand wrist must be within this*scale of the skeleton
                                       #   wrist to corroborate THAT arm (same pixel frame)
        self.hand_open_min_frac = 0.5  # >= this fraction of hand frames show an OPEN palm
        self.hand_amp_frac = 0.4       # hand-wrist sweep to corroborate, in PALM-lengths
        self.hand_min_reversals = 1    # some back-and-forth of the hand
        self.hand_min_frames = 3
        self.hand_swing_frac = 0.15    # hand oscillation swing floor (fraction of palm-length)...
        self.hand_swing_min_px = 6.0   # ...with an absolute px floor, so hand jitter can't corroborate
        self.hand_open_ext = 1.05      # a finger is "extended" if its tip is this* farther
                                       #   from the wrist than its PIP; open palm = >=3 of 4

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


def _swings(xs, swing_min):
    """Jitter-robust oscillation over a 1-D series -> (reversals, largest confirmed swing px).

    A hysteresis zig-zag: a turning point is CONFIRMED only once the series RETRACES by
    >= swing_min from the running extreme, so sub-swing_min pose jitter contributes ZERO
    reversals and ZERO amplitude. The first confirmed leg only establishes a direction (not a
    reversal); each later confirmed turn is a reversal. So a held/drifting hand -> (0, 0.0); a
    single out-and-back sweep -> (0-1, swing); a genuine back-and-forth wave -> (>=2, swing).
    `max_swing` is the largest confirmed extreme-to-extreme excursion (raw noise never inflates
    it). Pass swing_min in the SAME units as xs (pixels)."""
    if len(xs) < 2 or swing_min <= 0:
        return 0, 0.0
    pivot = hi = lo = xs[0]      # pivot = last confirmed turning point; hi/lo = extremes since it
    dirn = 0                     # +1 rising, -1 falling, 0 = no leg confirmed yet
    rev = 0
    max_swing = 0.0
    for x in xs[1:]:
        if x > hi:
            hi = x
        if x < lo:
            lo = x
        if dirn >= 0 and (hi - x) >= swing_min:      # rose to `hi`, now retraced down -> a PEAK
            if dirn != 0:
                rev += 1
            max_swing = max(max_swing, abs(hi - pivot))
            pivot, lo, dirn = hi, x, -1
        elif dirn <= 0 and (x - lo) >= swing_min:    # fell to `lo`, now retraced up -> a TROUGH
            if dirn != 0:
                rev += 1
            max_swing = max(max_swing, abs(lo - pivot))
            pivot, hi, dirn = lo, x, 1
    max_swing = max(max_swing, abs(xs[-1] - pivot))  # include the final unfinished leg's reach
    return rev, max_swing


def _smooth3(vals):
    """Centered 3-point moving average. Attenuates per-frame pose jitter (and a fast tremor)
    while barely touching a ~1-2 Hz wave, so the swing floor can be LOW enough to catch a small
    or far wave without a HELD hand's noise faking one. Endpoints pass through unchanged."""
    n = len(vals)
    if n < 3:
        return list(vals)
    out = [vals[0]]
    for i in range(1, n - 1):
        out.append((vals[i - 1] + vals[i] + vals[i + 1]) / 3.0)
    out.append(vals[-1])
    return out


def _arm_metrics(history, wri_idx, sho_idx, elb_idx, cfg):
    """Temporal metrics for ONE arm over the window, or None if too few good samples.

    history: list of (t, joints, scale). A frame contributes if the wrist is confident AND a
    RAISED reference is available -- the ELBOW (so 'wrist above elbow' = forearm up, which
    reads the same for an overhead OR a shoulder-level wave), falling back to the SHOULDER only
    when the elbow itself is occluded. Distances are normalized by the median scale."""
    ts, xs, ys, ups, scales = [], [], [], [], []
    for (t, joints, scale) in history:
        w = joints.get(wri_idx)
        if not w:
            continue
        ref = joints.get(elb_idx)              # forearm-raised test: wrist above the ELBOW
        if ref is None:
            ref = joints.get(sho_idx)          # elbow occluded -> fall back to the shoulder
        if ref is None:
            continue
        ts.append(t)
        xs.append(w[0])
        ys.append(w[1])
        # raised = wrist clears the reference by a margin (smaller y == higher in the image)
        ups.append(1 if w[1] < ref[1] - cfg.wave_elbow_margin_frac * scale else 0)
        scales.append(scale)
    if len(ts) < 2:
        return None
    scale = _median(scales) or 1.0
    sm = max(cfg.wave_swing_frac * scale, cfg.wave_swing_min_px)   # px floor beats fixed jitter
    sx, sy = (_smooth3(xs), _smooth3(ys)) if cfg.wave_smooth else (xs, ys)
    rev, xswing = _swings(sx, sm)              # jitter-robust: sub-swing_min wobble -> (0, 0.0)
    _, yswing = _swings(sy, sm)
    return {
        "n": len(ts),
        "span": ts[-1] - ts[0],
        "up_frac": sum(ups) / len(ups),
        "amp": xswing / scale,                 # amplitude of the largest CONFIRMED swing
        "rev": rev,
        "xswing": xswing,
        "yswing": yswing,
        "wrist": (_median(xs), _median(ys)),   # median wrist pos -> palm association
        "scale": scale,
    }


def _best_first(la, ra, key):
    """The two arm-metric dicts (dropping Nones), most-`key` first."""
    ms = [m for m in (la, ra) if m is not None]
    ms.sort(key=lambda m: m[key], reverse=True)
    return ms


def _arm_qualifies(m, cfg):
    """Does this arm look wave-ish enough to count toward the BIMANUAL veto? A LOWER frame/span
    bar than admission, so a hand briefly occluded mid-clap still registers as 'active'."""
    return (m is not None
            and m["n"] >= cfg.wave_veto_min_frames
            and m["span"] >= cfg.wave_veto_min_span_s
            and m["up_frac"] >= cfg.wave_up_frac
            and m["amp"] >= cfg.wave_amp_frac_lo
            and m["rev"] >= cfg.wave_min_reversals_lo)


def _skeleton_wave(history, cfg=DEFAULT_CFG):
    """Skeleton wave detector, with a STRENGTH so palm fusion can weigh it.

    Returns {"strength": "strong"|"weak", "wrist": (x, y), "scale": s} or None.
      strong -- forearm raised + a CLEAR oscillation (strict amplitude AND >=2 reversals):
                fires on its own, at any height (overhead OR shoulder-level).
      weak   -- forearm raised + a SUBTLE oscillation (small amplitude, still >=2 reversals):
                fires ONLY if the palm feed corroborates it.
    Two raised, oscillating arms at once are vetoed as a clap / two-handed gesture."""
    if len(history) < 2:
        return None
    la = _arm_metrics(history, L_WRI, L_SHO, L_ELB, cfg)
    ra = _arm_metrics(history, R_WRI, R_SHO, R_ELB, cfg)
    if cfg.wave_bimanual_veto and _arm_qualifies(la, cfg) and _arm_qualifies(ra, cfg):
        return None                            # clap / two-handed gesticulation, not a wave
    for m in _best_first(la, ra, key="up_frac"):
        if not (m["n"] >= cfg.wave_min_frames
                and m["span"] >= cfg.wave_min_span_s
                and m["up_frac"] >= cfg.wave_up_frac
                and m["rev"] >= cfg.wave_min_reversals_lo):   # enough back-and-forth to be a wave
            continue
        if m["xswing"] < cfg.wave_horiz_dom * max(m["yswing"], 1e-6):
            continue                           # stir / nod / vertical bob -- not a sideways sweep
        if m["span"] > 0 and (m["rev"] / m["span"]) > cfg.wave_max_rev_rate:
            continue                           # too fast -- tremor / fidget, not a deliberate wave
        if m["amp"] >= cfg.wave_amp_frac and m["rev"] >= cfg.wave_min_reversals:
            return {"strength": "strong", "wrist": m["wrist"], "scale": m["scale"]}
        if m["amp"] >= cfg.wave_amp_frac_lo:
            return {"strength": "weak", "wrist": m["wrist"], "scale": m["scale"]}
    return None


def classify(history, cfg=DEFAULT_CFG):
    """PURE skeleton-only classifier -> "wave" or None.

    Fires on a STRONG wave (forearm raised + a clear side-to-side oscillation, at any height).
    A weak/subtle wave needs palm corroboration and is handled by the reactor's fusion
    (GestureReactor.update), not here -- so this stays a strict, deterministic unit."""
    w = _skeleton_wave(history, cfg)
    return "wave" if (w and w["strength"] == "strong") else None


# ------------------------------------------------------------------- palm (hand) helpers
def parse_hands(frame):
    """Hand shm dict {w,h,items:[{hand,score,landmarks}]} or a bare items list -> items list."""
    if isinstance(frame, dict):
        return frame.get("items", []) or []
    return list(frame or [])


def _hand_scale(lm):
    """Palm-length pixel unit for a hand's landmarks: wrist -> middle-finger MCP. Falls back
    to the landmark bounding-box diagonal. Never 0 (guards divides)."""
    if len(lm) > H_MIDDLE_MCP:
        d = math.hypot(lm[H_WRIST][0] - lm[H_MIDDLE_MCP][0],
                       lm[H_WRIST][1] - lm[H_MIDDLE_MCP][1])
        if d > 1e-3:
            return d
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    d = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if lm else 0.0
    return d if d > 1e-3 else 1.0


def _hand_open(lm, ext=1.05):
    """True if the palm looks OPEN: >= 3 of the 4 reliable fingers extended, where a finger
    is 'extended' when its TIP is farther from the wrist than its PIP joint (orientation-
    independent, so it holds for a hand at any angle). A fist/point reads as closed."""
    if len(lm) <= H_PINKY_TIP:
        return False
    wx, wy = lm[H_WRIST][0], lm[H_WRIST][1]
    ext_n = 0
    for tip, pip in _FINGERS:
        dt = math.hypot(lm[tip][0] - wx, lm[tip][1] - wy)
        dp = math.hypot(lm[pip][0] - wx, lm[pip][1] - wy)
        if dt > ext * dp:
            ext_n += 1
    return ext_n >= 3


def _hand_obs(item, cfg):
    """One hand item -> (wrist_xy, palm_scale, open_bool), or None if it has no usable
    landmarks."""
    lm = item.get("landmarks") or []
    if len(lm) <= H_WRIST or len(lm[H_WRIST]) < 2:
        return None
    wrist = (float(lm[H_WRIST][0]), float(lm[H_WRIST][1]))
    return wrist, _hand_scale(lm), _hand_open(lm, cfg.hand_open_ext)


def _hand_corroborates(hand_traj, wrist_xy, scale, cfg):
    """Does the tracked hand confirm a wave at `wrist_xy` (the skeleton's waving wrist)?

    ALL three must hold over the hand window: the hand sits ON that wrist (within
    hand_assoc_frac*scale of it), an OPEN palm most of the time, and a horizontal
    oscillation (amplitude + a reversal). This is the 'palm agrees' half of the fusion."""
    pts = list(hand_traj)
    if len(pts) < cfg.hand_min_frames:
        return False
    hx = _median([wr[0] for (_, wr, _, _) in pts])
    hy = _median([wr[1] for (_, wr, _, _) in pts])
    if math.hypot(hx - wrist_xy[0], hy - wrist_xy[1]) > cfg.hand_assoc_frac * scale:
        return False
    if sum(1 for (_, _, _, op) in pts if op) / len(pts) < cfg.hand_open_min_frac:
        return False
    hscale = _median([hs for (_, _, hs, _) in pts]) or 1.0
    xs = [wr[0] for (_, wr, _, _) in pts]
    ys = [wr[1] for (_, wr, _, _) in pts]
    sm = max(cfg.hand_swing_frac * hscale, cfg.hand_swing_min_px)   # jitter-robust, px floor
    sx, sy = (_smooth3(xs), _smooth3(ys)) if cfg.wave_smooth else (xs, ys)
    rev, xswing = _swings(sx, sm)
    _, yswing = _swings(sy, sm)
    if xswing < cfg.wave_horiz_dom * max(yswing, 1e-6):
        return False                            # a still / vertical / jittering hand can't corroborate
    return (xswing / hscale) >= cfg.hand_amp_frac and rev >= cfg.hand_min_reversals


# ------------------------------------------------------------------- reactor
def parse_frame(frame):
    """Accept either the raw pose shm dict {w,h,items:[...]} or a bare items list; return
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
    """Stateful wrapper around the classifier: keeps per-track keypoint history + a tracked
    hand trajectory, picks ONE target person per frame, FUSES skeleton + palm, and DEBOUNCES
    so a continuous wave fires exactly once.

    Clean API:  reactor.update(frame, t, hands_frame) -> None | {gesture, track_id, ...}
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or DEFAULT_CFG
        self._hist = {}          # track_id -> deque[(t, joints, scale)]
        self._first_seen = {}    # track_id -> t first observed
        self._hand_traj = deque()  # (t, wrist_xy, palm_scale, open) for the tracked hand
        self._cooldown = {}      # gesture name -> t until which it may not re-fire
        self._last_fire_t = -1e9

    def reset(self):
        """Forget all history/debounce state (call when greeting mode toggles OFF so a
        stale held pose can't fire the instant it comes back ON)."""
        self._hist.clear()
        self._first_seen.clear()
        self._hand_traj.clear()
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

    def _ingest_hands(self, hands_frame, t):
        """Track the ONE most relevant hand across frames (the feed has no ids): pick the
        hand nearest the last tracked hand, else the highest (a waving hand is up). Keep a
        short trajectory (wrist / palm-scale / openness) for palm corroboration."""
        obs = [o for o in (_hand_obs(it, self.cfg)
                           for it in parse_hands(hands_frame)) if o is not None]
        chosen = None
        if obs:
            if self._hand_traj:
                px, py = self._hand_traj[-1][1]
                chosen = min(obs, key=lambda o: math.hypot(o[0][0] - px, o[0][1] - py))
            else:
                chosen = min(obs, key=lambda o: o[0][1])   # highest hand (smallest y)
            self._hand_traj.append((t, chosen[0], chosen[1], chosen[2]))
        # trim to the window; forget the hand entirely if it's gone stale
        cutoff = t - self.cfg.window_s
        while self._hand_traj and self._hand_traj[0][0] < cutoff:
            self._hand_traj.popleft()
        if self._hand_traj and (t - self._hand_traj[-1][0]) > self.cfg.hand_ttl_s:
            self._hand_traj.clear()

    def update(self, frame, t=None, hands_frame=None):
        """Feed one pose-track frame (+ optional hand-track frame); return an event or None.

        frame:       pose {w,h,items:[{id,box,kpts},...]} (shm dict) OR a bare items list.
        hands_frame: hand {w,h,items:[{landmarks:[[x,y,z]x21]},...]} OR None (skip palm fusion).
        t:           monotonic-ish seconds (defaults to wall clock). Passing your own clock
                     makes the reactor fully deterministic (the sim/tests rely on this)."""
        if t is None:
            t = time.time()
        items, w, h = parse_frame(frame)
        self._ingest(items, w, h, t)
        if self.cfg.use_hands:
            self._ingest_hands(hands_frame, t)

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

        # FUSE: a strong (shoulder-based) skeleton wave fires alone; a weak/partial one fires
        # only if the palm feed corroborates it at the same wrist.
        sk = _skeleton_wave(list(dq), self.cfg)
        if sk is None:
            return None
        if sk["strength"] == "strong":
            source = "skeleton"
        elif self.cfg.use_hands and _hand_corroborates(
                self._hand_traj, sk["wrist"], sk["scale"], self.cfg):
            source = "skeleton+palm"
        else:
            return None
        gesture = "wave"

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
            "source": source,          # "skeleton" or "skeleton+palm" (which evidence fired)
            "t": t,
            # Target's image box + frame size so a UI can anchor an on-feed label to the
            # right person even without the live pose overlay. Inert for existing callers.
            "box": target.get("box"), "w": w, "h": h,
        }


def describe(event):
    """Dashboard feedback string, e.g. 'saw wave -> waving back'."""
    if not event:
        return ""
    robot = event.get("robot_gesture") or "?"
    verb = {"high_wave": "waving back", "wave": "waving back"}.get(robot, robot)
    return f"saw {event.get('human', event.get('gesture'))} -> {verb}"


# =====================================================================================
# GreetingService -- the SAFETY-GATED bridge to the robot (shm -> reactor -> callback).
# =====================================================================================
class GreetingService:
    """Runs a GestureReactor in a daemon thread against the pose feed (and, optionally, the
    hand feed) and fires the mapped robot gesture through `fire_fn` -- but ONLY when BOTH
    gates pass:

        enabled_fn()  -- the "greeting mode" master toggle is ON (default: caller keeps it
                         OFF; auto-firing arms near clients must be opt-in).
        safe_fn()     -- it is safe RIGHT NOW: robot upright in walk, not moving, not
                         mid-gesture or mid-transition, no arm currently raised. (Supplied
                         by the controller; see the wiring block in robot_web_controller.py.)

    Every piece of robot coupling is a callback, so this class imports nothing from the
    controller and is exercised in tests with plain lambdas. Firing goes through the caller's
    EXISTING serialized path (the controller sets state.pending_cmd), so the _arm_lock,
    auto-release, and single-flight guarantees are reused unchanged.

    Additional safety layers beyond the reactor's own debounce:
      * stale-feed guard: only advances on a NEW pose frame (mtime change), so a frozen or
        dead pose feed can never fire.
      * demand heartbeat: while enabled, touches POSE_DEMAND (and, if fusing, HANDS_DEMAND)
        so the demand-gated pose/hands containers keep inferring -- greeting is a "viewer".
      * on OFF: resets the reactor so a pose held during the OFF window can't fire the
        instant it flips back ON.
    """

    def __init__(self, tracks_path, enabled_fn, safe_fn, fire_fn,
                 demand_path=None, hands_path=None, hands_demand_path=None,
                 on_event=None, on_skip=None,
                 cfg=None, period_s=0.1, idle_period_s=0.3):
        self.tracks_path = tracks_path
        self.demand_path = demand_path
        self.hands_path = hands_path
        self.hands_demand_path = hands_demand_path
        self.enabled_fn = enabled_fn
        self.safe_fn = safe_fn
        self.fire_fn = fire_fn
        self.on_event = on_event        # called on EVERY classified event (dashboard feedback)
        self.on_skip = on_skip          # called when an event was gated out by safe_fn
        self.reactor = GestureReactor(cfg)
        self.cfg = self.reactor.cfg
        self.period_s = period_s
        self.idle_period_s = idle_period_s
        self._stop = threading.Event()
        self._thread = None
        self._last_mtime = None
        self._hands_mtime = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="greeting", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _read_json(self, path, last_attr):
        """Return the parsed JSON at `path` ONLY if it changed since the last read (tracked
        on self.<last_attr>), else None -- so a static/frozen feed never advances anything."""
        try:
            mt = os.path.getmtime(path)
        except (OSError, TypeError):
            return None
        if mt == getattr(self, last_attr):
            return None
        setattr(self, last_attr, mt)
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _read_frame(self):
        return self._read_json(self.tracks_path, "_last_mtime")

    def _read_hands(self):
        if not (self.cfg.use_hands and self.hands_path):
            return None
        return self._read_json(self.hands_path, "_hands_mtime")

    def _touch_demand(self, path):
        if not path:
            return
        try:
            with open(path, "wb") as f:
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

    def poll_once(self, frame, t, hands_frame=None):
        """Run the reactor on ONE pose frame (+ optional hand frame) and dispatch any event
        through the safety gate. Assumes greeting mode is already enabled (the loop gates
        that upstream). Called by _loop with freshly-read shm, and by tests with synthetics."""
        return self._handle_event(self.reactor.update(frame, t, hands_frame))

    def _loop(self):
        while not self._stop.is_set():
            if not self.enabled_fn():
                self.reactor.reset()          # clean slate when greeting mode is OFF
                self._stop.wait(self.idle_period_s)
                continue
            self._touch_demand(self.demand_path)          # keep the pose GPU alive
            if self.cfg.use_hands:
                self._touch_demand(self.hands_demand_path)  # ...and the hands GPU
            frame = self._read_frame()
            if frame is not None:
                # freshest hands to fuse this tick (None if the hand feed didn't change)
                hands = self._read_hands()
                self.poll_once(frame, time.time(), hands)
            self._stop.wait(self.period_s)


# ------------------------------------------------------------------------- selftest
def selftest():
    """Fast smoke test of the PURE classifier + palm fusion + debounce, no camera/robot/shm.

    Synthesizes minimal wave / raised-hand / idle histories (and a shoulderless wave with
    and without a corroborating palm) directly; sim_gestures.py has the full harness."""
    cfg = ReactorConfig()
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and cond

    def person(lwri, rwri, cx=320.0, cy=240.0, sw=120.0):
        """One skeleton item; wrists given as (x,y). Shoulders sw apart at y=cy; elbows
        halfway down (so the elbow fallback has a reference when shoulders are dropped)."""
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
        kpts[L_ELB] = k(cx - 0.5 * sw, cy + 0.5 * sw)
        kpts[R_ELB] = k(cx + 0.5 * sw, cy + 0.5 * sw)
        kpts[L_HIP] = k(cx - 0.35 * sw, cy + 1.2 * sw)
        kpts[R_HIP] = k(cx + 0.35 * sw, cy + 1.2 * sw)
        kpts[L_WRI] = k(*lwri)
        kpts[R_WRI] = k(*rwri)
        box = [cx - sw, cy - 0.9 * sw, cx + sw, cy + 2.0 * sw]
        return {"id": 1, "name": "", "box": box, "kpts": kpts}

    def open_hand(x, y, scale=40.0, is_open=True):
        """A minimal 21-landmark hand centered at (x, y): tips far from the wrist = open
        palm, tips near = fist."""
        lm = [[x, y, 0.0] for _ in range(21)]
        lm[H_MIDDLE_MCP] = [x, y - scale, 0.0]         # palm length ~= scale
        reach = scale * (1.7 if is_open else 0.6)
        for tip, pip in _FINGERS:
            lm[pip] = [x, y - 0.9 * scale, 0.0]
            lm[tip] = [x, y - reach, 0.0]
        return {"hand": "Right", "score": 0.9, "landmarks": lm}

    def run(seq, hands=None, dt=0.08):
        r = GestureReactor(cfg)
        fired = []
        t = 100.0
        for i, it in enumerate(seq):
            hf = None
            if hands is not None and hands[i] is not None:
                hf = {"w": 640, "h": 480, "items": [hands[i]]}
            ev = r.update({"w": 640, "h": 480, "items": [it]}, t, hf)
            if ev:
                fired.append(ev)
            t += dt
        return fired

    cx, cy, sw = 320.0, 240.0, 120.0
    down = (cx + 0.5 * sw, cy + 1.15 * sw)   # right wrist hanging at the hip
    ldown = (cx - 0.5 * sw, cy + 1.15 * sw)

    def wave_frames(amp=0.25, wy=None, freq=1.3, n=20):
        wy = (cy - 0.5 * sw) if wy is None else wy      # default: wrist ABOVE the shoulder
        seq = []
        for i in range(n):
            t = i * 0.08
            wx = cx + 0.5 * sw + amp * sw * math.sin(2 * math.pi * freq * t)
            seq.append(person(ldown, (wx, wy)))
        return seq

    def drop_shoulders(seq):
        for it in seq:
            for j in (L_SHO, R_SHO):
                it["kpts"][j] = [0, 0, 0.0]
        return seq

    def palms_on(seq, is_open=True):
        return [open_hand(it["kpts"][R_WRI][0], it["kpts"][R_WRI][1], is_open=is_open)
                for it in seq]

    # WAVE (overhead): a clear wave above the shoulder fires on its own, once.
    f = [e["gesture"] for e in run(wave_frames())]
    c("overhead wave fires", "wave" in f)
    c("wave fires exactly once (cooldown)", f.count("wave") == 1)

    # SHOULDER-LEVEL wave (wrist AT shoulder height, NOT raised high) -- the case that used to
    # be missed. Elbow-based admission + a clear sweep -> fires on the skeleton ALONE.
    fe = run(wave_frames(wy=cy))                          # wrist on the shoulder line (shoulders at cy)
    c("shoulder-level wave fires (the fix)", "wave" in [e["gesture"] for e in fe])
    c("shoulder-level wave fires on skeleton alone", bool(fe) and fe[0]["source"] == "skeleton")

    # PARTIAL body (shoulders out of frame) but a CLEAR sweep -> also fires alone.
    c("shoulderless clear wave fires alone",
      "wave" in [e["gesture"] for e in run(drop_shoulders(wave_frames()))])

    # RAISE_HAND removed (auto high-five deleted): a still raised hand fires NOTHING now.
    raise_seq = [person(ldown, (cx + 0.5 * sw, cy - 0.95 * sw)) for _ in range(16)]
    c("raised hand fires nothing (high-five removed)", run(raise_seq) == [])

    # IDLE: both wrists at the hips -> NOTHING.
    c("idle fires nothing (no false positive)", run([person(ldown, down) for _ in range(20)]) == [])

    # SUBTLE wave (small amplitude) ALONE -> does NOT fire (needs palm).
    subtle = wave_frames(amp=0.11, freq=1.6, n=26)
    c("subtle wave alone does not fire (needs palm)", run(subtle) == [])

    # ...subtle wave + open palm oscillating on the wrist -> corroborated -> fires (fused).
    fe = run(subtle, hands=palms_on(subtle, is_open=True))
    c("subtle wave + open palm fires (corroborated)", "wave" in [e["gesture"] for e in fe])
    c("corroborated subtle wave tagged skeleton+palm",
      bool(fe) and fe[0]["source"] == "skeleton+palm")

    # ...subtle wave + closed fist -> no corroboration -> no fire.
    c("subtle wave + closed fist does not fire", run(subtle, hands=palms_on(subtle, is_open=False)) == [])

    # CLAP (both forearms raised + oscillating toward centre) -> bimanual veto -> nothing.
    clap = []
    for i in range(20):
        t = i * 0.08
        d = 0.30 * sw * abs(math.sin(2 * math.pi * 1.6 * t))
        cwy = cy - 0.35 * sw
        clap.append(person((cx - 0.15 * sw - d, cwy), (cx + 0.15 * sw + d, cwy)))
    c("clap fires nothing (bimanual veto)", run(clap) == [])

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
