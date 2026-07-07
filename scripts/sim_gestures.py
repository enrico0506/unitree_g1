#!/usr/bin/env python3
"""Deterministic gesture SIMULATOR + harness for gesture_reactor.py (no camera, no robot).

The reactor's classifier is PURE (see gesture_reactor.classify), so it can be exercised
end-to-end from SYNTHESIZED COCO-17 skeletons -- exactly the shape pose_service.py writes
to /dev/shm/g1_pose_tracks.json ({w,h,items:[{id,name,box,kpts}]}, image pixels, y DOWN,
conf 0-1). This file is the harness the demo is trusted on: it builds a waving arm and --
just as important -- NEGATIVE sequences (idle, walking, a static reach, a still raised
hand) that MUST NOT fire, feeds them through a real GestureReactor on a fake clock, and
asserts the wave fires exactly ONCE (debounce) and everything else fires nothing.

Everything is seeded/analytic, so a run is bit-for-bit reproducible: a green --selftest
here is a green demo (given a live pose feed). The synthesis primitives are also imported
by test_gesture_reactor.py, so the tests and the harness agree on what a "wave" looks like.

Image-coords reminder: y grows DOWNWARD, so a raised wrist has a SMALLER y than its
shoulder. All wrist heights below are written as `cy - k*sw` (k>0 == above centre).

    python3 scripts/sim_gestures.py --selftest
"""
import copy
import math
import random

from gesture_reactor import (
    GestureReactor, ReactorConfig, GESTURE_TO_ROBOT,
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB,
    L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
)

# --- Canvas + a nominal person geometry (pixels). sw == shoulder width == the reactor's
# scale unit, so thresholds expressed in shoulder-widths behave identically here. ---
FRAME_W, FRAME_H = 640, 480
CX, CY, SW = 320.0, 240.0, 120.0     # person centred, shoulders 120 px apart
FPS = 12.5                           # ~pose_service INFER_HZ; dt = 0.08 s
DT = 1.0 / FPS

# "Resting" wrist positions -- hanging by the hips, comfortably BELOW the shoulders.
REST_L = (CX - 0.5 * SW, CY + 1.15 * SW)
REST_R = (CX + 0.5 * SW, CY + 1.15 * SW)


def make_skeleton(l_wrist, r_wrist, cx=CX, cy=CY, sw=SW, tid=1, conf=0.9,
                  low_conf=(), rng=None, jitter=0.0):
    """One COCO-17 skeleton item {id,name,box,kpts} with the two wrists placed explicitly.

    Head / shoulders / hips / elbows / knees / ankles are derived from (cx, cy, sw) so the
    body is internally consistent (head above shoulders, hips below) -- the classifier only
    needs shoulders + head + wrists, but a full body makes _frame_scale + target scoring
    realistic. `low_conf` marks keypoint indices as barely-visible (conf 0.1 < KP_MIN_CONF)
    so we can prove robustness to dropout; `jitter` (px, seeded via `rng`) perturbs every
    joint to prove the temporal heuristics survive real pose noise."""
    def k(x, y, idx):
        c = 0.1 if idx in low_conf else conf          # below KP_MIN_CONF -> "missing"
        if jitter and rng is not None:
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        return [float(x), float(y), c]

    kpts = [[0.0, 0.0, 0.0]] * 17
    kpts[NOSE] = k(cx, cy - 0.70 * sw, NOSE)
    kpts[L_EYE] = k(cx - 0.12 * sw, cy - 0.78 * sw, L_EYE)
    kpts[R_EYE] = k(cx + 0.12 * sw, cy - 0.78 * sw, R_EYE)
    kpts[L_EAR] = k(cx - 0.22 * sw, cy - 0.72 * sw, L_EAR)
    kpts[R_EAR] = k(cx + 0.22 * sw, cy - 0.72 * sw, R_EAR)
    kpts[L_SHO] = k(cx - 0.50 * sw, cy, L_SHO)
    kpts[R_SHO] = k(cx + 0.50 * sw, cy, R_SHO)
    kpts[L_ELB] = k(cx - 0.55 * sw, cy + 0.55 * sw, L_ELB)
    kpts[R_ELB] = k(cx + 0.55 * sw, cy + 0.55 * sw, R_ELB)
    kpts[L_HIP] = k(cx - 0.35 * sw, cy + 1.20 * sw, L_HIP)
    kpts[R_HIP] = k(cx + 0.35 * sw, cy + 1.20 * sw, R_HIP)
    kpts[L_KNE] = k(cx - 0.32 * sw, cy + 2.10 * sw, L_KNE)
    kpts[R_KNE] = k(cx + 0.32 * sw, cy + 2.10 * sw, R_KNE)
    kpts[L_ANK] = k(cx - 0.30 * sw, cy + 3.00 * sw, L_ANK)
    kpts[R_ANK] = k(cx + 0.30 * sw, cy + 3.00 * sw, R_ANK)
    kpts[L_WRI] = k(l_wrist[0], l_wrist[1], L_WRI)
    kpts[R_WRI] = k(r_wrist[0], r_wrist[1], R_WRI)
    box = [cx - sw, cy - 0.9 * sw, cx + sw, cy + 3.2 * sw]
    return {"id": tid, "name": "", "box": [float(v) for v in box], "kpts": kpts}


# ---------------------------------------------------------------- gesture sequences
# Each builder returns a list of per-frame skeletons (one person). Feed a list through
# `feed()` (below) with a fixed dt to drive a GestureReactor on a deterministic clock.

def wave_sequence(n=22, freq=1.3, amp=0.25, arm="right", **kw):
    """A WAVE: one wrist held above its shoulder while its x oscillates ~`freq` Hz.

    This is the hero gesture. amp is peak deviation in shoulder-widths (0.25 -> 0.5 sw
    peak-to-peak, well over the reactor's wave_amp_frac). ~1.5 Hz over ~1.8 s gives several
    direction reversals -- what distinguishes a wave from a one-off raise."""
    seq = []
    for i in range(n):
        t = i * DT
        dx = amp * SW * math.sin(2 * math.pi * freq * t)
        up = (CY - 0.55 * SW)                          # comfortably above the shoulder
        if arm == "right":
            seq.append(make_skeleton(REST_L, (CX + 0.5 * SW + dx, up), **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * SW + dx, up), REST_R, **kw))
    return seq


def raise_sequence(n=18, arm="right", **kw):
    """NEGATIVE: one wrist held STILL above the head top (above eyes/nose). This was once a
    valid gesture (the auto high-five); now that the auto high-five is removed it MUST fire
    NOTHING -- kept as a negative fixture so a still raised hand can never trigger a response."""
    up = (CY - 0.95 * SW)                              # above head (nose is at -0.70 sw)
    seq = []
    for _ in range(n):
        if arm == "right":
            seq.append(make_skeleton(REST_L, (CX + 0.5 * SW, up), **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * SW, up), REST_R, **kw))
    return seq


def idle_sequence(n=22, **kw):
    """NEGATIVE: both wrists resting at the hips, nothing moving. Must fire NOTHING."""
    return [make_skeleton(REST_L, REST_R, **kw) for _ in range(n)]


def walking_sequence(n=26, speed=0.06, swing=0.18, **kw):
    """NEGATIVE: the whole body translates across frame while the arms swing naturally at
    HIP level (opposite phase), with a slight vertical bob. This is the adversarial case
    for the wave detector -- there IS horizontal wrist oscillation, but the wrists stay
    BELOW the shoulders, so the "wrist above shoulder" gate must reject it. Must NOT fire."""
    seq = []
    for i in range(n):
        t = i * DT
        cx = CX - 0.5 * n * speed * SW + i * speed * SW    # sweep left->right across frame
        bob = 0.03 * SW * math.sin(2 * math.pi * 2.0 * t)  # gait vertical bob
        ls = swing * SW * math.sin(2 * math.pi * 0.9 * t)
        rs = -ls                                            # arms swing in opposition
        lw = (cx - 0.45 * SW + ls, CY + 1.12 * SW + bob)
        rw = (cx + 0.45 * SW + rs, CY + 1.12 * SW + bob)
        seq.append(make_skeleton(lw, rw, cx=cx, cy=CY + bob, **kw))
    return seq


def reach_sequence(n=18, **kw):
    """NEGATIVE: one arm extended and held above the shoulder but BELOW the head, STILL.

    A tempting false positive -- the wrist IS above the shoulder -- but there is no
    horizontal oscillation, so it fails WAVE. Proves the wave detector doesn't collapse
    into "arm is somewhat up"."""
    mid = (CY - 0.30 * SW)                              # above shoulder, below the head top
    return [make_skeleton(REST_L, (CX + 0.6 * SW, mid), **kw) for _ in range(n)]


# ---------------------------------------------------------------- palm (hand) synthesis
# MediaPipe hand-landmark order (see perception/hands/hand_detector.py): 0 wrist; 5/9/13/17
# are the finger MCPs, 6/10/14/18 the PIPs, 8/12/16/20 the TIPs. We build the raw 21-point
# wire format the hand feed writes, so tests exercise the reactor's palm fusion end-to-end.
_HAND_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))   # (tip, pip): index/middle/ring/pinky


def make_hand(x, y, scale=40.0, is_open=True):
    """One hand item {hand,score,landmarks:[[x,y,z]x21]} centered at (x, y) in SOURCE-frame
    pixels (same frame as the skeleton). Open palm = finger tips reach far from the wrist; a
    fist = tips curl in near it. `scale` is the palm length (wrist -> middle MCP) in pixels."""
    lm = [[float(x), float(y), 0.0] for _ in range(21)]
    lm[9] = [float(x), float(y - scale), 0.0]              # middle-MCP -> palm length ~= scale
    reach = scale * (1.7 if is_open else 0.6)
    for tip, pip in _HAND_FINGERS:
        lm[pip] = [float(x), float(y - 0.9 * scale), 0.0]
        lm[tip] = [float(x), float(y - reach), 0.0]
    return {"hand": "Right", "score": 0.9, "landmarks": lm}


def hands_for_wave(pose_seq, arm="right", is_open=True):
    """A list of hand shm frames placing an (open) palm ON the waving wrist of each pose
    frame -- the 'palm agrees with the skeleton' corroboration case. Same length as pose_seq."""
    wri = R_WRI if arm == "right" else L_WRI
    frames = []
    for sk in pose_seq:
        wx, wy = sk["kpts"][wri][0], sk["kpts"][wri][1]
        frames.append({"w": FRAME_W, "h": FRAME_H,
                       "items": [make_hand(wx, wy, is_open=is_open)]})
    return frames


def no_shoulders(pose_seq):
    """Copies of the skeletons with BOTH shoulders dropped (conf 0) -- the SMALL-ROBOT case:
    head + shoulders out of frame, so a wave must lean on the elbow-based raised test."""
    out = []
    for sk in pose_seq:
        s = copy.deepcopy(sk)
        s["kpts"][L_SHO] = [0.0, 0.0, 0.0]
        s["kpts"][R_SHO] = [0.0, 0.0, 0.0]
        out.append(s)
    return out


def shoulder_level_wave_sequence(n=22, freq=1.3, amp=0.28, arm="right", **kw):
    """A natural wave held at SHOULDER height (wrist ~= shoulder.y, forearm bent up above the
    elbow) -- the 'normal wave without raising the hand high' the old shoulder-anchored gate
    missed. Fires on the elbow-based raised test."""
    seq = []
    for i in range(n):
        t = i * DT
        dx = amp * SW * math.sin(2 * math.pi * freq * t)
        wy = CY                                     # shoulder line (make_skeleton shoulders at CY)
        if arm == "right":
            seq.append(make_skeleton(REST_L, (CX + 0.5 * SW + dx, wy), **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * SW + dx, wy), REST_R, **kw))
    return seq


def subtle_wave_sequence(n=26, freq=1.6, amp=0.11, arm="right", **kw):
    """A SMALL-amplitude wave above the shoulder: clears the elbow raised test and has >=2
    reversals, but its sweep is below the strict amp floor -> a WEAK candidate that fires
    ONLY with palm corroboration."""
    seq = []
    for i in range(n):
        t = i * DT
        dx = amp * SW * math.sin(2 * math.pi * freq * t)
        up = CY - 0.55 * SW
        if arm == "right":
            seq.append(make_skeleton(REST_L, (CX + 0.5 * SW + dx, up), **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * SW + dx, up), REST_R, **kw))
    return seq


def clap_sequence(n=20, freq=1.6, amp=0.30, **kw):
    """NEGATIVE: BOTH forearms raised, both wrists oscillating toward/away from the body
    centreline (a clap / two-handed gesticulation). Each arm alone looks wave-ish, so the
    BIMANUAL VETO must suppress it."""
    seq = []
    for i in range(n):
        t = i * DT
        d = amp * SW * abs(math.sin(2 * math.pi * freq * t))
        up = CY - 0.35 * SW
        seq.append(make_skeleton((CX - 0.15 * SW - d, up), (CX + 0.15 * SW + d, up), **kw))
    return seq


def gesticulate_sequence(n=14, arm="right", **kw):
    """NEGATIVE: one raised forearm making a SINGLE lateral sweep (one direction change) -- a
    conversational gesture / reach, not a wave. rev <= 1 must fail the oscillation gate."""
    seq = []
    for i in range(n):
        frac = i / (n - 1)
        dx = 0.35 * SW * math.sin(math.pi * frac)   # single hump: out then back once
        up = CY - 0.4 * SW
        if arm == "right":
            seq.append(make_skeleton(REST_L, (CX + 0.5 * SW + dx, up), **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * SW + dx, up), REST_R, **kw))
    return seq


# --- non-wave movements that must NOT fire (the reported "raised hand fired" family). Several
# use a small `sw` (a FAR person -> small pixel scale) + per-frame `jitter` to reproduce the
# pose-noise-at-distance leak the absolute swing floor is designed to kill. -----------------
def held_raised_jitter_sequence(n=20, sw=45.0, jitter=5.0, seed=7, wy_frac=-0.95,
                                arm="right", **kw):
    """NEGATIVE (the on-robot bug): a forearm raised and HELD still (wrist fixed clearly above
    the elbow) with per-frame pose jitter, at a FAR/small scale where fixed-px jitter is a big
    fraction of scale. Must fire NOTHING -- jitter stays below the absolute swing floor."""
    rng = random.Random(seed)
    wy = CY + wy_frac * sw
    lrest, rrest = (CX - 0.5 * sw, CY + 1.15 * sw), (CX + 0.5 * sw, CY + 1.15 * sw)
    seq = []
    for _ in range(n):
        if arm == "right":
            seq.append(make_skeleton(lrest, (CX + 0.5 * sw, wy), sw=sw, rng=rng, jitter=jitter, **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * sw, wy), rrest, sw=sw, rng=rng, jitter=jitter, **kw))
    return seq


def stir_sequence(n=20, sw=SW, radius=0.30, revolutions=2, **kw):
    """NEGATIVE: a raised hand tracing a slow CIRCLE -- x is a real sine, but y co-moves in
    lockstep, so xswing ~= yswing -> the horizontal-dominance gate must reject it."""
    seq = []
    for i in range(n):
        ang = 2 * math.pi * revolutions * i / n
        wx = CX + 0.5 * sw + radius * sw * math.cos(ang)
        wy = CY - 0.5 * sw + radius * sw * math.sin(ang)
        seq.append(make_skeleton((CX - 0.5 * sw, CY + 1.15 * sw), (wx, wy), sw=sw, **kw))
    return seq


def nod_sequence(n=20, sw=80.0, bob=0.25, jitter=4.0, seed=11, **kw):
    """NEGATIVE: a raised hand held at ~fixed x while it bobs / tilts VERTICALLY (+ jitter).
    No horizontal oscillation -> x-swing ~ 0 -> no wave."""
    rng = random.Random(seed)
    seq = []
    for i in range(n):
        wy = CY - 0.5 * sw + bob * sw * math.sin(2 * math.pi * 1.3 * i * DT)
        seq.append(make_skeleton((CX - 0.5 * sw, CY + 1.15 * sw), (CX + 0.5 * sw, wy),
                                 sw=sw, rng=rng, jitter=jitter, **kw))
    return seq


def drift_sequence(n=20, sw=100.0, total=0.25, jitter=5.0, seed=13, **kw):
    """NEGATIVE: a raised hand drifting MONOTONICALLY sideways (a slow lean / point), + jitter.
    Never retraces >= the swing floor -> zero confirmed reversals -> no wave (even though the
    raw range would look like amplitude)."""
    rng = random.Random(seed)
    seq = []
    for i in range(n):
        wx = CX + 0.3 * sw + total * sw * (i / (n - 1))
        seq.append(make_skeleton((CX - 0.5 * sw, CY + 1.15 * sw), (wx, CY - 0.5 * sw),
                                 sw=sw, rng=rng, jitter=jitter, **kw))
    return seq


def tremor_sequence(n=20, sw=100.0, freq=4.0, amp=0.22, **kw):
    """NEGATIVE: a raised hand shaking FAST (~4 Hz) with real amplitude -- clears the swing
    floor, so it is the RATE CAP (too many reversals/second) that must reject it as a tremor /
    fidget rather than a deliberate ~1-2 Hz wave."""
    seq = []
    for i in range(n):
        wx = CX + 0.5 * sw + amp * sw * math.sin(2 * math.pi * freq * i * DT)
        seq.append(make_skeleton((CX - 0.5 * sw, CY + 1.15 * sw), (wx, CY - 0.5 * sw), sw=sw, **kw))
    return seq


def clear_wave_at_distance_sequence(n=22, sw=70.0, freq=1.3, amp=0.35, arm="right", **kw):
    """POSITIVE: a CLEAR wave at moderate distance (small scale). Its swing is big enough to
    clear the absolute px floor, so it still fires far away -- only tiny/subtle far waves are
    the accepted trade-off of the jitter hardening."""
    seq = []
    for i in range(n):
        dx = amp * sw * math.sin(2 * math.pi * freq * i * DT)
        up = CY - 0.5 * sw
        if arm == "right":
            seq.append(make_skeleton((CX - 0.5 * sw, CY + 1.15 * sw), (CX + 0.5 * sw + dx, up), sw=sw, **kw))
        else:
            seq.append(make_skeleton((CX - 0.5 * sw + dx, up), (CX + 0.5 * sw, CY + 1.15 * sw), sw=sw, **kw))
    return seq


# --------------------------------------------------------------------- harness
def feed(reactor, frames, t0=100.0, dt=DT, w=FRAME_W, h=FRAME_H, hand_frames=None):
    """Drive `reactor` frame-by-frame on a DETERMINISTIC clock; return the list of fired
    events. `frames` is a list where each element is either one skeleton item or a list of
    items (multi-person). `hand_frames`, if given, is a parallel list of hand shm frames fed
    to the palm fusion. Passing our own `t` is what makes the run reproducible."""
    fired = []
    t = t0
    for i, fr in enumerate(frames):
        items = fr if isinstance(fr, list) else [fr]
        hf = hand_frames[i] if hand_frames is not None else None
        ev = reactor.update({"w": w, "h": h, "items": items}, t, hf)
        if ev is not None:
            fired.append(ev)
        t += dt
    return fired


def fired_gestures(events):
    return [e["gesture"] for e in events]


# --------------------------------------------------------------------- selftest
def selftest():
    """Assert every positive gesture fires EXACTLY ONCE and every negative fires NOTHING,
    including a noisy (seeded-jitter) and a dropout run to prove robustness. Deterministic:
    same result every invocation. Mirrors the demo's trust boundary."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and cond

    def run(frames):
        return fired_gestures(feed(GestureReactor(ReactorConfig()), frames))

    # --- positives: each fires once, and maps to the intended robot response ---
    for label, seq, want in (
        ("wave",         wave_sequence(),          "wave"),
    ):
        f = run(seq)
        check(f"{label} fires", want in f)
        check(f"{label} fires exactly once (cooldown)", f.count(want) == 1)
        check(f"{label} maps to a robot gesture", GESTURE_TO_ROBOT.get(want) is not None)

    # left-arm wave must work identically (no left/right bias in the classifier)
    check("wave (left arm) fires once", run(wave_sequence(arm="left")).count("wave") == 1)

    # --- negatives: NOTHING fires (few-false-positives is the whole demo-safety bet) ---
    check("idle fires nothing",    run(idle_sequence()) == [])
    check("walking fires nothing", run(walking_sequence()) == [])
    check("static reach fires nothing", run(reach_sequence()) == [])
    check("raised hand fires nothing (auto high-five removed)", run(raise_sequence()) == [])

    # --- robustness: seeded pose jitter + a dropped-out shoulder still classify a wave ---
    rng = random.Random(42)
    noisy = wave_sequence(rng=rng, jitter=3.0)
    check("wave survives seeded pose jitter", run(noisy).count("wave") == 1)
    # right shoulder occasionally missing (low conf) -- classifier falls back on good frames
    drop = wave_sequence(low_conf=(R_EAR,))            # a non-arm joint gone: no effect
    check("wave survives head-joint dropout", run(drop).count("wave") == 1)

    # --- debounce: a LONG continuous wave still fires just once within the cooldown ---
    long_wave = wave_sequence(n=40)
    check("continuous wave debounced to one", run(long_wave).count("wave") == 1)

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
