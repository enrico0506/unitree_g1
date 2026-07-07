#!/usr/bin/env python3
"""Live wave-detector diagnostics -- READ-ONLY (no robot, no greeting, no arm motion).

Run this while you WAVE, and again while you HOLD a hand still, so we can tune the thresholds
to your REAL signal instead of guessing. It reads the same pose feed the reactor uses, and for
the target person prints the wave metrics each update plus -- if no wave would fire -- WHICH
gate blocked it (too small a sweep / not enough back-and-forth / too vertical / not raised ...).

    python3 scripts/wave_debug.py         # then wave at the head camera (pose container must be up)

Paste a few seconds of output for a wave AND for a held hand; that tells us exactly what to set.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gesture_reactor as gr

POSE_TRACKS = os.environ.get("POSE_TRACKS", "/dev/shm/g1_pose_tracks.json")
POSE_DEMAND = os.environ.get("POSE_DEMAND", "/dev/shm/g1_pose_demand")

cfg = gr.ReactorConfig()
reactor = gr.GestureReactor(cfg)


def explain(m):
    """One-line diagnosis of the best arm's metrics: the numbers + would-it-fire + why-not."""
    if m is None:
        return "no metrics -- wrist AND (elbow or shoulder) not both confident this window"
    sm = max(cfg.wave_swing_frac * m["scale"], cfg.wave_swing_min_px)
    rate = (m["rev"] / m["span"]) if m["span"] > 0 else 0.0
    blocked = []
    if m["n"] < cfg.wave_min_frames:
        blocked.append(f"n {m['n']}<{cfg.wave_min_frames}")
    if m["span"] < cfg.wave_min_span_s:
        blocked.append(f"span {m['span']:.2f}<{cfg.wave_min_span_s}")
    if m["up_frac"] < cfg.wave_up_frac:
        blocked.append(f"up_frac {m['up_frac']:.2f}<{cfg.wave_up_frac} (not raised above elbow)")
    if m["rev"] < cfg.wave_min_reversals_lo:
        blocked.append(f"rev {m['rev']}<{cfg.wave_min_reversals_lo} (not enough back-and-forth)")
    elif m["xswing"] < cfg.wave_horiz_dom * max(m["yswing"], 1e-6):
        blocked.append(f"xswing {m['xswing']:.0f}<{cfg.wave_horiz_dom}xyswing {m['yswing']:.0f} (too vertical/circular)")
    elif m["span"] > 0 and rate > cfg.wave_max_rev_rate:
        blocked.append(f"rate {rate:.1f}/s>{cfg.wave_max_rev_rate} (too fast)")
    elif m["amp"] < cfg.wave_amp_frac_lo:
        blocked.append(f"amp {m['amp']:.2f}<{cfg.wave_amp_frac_lo} (sweep too small; swing_min={sm:.0f}px)")
    if not blocked:
        tier = "STRONG->fires alone" if (m["amp"] >= cfg.wave_amp_frac and m["rev"] >= cfg.wave_min_reversals) \
            else "WEAK->needs open palm"
        verdict = f"WOULD FIRE ({tier})"
    else:
        verdict = "blocked: " + "; ".join(blocked)
    return (f"up={m['up_frac']:.2f} rev={m['rev']} amp={m['amp']:.2f} "
            f"xswing={m['xswing']:.0f}px yswing={m['yswing']:.0f}px scale={m['scale']:.0f}px "
            f"rate={rate:.1f}/s (swing_min={sm:.0f}px) -> {verdict}")


def main():
    last_mtime = None
    print("watching pose feed (Ctrl-C to stop). WAVE now; then try HOLDING a hand still.\n"
          "swing_min is the px a sweep must exceed to count; amp is normalized to body scale.\n",
          flush=True)
    while True:
        try:
            with open(POSE_DEMAND, "wb") as f:
                f.write(b"1")               # keep the demand-gated pose container inferring
        except OSError:
            pass
        try:
            mt = os.path.getmtime(POSE_TRACKS)
        except OSError:
            time.sleep(0.15)
            continue
        if mt == last_mtime:
            time.sleep(0.05)
            continue
        last_mtime = mt
        try:
            with open(POSE_TRACKS) as f:
                frame = json.load(f)
        except (OSError, ValueError):
            continue
        items = frame.get("items", []) or []
        t = time.time()
        reactor._ingest(items, frame.get("w"), frame.get("h"), t)
        target, best = None, float("-inf")
        for it in items:
            if it.get("id") is None:
                continue
            s = gr._target_score(it, frame.get("w"), frame.get("h"))
            if s > best:
                best, target = s, it
        if target is None:
            continue
        dq = reactor._hist.get(target["id"])
        if not dq:
            continue
        hist = list(dq)
        la = gr._arm_metrics(hist, gr.L_WRI, gr.L_SHO, gr.L_ELB, cfg)
        ra = gr._arm_metrics(hist, gr.R_WRI, gr.R_SHO, gr.R_ELB, cfg)
        cands = [x for x in (la, ra) if x is not None]
        m = max(cands, key=lambda z: z["up_frac"]) if cands else None
        print(f"[{time.strftime('%H:%M:%S')}] id={target['id']} {explain(m)}", flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
