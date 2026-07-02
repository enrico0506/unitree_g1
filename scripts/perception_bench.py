#!/usr/bin/env python3
"""perception_bench.py -- the measurement foundation (INSTITUTIONAL_ROADMAP.md Phase 0).

Turns the pipeline's qualitative "seems better" into MEASURED numbers. Drives the REAL
obstacle_node math (via perception_harness's rclpy-stubbed node) over the sim scenarios,
comparing the perceived 72-sector ring against a per-sector GROUND-TRUTH ring rasterized
from the sim obstacle footprints. Emits:

  * per-sector confusion matrix -> Precision / Recall / F1, overall and stratified by range
  * distance RMSE on true positives
  * forward-detection latency (frames from first-true to first-detected)
  * false-stop rate (false-positive sectors)
  * a coverage-gap statement (hazard space the geometry cannot monitor -- not hidden)

Entirely OFFLINE, pure-numpy, zero GPU, cannot contend with the pose service. Deterministic
via the sim's seed. Run:  python3 scripts/perception_bench.py [--json] [--selftest]

Honest caveat (printed in every report): the ray-cast sim is idealized (flat z=0 floor, no
specular holes, no real deskew noise) -- sim numbers are necessary-but-optimistic; real-log
validation is the ceiling for "institutional".
"""
import os, sys, math, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import sim_perception as sp   # brings make_node, detect, livox_cloud, obstacles_at, scenarios, DT...

RANGE_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
FRONT_CONE_DEG = 20.0        # |bearing| < this = "forward" for the latency metric


def true_ring(x, y, theta, obs, n, start_deg, bin_deg, max_range, min_h):
    """Ground-truth per-sector nearest horizontal distance, by densely sampling each obstacle
    footprint boundary and binning into the sensor-frame polar grid. Height-agnostic beyond a
    detectability floor (min_h): any obstacle taller than the ground clearance is a hazard the
    ring is expected to report. Returns a list[n] of float or None."""
    out = [None] * n
    ct, st = math.cos(theta), math.sin(theta)

    def add(wx, wy):
        dx, dy = wx - x, wy - y
        rx = ct * dx + st * dy          # world -> sensor frame (same convention as livox_cloud)
        ry = -st * dx + ct * dy
        d = math.hypot(rx, ry)
        if d > max_range:
            return
        ang = math.degrees(math.atan2(ry, rx))
        s = int(math.floor((ang - start_deg) / bin_deg)) % n
        if out[s] is None or d < out[s]:
            out[s] = d

    for o in obs:
        if o.get("h", 1.2) < min_h:
            continue
        if o["type"] == "circle":
            cx, cy, r = o["cx"], o["cy"], o["r"]
            for k in range(120):
                a = 2.0 * math.pi * k / 120.0
                add(cx + r * math.cos(a), cy + r * math.sin(a))
        else:
            ax, ay, bx, by = o["ax"], o["ay"], o["bx"], o["by"]
            for k in range(201):
                u = k / 200.0
                add(ax + u * (bx - ax), ay + u * (by - ay))
    return out


def _bin_idx(d):
    for i, (lo, hi) in enumerate(RANGE_BINS):
        if lo <= d < hi:
            return i
    return None


def run_bench(noise=0.02, sway_amp=2.0, seed=0, scns=None):
    """Sweep every scenario open-loop (robot drives forward so obstacles pass far->near),
    accumulating a per-sector TP/FP/FN ledger vs the GT ring. Returns a KPI dict."""
    scns = scns if scns is not None else sp.scenarios()
    per_scn = {}

    def _rate(a, b):
        return (a / b) if b else float("nan")
    # confusion, overall + per range bin
    TP = FP = FN = TN = 0
    tp_by = [0] * len(RANGE_BINS); fn_by = [0] * len(RANGE_BINS)
    sq_err = []; sq_err_by = [[] for _ in RANGE_BINS]
    latencies = []; misses = 0
    fp_per_scn = []
    frames = 0

    for name, scn in scns.items():
        node = sp.make_node()
        horizon = float(node.range_m)
        min_h = float(node.ground_clear)
        rng = np.random.RandomState(seed)
        x, y, theta = scn["start"]
        nsteps = int(scn["dur"] / sp.DT)
        first_true = None; first_det = None; scn_fp = 0; scn_tp = 0; scn_fn = 0
        for step in range(nsteps):
            t = step * sp.DT
            obs = sp.obstacles_at(scn["obstacles"], t)
            pitch = math.radians(sway_amp) * math.sin(2 * math.pi * 1.2 * t)
            roll = math.radians(sway_amp) * math.sin(2 * math.pi * 1.2 * t + 1.3)
            cloud = sp.livox_cloud(x, y, theta, obs, node.sensor_h, noise=noise,
                                   rng=rng, pitch=pitch, roll=roll)
            ring = sp.detect(node, cloud)["ring"]
            gt = true_ring(x, y, theta, obs, node.ring_n, node.ring_start_deg,
                           node.ring_bin_deg, horizon, min_h)
            frames += 1

            # occupied sets within range; match with +/-1 sector tolerance so a 5 deg
            # discretization/noise boundary shift is not scored as both an FP and an FN.
            N = node.ring_n
            gt_occ = {s: gt[s] for s in range(N) if gt[s] is not None and gt[s] <= horizon}
            pc_occ = {s: ring[s] for s in range(N) if ring[s] is not None and ring[s] <= horizon}
            near_true = min(gt_occ.values(), default=None)
            TOL = 1

            def _near(s, keys):
                return any(((s + d) % N) in keys for d in range(-TOL, TOL + 1))

            # recall side: each GT-occupied sector matched to a perceived sector within tol
            for s, tv in gt_occ.items():
                b = _bin_idx(tv)
                if _near(s, pc_occ):
                    TP += 1; scn_tp += 1
                    if b is not None:
                        tp_by[b] += 1
                        pv = min((pc_occ[(s + d) % N] for d in range(-TOL, TOL + 1)
                                  if ((s + d) % N) in pc_occ), key=lambda v: abs(v - tv))
                        e = (pv - tv) ** 2; sq_err.append(e); sq_err_by[b].append(e)
                else:
                    FN += 1; scn_fn += 1
                    if b is not None:
                        fn_by[b] += 1
            # precision side: a perceived sector with NO GT within tol is a false stop
            for s in pc_occ:
                if not _near(s, gt_occ):
                    FP += 1; scn_fp += 1
            TN += N - len(gt_occ) - sum(1 for s in pc_occ if not _near(s, gt_occ))

            # forward-cone detection latency (first hazard in front -> first detected in front)
            cone = int(FRONT_CONE_DEG / node.ring_bin_deg)
            fwd_sectors = [(-i) % node.ring_n for i in range(cone + 1)] + \
                          [i % node.ring_n for i in range(cone + 1)]
            ft = any(gt[s] is not None and gt[s] <= horizon for s in fwd_sectors)
            fp_ = any(ring[s] is not None and ring[s] <= horizon for s in fwd_sectors)
            if ft and first_true is None:
                first_true = step
            if ft and fp_ and first_det is None and first_true is not None:
                first_det = step

            # advance open-loop; stop before driving THROUGH the nearest obstacle
            if near_true is not None and near_true < 0.4:
                break
            vx = sp.MAXES[0]
            x += vx * math.cos(theta) * sp.DT
            y += vx * math.sin(theta) * sp.DT

        fp_per_scn.append(scn_fp)
        per_scn[name] = {"recall": _rate(scn_tp, scn_tp + scn_fn), "tp": scn_tp,
                         "fn": scn_fn, "fp": scn_fp}
        if first_true is not None:
            if first_det is not None:
                latencies.append((first_det - first_true) * sp.DT)
            else:
                misses += 1

    prec = _rate(TP, TP + FP); rec = _rate(TP, TP + FN)
    f1 = _rate(2 * prec * rec, prec + rec) if (prec == prec and rec == rec and prec + rec) else float("nan")
    return {
        "frames": frames, "scenarios": len(scns),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": prec, "recall": rec, "f1": f1,
        "recall_by_range": [_rate(tp_by[i], tp_by[i] + fn_by[i]) for i in range(len(RANGE_BINS))],
        "dist_rmse_m": (math.sqrt(np.mean(sq_err)) if sq_err else float("nan")),
        "dist_rmse_by_range": [(math.sqrt(np.mean(sq_err_by[i])) if sq_err_by[i] else float("nan"))
                               for i in range(len(RANGE_BINS))],
        "latency_median_s": (float(np.median(latencies)) if latencies else float("nan")),
        "latency_max_s": (float(np.max(latencies)) if latencies else float("nan")),
        "front_misses": misses,
        "false_stop_sectors_per_scenario": float(np.mean(fp_per_scn)) if fp_per_scn else 0.0,
        "per_scenario": per_scn,
    }


def small_scenarios():
    """Fine-detection stress set: thin poles at range + a small pole. These are the
    body-height objects the Mid-360 CAN see; cables ON THE FLOOR are depth-only (the sim
    models only the lidar, so they are exercised by obstacle/test_depth_ring.py instead)."""
    C = lambda cx, cy, r, **k: dict(type="circle", cx=cx, cy=cy, r=r, **k)
    fwd = lambda: dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=14.0)
    return {
        "thin_pole_1m":  dict(**fwd(), obstacles=[C(3.0, 0.0, 0.02, h=1.2)]),   # 4 cm dia
        "thin_pole_2m":  dict(**fwd(), obstacles=[C(4.0, 0.0, 0.02, h=1.2)]),
        "wire_pole_3m":  dict(**fwd(), obstacles=[C(5.0, 0.0, 0.015, h=1.2)]),  # 3 cm dia wire
        "small_pole_3m": dict(**fwd(), obstacles=[C(5.0, 0.0, 0.05, h=1.2)]),   # 10 cm dia
        "thin_offaxis":  dict(**fwd(), obstacles=[C(3.5, 0.8, 0.02, h=1.2)]),
    }


def print_report(k):
    def pct(v):
        return "  n/a" if v != v else f"{100 * v:5.1f}%"
    def m(v):
        return " n/a" if v != v else f"{v:.3f}"
    rb = "  ".join(f"{lo:.0f}-{hi:.0f}m {pct(k['recall_by_range'][i])}"
                   for i, (lo, hi) in enumerate(RANGE_BINS))
    db = "  ".join(f"{lo:.0f}-{hi:.0f}m {m(k['dist_rmse_by_range'][i])}"
                   for i, (lo, hi) in enumerate(RANGE_BINS))
    print("=" * 66)
    print(f"PERCEPTION BENCH  ({k['frames']} frames, {k['scenarios']} scenarios)")
    print("=" * 66)
    print(f"  Precision {pct(k['precision'])}   Recall {pct(k['recall'])}   F1 {pct(k['f1'])}")
    print(f"  Confusion  TP={k['TP']}  FP={k['FP']}  FN={k['FN']}  TN={k['TN']}")
    print(f"  Recall by range:     {rb}")
    print(f"  Dist RMSE (TP): {m(k['dist_rmse_m'])} m   by range: {db}")
    print(f"  Front detection latency: median {m(k['latency_median_s'])} s  "
          f"max {m(k['latency_max_s'])} s   misses {k['front_misses']}")
    print(f"  False-stop sectors / scenario: {k['false_stop_sectors_per_scenario']:.2f}")
    if k.get("per_scenario"):
        print("-" * 66)
        print("  Per-scenario recall (TP/FN, FP):")
        for nm, s in k["per_scenario"].items():
            print(f"    {nm:16s} {pct(s['recall'])}   (TP {s['tp']}  FN {s['fn']}  FP {s['fp']})")
    print("-" * 66)
    print("  COVERAGE GAPS (geometry cannot monitor -- NOT scored, not laundered):")
    print("    - low objects to the SIDES / REAR (depth forward-only, lidar near-floor-blind)")
    print("    - NEGATIVE obstacles: drop-offs / descending stairs (zero detection)")
    print("    - at-feet inside ~0.45 m (self-mask); sub-~2 cm debris (physics floor)")
    print("  CAVEAT: idealized ray-cast sim -> numbers are necessary-but-optimistic;")
    print("          real-log validation is the ceiling for 'institutional'.")
    print("=" * 66)


def selftest():
    k = run_bench()
    assert k["frames"] > 200, "too few frames"
    assert k["TP"] > 0 and k["recall"] == k["recall"], "no true positives / recall nan"
    # easy near/mid obstacles must be recalled well; wiring sanity, not a tuned target
    assert k["recall"] > 0.5, f"overall recall implausibly low: {k['recall']:.2f}"
    assert k["recall_by_range"][0] > 0.6, f"near recall low: {k['recall_by_range'][0]:.2f}"
    assert k["dist_rmse_m"] < 0.6, f"distance RMSE too high: {k['dist_rmse_m']:.2f}"
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--small", action="store_true", help="fine-detection stress set (thin poles)")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sway", type=float, default=2.0)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        scns = small_scenarios() if args.small else None
        k = run_bench(noise=args.noise, sway_amp=args.sway, scns=scns)
        if args.json:
            print(json.dumps(k, indent=2))
        else:
            print_report(k)
