#!/usr/bin/env python3
"""sim_odometry.py -- headless proof that fused_odometry beats leg-only, WITHOUT the robot.

Mirrors the sim_perception.py / perception_bench.py culture: a deterministic (seeded)
synthetic world, the REAL module under test driven closed-loop over it, and MEASURED
numbers with a --selftest that ASSERTS the win.

WHAT IT DOES
    1. Generates a GROUND-TRUTH SE2 trajectory (forward motion + a sinusoidal yaw rate, so
       the heading matters and position curves).
    2. Synthesizes the THREE real sources against that truth:
         leg-odom  -- dead-reckoned from body velocities with a SCALE error, a YAW-RATE
                      BIAS (the drift driver -> the frame slowly rotates, position spirals),
                      and white noise. This is what OdomReader would report today.
         IMU gyro  -- true yaw-rate + a (different) constant bias + white noise, at high rate.
         lidar-odom-- true SE2 pose + noise, at a LOWER rate, with a scheduled DROPOUT
                      window (no samples) and a few injected OUTLIER jumps (bad scan match).
    3. Runs FusedOdometry over the interleaved, stamp-ordered stream.
    4. MEASURES position + yaw RMSE of FUSED vs GROUND TRUTH and of LEG-ONLY vs GROUND TRUTH,
       plus dropout-window behaviour and outlier-rejection behaviour.

Everything is a single master clock at the IMU rate; each source emits on its own sub-rate,
so ground truth, the leg baseline, and the fused estimate all advance together and logging
is a straight aligned comparison. Pure numpy/math, no ROS, no robot.

Run:  python3 scripts/sim_odometry.py                 # default scenario, human report
      python3 scripts/sim_odometry.py --json
      python3 scripts/sim_odometry.py --ascii          # ASCII overlay of the three paths
      python3 scripts/sim_odometry.py --scenario spiral_dropout
      python3 scripts/sim_odometry.py --list
      python3 scripts/sim_odometry.py --selftest       # ASSERTS fused <= 0.5x leg + graceful
"""
import argparse
import json
import math
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_odometry import FusedOdometry, wrap, yaw_rmse   # noqa: E402


# ---------------------------------------------------------------------------
# scenarios -- each defines the truth motion + the source error/rate/dropout knobs.
# ---------------------------------------------------------------------------
def scenarios():
    return {
        # the headline case: a long curving run, strong leg drift, a lidar dropout, outliers.
        "spiral_dropout": dict(
            dur=30.0, v_fwd=0.8, yaw_amp=0.35, yaw_period=12.0,
            leg_scale=1.03, leg_yaw_bias=0.030, leg_vel_noise=0.010, leg_yaw_noise=0.004,
            imu_bias=0.010, imu_noise=0.010,
            lidar_pos_noise=0.030, lidar_yaw_noise=0.009,
            dropout=(12.0, 17.0), outliers=[(8.0, 3.0, 0.0), (20.0, -2.5, 0.0), (25.0, 0.0, 2.5)],
        ),
        # straight-ish, milder drift, no dropout -- a sanity baseline.
        "gentle": dict(
            dur=20.0, v_fwd=0.7, yaw_amp=0.15, yaw_period=15.0,
            leg_scale=1.02, leg_yaw_bias=0.020, leg_vel_noise=0.008, leg_yaw_noise=0.003,
            imu_bias=0.008, imu_noise=0.008,
            lidar_pos_noise=0.025, lidar_yaw_noise=0.007,
            dropout=None, outliers=[],
        ),
        # aggressive turning + a long dropout: stresses coasting on leg+imu heading.
        "loops_long_dropout": dict(
            dur=36.0, v_fwd=0.9, yaw_amp=0.6, yaw_period=9.0,
            leg_scale=1.04, leg_yaw_bias=0.040, leg_vel_noise=0.012, leg_yaw_noise=0.005,
            imu_bias=0.015, imu_noise=0.012,
            lidar_pos_noise=0.035, lidar_yaw_noise=0.010,
            dropout=(15.0, 23.0), outliers=[(6.0, 2.5, 0.0), (30.0, 0.0, -2.0)],
        ),
    }


# ---------------------------------------------------------------------------
# rates (Hz) -- realistic for the G1: odommodestate ~50 Hz, IMU faster, FAST-LIO ~10 Hz.
# ---------------------------------------------------------------------------
IMU_HZ = 200.0
LEG_HZ = 50.0
LIDAR_HZ = 10.0


def run(scn, seed=0, cfg=None):
    """Drive FusedOdometry over one scenario. Returns a metrics dict (+ the logged paths)."""
    rng = np.random.RandomState(seed)
    dt = 1.0 / IMU_HZ
    nsteps = int(round(scn["dur"] * IMU_HZ))
    leg_every = int(round(IMU_HZ / LEG_HZ))
    lidar_every = int(round(IMU_HZ / LIDAR_HZ))

    w = 2.0 * math.pi / scn["yaw_period"]
    v = scn["v_fwd"]

    f = FusedOdometry(cfg=cfg)

    # ground truth (integrated), leg dead-reckoning (integrated with its errors)
    gx = gy = gyaw = 0.0
    lx = ly = lyaw = 0.0

    # per-sample logs at leg rate (the common comparison grid)
    log = {"t": [], "gt": [], "leg": [], "fused": [], "in_dropout": [], "lidar_healthy": []}

    dropout = scn.get("dropout")
    outliers = list(scn.get("outliers", []))
    outlier_fired = []          # (t, epos_fused_at_that_time) for reporting/asserts

    for i in range(nsteps):
        t = i * dt
        vyaw_true = scn["yaw_amp"] * math.sin(w * t)

        # --- advance GROUND TRUTH (midpoint integration) ---
        gyaw_mid = wrap(gyaw + 0.5 * vyaw_true * dt)
        gx += v * dt * math.cos(gyaw_mid)
        gy += v * dt * math.sin(gyaw_mid)
        gyaw = wrap(gyaw + vyaw_true * dt)

        # --- advance LEG dead-reckoning (scale err + yaw bias + noise) ---
        v_leg = v * scn["leg_scale"] + rng.normal(0.0, scn["leg_vel_noise"])
        vyaw_leg = vyaw_true + scn["leg_yaw_bias"] + rng.normal(0.0, scn["leg_yaw_noise"])
        lyaw_mid = wrap(lyaw + 0.5 * vyaw_leg * dt)
        lx += v_leg * dt * math.cos(lyaw_mid)
        ly += v_leg * dt * math.sin(lyaw_mid)
        lyaw = wrap(lyaw + vyaw_leg * dt)

        # --- IMU sample every tick: true yaw-rate + bias + noise ---
        gyro_z = vyaw_true + scn["imu_bias"] + rng.normal(0.0, scn["imu_noise"])
        f.add_imu(gyro_z, t)

        # --- LEG sample at LEG_HZ (feed the filter the current leg pose) ---
        if i % leg_every == 0:
            f.add_leg(lx, ly, lyaw, t)

        # --- LIDAR sample at LIDAR_HZ, unless inside the dropout window ---
        in_dropout = bool(dropout and dropout[0] <= t < dropout[1])
        if i % lidar_every == 0 and not in_dropout:
            # is there a scheduled outlier near this stamp? (fire once)
            fired = None
            for k, (ot, dx, dyaw) in enumerate(outliers):
                if abs(t - ot) < 0.5 / LIDAR_HZ:
                    fired = (k, dx, dyaw)
                    break
            if fired is not None:
                k, dx, dyaw = fired
                outliers.pop(k)
                # a corrupt scan-match: truth + a big position and/or yaw jump. Feed it and
                # record whether the filter's pose stayed put (i.e. it rejected the outlier).
                f.add_lidar(gx + dx, gy, wrap(gyaw + dyaw), t)
                fx, fy, _ = f.get_pose()
                outlier_fired.append((t, math.hypot(fx - gx, fy - gy)))
            else:
                mx = gx + rng.normal(0.0, scn["lidar_pos_noise"])
                my = gy + rng.normal(0.0, scn["lidar_pos_noise"])
                myaw = wrap(gyaw + rng.normal(0.0, scn["lidar_yaw_noise"]))
                f.add_lidar(mx, my, myaw, t)

        # --- log at leg rate (aligned three-way comparison) ---
        if i % leg_every == 0:
            fx, fy, fyaw = f.get_pose()
            log["t"].append(t)
            log["gt"].append((gx, gy, gyaw))
            log["leg"].append((lx, ly, lyaw))
            log["fused"].append((fx, fy, fyaw))
            log["in_dropout"].append(in_dropout)
            log["lidar_healthy"].append(f.lidar_healthy())

    return _metrics(scn, f, log, outlier_fired)


def _pos_rmse(a_list, b_list):
    s = 0.0
    for (ax, ay, _), (bx, by, _) in zip(a_list, b_list):
        s += (ax - bx) ** 2 + (ay - by) ** 2
    return math.sqrt(s / len(a_list)) if a_list else float("nan")


def _yaw_rmse(a_list, b_list):
    return yaw_rmse([a[2] - b[2] for a, b in zip(a_list, b_list)])


def _metrics(scn, f, log, outlier_fired):
    gt, leg, fused = log["gt"], log["leg"], log["fused"]
    fused_pos = _pos_rmse(fused, gt)
    leg_pos = _pos_rmse(leg, gt)
    fused_yaw = _yaw_rmse(fused, gt)
    leg_yaw = _yaw_rmse(leg, gt)

    # per-sample fused/leg position error, and the subset inside the dropout window.
    fused_err = [math.hypot(fx - gx, fy - gy)
                 for (fx, fy, _), (gx, gy, _) in zip(fused, gt)]
    leg_err = [math.hypot(lx_ - gx, ly_ - gy)
               for (lx_, ly_, _), (gx, gy, _) in zip(leg, gt)]
    drop_idx = [i for i, d in enumerate(log["in_dropout"]) if d]
    drop_max_fused = max((fused_err[i] for i in drop_idx), default=0.0)
    drop_max_leg = max((leg_err[i] for i in drop_idx), default=0.0)
    # lidar_healthy must flip False while coasting through the dropout, then flip back True
    # once lidar returns and is re-acquired (the honest "confidence degraded" signal).
    health = log["lidar_healthy"]
    lidar_unhealthy_in_dropout = any(not health[i] for i in drop_idx)
    lidar_healthy_after_dropout = bool(health[-1]) if health else False
    # fused error at the END of the run (drift never accumulates for fused -> should be small)
    final_fused = fused_err[-1] if fused_err else float("nan")
    final_leg = leg_err[-1] if leg_err else float("nan")

    st = f.get_state()
    n_outliers = len(scn.get("outliers", []))
    # every outlier we fired: did the fused pose stay near truth (i.e. was it rejected)?
    outlier_max_jump = max((e for (_, e) in outlier_fired), default=0.0)

    ratio = (fused_pos / leg_pos) if leg_pos > 1e-9 else float("nan")
    return {
        "scenario": scn.get("name", "?"),
        "samples": len(gt),
        "fused_pos_rmse_m": fused_pos,
        "leg_pos_rmse_m": leg_pos,
        "pos_rmse_ratio": ratio,
        "fused_yaw_rmse_rad": fused_yaw,
        "leg_yaw_rmse_rad": leg_yaw,
        "yaw_rmse_ratio": (fused_yaw / leg_yaw) if leg_yaw > 1e-9 else float("nan"),
        "final_fused_err_m": final_fused,
        "final_leg_err_m": final_leg,
        "dropout_max_fused_err_m": drop_max_fused,
        "dropout_max_leg_err_m": drop_max_leg,
        "lidar_unhealthy_in_dropout": lidar_unhealthy_in_dropout,
        "lidar_healthy_after_dropout": lidar_healthy_after_dropout,
        "n_outliers_injected": n_outliers,
        "n_lidar_reject": st["n_lidar_reject"],
        "n_lidar_accept": st["n_lidar_accept"],
        "outlier_max_fused_jump_m": outlier_max_jump,
        "est_gyro_bias": st["gyro_bias"],
        "true_imu_bias": scn.get("imu_bias", 0.0),
        "final_confidence": st["confidence"],
        "_log": log,
    }


# ---------------------------------------------------------------------------
# ASCII overlay of the three paths (truth '.', leg 'x', fused 'o')
# ---------------------------------------------------------------------------
def ascii_paths(log, w=78, h=26):
    gt, leg, fused = log["gt"], log["leg"], log["fused"]
    xs = [p[0] for p in gt + leg + fused]
    ys = [p[1] for p in gt + leg + fused]
    x0, x1 = min(xs) - 0.3, max(xs) + 0.3
    y0, y1 = min(ys) - 0.3, max(ys) + 0.3
    x1 = max(x1, x0 + 1e-3); y1 = max(y1, y0 + 1e-3)
    cl = lambda v, n: 0 if v < 0 else n - 1 if v >= n else v
    col = lambda px: cl(int((px - x0) / (x1 - x0) * (w - 1)), w)
    row = lambda py: cl(int((y1 - py) / (y1 - y0) * (h - 1)), h)
    g = [[" "] * w for _ in range(h)]
    for (px, py, _) in leg:
        g[row(py)][col(px)] = "x"           # leg drift path
    for (px, py, _) in gt:
        g[row(py)][col(px)] = "."           # ground truth
    for (px, py, _) in fused:
        g[row(py)][col(px)] = "o"           # fused
    g[row(gt[0][1])][col(gt[0][0])] = "S"
    g[row(gt[-1][1])][col(gt[-1][0])] = "T"      # truth end
    g[row(leg[-1][1])][col(leg[-1][0])] = "L"    # leg end (spiralled away)
    return "\n".join("".join(r) for r in g)


# ---------------------------------------------------------------------------
def print_report(m):
    def f(v):
        return "  n/a" if v != v else f"{v:.3f}"
    print("=" * 68)
    print(f"FUSED ODOMETRY SIM  --  scenario '{m['scenario']}'  ({m['samples']} samples)")
    print("=" * 68)
    print(f"  Position RMSE   fused {f(m['fused_pos_rmse_m'])} m   "
          f"leg-only {f(m['leg_pos_rmse_m'])} m   ratio {f(m['pos_rmse_ratio'])}")
    print(f"  Yaw RMSE        fused {f(m['fused_yaw_rmse_rad'])} rad "
          f"leg-only {f(m['leg_yaw_rmse_rad'])} rad ratio {f(m['yaw_rmse_ratio'])}")
    print(f"  Final error     fused {f(m['final_fused_err_m'])} m   leg-only {f(m['final_leg_err_m'])} m")
    print(f"  Dropout window  max fused err {f(m['dropout_max_fused_err_m'])} m   "
          f"max leg err {f(m['dropout_max_leg_err_m'])} m")
    print(f"  Outliers        injected {m['n_outliers_injected']}   lidar rejects {m['n_lidar_reject']}   "
          f"accepts {m['n_lidar_accept']}   max fused jump at outlier {f(m['outlier_max_fused_jump_m'])} m")
    print(f"  Gyro bias       estimated {f(m['est_gyro_bias'])} rad/s   (true IMU bias {f(m['true_imu_bias'])})")
    print(f"  Final confidence {f(m['final_confidence'])}")
    print("-" * 68)
    better = m["pos_rmse_ratio"]
    print(f"  => fused position RMSE is {1.0/better:.1f}x better than leg-only"
          if better == better and better > 0 else "  => n/a")
    print("=" * 68)


# ---------------------------------------------------------------------------
def selftest():
    """ASSERTS the fusion win + graceful behaviour, deterministically, over all scenarios."""
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and bool(cond)

    for name, base in scenarios().items():
        scn = dict(base); scn["name"] = name
        m = run(scn, seed=0)
        print(f"\n--- {name} ---")
        print(f"  fused_pos={m['fused_pos_rmse_m']:.3f}m leg_pos={m['leg_pos_rmse_m']:.3f}m "
              f"ratio={m['pos_rmse_ratio']:.3f}  fused_yaw={m['fused_yaw_rmse_rad']:.3f} "
              f"leg_yaw={m['leg_yaw_rmse_rad']:.3f}  bias_est={m['est_gyro_bias']:.4f} "
              f"(true {m['true_imu_bias']:.4f})")

        # --- HEADLINE DRIFT REDUCTION (the requirement) ---
        # (1) fused position RMSE <= 0.5x leg-only  (>= 2x better).
        c(f"[{name}] fused pos RMSE <= 0.5x leg-only", m["pos_rmse_ratio"] <= 0.5)
        # (2) fused yaw RMSE also <= 0.5x leg-only  (bias estimation working; heading drives drift).
        c(f"[{name}] fused yaw RMSE <= 0.5x leg-only", m["yaw_rmse_ratio"] <= 0.5)

        # --- ABSOLUTE SMALLNESS (not merely better than a bad baseline) ---
        # (3) fused position RMSE genuinely small.
        c(f"[{name}] fused pos RMSE < 0.30 m (absolute)", m["fused_pos_rmse_m"] < 0.30)
        # (4) fused yaw RMSE genuinely small.
        c(f"[{name}] fused yaw RMSE < 0.05 rad (absolute)", m["fused_yaw_rmse_rad"] < 0.05)
        # (5) no NaN / explosion anywhere.
        c(f"[{name}] fused finite & bounded", math.isfinite(m["fused_pos_rmse_m"])
          and m["final_fused_err_m"] < 1.0)

        # --- BIAS ESTIMATION (makes dropout coasting accurate) ---
        # (6) learned gyro bias converges to the TRUE injected IMU bias. With the gyro weighted
        #     high the estimator isolates the IMU bias (the leg's own yaw bias barely leaks in).
        c(f"[{name}] gyro bias estimate within 0.01 rad/s of truth",
          abs(m["est_gyro_bias"] - m["true_imu_bias"]) < 0.01)

        # --- GRACEFUL DROPOUT COASTING (only where a dropout exists) ---
        if base.get("dropout"):
            # (7) coasting on leg+imu stays bounded (no divergence).
            c(f"[{name}] dropout: fused coast bounded (< 0.50 m)", m["dropout_max_fused_err_m"] < 0.50)
            # (8) fused still beats leg during the coast.
            c(f"[{name}] dropout: fused still beats leg", m["dropout_max_fused_err_m"] < m["dropout_max_leg_err_m"])
            # (9) lidar_healthy flips False during the window, then True again after re-acquire.
            c(f"[{name}] dropout: lidar_healthy flipped False while coasting", m["lidar_unhealthy_in_dropout"])
            c(f"[{name}] dropout: lidar_healthy recovered True after", m["lidar_healthy_after_dropout"])

        # --- OUTLIER REJECTION (only where outliers are injected) ---
        if base.get("outliers"):
            # (10) EVERY injected teleport is gated -- exact count, no leaks, no false rejects.
            c(f"[{name}] outliers rejected == injected", m["n_lidar_reject"] == m["n_outliers_injected"])
            # (11) no teleport leaked into the output pose (fused stayed put at the outlier stamp).
            c(f"[{name}] fused never jumped to an outlier (< 0.10 m)", m["outlier_max_fused_jump_m"] < 0.10)

        # --- DETERMINISM: same seed -> byte-identical metrics dict (excluding the raw log) ---
        m2 = run(scn, seed=0)
        keys = [k for k in m if k != "_log"]
        c(f"[{name}] deterministic (same-seed metrics identical)",
          all(m[k] == m2[k] for k in keys))

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="spiral_dropout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ascii", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    scns = scenarios()
    if args.list:
        for k in scns:
            print(k)
        return 0

    names = list(scns) if args.all else [args.scenario]
    out = []
    for name in names:
        scn = dict(scns[name]); scn["name"] = name
        out.append(run(scn, seed=args.seed))

    if args.json:
        print(json.dumps([{k: v for k, v in m.items() if k != "_log"} for m in out], indent=2))
    else:
        for m in out:
            print_report(m)
            if args.ascii:
                print("  legend: '.'=truth  'x'=leg-only  'o'=fused   S=start T=truth-end L=leg-end")
                print(ascii_paths(m["_log"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
