#!/usr/bin/env python3
"""Unit tests for scripts/fused_odometry.py -- the SE2 complementary fusion filter.

Pure numpy/math, no ROS, no robot. Runs under pytest:
    pytest scripts/test_fused_odometry.py -q
and also standalone (prints PASS/FAIL, exits non-zero on failure):
    python3 scripts/test_fused_odometry.py

Covers the filter math (straight / rotation), yaw wraparound, the no-lidar fallback
(== leg-only exactly), lidar correction, dropout coasting, outlier rejection + re-acquire,
gyro-bias-aided coasting, and the confidence proxy.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_odometry import (   # noqa: E402
    FusedOdometry, wrap, yaw_rmse, quat_to_yaw, DEFAULTS,
)


# --------------------------------------------------------------------------- #
# helpers: synthesize a self-consistent leg dead-reckoning stream
# --------------------------------------------------------------------------- #
def integrate_leg(v_fwd, vyaw, dur, hz=50.0, x0=0.0, y0=0.0, yaw0=0.0):
    """Integrate a leg pose from constant (v_fwd, vyaw) at `hz`. Yields (x, y, yaw, t)."""
    dt = 1.0 / hz
    x, y, yaw = x0, y0, yaw0
    n = int(round(dur * hz))
    out = []
    for i in range(n):
        t = i * dt
        out.append((x, y, yaw, t))
        yaw_mid = wrap(yaw + 0.5 * vyaw * dt)
        x += v_fwd * dt * math.cos(yaw_mid)
        y += v_fwd * dt * math.sin(yaw_mid)
        yaw = wrap(yaw + vyaw * dt)
    return out


# --------------------------------------------------------------------------- #
# wrap()
# --------------------------------------------------------------------------- #
def test_wrap_basic():
    assert abs(wrap(0.0)) < 1e-12
    assert abs(wrap(math.pi) - math.pi) < 1e-12          # +pi maps to +pi
    assert abs(wrap(-math.pi) - math.pi) < 1e-12         # -pi maps to +pi (half-open on the low side)
    assert abs(wrap(3.0 * math.pi) - math.pi) < 1e-9
    assert abs(wrap(2.0 * math.pi)) < 1e-9
    # a +179 -> -179 deg step is a small (~2 deg) wrapped error, not ~358 deg
    e = wrap(math.radians(-179.0) - math.radians(179.0))
    assert abs(e - math.radians(2.0)) < 1e-6


def test_yaw_rmse_wraps():
    # errors straddling the seam average correctly once wrapped
    errs = [math.radians(179.0) - math.radians(-179.0)] * 4   # each really +2 deg
    assert abs(yaw_rmse(errs) - math.radians(2.0)) < 1e-6
    assert math.isnan(yaw_rmse([]))


# --------------------------------------------------------------------------- #
# basic filter math (no lidar): straight line and pure rotation
# --------------------------------------------------------------------------- #
def test_straight_line_no_sources_but_leg():
    f = FusedOdometry()
    for (x, y, yaw, t) in integrate_leg(0.5, 0.0, 4.0):
        f.add_leg(x, y, yaw, t)
    fx, fy, fyaw = f.get_pose()
    assert abs(fx - 0.5 * (4.0 - 1.0 / 50.0)) < 1e-6   # last sample stamp is dur - dt
    assert abs(fy) < 1e-9
    assert abs(fyaw) < 1e-9


def test_pure_rotation_leg_only():
    f = FusedOdometry()
    for (x, y, yaw, t) in integrate_leg(0.0, 0.5, 4.0):   # spin in place
        f.add_leg(x, y, yaw, t)
    fx, fy, fyaw = f.get_pose()
    assert abs(fx) < 1e-9 and abs(fy) < 1e-9              # no translation
    # yaw advanced ~0.5 rad/s for (4 - dt) s
    assert abs(wrap(fyaw - 0.5 * (4.0 - 1.0 / 50.0))) < 1e-6


# --------------------------------------------------------------------------- #
# no-lidar fallback == leg-only EXACTLY (graceful degradation)
# --------------------------------------------------------------------------- #
def test_no_lidar_no_imu_fallback_equals_leg():
    """With neither lidar nor IMU, the fused pose must reproduce leg odometry exactly
    (it integrates the leg's own body deltas at the leg's own heading)."""
    f = FusedOdometry()
    leg = integrate_leg(0.7, 0.2, 6.0)
    for (x, y, yaw, t) in leg:
        f.add_leg(x, y, yaw, t)
    fx, fy, fyaw = f.get_pose()
    lx, ly, lyaw, _ = leg[-1]
    assert abs(fx - lx) < 1e-9
    assert abs(fy - ly) < 1e-9
    assert abs(wrap(fyaw - lyaw)) < 1e-9


def test_no_lidar_with_imu_close_to_leg():
    """With IMU (unbiased) but no lidar, the fused pose stays very close to leg-only
    (the gyro blend only smooths the heading; it does not inject drift of its own)."""
    f = FusedOdometry()
    leg = integrate_leg(0.7, 0.2, 6.0)
    for (x, y, yaw, t) in leg:
        f.add_imu(0.2, t)                 # gyro == true rate, no bias
        f.add_leg(x, y, yaw, t)
    fx, fy, fyaw = f.get_pose()
    lx, ly, lyaw, _ = leg[-1]
    assert math.hypot(fx - lx, fy - ly) < 0.05
    assert abs(wrap(fyaw - lyaw)) < 0.02


# --------------------------------------------------------------------------- #
# lidar correction pulls drift out
# --------------------------------------------------------------------------- #
def test_lidar_corrects_drifting_heading():
    """Leg heading drifts (yaw-rate bias); lidar reports truth. Fused must track truth."""
    f = FusedOdometry()
    dt = 1.0 / 50.0
    n = 300
    # truth: straight +x at 0.5 m/s, yaw 0. leg: same forward speed but +0.4 rad/s bias.
    lx = ly = lyaw = 0.0
    for i in range(n):
        t = i * dt
        lyaw_mid = wrap(lyaw + 0.5 * 0.4 * dt)
        lx += 0.5 * dt * math.cos(lyaw_mid)
        ly += 0.5 * dt * math.sin(lyaw_mid)
        lyaw = wrap(lyaw + 0.4 * dt)
        f.add_imu(0.0, t)                 # gyro agrees with truth (no rotation)
        f.add_leg(lx, ly, lyaw, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)     # truth
    fx, fy, fyaw = f.get_pose()
    tx = 0.5 * (n - 1) * dt
    assert abs(fx - tx) < 0.05
    assert abs(fy) < 0.05
    assert abs(fyaw) < 0.03
    # and the leg baseline really did drift badly (so the correction is meaningful)
    assert math.hypot(lx - tx, ly - 0.0) > 1.0


def test_lidar_bootstraps_pose():
    """If lidar arrives before any leg sample it seeds the pose."""
    f = FusedOdometry()
    f.add_lidar(3.0, -2.0, 1.0, 0.0)
    x, y, yaw = f.get_pose()
    assert (abs(x - 3.0) < 1e-9 and abs(y + 2.0) < 1e-9 and abs(yaw - 1.0) < 1e-9)
    assert f.get_state()["have_pose"] is True


# --------------------------------------------------------------------------- #
# dropout coasting
# --------------------------------------------------------------------------- #
def test_dropout_coasts_without_explosion():
    """During a lidar dropout the fused pose keeps integrating leg+imu and stays bounded;
    lidar_healthy flips false; confidence decays; then recovers when lidar returns."""
    f = FusedOdometry()
    dt = 1.0 / 50.0
    n = 600
    lx = ly = lyaw = 0.0
    healthy_flipped_false = False
    max_err_during_dropout = 0.0
    for i in range(n):
        t = i * dt
        # truth straight +x @0.5; leg drifts +0.3 rad/s.
        gx = 0.5 * t
        lyaw_mid = wrap(lyaw + 0.5 * 0.3 * dt)
        lx += 0.5 * dt * math.cos(lyaw_mid)
        ly += 0.5 * dt * math.sin(lyaw_mid)
        lyaw = wrap(lyaw + 0.3 * dt)
        f.add_imu(0.0, t)
        f.add_leg(lx, ly, lyaw, t)
        in_dropout = 4.0 <= t < 8.0            # 4 s dropout
        if i % 5 == 0 and not in_dropout:
            f.add_lidar(gx, 0.0, 0.0, t)
        if in_dropout:
            if not f.lidar_healthy():
                healthy_flipped_false = True
            fx, fy, _ = f.get_pose()
            max_err_during_dropout = max(max_err_during_dropout, math.hypot(fx - gx, fy))
    # coasting stayed bounded and finite (heading was corrected before dropout began)
    assert math.isfinite(max_err_during_dropout)
    assert max_err_during_dropout < 0.6
    assert healthy_flipped_false is True
    # after lidar returns, it re-converges
    fx, fy, _ = f.get_pose()
    assert abs(fx - 0.5 * (n - 1) * dt) < 0.1 and abs(fy) < 0.1
    assert f.lidar_healthy() is True


def test_confidence_decays_then_recovers():
    f = FusedOdometry()
    dt = 1.0 / 50.0
    # warm up with lidar -> high confidence
    for i in range(100):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    conf_healthy = f.get_state()["confidence"]
    # now coast (no lidar) for a while -> confidence must drop
    for i in range(100, 400):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
    conf_coast = f.get_state()["confidence"]
    assert conf_coast < conf_healthy
    # lidar returns -> confidence recovers
    for i in range(400, 460):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    assert f.get_state()["confidence"] > conf_coast


# --------------------------------------------------------------------------- #
# outlier rejection + re-acquire
# --------------------------------------------------------------------------- #
def test_single_outlier_rejected():
    """A lone huge lidar jump is rejected and does not move the fused pose."""
    f = FusedOdometry()
    dt = 1.0 / 50.0
    for i in range(200):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    x0, y0, _ = f.get_pose()
    rej0 = f.get_state()["n_lidar_reject"]
    # inject one 5 m outlier at the next lidar stamp
    t = 200 * dt
    f.add_leg(0.5 * t, 0.0, 0.0, t)
    accepted = f.add_lidar(0.5 * t + 5.0, 0.0, 0.0, t)
    x1, y1, _ = f.get_pose()
    assert accepted is False
    assert f.get_state()["n_lidar_reject"] == rej0 + 1
    assert math.hypot(x1 - (x0 + 0.5 * dt), y1 - y0) < 0.05   # pose barely moved (just the leg step)


def test_yaw_outlier_rejected():
    f = FusedOdometry()
    dt = 1.0 / 50.0
    for i in range(200):
        t = i * dt
        f.add_leg(0.0, 0.0, 0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.0, 0.0, 0.0, t)
    _, _, yaw0 = f.get_pose()
    t = 200 * dt
    accepted = f.add_lidar(0.0, 0.0, 2.5, t)   # 2.5 rad yaw jump
    _, _, yaw1 = f.get_pose()
    assert accepted is False
    assert abs(wrap(yaw1 - yaw0)) < 0.05


def test_persistent_disagreement_reacquires():
    """A SUSTAINED new location (FAST-LIO relocalized / kidnapped) is rejected a few times
    then force-accepted so the filter cannot latch onto a stale pose forever."""
    f = FusedOdometry()
    dt = 1.0 / 50.0
    for i in range(100):
        t = i * dt
        f.add_leg(0.0, 0.0, 0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.0, 0.0, 0.0, t)
    # now lidar consistently reports a pose 5 m away (a real relocalization)
    reacquired = False
    for j in range(1, DEFAULTS["max_reject"] + 3):
        t = (100 + j) * dt
        f.add_leg(0.0, 0.0, 0.0, t)
        acc = f.add_lidar(5.0, 5.0, 0.0, t)
        if acc:
            reacquired = True
            break
    assert reacquired is True
    x, y, _ = f.get_pose()
    assert math.hypot(x - 5.0, y - 5.0) < 0.2       # snapped to the new location


# --------------------------------------------------------------------------- #
# gyro-bias estimation aids coasting
# --------------------------------------------------------------------------- #
def test_gyro_bias_reduces_coast_error():
    """A biased gyro during a dropout: with bias estimation ON (the module default) the
    coast error is smaller than with bias estimation disabled (gain 0)."""
    dt = 1.0 / 50.0
    n = 500
    drop = (5.0, 9.0)

    def run(bias_gain):
        f = FusedOdometry(cfg={"gyro_bias_gain": bias_gain})
        lx = ly = lyaw = 0.0
        worst = 0.0
        for i in range(n):
            t = i * dt
            gx = 0.5 * t
            # leg drifts +0.25 rad/s; gyro has a +0.15 rad/s bias (so coasting on gyro
            # alone would rotate away unless the bias is learned before the dropout).
            lyaw_mid = wrap(lyaw + 0.5 * 0.25 * dt)
            lx += 0.5 * dt * math.cos(lyaw_mid)
            ly += 0.5 * dt * math.sin(lyaw_mid)
            lyaw = wrap(lyaw + 0.25 * dt)
            f.add_imu(0.15, t)
            f.add_leg(lx, ly, lyaw, t)
            in_drop = drop[0] <= t < drop[1]
            if i % 5 == 0 and not in_drop:
                f.add_lidar(gx, 0.0, 0.0, t)
            if in_drop:
                fx, fy, _ = f.get_pose()
                worst = max(worst, math.hypot(fx - gx, fy))
        return worst

    worst_on = run(DEFAULTS["gyro_bias_gain"])
    worst_off = run(0.0)
    assert math.isfinite(worst_on) and math.isfinite(worst_off)
    assert worst_on <= worst_off + 1e-9          # bias estimation never hurts, usually helps


# --------------------------------------------------------------------------- #
# misc API
# --------------------------------------------------------------------------- #
def test_quat_to_yaw():
    # yaw of 90 deg about z: q = (0,0,sin(45),cos(45))
    y = quat_to_yaw(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert abs(wrap(y - math.pi / 2)) < 1e-9
    # identity quaternion -> 0
    assert abs(quat_to_yaw(0.0, 0.0, 0.0, 1.0)) < 1e-12


def test_reset_and_out_of_order():
    f = FusedOdometry()
    f.add_leg(1.0, 2.0, 0.5, 1.0)
    f.reset(0.0, 0.0, 0.0)
    assert f.get_pose() == (0.0, 0.0, 0.0)
    assert f.get_state()["have_pose"] is False
    # out-of-order leg stamp must not integrate a negative dt into a blow-up
    f.add_leg(0.0, 0.0, 0.0, 5.0)
    f.add_leg(1.0, 0.0, 0.0, 4.0)     # earlier stamp: ignored for integration
    x, y, yaw = f.get_pose()
    assert math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)


# --------------------------------------------------------------------------- #
# non-finite input is dropped, never fused (guards the outlier gate)
# --------------------------------------------------------------------------- #
def test_nonfinite_inputs_dropped_and_recover():
    """NaN/inf on any add_*() must be DROPPED as absent, never poison the pose. A NaN
    lidar frame would otherwise sail through the outlier gate -- every comparison with
    NaN is False, so `epos > gate` is False and the frame is scored 'accepted' -- writing
    NaN into the state, from which the filter could never recover even after good data
    resumes. Assert: NaN/inf lidar frames are rejected (return False) without moving the
    pose and without being charged as outlier rejects; NaN/inf leg & imu samples are
    silently dropped; the pose stays finite throughout; and good data fully recovers."""
    f = FusedOdometry()
    dt = 1.0 / 50.0
    # warm up on truth (straight +x @0.5 m/s) so we hold a good, finite, corrected pose.
    for i in range(100):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
        f.add_imu(0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    x0, y0, yaw0 = f.get_pose()
    rej0 = f.get_state()["n_lidar_reject"]
    assert all(math.isfinite(v) for v in (x0, y0, yaw0))

    # NaN / inf lidar frames: each must be REJECTED (False), leave the pose untouched,
    # and NOT be counted as an outlier reject (a NaN is a dropped sample, not a disagreement).
    t = 100 * dt
    assert f.add_lidar(float("nan"), 0.0, 0.0, t) is False
    assert f.add_lidar(0.0, float("inf"), 0.0, t) is False
    assert f.add_lidar(0.0, 0.0, float("nan"), t) is False
    assert f.add_lidar(0.0, 0.0, 0.0, float("nan")) is False
    x1, y1, yaw1 = f.get_pose()
    assert (x1, y1, yaw1) == (x0, y0, yaw0)                  # pose did not move at all
    assert all(math.isfinite(v) for v in (x1, y1, yaw1))
    assert f.get_state()["n_lidar_reject"] == rej0          # not charged as an outlier

    # NaN / inf leg and imu samples are silently dropped and never corrupt the state.
    f.add_imu(float("nan"), t)
    f.add_leg(float("nan"), 0.0, 0.0, t)
    f.add_leg(0.0, float("inf"), 0.0, t)
    x2, y2, yaw2 = f.get_pose()
    assert all(math.isfinite(v) for v in (x2, y2, yaw2))

    # good data resumes -> the filter still tracks truth and stays finite (full recovery).
    for i in range(101, 300):
        t = i * dt
        f.add_leg(0.5 * t, 0.0, 0.0, t)
        f.add_imu(0.0, t)
        if i % 5 == 0:
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    fx, fy, fyaw = f.get_pose()
    tx = 0.5 * (300 - 1) * dt
    assert all(math.isfinite(v) for v in (fx, fy, fyaw))
    assert abs(fx - tx) < 0.05 and abs(fy) < 0.05 and abs(fyaw) < 0.03


# --------------------------------------------------------------------------- #
# standalone runner (pytest-free)
# --------------------------------------------------------------------------- #
def test_stale_gyro_falls_back_to_leg():
    """A stuck (frozen-but-finite) gyro must NOT keep steering fused heading.

    Regression for the IMU-freshness gap: add_leg() blends gyro at w_gyro~0.98,
    but the gyro cache was used without checking its age. A gyro that stops
    updating stays finite (so the NaN guard can't help) and would pin fused yaw
    to a dead rate while the robot really turns. With imu_timeout_s gating, once
    the cached gyro is older than the timeout we fall back to the leg yaw-delta.
    """
    dt = 1.0 / 30.0
    N = 60
    omega = 1.99 / (N * dt)          # leg reports a real ~1.99 rad turn in place

    # STALE case: one gyro sample at t=0 (stuck at 0), then it goes silent while
    # leg keeps coming with stamps >> imu_timeout_s later. Every integrated step
    # sees a stale gyro -> pure leg -> fused heading == leg heading exactly.
    f = FusedOdometry()
    f.add_imu(0.0, 0.0)              # gyro says "not turning", then never updates again
    f.add_leg(0.0, 0.0, 0.0, 1.0)   # seed leg_prev at t=1.0 (gyro already 1 s stale)
    lyaw = 0.0
    for i in range(1, N + 1):
        t = 1.0 + i * dt
        lyaw = wrap(lyaw + omega * dt)
        f.add_leg(0.0, 0.0, lyaw, t)
    _, _, yaw = f.get_pose()
    assert abs(wrap(yaw - lyaw)) < 1e-9, (
        "stale gyro should be ignored -> fused yaw must equal leg yaw exactly, "
        "got %.6f vs leg %.6f" % (yaw, lyaw))
    assert yaw > 1.9, "fused heading must follow the real turn, got %.6f" % yaw

    # CONTRAST: if the same stuck-at-0 gyro were kept FRESH (updated every step),
    # w_gyro=0.98 legitimately pins fused heading near 0. This is exactly the
    # corruption the freshness gate removes when the gyro is instead stale.
    g = FusedOdometry()
    g.add_leg(0.0, 0.0, 0.0, 0.0)
    lyaw2 = 0.0
    for i in range(1, N + 1):
        t = i * dt
        lyaw2 = wrap(lyaw2 + omega * dt)
        g.add_imu(0.0, t)           # fresh but stuck-at-zero gyro
        g.add_leg(0.0, 0.0, lyaw2, t)
    _, _, yaw2 = g.get_pose()
    assert abs(yaw2) < 0.1, (
        "fresh stuck gyro should dominate and pin heading near 0 (the bug the "
        "gate protects against elsewhere), got %.6f" % yaw2)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    npass = 0
    for fn in tests:
        try:
            fn()
            print(f"[ PASS ] {fn.__name__}")
            npass += 1
        except Exception:
            import traceback
            print(f"[ FAIL ] {fn.__name__}")
            traceback.print_exc()
    print("=" * 60)
    print(f"RESULT: {npass}/{len(tests)} passed")
    return 0 if npass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
