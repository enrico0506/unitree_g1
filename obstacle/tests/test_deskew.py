#!/usr/bin/env python3
"""Standalone tests + benchmark for obstacle/deskew.py.

Run:  python3 obstacle/test_deskew.py   (or from inside obstacle/: python3 test_deskew.py)

Pure numpy synthetic data, plain asserts, prints PASS/FAIL per case and exits
non-zero if anything fails. The ground-truth rotations here are built with an
INDEPENDENT reference implementation (explicit trig / a plain Rodrigues loop, and
an optional scipy cross-check) so a bug in the module's vectorised math is
actually caught rather than mirrored.
"""

import os
import sys
import time

import numpy as np

# Import the module whether we're run from the repo root or from obstacle/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from deskew import (ImuGyroDeskewer, rotvec_to_matrix, matrix_to_rotvec,
                        apply_rotvec, integrate_gyro)
except ImportError:
    from obstacle.deskew import (ImuGyroDeskewer, rotvec_to_matrix,
                                 matrix_to_rotvec, apply_rotvec, integrate_gyro)


# --------------------------------------------------------------------------- #
#  Independent reference helpers (NOT the module's code path).
# --------------------------------------------------------------------------- #

def ref_rodrigues(rv):
    """Plain single-rotvec Rodrigues, independent of the module."""
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rz(angle):
    """Explicit yaw (about +z) matrix built from raw trig - fully independent."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def scipy_rotvec_to_matrix(rv):
    """scipy cross-check reference, or None if scipy/this API is unavailable."""
    try:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(np.asarray(rv, dtype=np.float64))
        return r.as_matrix() if hasattr(r, "as_matrix") else r.as_dcm()
    except Exception:
        return None


def make_gyro(deskewer, t_start, t_end, w, hz=200.0):
    """Feed constant angular velocity w=(wx,wy,wz) over [t_start, t_end] at hz."""
    n = int(round((t_end - t_start) * hz)) + 1
    for t in np.linspace(t_start, t_end, n):
        deskewer.add_gyro(t, w[0], w[1], w[2])


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

def test_free_functions():
    """rotvec_to_matrix is a valid rotation, round-trips, and matches scipy."""
    rng = np.random.RandomState(0)

    # Zero angle -> identity.
    assert np.allclose(rotvec_to_matrix(np.zeros(3)), np.eye(3))
    # Tiny angle stays finite and near-identity (no divide-by-zero).
    R_tiny = rotvec_to_matrix(np.array([1e-12, 0.0, -1e-12]))
    assert np.all(np.isfinite(R_tiny)) and np.allclose(R_tiny, np.eye(3))

    rvs = rng.uniform(-1.0, 1.0, size=(200, 3)) * rng.uniform(0, 3, (200, 1))
    R = rotvec_to_matrix(rvs)                                  # (200, 3, 3)
    # Orthonormal, right-handed.
    RRt = np.einsum('nij,nkj->nik', R, R)
    assert np.allclose(RRt, np.broadcast_to(np.eye(3), RRt.shape), atol=1e-10)
    dets = np.linalg.det(R)
    assert np.allclose(dets, 1.0, atol=1e-10)

    # Round-trip through the log map.
    rvs2 = matrix_to_rotvec(R)
    R2 = rotvec_to_matrix(rvs2)
    assert np.allclose(R, R2, atol=1e-9), "rotvec<->matrix round-trip failed"

    # Independent scipy cross-check (if available).
    ref = scipy_rotvec_to_matrix(rvs)
    if ref is not None:
        assert np.allclose(R, ref, atol=1e-9), "rotvec_to_matrix != scipy"
    else:
        print("      (scipy cross-check skipped: not available)")

    # apply_rotvec consistent with building the matrix explicitly.
    pts = rng.uniform(-2, 2, size=(200, 3))
    assert np.allclose(apply_rotvec(pts, rvs),
                       np.einsum('nij,nj->ni', R, pts), atol=1e-12)
    # Single rotvec broadcast form.
    assert np.allclose(apply_rotvec(pts, rvs[0]), pts @ R[0].T, atol=1e-12)


def test_integrate_gyro_linear():
    """Constant gyro => cumulative angle grows linearly with time."""
    hz, dur, wz = 200.0, 0.5, 0.7
    t = np.linspace(0.0, dur, int(dur * hz) + 1)
    W = np.tile([0.0, 0.0, wz], (t.size, 1))
    cum = integrate_gyro(t, W)                                  # (K, 3)
    # Final cumulative yaw should equal wz * dur about +z.
    assert np.allclose(cum[0], 0.0)
    assert abs(np.linalg.norm(cum[-1]) - wz * dur) < 1e-6
    assert np.allclose(cum[-1] / np.linalg.norm(cum[-1]), [0, 0, 1], atol=1e-9)
    # A mid knot: angle proportional to elapsed time.
    mid = t.size // 2
    assert abs(np.linalg.norm(cum[mid]) - wz * (t[mid] - t[0])) < 1e-6


def test_static_wall_yaw():
    """Wall at x=2 m, sweep captured while yawing at 1.0 rad/s -> de-skew it back."""
    rng = np.random.RandomState(1)
    N = 5000
    t0 = 1000.0                       # large base => exercises absolute-time handling
    dur = 0.1
    omega = 1.0                       # rad/s yaw about +z
    ref_time = t0 + dur               # de-skew to scan end

    # True wall (== what a snapshot at ref_time would see): plane x = 2.
    P0 = np.empty((N, 3), dtype=np.float64)
    P0[:, 0] = 2.0
    P0[:, 1] = rng.uniform(-1.5, 1.5, N)   # left/right
    P0[:, 2] = rng.uniform(-1.0, 1.0, N)   # up/down

    # Per-point capture times spread across the sweep.
    point_times = np.linspace(t0, t0 + dur, N)

    # Corrupt: a point seen at t_i sits, in the sensor frame, rotated by the
    # yaw accumulated between ref_time and t_i => Rz(omega*(t_ref - t_i)) @ P0.
    # (Built with explicit trig, independent of the module.)
    ang = omega * (ref_time - point_times)                     # (N,)
    c, s = np.cos(ang), np.sin(ang)
    P_meas = np.empty_like(P0)
    P_meas[:, 0] = c * P0[:, 0] - s * P0[:, 1]
    P_meas[:, 1] = s * P0[:, 0] + c * P0[:, 1]
    P_meas[:, 2] = P0[:, 2]

    raw_err = np.max(np.linalg.norm(P_meas - P0, axis=1))
    assert raw_err > 0.05, "test setup: smear should be large before de-skew"

    dsk = ImuGyroDeskewer(buffer_s=0.5)
    make_gyro(dsk, t0 - 0.05, t0 + 0.15, (0.0, 0.0, omega))

    out = dsk.deskew(P_meas.astype(np.float32), point_times, ref_time)
    res = np.max(np.linalg.norm(out.astype(np.float64) - P0, axis=1))
    print("      raw smear max = %.4f m,  de-skewed residual max = %.6f m"
          % (raw_err, res))
    assert res < 0.01, "de-skewed residual too large: %.4f m" % res
    assert out.dtype == np.float32


def test_zero_motion():
    """No rotation => de-skew is (essentially) the identity; empty buffer too."""
    rng = np.random.RandomState(2)
    N = 3000
    t0 = 50.0
    pts = rng.uniform(-3, 3, size=(N, 3)).astype(np.float32)
    point_times = np.linspace(t0, t0 + 0.1, N)

    # Zero gyro.
    dsk = ImuGyroDeskewer(buffer_s=0.5)
    make_gyro(dsk, t0 - 0.05, t0 + 0.15, (0.0, 0.0, 0.0))
    out = dsk.deskew(pts, point_times, t0 + 0.1)
    assert np.allclose(out, pts, atol=1e-6), "zero-motion de-skew not identity"

    # Empty buffer -> unchanged copy.
    dsk2 = ImuGyroDeskewer(buffer_s=0.5)
    out2 = dsk2.deskew(pts, point_times, t0 + 0.1)
    assert np.array_equal(out2, pts) and out2 is not pts


def test_pitch_plus_yaw():
    """Combined constant pitch+yaw rotation is recovered on scattered points."""
    rng = np.random.RandomState(3)
    N = 2000
    t0 = 200.0
    dur = 0.1
    w = np.array([0.0, 0.6, -0.8])     # pitch about +y, yaw about +z, rad/s
    ref_time = t0 + dur

    # Scattered points a few metres out (not co-planar).
    P0 = rng.uniform(-1.0, 1.0, size=(N, 3))
    P0[:, 0] += 3.0                    # push forward so ranges ~2-4 m
    point_times = np.linspace(t0, t0 + dur, N)

    # Corrupt with the true relative rotation, built independently per point:
    # for constant body rate, C(t) = Rodrigues(w*(t-t0)) and the measured point
    # is C(t_i)^T C(t_r) @ P0 == Rodrigues(w*(t_r - t_i)) @ P0.
    P_meas = np.empty_like(P0)
    for i in range(N):
        P_meas[i] = ref_rodrigues(w * (ref_time - point_times[i])) @ P0[i]

    raw_err = np.max(np.linalg.norm(P_meas - P0, axis=1))
    assert raw_err > 0.05, "test setup: smear should be large before de-skew"

    dsk = ImuGyroDeskewer(buffer_s=0.5)
    make_gyro(dsk, t0 - 0.05, t0 + 0.15, w)
    out = dsk.deskew(P_meas.astype(np.float32), point_times, ref_time)
    res = np.max(np.linalg.norm(out.astype(np.float64) - P0, axis=1))
    print("      raw smear max = %.4f m,  de-skewed residual max = %.6f m"
          % (raw_err, res))
    assert res < 0.01, "pitch+yaw residual too large: %.4f m" % res


def test_translation_stub():
    """deskew_full adds the expected constant-velocity shift on top of rotation."""
    rng = np.random.RandomState(4)
    N = 1000
    t0 = 10.0
    pts = rng.uniform(-2, 2, size=(N, 3)).astype(np.float32)
    point_times = np.linspace(t0, t0 + 0.1, N)
    ref_time = t0 + 0.1
    vel = np.array([0.8, 0.0, 0.0])    # walking forward 0.8 m/s

    dsk = ImuGyroDeskewer(buffer_s=0.5)
    make_gyro(dsk, t0 - 0.05, t0 + 0.15, (0.0, 0.0, 0.0))   # no rotation
    rot = dsk.deskew(pts, point_times, ref_time).astype(np.float64)
    full = dsk.deskew_full(pts, point_times, ref_time, vel).astype(np.float64)
    expect = rot + vel[None, :] * (point_times - ref_time)[:, None]
    assert np.allclose(full, expect, atol=1e-5)
    # Earlier points (t_i < t_r) get pulled toward -x since the body moved +x.
    assert full[0, 0] < rot[0, 0]


def benchmark():
    """15 000 points, ~100 buffered gyro samples: print ms/call."""
    rng = np.random.RandomState(5)
    N = 15000
    t0 = 500.0
    dur = 0.1
    pts = (rng.uniform(-1, 1, size=(N, 3)).astype(np.float32) +
           np.array([3, 0, 0], dtype=np.float32))
    point_times = np.linspace(t0, t0 + dur, N)
    ref_time = t0 + dur

    dsk = ImuGyroDeskewer(buffer_s=0.5)
    # Fill a realistic full 0.5 s buffer at 200 Hz (~100 samples).
    make_gyro(dsk, t0 - 0.4, t0 + 0.1, (0.05, 0.6, 1.0))

    dsk.deskew(pts, point_times, ref_time)   # warm-up (allocations, caches)
    iters = 200
    samples = np.empty(iters)
    for i in range(iters):
        t = time.perf_counter()
        dsk.deskew(pts, point_times, ref_time)
        samples[i] = (time.perf_counter() - t) * 1e3
    # Report min/median (true compute cost); mean/max are inflated by CPU
    # contention from other processes on a live robot Jetson.
    lo, med = samples.min(), np.median(samples)
    print("      %d pts, %d gyro samples: min %.3f ms, median %.3f ms  (%.1f Mpts/s median)"
          % (N, dsk.n_samples(), lo, med, N / (med * 1e3)))


# --------------------------------------------------------------------------- #
#  Harness
# --------------------------------------------------------------------------- #

def main():
    tests = [
        ("free functions (Rodrigues/log/apply)", test_free_functions),
        ("integrate_gyro linear growth", test_integrate_gyro_linear),
        ("static wall, 1.0 rad/s yaw", test_static_wall_yaw),
        ("zero motion == identity", test_zero_motion),
        ("pitch + yaw combined", test_pitch_plus_yaw),
        ("translation stub (deskew_full)", test_translation_stub),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:
            n_fail += 1
            print("FAIL  %s  ->  %s: %s" % (name, type(e).__name__, e))

    print("\n-- benchmark --")
    try:
        benchmark()
    except Exception as e:
        print("benchmark error: %s" % e)

    print("\n%d/%d tests passed" % (len(tests) - n_fail, len(tests)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
