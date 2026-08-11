#!/usr/bin/env python3
"""test_filters.py -- standalone unit tests + benchmark for obstacle/filters.py.

Run:  python3 obstacle/tests/test_filters.py
No ROS / rclpy needed.  Prints PASS/FAIL per test, a final summary, and a per-function
benchmark on 15000 points.  Exit code 0 iff all tests pass.
"""

import os
import sys
import time
from collections import namedtuple

import numpy as np

# obstacle/filters.py lives one directory up from obstacle/tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import (                                  # noqa: E402
    CUSTOMPOINT_DTYPE,
    SCIPY_CAPS,
    custommsg_to_numpy,
    dror_filter,
    merge_min,
    required_points,
    sector_kth_nearest,
    sector_tripwire,
)

# -------------------------------------------------------------------------------------
# tiny test harness
# -------------------------------------------------------------------------------------
_RESULTS = []


def check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond)))
    tag = "PASS" if cond else "FAIL"
    line = "  [%s] %s" % (tag, name)
    if detail:
        line += "  -- " + detail
    print(line)
    return bool(cond)


# -------------------------------------------------------------------------------------
# fake CustomMsg builders
# -------------------------------------------------------------------------------------
# A CustomPoint stand-in whose attribute names match the real rosidl object.
FakePoint = namedtuple("CustomPoint",
                       "offset_time x y z reflectivity tag line")


class FakeCustomMsgList(object):
    """Mimics the real rosidl CustomMsg: .points is a list of point objects."""
    def __init__(self, timebase, rec):
        self.timebase = int(timebase)
        self.point_num = int(rec.shape[0])
        self.points = [FakePoint(int(r["offset_time"]), float(r["x"]), float(r["y"]),
                                 float(r["z"]), int(r["reflectivity"]),
                                 int(r["tag"]), int(r["line"])) for r in rec]


class FakeCustomMsgBuf(object):
    """Mimics a zero-copy driver: exposes a contiguous record buffer via .point_data."""
    def __init__(self, timebase, rec):
        self.timebase = int(timebase)
        self.point_num = int(rec.shape[0])
        self.point_data = rec.tobytes()


def make_records(n, timebase_offset_step=50000):
    """Build a ground-truth structured array of n CustomPoint records."""
    rng = np.random.default_rng(1234)
    rec = np.zeros(n, dtype=CUSTOMPOINT_DTYPE)
    rec["x"] = rng.uniform(-3.0, 3.0, n).astype(np.float32)
    rec["y"] = rng.uniform(-3.0, 3.0, n).astype(np.float32)
    rec["z"] = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    # strictly increasing offsets, spaced well above float64-at-1.7e18 resolution.
    rec["offset_time"] = (np.arange(n, dtype=np.uint32) * timebase_offset_step)
    rec["reflectivity"] = rng.integers(0, 256, n).astype(np.uint8)
    rec["tag"] = rng.integers(0, 4, n).astype(np.uint8)
    rec["line"] = rng.integers(0, 4, n).astype(np.uint8)
    return rec


def slow_decode(rec, timebase, decimate=1):
    """Reference decode straight from the record array (independent of filters.py)."""
    r = rec[::decimate] if decimate > 1 else rec
    pts = np.stack([r["x"], r["y"], r["z"]], axis=1).astype(np.float32)
    times = timebase * 1e-9 + r["offset_time"].astype(np.float64) * 1e-9
    refl = r["reflectivity"].astype(np.uint8)
    return pts, times, refl


# =====================================================================================
# TEST 1 -- fast ingest
# =====================================================================================
def test_ingest():
    print("\n== TEST 1: custommsg_to_numpy (fast ingest) ==")
    print("  CustomPoint dtype itemsize = %d bytes, fields = %s"
          % (CUSTOMPOINT_DTYPE.itemsize, list(CUSTOMPOINT_DTYPE.names)))

    check("CustomPoint itemsize == 20", CUSTOMPOINT_DTYPE.itemsize == 20)

    timebase = 1_700_000_000_000_000_000          # ~1.7e18 ns (realistic Livox epoch)
    n = 2000
    rec = make_records(n)
    ref_pts, ref_t, ref_refl = slow_decode(rec, timebase)

    # ---- buffer (fast) path ----------------------------------------------------------
    msg_buf = FakeCustomMsgBuf(timebase, rec)
    p_b, t_b, r_b = custommsg_to_numpy(msg_buf)
    check("buffer: shape (N,3) float32",
          p_b.shape == (n, 3) and p_b.dtype == np.float32,
          "got %s %s" % (p_b.shape, p_b.dtype))
    check("buffer: xyz matches reference", np.allclose(p_b, ref_pts))
    check("buffer: times matches reference", np.allclose(t_b, ref_t))
    check("buffer: reflectivity matches (uint8)",
          r_b.dtype == np.uint8 and np.array_equal(r_b, ref_refl))
    check("buffer: times strictly monotonic", np.all(np.diff(t_b) > 0))

    # ---- list (fallback) path --------------------------------------------------------
    msg_list = FakeCustomMsgList(timebase, rec)
    p_l, t_l, r_l = custommsg_to_numpy(msg_list)
    check("list: shape (N,3) float32",
          p_l.shape == (n, 3) and p_l.dtype == np.float32)
    check("list: xyz matches reference", np.allclose(p_l, ref_pts))
    check("list: times matches reference", np.allclose(t_l, ref_t))
    check("list: reflectivity matches", np.array_equal(r_l, ref_refl))
    check("list path == buffer path", np.allclose(p_l, p_b) and np.allclose(t_l, t_b))

    # ---- decimate --------------------------------------------------------------------
    dec = 5
    p_d, t_d, r_d = custommsg_to_numpy(msg_buf, decimate=dec)
    rp, rt, rr = slow_decode(rec, timebase, decimate=dec)
    check("decimate=5: count", p_d.shape[0] == rp.shape[0],
          "got %d expected %d" % (p_d.shape[0], rp.shape[0]))
    check("decimate=5: xyz matches reference", np.allclose(p_d, rp))
    check("decimate=5: times matches reference", np.allclose(t_d, rt))
    # decimate on the list path too
    p_dl, _, _ = custommsg_to_numpy(msg_list, decimate=dec)
    check("decimate=5: list path == buffer path", np.allclose(p_dl, p_d))

    # ---- empty ----------------------------------------------------------------------
    empty = FakeCustomMsgList(timebase, make_records(0))
    pe, te, re = custommsg_to_numpy(empty)
    check("empty msg -> (0,3)", pe.shape == (0, 3) and te.shape == (0,))


# =====================================================================================
# TEST 2 -- DROR
# =====================================================================================
def test_dror():
    print("\n== TEST 2: dror_filter (range-scaled outlier rejection) ==")
    rng = np.random.default_rng(7)
    parts = []

    # (a) a dense wall patch at ~2 m ahead (~300 pts in a small slab).
    dense = np.column_stack([
        rng.uniform(1.95, 2.05, 300),
        rng.uniform(-0.15, 0.15, 300),
        rng.uniform(-0.10, 0.10, 300),
    ]).astype(np.float32)
    dense_sl = slice(0, 300)
    parts.append(dense)

    # (b) a legit THIN 3-point object at ~1 m (a cable cross-section): 3 points ~2 cm
    #     apart, off to the left, isolated from everything else.
    thin = np.array([
        [0.90, 0.60, 0.00],
        [0.90, 0.62, 0.00],
        [0.90, 0.64, 0.00],
    ], dtype=np.float32)
    thin_sl = slice(300, 303)
    parts.append(thin)

    # (c) a single lone flier way out, no neighbours anywhere near.
    flier = np.array([[5.0, -3.0, 0.5]], dtype=np.float32)
    flier_idx = 303
    parts.append(flier)

    pts = np.concatenate(parts, axis=0)
    keep = dror_filter(pts, alpha_rad=0.004, beta=3.0, r_min=0.08, min_neighbors=3)

    check("DROR: lone flier removed", keep[flier_idx] == False)  # noqa: E712
    check("DROR: thin 3-point object survives", bool(np.all(keep[thin_sl])),
          "kept %d/3" % int(keep[thin_sl].sum()))
    check("DROR: dense wall mostly kept", keep[dense_sl].mean() > 0.95,
          "kept fraction %.3f" % keep[dense_sl].mean())

    # sanity: a truly isolated pair (2 pts) fails min_neighbors=3 but a k_for_scale=2
    # override keeps them (exercises the k_for_scale knob).
    pair = np.array([[8.0, 0.0, 0.0], [8.0, 0.02, 0.0]], dtype=np.float32)
    p2 = np.concatenate([pts, pair], axis=0)
    k3 = dror_filter(p2, min_neighbors=3)
    k2 = dror_filter(p2, min_neighbors=3, k_for_scale=2)
    check("DROR: isolated pair dropped at min_neighbors=3",
          not k3[-1] and not k3[-2])
    check("DROR: k_for_scale=2 keeps the pair", k2[-1] and k2[-2])


# =====================================================================================
# TEST 3 -- sector k-th nearest
# =====================================================================================
def _ring_of(x, y):
    ang = np.degrees(np.arctan2(y, x))
    horiz = np.hypot(x, y)
    return ang, horiz


def test_sector():
    print("\n== TEST 3: sector_kth_nearest (range-scaled per-sector distance) ==")
    n_sectors = 180
    start = -180.0
    bin_deg = 360.0 / n_sectors

    xs, ys = [], []

    # (A) thin 3-point object at ~1.2 m, all in ONE sector. Center mid-bin (31 deg is
    #     the middle of the 30..32 deg sector) so all 3 points land together.
    thin_center = 31.0
    a0 = np.radians(thin_center)
    r_thin = 1.2
    for da in (-0.3, 0.0, 0.3):                       # stays inside the 2 deg sector
        aa = a0 + np.radians(da)
        xs.append(r_thin * np.cos(aa)); ys.append(r_thin * np.sin(aa))
    thin_sector = int(np.floor((thin_center - start) / bin_deg)) % n_sectors

    # (B) a single flier at 1.0 m near -60 deg (its own sector).
    af = np.radians(-60.0)
    xs.append(1.0 * np.cos(af)); ys.append(1.0 * np.sin(af))
    flier_sector = int(np.floor((-60.0 - start) / bin_deg)) % n_sectors

    # (C) a genuine dense near hazard: 20 points at ~0.6 m near +90 deg.
    rng = np.random.default_rng(3)
    an = np.radians(90.0) + rng.uniform(-0.005, 0.005, 20)   # tight, one sector
    rn = 0.6 + rng.uniform(-0.01, 0.01, 20)
    xs.extend((rn * np.cos(an)).tolist()); ys.extend((rn * np.sin(an)).tolist())
    near_sector = int(np.floor((90.0 - start) / bin_deg)) % n_sectors

    x = np.array(xs); y = np.array(ys)
    ang, horiz = _ring_of(x, y)

    dist, counts = sector_kth_nearest(ang, horiz, n_sectors=n_sectors, start_deg=start,
                                      k=3, range_scaled_mincount=True,
                                      k0=3, r0=1.0, r_max=3.0)

    check("sector: thin object sector reports ~1.2 m",
          np.isfinite(dist[thin_sector]) and abs(dist[thin_sector] - 1.2) < 0.02,
          "dist=%.3f count=%d req=%d" %
          (dist[thin_sector], counts[thin_sector],
           int(required_points(1.2, 3, 1.0, 3.0))))

    check("sector: lone-flier sector rejected (NaN)",
          np.isnan(dist[flier_sector]),
          "count=%d req(@1.0)=%d" %
          (counts[flier_sector], int(required_points(1.0, 3, 1.0, 3.0))))

    check("sector: dense near hazard kept ~0.6 m",
          np.isfinite(dist[near_sector]) and abs(dist[near_sector] - 0.6) < 0.05,
          "dist=%.3f count=%d req=%d" %
          (dist[near_sector], counts[near_sector],
           int(required_points(0.6, 3, 1.0, 3.0))))

    # without the min-count gate the lone flier IS reported (k clamps to count=1).
    d2, _ = sector_kth_nearest(ang, horiz, n_sectors=n_sectors, start_deg=start,
                               k=3, range_scaled_mincount=False)
    check("sector: no-gate reports the lone flier (~1.0 m)",
          np.isfinite(d2[flier_sector]) and abs(d2[flier_sector] - 1.0) < 0.02)

    # ---- tripwire override -----------------------------------------------------------
    # a 2-point sparse object at ~1.0 m: fails the k-th min-count gate (req@1.0 = 3),
    # but the near-field tripwire rescues it.
    at = np.radians(150.0)
    tx = [1.0 * np.cos(at), 1.0 * np.cos(at + np.radians(0.3))]
    ty = [1.0 * np.sin(at), 1.0 * np.sin(at + np.radians(0.3))]
    x2 = np.concatenate([x, tx]); y2 = np.concatenate([y, ty])
    ang2, horiz2 = _ring_of(x2, y2)
    trip_sector = int(np.floor((150.0 - start) / bin_deg)) % n_sectors

    base, _ = sector_kth_nearest(ang2, horiz2, n_sectors=n_sectors, start_deg=start,
                                 k=3, range_scaled_mincount=True, k0=3, r0=1.0, r_max=3.0)
    trip = sector_tripwire(ang2, horiz2, n_sectors=n_sectors, start_deg=start,
                           tripwire_range=1.6, tripwire_min=2)
    merged = merge_min(base, trip)

    check("tripwire: 2-pt near object missed by k-th gate",
          np.isnan(base[trip_sector]))
    check("tripwire: rescued by tripwire (~1.0 m)",
          np.isfinite(merged[trip_sector]) and abs(merged[trip_sector] - 1.0) < 0.02,
          "merged=%.3f" % merged[trip_sector])
    check("tripwire: thin object still ~1.2 m after merge",
          abs(merged[thin_sector] - 1.2) < 0.02)


# =====================================================================================
# TEST 4 -- benchmark
# =====================================================================================
def _bench(fn, iters=20):
    fn()                                              # warm-up
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3   # ms/call


def test_benchmark():
    print("\n== TEST 4: benchmark (15000 points) ==")
    print("  scipy caps: %s" % SCIPY_CAPS)
    n = 15000
    timebase = 1_700_000_000_000_000_000
    rec = make_records(n)
    msg_buf = FakeCustomMsgBuf(timebase, rec)
    msg_list = FakeCustomMsgList(timebase, rec)

    pts, _, _ = custommsg_to_numpy(msg_buf)
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    horiz = np.hypot(pts[:, 0], pts[:, 1])

    ms_buf = _bench(lambda: custommsg_to_numpy(msg_buf))
    ms_list = _bench(lambda: custommsg_to_numpy(msg_list), iters=10)
    ms_dror = _bench(lambda: dror_filter(pts), iters=10)
    ms_sec = _bench(lambda: sector_kth_nearest(ang, horiz))
    ms_trip = _bench(lambda: sector_tripwire(ang, horiz))

    print("  custommsg_to_numpy (buffer/frombuffer): %8.3f ms" % ms_buf)
    print("  custommsg_to_numpy (list/fallback)    : %8.3f ms" % ms_list)
    print("  dror_filter                           : %8.3f ms" % ms_dror)
    print("  sector_kth_nearest                    : %8.3f ms" % ms_sec)
    print("  sector_tripwire                       : %8.3f ms" % ms_trip)
    print("  speedup buffer vs list                : %8.1fx" % (ms_list / ms_buf))

    # a loose budget so the benchmark also acts as a regression guard on this Jetson.
    check("benchmark: full ingest+DROR+sector < 100 ms/frame",
          (ms_buf + ms_dror + ms_sec + ms_trip) < 100.0,
          "sum=%.2f ms" % (ms_buf + ms_dror + ms_sec + ms_trip))


# =====================================================================================
def main():
    print("filters.py self-test  |  scipy caps: %s" % SCIPY_CAPS)
    test_ingest()
    test_dror()
    test_sector()
    test_benchmark()

    total = len(_RESULTS)
    passed = sum(1 for _, ok in _RESULTS if ok)
    print("\n" + "=" * 60)
    print("SUMMARY: %d / %d passed" % (passed, total))
    failed = [n for n, ok in _RESULTS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print("   - " + n)
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
