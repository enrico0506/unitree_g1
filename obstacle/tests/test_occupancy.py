#!/usr/bin/env python3
"""Standalone unit tests + benchmark for obstacle/occupancy.py.

Run:  python3 obstacle/tests/test_occupancy.py     (from the repo root)
      python3 test_occupancy.py                     (from inside obstacle/tests/)

No ROS / rclpy needed -- pure numpy. Prints PASS/FAIL per test and a per-update
benchmark; exits non-zero if any test fails.
"""

import os
import sys
import time

import numpy as np

# obstacle/occupancy.py lives one directory up from obstacle/tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from occupancy import PolarOccupancyGrid, UNKNOWN, FREE, OCCUPIED  # noqa: E402


# ------------------------------------------------------------------ test cases
def test_convergence():
    """Repeatedly observing a return at 1.00 m drives that bin occupied, nearer
    bins free, farther bins ~unknown; the sector state becomes OCCUPIED."""
    g = PolarOccupancyGrid(n_sectors=180, r_max=3.0, r_bin=0.05)
    s = g.sector_index(0.0)                     # straight-ahead sector
    ret = np.full(g.n_sectors, np.nan)
    ret[s] = 1.00
    return_bin = int(1.00 / g.r_bin)            # -> 20; center 1.025 m

    # After 3 frames the return bin should already exceed p > 0.7.
    for _ in range(3):
        g.update(ret, dt=0.1)
    p3 = g.prob_matrix()[s]
    assert p3[return_bin] > 0.7, "return bin p=%.3f not > 0.7 after 3 frames" % p3[return_bin]

    for _ in range(2):                          # 5 frames total
        g.update(ret, dt=0.1)
    p = g.prob_matrix()[s]
    assert p[return_bin] > 0.7, "return bin p=%.3f not occupied" % p[return_bin]
    assert p[return_bin - 1] < 0.5, "nearer bin p=%.3f not free" % p[return_bin - 1]
    # A farther, never-touched bin stays at the prior ~0.5.
    assert abs(p[return_bin + 6] - 0.5) < 1e-3, "farther bin p=%.3f drifted from 0.5" % p[return_bin + 6]

    dist, state = g.nearest_occupied()
    assert state[s] == OCCUPIED, "state=%d not OCCUPIED" % state[s]
    assert abs(dist[s] - 1.025) < 1e-6, "nearest dist=%.3f not the return bin center" % dist[s]
    # An unrelated sector must remain UNKNOWN, not silently FREE.
    other = (s + 40) % g.n_sectors
    assert state[other] == UNKNOWN, "untouched sector state=%d not UNKNOWN" % state[other]


def test_decay_clear():
    """After feeding stops, a decaying occupied sector leaves OCCUPIED within a
    few tau and returns to UNKNOWN -- gradually, not instantly."""
    g = PolarOccupancyGrid(n_sectors=180, r_max=3.0, r_bin=0.05, tau_s=0.7)
    s = g.sector_index(0.0)
    ret = np.full(g.n_sectors, np.nan)
    ret[s] = 1.00
    for _ in range(12):                         # drive it well past l_high
        g.update(ret, dt=0.1)
    _, state = g.nearest_occupied()
    assert state[s] == OCCUPIED, "did not reach OCCUPIED before decay"

    nan_frame = np.full(g.n_sectors, np.nan)

    # One missed frame must NOT instantly clear it.
    g.update(nan_frame, dt=0.1)
    _, state = g.nearest_occupied()
    assert state[s] == OCCUPIED, "cleared after a single missed frame (too eager)"

    # Within a few tau it must return to UNKNOWN (not FREE).
    frames_to_clear = None
    for k in range(2, 200):                     # up to 20 s of missed frames
        g.update(nan_frame, dt=0.1)
        _, state = g.nearest_occupied()
        if state[s] != OCCUPIED:
            frames_to_clear = k
            break
    assert frames_to_clear is not None, "never left OCCUPIED"
    t_clear = frames_to_clear * 0.1
    assert t_clear <= 6.0 * g.tau_s, "took %.2fs (> 6 tau) to clear" % t_clear
    assert t_clear >= 1.0 * g.tau_s, "cleared in %.2fs (< 1 tau) -- not gradual" % t_clear
    _, state = g.nearest_occupied()
    assert state[s] == UNKNOWN, "settled to state=%d, expected UNKNOWN" % state[s]


def test_hysteresis():
    """A sector whose peak log-odds oscillates around l_high must NOT toggle the
    blocked/clear latch every frame -- clearing requires crossing back down."""
    g = PolarOccupancyGrid(n_sectors=180, r_max=3.0, r_bin=0.05, l_high=0.85, l_low=0.0)
    s = g.sector_index(0.0)
    ret = np.full(g.n_sectors, np.nan)
    ret[s] = 1.00
    nan_frame = np.full(g.n_sectors, np.nan)

    # Latch it blocked (two feeds -> peak ~1.59 > l_high).
    g.update(ret, dt=0.1)
    g.update(ret, dt=0.1)
    assert g.blocked[s], "did not latch blocked"

    # Now oscillate: a few missed frames (peak decays below l_high) then a feed
    # (peak jumps back above). A naive per-frame threshold on l_high would toggle;
    # the hysteresis latch must stay blocked the whole time.
    blocked_hist = []
    naive_hist = []
    for _ in range(3):                          # 3 oscillation cycles
        for _ in range(6):                      # decay below l_high
            g.update(nan_frame, dt=0.1)
            peak = float(g.logodds[s].max())
            blocked_hist.append(bool(g.blocked[s]))
            naive_hist.append(peak > g.l_high)
        g.update(ret, dt=0.1)                   # push back above l_high
        peak = float(g.logodds[s].max())
        blocked_hist.append(bool(g.blocked[s]))
        naive_hist.append(peak > g.l_high)

    assert all(blocked_hist), "hysteresis latch toggled off during oscillation"
    assert any(naive_hist) and not all(naive_hist), (
        "test scenario did not actually straddle l_high (naive=%r)" % naive_hist
    )

    # Sanity: sustained decay well past the prior DOES eventually clear the latch.
    for _ in range(60):
        g.update(nan_frame, dt=0.1)
    assert not g.blocked[s], "latch never cleared after prolonged decay"


def test_unknown_honesty():
    """A never-observed sector reports UNKNOWN (0), never FREE, and p ~ 0.5."""
    g = PolarOccupancyGrid(n_sectors=180, r_max=3.0, r_bin=0.05)
    # Feed only one sector; every other sector was never scanned.
    hit = g.sector_index(90.0)
    ret = np.full(g.n_sectors, np.nan)
    ret[hit] = 0.75
    for _ in range(5):
        g.update(ret, dt=0.1)

    _, state = g.nearest_occupied()
    untouched = [i for i in range(g.n_sectors) if i != hit]
    assert np.all(state[untouched] == UNKNOWN), "some never-observed sector is not UNKNOWN"
    assert not np.any(state[untouched] == FREE), "a never-observed sector reported FREE"

    psec = g.prob_per_sector()
    assert np.all(np.abs(psec[untouched] - 0.5) < 1e-6), "unobserved sector p drifted from 0.5"

    # And with covered_mask, an explicitly scanned-but-clear NaN sector DOES free.
    clear_sector = g.sector_index(-90.0)
    covered = np.zeros(g.n_sectors, dtype=bool)
    covered[clear_sector] = True
    clear_ret = np.full(g.n_sectors, np.nan)
    for _ in range(3):
        g.update(clear_ret, dt=0.1, covered_mask=covered)
    _, state = g.nearest_occupied()
    assert state[clear_sector] == FREE, "covered scanned-clear sector not FREE (state=%d)" % state[clear_sector]


def benchmark():
    """180 sectors x 60 bins, 100 updates; print ms/update (target < ~1-2 ms)."""
    g = PolarOccupancyGrid(n_sectors=180, r_max=3.0, r_bin=0.05)
    assert g.n_rbins == 60, "expected 60 range bins, got %d" % g.n_rbins

    rng = np.random.RandomState(0)
    # Mixed frame: ~60% of sectors carry a return in [0.3, 3.0) m, rest NaN.
    frames = []
    for _ in range(100):
        r = rng.uniform(0.3, 3.0, size=g.n_sectors).astype(np.float64)
        mask = rng.rand(g.n_sectors) < 0.4
        r[mask] = np.nan
        frames.append(r)

    # Warm up (first call pays one-time numpy setup costs).
    g.update(frames[0], dt=0.1)
    g.reset()

    t0 = time.perf_counter()
    for f in frames:
        g.update(f, dt=0.1)
    elapsed = time.perf_counter() - t0
    ms = 1000.0 * elapsed / len(frames)
    print("  benchmark: %.4f ms/update  (180x60 grid, 100 updates; target < ~1-2 ms)" % ms)
    assert ms < 5.0, "update too slow: %.3f ms/update" % ms
    return ms


# ---------------------------------------------------------------------- runner
def main():
    tests = [
        ("convergence (return -> occupied, nearer free, farther unknown)", test_convergence),
        ("decay/clear (self-clears to unknown over a few tau, not instant)", test_decay_clear),
        ("hysteresis (no per-frame toggle around l_high)", test_hysteresis),
        ("unknown honesty (never-observed = unknown, not free)", test_unknown_honesty),
    ]
    n_pass = 0
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            n_fail += 1
            print("FAIL: %s\n      %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print("ERROR: %s\n      %r" % (name, e))
        else:
            n_pass += 1
            print("PASS: %s" % name)

    print("-" * 60)
    try:
        benchmark()
        print("PASS: benchmark")
        n_pass += 1
    except AssertionError as e:
        n_fail += 1
        print("FAIL: benchmark\n      %s" % e)
    print("-" * 60)
    print("%d passed, %d failed" % (n_pass, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
