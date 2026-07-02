#!/usr/bin/env python3
"""Standalone tests for obstacle/ground.py.

Run:  python3 obstacle/test_ground.py

Pure numpy, no ROS. Prints PASS/FAIL per test, a benchmark, and reports whether
the optional `pypatchworkpp` backend was importable in this environment.
Exits non-zero if any test fails.
"""

import os
import sys
import time

import numpy as np

# Allow both `python3 obstacle/test_ground.py` (from repo root) and running
# from inside obstacle/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ground  # noqa: E402


RNG = np.random.default_rng(42)
FLOOR_Z = -1.10                     # floor is 1.10 m below a level sensor


# --------------------------------------------------------------------------- #
# Scene builders
# --------------------------------------------------------------------------- #
def make_floor(x0=0.5, x1=3.0, y0=-1.5, y1=1.5, step=0.05, noise=0.01):
    """Dense noisy floor plane at z = FLOOR_Z. Returns (N, 3) float32."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    gz = FLOOR_Z + RNG.normal(0.0, noise, size=gx.shape)
    return np.stack([gx, gy, gz], axis=1).astype(np.float32)


def make_box(cx=1.5, cy=0.0, half=0.1, height=0.3, n=400):
    """Filled box sitting ON the floor: points span the full 0..height range
    so the cell's low percentile catches its base. Returns (N, 3)."""
    x = cx + RNG.uniform(-half, half, n)
    y = cy + RNG.uniform(-half, half, n)
    z = FLOOR_Z + RNG.uniform(0.0, height, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def make_thin_bar(cx=2.15, cy=0.0, half_x=0.15, half_y=0.03, top=0.05, n=60):
    """A thin bar (e.g. a cable) whose top sits `top` metres above the floor.
    Its cells also see the floor around it, so the local floor is FLOOR_Z."""
    x = cx + RNG.uniform(-half_x, half_x, n)
    y = cy + RNG.uniform(-half_y, half_y, n)
    z = FLOOR_Z + top + RNG.normal(0.0, 0.002, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def rot_y(theta):
    """Rotation about +y (pitch). Positive tilts the +x axis downward."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_flat_floor_and_box():
    """Floor removed, box kept, thin bar survives with a small clearance."""
    floor = make_floor()
    box = make_box()
    bar = make_thin_bar()
    pts = np.vstack([floor, box, bar]).astype(np.float32)

    nf, nb = len(floor), len(box)
    fl = slice(0, nf)
    bx = slice(nf, nf + nb)
    br = slice(nf + nb, len(pts))

    mask = ground.segment_obstacles_elevation(
        pts, cell=0.20, ground_clearance=0.08, max_above=1.9, floor_pct=10.0)

    floor_kept = mask[fl].mean()
    box_kept = mask[bx].mean()
    print(f"    floor flagged-as-obstacle: {floor_kept*100:5.1f}%  "
          f"box kept: {box_kept*100:5.1f}%")

    ok = True
    ok &= floor_kept < 0.02        # floor overwhelmingly removed
    ok &= box_kept > 0.55          # majority of box (its upper 0.22 m) kept

    # thin 0.05 m bar with a 0.04 m clearance must survive.
    mask_bar = ground.segment_obstacles_elevation(
        pts, cell=0.20, ground_clearance=0.04, max_above=1.9, floor_pct=10.0)
    bar_kept = mask_bar[br].mean()
    print(f"    thin-bar kept (clearance 0.04): {bar_kept*100:5.1f}%")
    ok &= bar_kept > 0.80
    return bool(ok)


def test_tilt_robustness():
    """A 12 deg-tilted scene, gravity-leveled, segments like the level one."""
    floor = make_floor()
    box = make_box()
    world = np.vstack([floor, box]).astype(np.float32)
    nf = len(floor)
    fl, bx = slice(0, nf), slice(nf, len(world))

    # Reference: untilted (already level).
    _, mask_level = ground.segment_obstacles(
        world, gravity_vec=None, method="elevation",
        cell=0.20, ground_clearance=0.08)

    # Tilt the whole scene 12 deg (sensor nose-up). p_sensor = R @ p_world.
    theta = np.deg2rad(12.0)
    R = rot_y(theta)
    tilted = (world @ R.T).astype(np.float32)
    # Gravity in the sensor frame: world-down [0,0,-1] rotated into the frame.
    gravity_vec = (R @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

    _, mask_tilt = ground.segment_obstacles(
        tilted, gravity_vec=gravity_vec, method="elevation",
        cell=0.20, ground_clearance=0.08)

    floor_lvl, floor_tlt = mask_level[fl].mean(), mask_tilt[fl].mean()
    box_lvl, box_tlt = mask_level[bx].mean(), mask_tilt[bx].mean()

    # Sanity: WITHOUT levelling the tilted floor should leak badly.
    mask_raw = ground.segment_obstacles_elevation(tilted, cell=0.20,
                                                  ground_clearance=0.08)
    floor_raw = mask_raw[fl].mean()

    print(f"    floor obstacle%%  level={floor_lvl*100:4.1f}  "
          f"tilted+leveled={floor_tlt*100:4.1f}  tilted+RAW={floor_raw*100:4.1f}")
    print(f"    box   kept%%      level={box_lvl*100:4.1f}  "
          f"tilted+leveled={box_tlt*100:4.1f}")

    ok = True
    ok &= floor_tlt < 0.05                     # leveled floor still removed
    ok &= box_tlt > 0.55                       # leveled box still kept
    ok &= abs(box_tlt - box_lvl) < 0.20        # close to the level result
    # The elevation grid is itself tilt-tolerant (per-cell LOCAL floor: a 12 deg
    # tilt over a 0.2 m cell is only ~4 cm < ground_clearance), so raw-vs-leveled
    # floor leakage is NOT the right signal. Assert what level_by_gravity is FOR:
    # it flattens the tilted floor (z variance collapses to sensor noise).
    tilted_floor = tilted[fl]
    leveled_floor = ground.level_by_gravity(tilted_floor, gravity_vec)
    ok &= leveled_floor[:, 2].std() < tilted_floor[:, 2].std() * 0.5
    print(f"    floor z-std      tilted={tilted_floor[:,2].std():.3f}  "
          f"leveled={leveled_floor[:,2].std():.3f}  (raw-grid leak={floor_raw*100:.1f}%)")

    # Levelling an already-level cloud must be a near no-op.
    lvl = ground.level_by_gravity(world, np.array([0.0, 0.0, -1.0]))
    ok &= np.allclose(lvl, world, atol=1e-4)
    return bool(ok)


def test_self_mask():
    """Footprint low returns dropped; a chest-height point above is kept."""
    pts = np.array([
        [0.10, 0.00, 0.10],    # inside footprint, low  -> self (drop)
        [-0.20, 0.15, 0.30],   # inside footprint, low  -> self (drop)
        [0.10, 0.00, 1.20],    # inside footprint column but chest height -> keep
        [2.00, 0.00, 0.10],    # well ahead                              -> keep
        [0.10, 0.50, 0.10],    # outside |y| half-width                  -> keep
    ], dtype=np.float32)
    keep = ground.self_mask(pts)   # defaults front/back 0.35, half_w 0.30, max_h 0.6
    expected = np.array([False, False, True, True, True])
    print(f"    self keep-mask: {keep.tolist()}  expected: {expected.tolist()}")
    return bool(np.array_equal(keep, expected))


def test_benchmark():
    """15000-point frame; report ms/call for the elevation grid."""
    n = 15000
    floor = make_floor(step=0.03)                 # denser -> ~a lot of points
    # trim/pad to exactly n points for a stable number
    if len(floor) >= n:
        pts = floor[:n]
    else:
        extra = RNG.uniform([0.5, -1.5, FLOOR_Z], [3.0, 1.5, FLOOR_Z + 0.5],
                            size=(n - len(floor), 3))
        pts = np.vstack([floor, extra]).astype(np.float32)
    pts = pts.astype(np.float32)
    grav = np.array([0.05, 0.0, -1.0], dtype=np.float32)

    iters = 100
    # warm up
    ground.segment_obstacles(pts, gravity_vec=grav, method="elevation")
    t0 = time.perf_counter()
    for _ in range(iters):
        ground.segment_obstacles(pts, gravity_vec=grav, method="elevation")
    dt_full = (time.perf_counter() - t0) / iters * 1e3

    t0 = time.perf_counter()
    for _ in range(iters):
        ground.segment_obstacles_elevation(pts)
    dt_grid = (time.perf_counter() - t0) / iters * 1e3

    print(f"    {n} pts | elevation grid only: {dt_grid:6.2f} ms/call | "
          f"full (level+grid+self): {dt_full:6.2f} ms/call")
    # target < ~10-15 ms for the elevation grid; allow headroom for CI noise.
    return bool(dt_grid < 25.0)


# --------------------------------------------------------------------------- #
def main():
    print(f"pypatchworkpp available: {ground.patchwork_available()}")
    print("-" * 68)
    tests = [
        ("flat floor + box + thin bar", test_flat_floor_and_box),
        ("tilt robustness (12 deg + gravity)", test_tilt_robustness),
        ("self_mask footprint", test_self_mask),
        ("benchmark 15000 pts", test_benchmark),
    ]
    results = []
    for name, fn in tests:
        print(f"[ RUN  ] {name}")
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok = False
        print(f"[ {'PASS' if ok else 'FAIL'} ] {name}\n")
        results.append(ok)

    npass = sum(results)
    print("=" * 68)
    print(f"RESULT: {npass}/{len(results)} passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
