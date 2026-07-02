"""Unit tests for the D435i per-sector depth ring + guard merge (QW4 / depth fusion).

Camera integration (V4L2 /dev/video0) can only be validated on the robot; these cover the
math: the per-sector ring compute and the NaN-aware min-merge (UNKNOWN/fail-safe rule).
"""
import os, sys, math
import numpy as np
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "obstacle"))
YAML = os.path.join(ROOT, "obstacle", "obstacle.yaml")

# guard.py has no ROS deps -> import directly
_g = importlib.util.spec_from_file_location("guard", os.path.join(ROOT, "obstacle", "guard.py"))
guard = importlib.util.module_from_spec(_g); _g.loader.exec_module(guard)
Guard = guard.ObstacleGuard

from depth_nearfield import DepthNearField


def _mk_dnf():
    """Bare DepthNearField with the validated geometry; no camera/thread."""
    d = DepthNearField.__new__(DepthNearField)
    d.ls_mount_height = 1.30
    theta = math.radians(47.6)
    d._ct, d._st = math.cos(theta), math.sin(theta)
    d.cam_h = 1.30
    d.ground_clear = 0.08
    d.max_height = 1.8
    d.min_range = 0.3
    d.max_range = 2.5
    d.ring_bin_deg = 5.0
    d.ring_n = 72
    d.ring_start_deg = -180.0
    d.ring_fov_half = 43.0
    d.ring_min_pts = 2
    d.kth = 4
    return d


def _pt(fwd0, left0, height, d):
    """Inverse of _compute_ring's derotation: world (fwd,left,height) -> input cloud xyz
    (lidar_source level frame with baked mount height)."""
    up0 = height - d.cam_h
    X = fwd0 * d._ct - up0 * d._st
    Zc = fwd0 * d._st + up0 * d._ct
    return [X, left0, Zc + d.ls_mount_height]


def _sector(ang_deg):
    return int(math.floor((ang_deg - (-180.0)) / 5.0)) % 72


def test_ring_empty_and_floor():
    d = _mk_dnf()
    out, n = d._compute_ring(np.zeros((0, 3), np.float32))
    assert np.isnan(out).all() and n == 0, "empty -> all NaN"
    # floor-only: many points at height ~0 (below ground_clear) -> no obstacle sector
    floor = np.array([_pt(fw, lf, 0.0, d)
                      for fw in np.arange(0.5, 2.4, 0.05) for lf in np.arange(-0.6, 0.6, 0.05)],
                     np.float32)
    out, n = d._compute_ring(floor)
    assert np.isnan(out).all(), "floor must NOT produce obstacle sectors (no false positive)"
    print("PASS  empty + floor-only -> no reading")


def test_ring_detects_near_obstacle():
    d = _mk_dnf()
    # a 20 cm-tall small object at 1.0 m ahead (ang 0), plus floor around it
    obj = np.array([_pt(1.0 + dx, dy, h, d)
                    for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)
                    for h in (0.12, 0.16, 0.20)], np.float32)
    floor = np.array([_pt(fw, lf, 0.0, d)
                      for fw in np.arange(0.5, 2.4, 0.08) for lf in np.arange(-0.6, 0.6, 0.08)],
                     np.float32)
    out, n = d._compute_ring(np.vstack([floor, obj]))
    s = _sector(0.0)
    assert np.isfinite(out[s]), f"forward sector {s} should report the object"
    assert abs(out[s] - 1.0) < 0.15, f"reported dist {out[s]:.2f} ~ 1.0 m"
    print(f"PASS  near 20cm object @1.0m -> sector {s} reads {out[s]:.2f} m")


def test_ring_fov_clip():
    d = _mk_dnf()
    # object at ~70 deg left (outside the 43 deg half-FOV) -> must NOT report
    ang = math.radians(70.0)
    obj = np.array([_pt(1.0 * math.cos(ang), 1.0 * math.sin(ang), h, d)
                    for h in (0.15, 0.18, 0.2)] * 3, np.float32)
    out, _ = d._compute_ring(obj)
    assert np.isnan(out).all(), "object outside camera FOV must not report"
    print("PASS  object outside FOV -> no reading (no phantom)")


def test_merge_fail_safe_and_monotonic():
    m = Guard._merge_depth_ring
    base = {"n": 4, "bin_deg": 90.0, "start_deg": -180.0, "dist": [2.0, None, 1.5, None]}
    # depth closer in s0, present where lidar None in s1, farther in s2, None in s3
    dep = {"n": 4, "bin_deg": 90.0, "start_deg": -180.0, "dist": [1.0, 0.8, 3.0, None]}
    out = m(base, dep)
    assert out["dist"] == [1.0, 0.8, 1.5, None], f"min-merge wrong: {out['dist']}"
    # depth never CLEARS: a lidar reading is never removed / increased
    for i, (b, o) in enumerate(zip(base["dist"], out["dist"])):
        if b is not None:
            assert o is not None and o <= b, f"sector {i}: depth cleared/raised lidar"
    # geometry mismatch -> lidar returned unchanged (identity)
    bad = {"n": 8, "bin_deg": 45.0, "start_deg": -180.0, "dist": [0.1] * 8}
    assert m(base, bad) is base, "geometry mismatch must return the lidar ring unchanged"
    assert m(base, None) is base and m(base, {}) is base, "None/empty -> unchanged"
    # input not mutated
    assert base["dist"] == [2.0, None, 1.5, None], "must not mutate the shm ring"
    print("PASS  merge: min-merge, never-clear, geometry-guarded, non-mutating")


if __name__ == "__main__":
    test_ring_empty_and_floor()
    test_ring_detects_near_obstacle()
    test_ring_fov_clip()
    test_merge_fail_safe_and_monotonic()
    print("\nALL DEPTH-RING TESTS PASS")
