#!/usr/bin/env python3
"""Load the REAL obstacle-perception code (obstacle/obstacle_node.py) WITHOUT ROS,
so its detection math can be unit-tested / fuzzed against synthetic point clouds.

obstacle_node imports rclpy at module load; we stub the three ROS modules it needs,
then build a BARE ObstacleNode via __new__ (skipping __init__/super) and set exactly
the tuning attributes its pure-math methods read. This exercises the ACTUAL
_ground_mask / _fit_floor / _robust_nearest / _ring_distances code paths
-- the ones that decide whether an obstacle is detected -- not a reimplementation.

Use:
    from perception_harness import make_node, detect, cloud_box, cloud_points
    node = make_node()                       # real methods, config-tuned attrs
    res  = detect(node, cloud)               # full pipeline: ground-mask -> ring + wedges
    # res = {front,left,right,back, ring(list), n_in, n_keep}
"""
import os
import sys
import types
import importlib.util

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_ros():
    if "rclpy" in sys.modules and getattr(sys.modules["rclpy"], "_stub", False):
        return
    rclpy = types.ModuleType("rclpy"); rclpy._stub = True
    rclpy.init = lambda *a, **k: None
    node_mod = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, *a, **k):
            pass
    node_mod.Node = _Node
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.qos_profile_sensor_data = None
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_mod
    sys.modules["rclpy.qos"] = qos_mod


def load_obstacle_node():
    _stub_ros()
    path = os.path.join(ROOT, "obstacle", "obstacle_node.py")
    spec = importlib.util.spec_from_file_location("obstacle_node_real", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_node(overrides=None):
    """Bare ObstacleNode (no ROS) with attrs sourced from obstacle.yaml + DEFAULTS."""
    mod = load_obstacle_node()
    cfg = dict(mod.DEFAULTS)
    try:
        y = yaml.safe_load(open(os.path.join(ROOT, "obstacle", "obstacle.yaml"))) or {}
        for k, v in y.items():
            if v is not None:
                cfg[k] = v
    except OSError:
        pass
    if overrides:
        cfg.update(overrides)

    n = mod.ObstacleNode.__new__(mod.ObstacleNode)
    # mirror the attribute set obstacle_node.__init__ builds (only what the pure
    # detection methods read).
    n.range_m = float(cfg["obstacle_range_m"])
    n.blind = float(cfg["blind_radius_m"])
    n.sensor_h = float(cfg["sensor_height_m"])
    n.ground_removal = bool(cfg["ground_removal"])
    n.ground_clear = float(cfg["ground_clearance_m"])
    n.ground_band = float(cfg["ground_band_m"])
    n.ground_min_pts = max(3, int(cfg["ground_min_points"]))
    n.floor_seed_pct = float(cfg.get("floor_seed_percentile", 25.0))
    n.floor_inlier_band = float(cfg.get("floor_inlier_band_m", 0.12))
    n.floor_fit_iters = max(1, int(cfg.get("floor_fit_iters", 3)))
    n.max_above = float(cfg["obstacle_max_above_m"])
    n.self_front = float(cfg["self_front_m"])
    n.self_back = float(cfg["self_back_m"])
    n.self_half_w = float(cfg["self_half_width_m"])
    n.self_mask_max_h = float(cfg.get("self_mask_max_height_m", 0.6))
    n.min_h = float(cfg["min_height_m"])
    n.max_h = float(cfg["max_height_m"])
    n.k_cluster = max(1, int(cfg["min_cluster_points"]))
    n.ring_bin_deg = float(cfg["ring_bin_deg"])
    n.ring_min = max(1, int(cfg["ring_min_points"]))
    n.tripwire_range = float(cfg.get("tripwire_range_m", 1.6))
    n.tripwire_min = max(1, int(cfg.get("tripwire_min_points", 1)))
    n.ring_n = max(1, int(round(360.0 / n.ring_bin_deg)))
    n.ring_start_deg = -180.0
    n.floor_abc = None
    n.cfg = cfg
    # precision pipeline attrs (mirror obstacle_node.__init__). Occupancy is stateful
    # across detect() calls, since sim_perception reuses one node for a whole run;
    # de-skew is identity here (no ROS/IMU / per-point times in the harness).
    n.use_occupancy = bool(cfg.get("use_occupancy", True)) and mod.PolarOccupancyGrid is not None
    n.use_deskew = False
    n._deskewer = None
    n._last_proc_t = None
    n.occ = None
    if n.use_occupancy:
        n.occ = mod.PolarOccupancyGrid(
            n_sectors=n.ring_n, r_max=n.range_m,
            r_bin=float(cfg.get("occ_r_bin_m", 0.05)),
            start_deg=n.ring_start_deg, tau_s=float(cfg.get("occ_decay_s", 0.7)),
            l_high=float(cfg.get("occ_l_high", 1.5)))
    return n


def detect(node, pts, dt=0.1):
    """Run the REAL ground-mask + ring + 4-wedge detection on a (N,3) cloud.
    Returns dict with the wedge distances, the full ring, and point counts.
    Mirrors obstacle_node._on_cloud steps 1-3 + 5b/5c exactly (incl. the log-odds
    occupancy finalize; `dt` drives the occupancy decay across calls)."""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        ring0, st0, pr0 = node._finalize_ring([None] * node.ring_n, dt)
        return dict(front=None, left=None, right=None, back=None,
                    ring=ring0, ring_state=st0, ring_prob=pr0, n_in=0, n_keep=0)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    horiz = np.hypot(x, y)
    if node.ground_removal:
        keep = node._ground_mask(x, y, z, horiz)
    else:
        keep = ((horiz >= node.blind) & (horiz <= node.range_m)
                & (z >= node.min_h) & (z <= node.max_h))
    x, y, z, horiz = x[keep], y[keep], z[keep], horiz[keep]
    ang = np.degrees(np.arctan2(y, x))
    front = node._robust_nearest(horiz[np.abs(ang) < 45.0])
    left = node._robust_nearest(horiz[(ang >= 45.0) & (ang < 135.0)])
    right = node._robust_nearest(horiz[(ang <= -45.0) & (ang > -135.0)])
    back = node._robust_nearest(horiz[np.abs(ang) >= 135.0])
    raw_ring = node._ring_distances(ang, horiz)
    ring, state, prob = node._finalize_ring(raw_ring, dt)
    return dict(front=front, left=left, right=right, back=back, ring=ring,
                ring_state=state, ring_prob=prob,
                n_in=len(pts), n_keep=int(keep.sum()))


# ---- synthetic cloud builders -------------------------------------------------
def cloud_box(cx, cy, w, d, z0, z1, density=0.03, jitter=0.0):
    """Filled box of points centred at (cx,cy), width w (y), depth d (x),
    spanning z0..z1. Sensor frame x-fwd/y-left/z-up, origin at sensor."""
    xs = np.arange(cx - d / 2, cx + d / 2 + 1e-9, density)
    ys = np.arange(cy - w / 2, cy + w / 2 + 1e-9, density)
    zs = np.arange(z0, z1 + 1e-9, max(density, 0.02))
    X, Y, Z = np.meshgrid(xs, ys, zs)
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    if jitter:
        p = p + np.random.uniform(-jitter, jitter, p.shape).astype(np.float32)
    return p


def cloud_floor(extent=4.0, density=0.06, z=None, tilt=(0.0, 0.0), sensor_h=1.10):
    """A flat (or tilted) floor patch at z = -sensor_h (+ tilt*x + tilt*y)."""
    g = np.arange(-extent, extent + 1e-9, density)
    X, Y = np.meshgrid(g, g)
    a, b = tilt
    Z = (-sensor_h if z is None else z) + a * X + b * Y
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def cloud_pole(cx, cy, r=0.05, z0=-0.5, z1=1.0, n=400):
    """A thin vertical pole (cylinder shell) at (cx,cy)."""
    th = np.random.uniform(0, 2 * np.pi, n)
    zz = np.random.uniform(z0, z1, n)
    X = cx + r * np.cos(th)
    Y = cy + r * np.sin(th)
    return np.stack([X, Y, zz], axis=1).astype(np.float32)


def cloud_points(lst):
    return np.asarray(lst, dtype=np.float32)


if __name__ == "__main__":
    # smoke test: a box 2 m ahead must be detected in the front wedge + a ring bin.
    node = make_node()
    floor = cloud_floor(sensor_h=node.sensor_h)
    box = cloud_box(2.0, 0.0, 0.6, 0.3, -0.3, 0.8)
    res = detect(node, np.vstack([floor, box]))
    print("front:", res["front"], "ring bins hit:",
          sum(1 for d in res["ring"] if d is not None), "/", node.ring_n)
    assert res["front"] is not None and abs(res["front"] - 1.85) < 0.3, res["front"]
    print("perception harness OK -- real obstacle_node detection reachable without ROS")
