#!/usr/bin/env python3
"""REAL-PERCEPTION closed-loop simulator -- the high-fidelity successor to
sim_walk.py. Instead of an idealized ray-cast RING, this ray-casts a faithful 3D
Livox-style point cloud from the robot pose each tick and runs the ACTUAL
obstacle_node detection (ground-mask + 360 ring + wedges, via perception_harness),
then the real guard + real shaper, then integrates the pose. NO ROS, NO robot.

So the FULL pipeline is exercised closed-loop with the robot moving:
    world (3D obstacles + floor)
      -> livox_cloud()        faithful angular ray-cast (density falls with range;
                              optional gaussian noise + decimation + blind cone)
      -> obstacle_node detect()   REAL ground-plane fit + ring + wedges
      -> ObstacleGuard.apply()    REAL braking + hard stops + predictor
      -> CommandShaper            REAL jerk/accel smoothing
      -> integrate pose
This catches perception<->control interaction bugs the idealized sim cannot:
sparse-return misses while moving, floor-fit corruption near obstacles, the front
blind cone, detection latency vs closing speed, noise-induced phantom stops, etc.

Run:  python3 scripts/sim_perception.py <scenario> [--noise 0.02] [--ascii] [--json]
      python3 scripts/sim_perception.py --selftest
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "obstacle"))

import yaml  # noqa: E402
from cmd_shaper import CommandShaper  # noqa: E402
from perception_harness import make_node, detect  # noqa: E402
import importlib.util  # noqa: E402

_g = importlib.util.spec_from_file_location("guard", os.path.join(ROOT, "obstacle", "guard.py"))
_gm = importlib.util.module_from_spec(_g); _g.loader.exec_module(_gm)
ObstacleGuard = _gm.ObstacleGuard
import time as _realtime  # noqa: E402


class _SimClock:
    """Virtual monotonic clock so the guard's wall-clock slew advances by DT/tick."""
    def __init__(self): self.t = 1000.0
    def monotonic(self): return self.t
    def __getattr__(self, k): return getattr(_realtime, k)


CFG = yaml.safe_load(open(os.path.join(ROOT, "config", "robot.yaml")))
MAXES = (CFG["speeds"]["max_vx"], CFG["speeds"]["max_vy"], CFG["speeds"]["max_vyaw"])
DT = 1.0 / CFG["control"]["send_hz"]
ACCEL = (CFG["motion"]["accel_vx"], CFG["motion"]["accel_vy"], CFG["motion"]["accel_vyaw"])
ROBOT_RADIUS = 0.30


# --------------------------------------------------------------- faithful LiDAR
def livox_cloud(x, y, theta, obs, sensor_h,
                az_step=0.4, el_lo=-16.0, el_hi=28.0, el_step=1.2,
                max_range=8.0, noise=0.0, decimate=1, rng=None,
                pitch=0.0, roll=0.0, dropout=0.0):
    """Ray-cast a 3D point cloud from the sensor at world (x, y, sensor_h), heading
    theta. Returns (N,3) in the SENSOR frame (x-fwd / y-left / z-up, origin at the
    sensor, floor at z=-sensor_h) -- exactly what obstacle_node consumes.

    Obstacles: circle -> vertical cylinder (radius r, height h, default 1.2 m);
    wall -> vertical panel (segment in xy, height h). The floor (world z=0) is hit
    by downward rays. Point DENSITY falls with range (rays are angular), so far/thin
    objects yield few returns -- the realism that drives sparse-detection behaviour."""
    az = np.deg2rad(np.arange(-180.0, 180.0, az_step))
    el = np.deg2rad(np.arange(el_lo, el_hi + 1e-9, el_step))
    AZ, EL = np.meshgrid(az, el)
    AZ = AZ.ravel(); EL = EL.ravel()
    ce = np.cos(EL)
    # world-frame ray directions (unit)
    dwx = ce * np.cos(theta + AZ)
    dwy = ce * np.sin(theta + AZ)
    dwz = np.sin(EL)

    tmin = np.full(AZ.shape, np.inf)
    # floor (world z = 0): sensor_h + t*dwz = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(dwz < -1e-6, -sensor_h / dwz, np.inf)
    tmin = np.minimum(tmin, tf)

    for o in obs:
        if o["type"] == "circle":
            t = _ray_cylinder(x, y, sensor_h, dwx, dwy, dwz,
                              o["cx"], o["cy"], o["r"], o.get("h", 1.2))
        else:
            t = _ray_panel(x, y, sensor_h, dwx, dwy, dwz, o)
        tmin = np.minimum(tmin, t)

    hit = np.isfinite(tmin) & (tmin > 0.05) & (tmin < max_range)
    t = tmin[hit]
    if t.size == 0:
        return np.zeros((0, 3), np.float32)
    hwx = x + t * dwx[hit]; hwy = y + t * dwy[hit]; hwz = sensor_h + t * dwz[hit]
    # world -> sensor frame (rotate by -theta about z, origin at sensor)
    dx = hwx - x; dy = hwy - y
    ct, st = math.cos(theta), math.sin(theta)
    rx = ct * dx + st * dy
    ry = -st * dx + ct * dy
    rz = hwz - sensor_h
    # robot SWAY: the sensor pitches/rolls with the gait, tilting the WHOLE cloud (so
    # the floor is no longer flat at z=-sensor_h -- this exercises the tilt-aware floor
    # fit + ground removal under realistic motion). pitch about y, roll about x.
    if pitch != 0.0:
        cp, sp = math.cos(pitch), math.sin(pitch)
        rx, rz = rx * cp + rz * sp, -rx * sp + rz * cp
    if roll != 0.0:
        cr, sr = math.cos(roll), math.sin(roll)
        ry, rz = ry * cr - rz * sr, ry * sr + rz * cr
    cloud = np.stack([rx, ry, rz], axis=1).astype(np.float32)
    r = rng if rng is not None else np.random
    if decimate > 1:
        cloud = cloud[::decimate]
    if dropout > 0.0 and len(cloud):                 # random sensor return dropout
        cloud = cloud[r.random(len(cloud)) >= dropout]
    if noise > 0.0:
        cloud = cloud + r.normal(0.0, noise, cloud.shape).astype(np.float32)
    return cloud


def _ray_cylinder(ox, oy, oh, dx, dy, dz, cx, cy, r, h):
    """t for each ray hitting an infinite-in-z circle (cx,cy,r) THEN z within [0,h]."""
    fx = ox - cx; fy = oy - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    out = np.full(dx.shape, np.inf)
    ok = (disc >= 0.0) & (a > 1e-12)
    s = np.sqrt(np.where(ok, disc, 0.0))
    t1 = np.where(ok, (-b - s) / (2.0 * a), np.inf)
    t2 = np.where(ok, (-b + s) / (2.0 * a), np.inf)
    for t in (t1, t2):
        z = oh + t * dz
        good = ok & (t > 0.05) & (z >= 0.0) & (z <= h)
        out = np.where(good & (t < out), t, out)
    return out


def _ray_panel(ox, oy, oh, dx, dy, dz, o):
    """t for each ray hitting a vertical wall panel: segment (ax,ay)-(bx,by), z in [0,h]."""
    ax, ay, bx, by = o["ax"], o["ay"], o["bx"], o["by"]
    h = o.get("h", 1.2)
    ex = bx - ax; ey = by - ay
    denom = dx * ey - dy * ex
    out = np.full(dx.shape, np.inf)
    ok = np.abs(denom) > 1e-12
    t = np.where(ok, ((ax - ox) * ey - (ay - oy) * ex) / np.where(ok, denom, 1.0), np.inf)
    u = np.where(ok, ((ax - ox) * dy - (ay - oy) * dx) / np.where(ok, denom, 1.0), np.inf)
    z = oh + t * dz
    good = ok & (t > 0.05) & (u >= 0.0) & (u <= 1.0) & (z >= 0.0) & (z <= h)
    out = np.where(good, t, out)
    return out


# ------------------------------------------------------------------- operator
def policy_forward(x, y, theta, goal):
    return (MAXES[0], 0.0, 0.0)


def policy_goal(x, y, theta, goal):
    gx, gy = goal
    if math.hypot(gx - x, gy - y) < 0.35:
        return (0.0, 0.0, 0.0)
    bearing = ((math.atan2(gy - y, gx - x) - theta + math.pi) % (2 * math.pi)) - math.pi
    vyaw = max(-MAXES[2], min(MAXES[2], 1.6 * bearing))
    vx = MAXES[0] * max(0.0, math.cos(bearing))
    return (vx, 0.0, vyaw)


POLICIES = {"forward": policy_forward, "goal": policy_goal}


def obstacles_at(world, t):
    """Active obstacles at sim-time t, with MOVING obstacles advanced by their
    velocity (ovx, ovy m/s) -- people crossing / approaching / receding."""
    out = []
    for o in world:
        if not (o.get("appear_at", -1e9) <= t <= o.get("disappear_at", 1e9)):
            continue
        ovx, ovy = o.get("ovx", 0.0), o.get("ovy", 0.0)
        if ovx == 0.0 and ovy == 0.0:
            out.append(o); continue
        m = dict(o); dt_x, dt_y = ovx * t, ovy * t
        if o["type"] == "circle":
            m["cx"] += dt_x; m["cy"] += dt_y
        else:
            m["ax"] += dt_x; m["ay"] += dt_y; m["bx"] += dt_x; m["by"] += dt_y
        out.append(m)
    return out


# ------------------------------------------------------------------- scenarios
def scenarios():
    W = lambda ax, ay, bx, by, **k: dict(type="wall", ax=ax, ay=ay, bx=bx, by=by, **k)
    C = lambda cx, cy, r, **k: dict(type="circle", cx=cx, cy=cy, r=r, **k)
    return {
        "open_field": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=8.0, obstacles=[]),
        "wall": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                     obstacles=[W(4, -3, 4, 3)]),
        "pole": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                     obstacles=[C(3.0, 0.0, 0.06, h=1.4)]),
        "thin_pole": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                          obstacles=[C(2.5, 0.0, 0.03, h=1.2)]),
        "diagonal": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                         obstacles=[C(2.2, 1.4, 0.35, h=1.2)]),
        "corridor": dict(policy="goal", start=(0, 0, 0), goal=(8, 0), dur=18.0,
                         obstacles=[W(1, 0.9, 7, 0.9), W(1, -0.9, 7, -0.9)]),
        "slalom": dict(policy="goal", start=(0, 0, 0), goal=(9, 0), dur=24.0,
                       obstacles=[C(2.5, 0.6, 0.35, h=1.2), C(4.5, -0.6, 0.35, h=1.2),
                                  C(6.5, 0.6, 0.35, h=1.2)]),
        "dynamic": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=8.0,
                        obstacles=[W(2.2, -2, 2.2, 2, appear_at=1.5)]),
        "low_box": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                        obstacles=[dict(type="wall", ax=3, ay=-2, bx=3, by=2, h=0.35)]),
        # --- dynamic (moving) obstacles: ovx/ovy m/s ---
        "crossing": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=10.0,
                         obstacles=[C(3.0, -2.0, 0.30, h=1.7, ovx=0.0, ovy=0.9)]),
        "approaching": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=10.0,
                            obstacles=[C(5.0, 0.0, 0.30, h=1.7, ovx=-0.8, ovy=0.0)]),
        "overtaking": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=12.0,
                           obstacles=[C(-1.0, 0.2, 0.30, h=1.7, ovx=2.2, ovy=0.0)]),
    }


# ------------------------------------------------------------------------ runner
def run(scn, noise=0.0, decimate=1, el_lo=-16.0, gap_follow=False, recovery=False,
        seed=0, sway_amp=0.0, sway_freq=1.2, dropout=0.0):
    rng = np.random.RandomState(seed)
    clock = _SimClock(); _gm.time = clock
    node = make_node()
    shaper = CommandShaper(cfg=CFG["motion"], max_speeds=MAXES)
    guard = ObstacleGuard(os.path.join(ROOT, "obstacle", "obstacle.yaml"), audio=None, limits=MAXES)
    guard.enabled = True; guard.mode = "ACTIVE"
    guard.set_gap_follow(gap_follow); guard.set_recovery(recovery)
    sensor_h = node.sensor_h

    x, y, theta = scn["start"]
    goal = scn.get("goal", (1e9, 0))
    policy = POLICIES[scn["policy"]]
    release_at = scn.get("release_at", 1e18)
    nsteps = int(scn["dur"] / DT)

    traj = [(x, y)]; sent = []; min_surf = 1e9; reached = False
    detected_any = False; fault = False; t = 0.0
    for _ in range(nsteps):
        obs = obstacles_at(scn["obstacles"], t)
        # gait sway: pitch + (phase-shifted) roll oscillation, radians.
        pitch = math.radians(sway_amp) * math.sin(2 * math.pi * sway_freq * t)
        roll = math.radians(sway_amp) * math.sin(2 * math.pi * sway_freq * t + 1.3)
        cloud = livox_cloud(x, y, theta, obs, sensor_h, el_lo=el_lo,
                            noise=noise, decimate=decimate, rng=rng,
                            pitch=pitch, roll=roll, dropout=dropout)
        res = detect(node, cloud)
        ring = {"n": node.ring_n, "bin_deg": node.ring_bin_deg,
                "start_deg": node.ring_start_deg, "dist": res["ring"]}
        if any(d is not None for d in res["ring"]):
            detected_any = True

        # true nearest obstacle surface distance (ground truth, for the collision check)
        msurf = _true_clearance(x, y, obs)
        if msurf is not None:
            min_surf = min(min_surf, msurf)

        intent = policy(x, y, theta, goal) if t < release_at else (0.0, 0.0, 0.0)
        intent = shaper.normalize(tuple(_clamp(intent[i], -MAXES[i], MAXES[i]) for i in range(3)))
        data = {"ring": ring, "front_m": res["front"], "left_m": res["left"],
                "right_m": res["right"], "back_m": res["back"], "tight_factor": 1.0,
                "gap": {"state": "FOLLOW", "center_deg": 0.0, "passable": True,
                        "yaw_cmd": 0.0, "vy_cmd": 0.0, "turn_factor": 1.0}}
        with guard._lock:
            guard._data = data; guard._last_fresh = clock.monotonic(); guard._live = True
        governed = guard.apply(intent)
        hard = guard.hard_stop_flags()
        vx, vy, vyaw = shaper.shape(governed, DT, bypass=hard)
        sent.append((vx, vy, vyaw))

        # robot-FAULT collision: in contact AND the robot was driving TOWARD the
        # obstacle (a stopped robot that a moving obstacle walks into is NOT a fault).
        if msurf is not None and msurf < ROBOT_RADIUS:
            wvx = vx * math.cos(theta) - vy * math.sin(theta)
            wvy = vx * math.sin(theta) + vy * math.cos(theta)
            nx, ny = _nearest_dir(x, y, obs)
            if nx is not None and (wvx * nx + wvy * ny) > 0.1:   # closing on the obstacle
                fault = True

        x += (vx * math.cos(theta) - vy * math.sin(theta)) * DT
        y += (vx * math.sin(theta) + vy * math.cos(theta)) * DT
        theta += vyaw * DT
        traj.append((x, y))
        if math.hypot(goal[0] - x, goal[1] - y) < 0.35:
            reached = True
        t += DT
        clock.t += DT

    return _metrics(scn, traj, sent, min_surf, reached, detected_any, fault)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _true_clearance(x, y, obs):
    best = None
    for o in obs:
        if o.get("h", 1.2) < 0.10:        # too low to be a body-collision (e.g. a curb)
            continue
        if o["type"] == "circle":
            d = math.hypot(o["cx"] - x, o["cy"] - y) - o["r"]
        else:
            d = _pt_seg(x, y, o["ax"], o["ay"], o["bx"], o["by"])
        best = d if best is None else min(best, d)
    return best


def _pt_seg(px, py, ax, ay, bx, by):
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    tt = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / L2))
    cx, cy = ax + tt * ex, ay + tt * ey
    return math.hypot(px - cx, py - cy)


def _nearest_dir(x, y, obs):
    """Unit direction from the robot toward the nearest body-height obstacle surface."""
    best = None; bd = 1e18
    for o in obs:
        if o.get("h", 1.2) < 0.10:
            continue
        if o["type"] == "circle":
            d = math.hypot(o["cx"] - x, o["cy"] - y) - o["r"]
            cx, cy = o["cx"], o["cy"]
        else:
            d = _pt_seg(x, y, o["ax"], o["ay"], o["bx"], o["by"])
            ex, ey = o["bx"] - o["ax"], o["by"] - o["ay"]; L2 = ex * ex + ey * ey
            tt = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - o["ax"]) * ex + (y - o["ay"]) * ey) / L2))
            cx, cy = o["ax"] + tt * ex, o["ay"] + tt * ey
        if d < bd:
            bd = d; dx, dy = cx - x, cy - y
            n = math.hypot(dx, dy) or 1.0
            best = (dx / n, dy / n)
    return best if best else (None, None)


def _metrics(scn, traj, sent, min_surf, reached, detected_any, fault=False):
    v = [s[0] for s in sent]
    acc = [(v[i + 1] - v[i]) / DT for i in range(len(v) - 1)]
    jrk = [abs(acc[i + 1] - acc[i]) / DT for i in range(len(acc) - 1)]
    viol = 0
    for idx, lim in ((0, ACCEL[0]), (1, ACCEL[1]), (2, ACCEL[2])):
        vv = [s[idx] for s in sent]
        for i in range(len(vv) - 1):
            snap = abs(vv[i + 1]) < 1e-9 and abs(vv[i]) > lim * DT
            if not snap and abs(vv[i + 1] - vv[i]) / DT > lim * 1.15:
                viol += 1
    return {
        "scenario": scn.get("name", "?"),
        "collided": min_surf < ROBOT_RADIUS,
        "robot_fault_collision": fault,    # robot DROVE into it (vs an obstacle moving into a stopped robot)
        "min_surface_dist_m": round(min_surf, 3) if min_surf < 1e8 else None,
        "detected_obstacle": detected_any,
        "reached_goal": reached,
        "max_accel_vx": round(max((abs(a) for a in acc), default=0.0), 2),
        "max_jerk_vx": round(max(jrk, default=0.0), 1),
        "accel_violations": viol,
        "final_pose": [round(traj[-1][0], 2), round(traj[-1][1], 2)],
        "traj": traj,
    }


def ascii_map(scn, traj, w=70, h=22):
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
    for o in scn["obstacles"]:
        if o["type"] == "circle":
            xs += [o["cx"] - o["r"], o["cx"] + o["r"]]; ys += [o["cy"] - o["r"], o["cy"] + o["r"]]
        else:
            xs += [o["ax"], o["bx"]]; ys += [o["ay"], o["by"]]
    x0, x1 = min(xs) - 0.5, max(xs) + 0.5; y0, y1 = min(ys) - 0.5, max(ys) + 0.5
    x1 = max(x1, x0 + 1e-3); y1 = max(y1, y0 + 1e-3)
    cl = lambda v, n: 0 if v < 0 else n - 1 if v >= n else v
    col = lambda px: cl(int((px - x0) / (x1 - x0) * (w - 1)), w)
    row = lambda py: cl(int((y1 - py) / (y1 - y0) * (h - 1)), h)
    g = [[" "] * w for _ in range(h)]
    for o in scn["obstacles"]:
        if o["type"] == "circle":
            for a in range(0, 360, 10):
                g[row(o["cy"] + o["r"] * math.sin(math.radians(a)))][col(o["cx"] + o["r"] * math.cos(math.radians(a)))] = "#"
        else:
            for k in range(41):
                g[row(o["ay"] + (o["by"] - o["ay"]) * k / 40)][col(o["ax"] + (o["bx"] - o["ax"]) * k / 40)] = "#"
    for (px, py) in traj:
        g[row(py)][col(px)] = "."
    g[row(traj[0][1])][col(traj[0][0])] = "S"
    g[row(traj[-1][1])][col(traj[-1][0])] = "E"
    return "\n".join("".join(r) for r in g)


# -------------------------------------------------------------------------- main
def selftest():
    """Validate the LiDAR + closed loop against known-good expectations."""
    node = make_node(); sh = node.sensor_h; ok = True
    def c(n, cond):
        nonlocal ok; print(("PASS" if cond else "FAIL") + "  " + n); ok = ok and cond
    # floor only -> no detection
    cloud = livox_cloud(0, 0, 0, [], sh)
    c("floor-only cloud -> NO detection", not any(d is not None for d in detect(node, cloud)["ring"]))
    # wall 2 m ahead -> front ~2 m
    cloud = livox_cloud(0, 0, 0, [dict(type="wall", ax=2, ay=-2, bx=2, by=2)], sh)
    fr = detect(node, cloud)["front"]
    c("wall @2m -> front detected ~2m", fr is not None and abs(fr - 2.0) < 0.4)
    # pole 3 m ahead -> detected
    cloud = livox_cloud(0, 0, 0, [dict(type="circle", cx=3, cy=0, r=0.06, h=1.4)], sh)
    c("6cm pole @3m -> detected", detect(node, cloud)["front"] is not None)
    # closed loop: drive at wall -> stops, no collision
    m = run(dict(name="wall", **scenarios()["wall"]))
    c("closed-loop wall: detected", m["detected_obstacle"])
    c("closed-loop wall: NO collision", not m["collided"])
    c("closed-loop wall: smooth (0 accel violations)", m["accel_violations"] == 0)
    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="wall")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--el-lo", type=float, default=-16.0, help="lowest ray elevation (deg); raise to model the front blind cone")
    ap.add_argument("--sway", type=float, default=0.0, help="gait sway amplitude (deg pitch/roll)")
    ap.add_argument("--dropout", type=float, default=0.0, help="random sensor return dropout fraction 0..1")
    ap.add_argument("--gap", action="store_true"); ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--ascii", action="store_true"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--scenario-file")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    scns = scenarios()
    if args.list:
        for k in scns: print(k)
        return 0
    names = list(scns) if args.all else [args.scenario]
    out = []
    for name in names:
        if args.scenario_file:
            scn = json.load(open(args.scenario_file)); scn.setdefault("name", "custom")
        else:
            scn = dict(scns[name]); scn["name"] = name
        m = run(scn, noise=args.noise, decimate=args.decimate, el_lo=args.el_lo,
                gap_follow=args.gap, recovery=args.recovery,
                sway_amp=args.sway, dropout=args.dropout)
        out.append((scn, m))
    if args.json:
        print(json.dumps([{k: v for k, v in m.items() if k != "traj"} for (_, m) in out], indent=2))
    else:
        for (scn, m) in out:
            verdict = "PASS" if (not m["robot_fault_collision"] and m["accel_violations"] == 0) else "FAIL"
            print(f"\n=== {m['scenario']} === {verdict}")
            print(f"  fault_collision={m['robot_fault_collision']} (raw_collided={m['collided']}) "
                  f"min_surf={m['min_surface_dist_m']} detected={m['detected_obstacle']} "
                  f"reached={m['reached_goal']}  max_accel={m['max_accel_vx']} max_jerk={m['max_jerk_vx']} viol={m['accel_violations']}")
            if args.ascii:
                print(ascii_map(scn, m["traj"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
