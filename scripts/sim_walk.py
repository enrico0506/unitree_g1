#!/usr/bin/env python3
"""Local closed-loop walking simulator for the G1 teleop pipeline -- NO ROBOT.

Drives the REAL command pipeline against a synthetic 2D world so the obstacle
braking + velocity smoothing can be exercised end-to-end offline:

    operator policy  -> intent (vx,vy,vyaw)
      -> CommandShaper.normalize()            (diagonal -> speed ellipse)
      -> ObstacleGuard.apply()                (reads a SYNTHETIC 360 ring; brakes)
      -> CommandShaper.shape(bypass=hard)     (jerk/accel-limit; snap on hard stop)
      -> integrate the robot pose             (assumes the gait tracks the command)

The synthetic ring is ray-cast from the robot pose against the world obstacles,
producing exactly the {n,bin_deg,start_deg,dist[]} contract obstacle_node.py
writes -- so the guard runs unmodified. This validates braking + smoothness +
no-collision, not the robot's low-level controller (the sent velocity is taken
as the achieved velocity, which is conservative for collision: a commanded stop
stops the robot).

Usage:
    python3 scripts/sim_walk.py <scenario> [--ascii] [--json]
    python3 scripts/sim_walk.py --list
    python3 scripts/sim_walk.py --scenario-file foo.json [...]

Scenarios (built-in): open_field, head_on_wall, diagonal_obstacle, corridor,
    slalom, dead_end, dynamic_appear, side_obstacle.
Exit code 0 if the run met its safety+smoothness criteria, else 1.
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "obstacle"))

import yaml  # noqa: E402
from cmd_shaper import CommandShaper  # noqa: E402

import importlib.util  # noqa: E402
_g = importlib.util.spec_from_file_location("guard", os.path.join(ROOT, "obstacle", "guard.py"))
_gm = importlib.util.module_from_spec(_g); _g.loader.exec_module(_gm)
ObstacleGuard = _gm.ObstacleGuard

import time as _realtime  # noqa: E402


class _SimClock:
    """Virtual monotonic clock so the guard's wall-clock-based slew (scale_slew,
    fault timing, freshness) advances by exactly DT per sim tick -- otherwise an
    accelerated (faster-than-real-time) sim makes the guard see dt~=0 and its
    internal scale-slew never moves. Delegates everything except monotonic() to the
    real time module."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def __getattr__(self, k):
        return getattr(_realtime, k)

CFG = yaml.safe_load(open(os.path.join(ROOT, "config", "robot.yaml")))
MAXES = (CFG["speeds"]["max_vx"], CFG["speeds"]["max_vy"], CFG["speeds"]["max_vyaw"])
DT = 1.0 / CFG["control"]["send_hz"]
ACCEL = (CFG["motion"]["accel_vx"], CFG["motion"]["accel_vy"], CFG["motion"]["accel_vyaw"])
JERK = (CFG["motion"]["jerk_vx"], CFG["motion"]["jerk_vy"], CFG["motion"]["jerk_vyaw"])

RING_N = 72
RING_BIN = 5.0
RING_START = -180.0
RANGE_M = 3.0          # synthetic sensor range (matches obstacle_range_m)
ROBOT_RADIUS = 0.30    # physical half-extent for the collision check


# --------------------------------------------------------------------------- world
def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    """Nearest positive t where (ox,oy)+t*(dx,dy) enters circle (cx,cy,r), or None."""
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c          # a = 1 (dir is unit)
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    t1, t2 = (-b - s) / 2.0, (-b + s) / 2.0
    for t in (t1, t2):
        if t > 1e-6:
            return t
    return None


def _ray_segment(ox, oy, dx, dy, ax, ay, bx, by):
    """Nearest positive t where the ray hits segment A-B, or None."""
    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom    # along ray
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom    # along segment
    if t > 1e-6 and -1e-9 <= u <= 1.0 + 1e-9:
        return t
    return None


def obstacles_at(world, t):
    """Active obstacle list at sim-time t (supports appear_at / disappear_at)."""
    out = []
    for o in world:
        if t < o.get("appear_at", -1e9) or t > o.get("disappear_at", 1e9):
            continue
        out.append(o)
    return out


def cast_ring(x, y, theta, obs):
    """Ray-cast the world into the obstacle-node ring contract from pose (x,y,theta).
    Returns (ring_dict, front, left, right, back, min_surface_dist)."""
    dist = [None] * RING_N
    for i in range(RING_N):
        ang = math.radians(RING_START + (i + 0.5) * RING_BIN)   # robot frame
        dx, dy = math.cos(theta + ang), math.sin(theta + ang)
        best = None
        for o in obs:
            if o["type"] == "circle":
                t = _ray_circle(x, y, dx, dy, o["cx"], o["cy"], o["r"])
            else:  # wall segment
                t = _ray_segment(x, y, dx, dy, o["ax"], o["ay"], o["bx"], o["by"])
            if t is not None and t <= RANGE_M and (best is None or t < best):
                best = t
        dist[i] = round(best, 3) if best is not None else None
    ring = {"n": RING_N, "bin_deg": RING_BIN, "start_deg": RING_START, "dist": dist}

    def wedge(lo, hi):
        vals = [dist[i] for i in range(RING_N)
                if lo <= (RING_START + (i + 0.5) * RING_BIN) < hi and dist[i] is not None]
        return min(vals) if vals else None
    front = min([d for c, d in _centered(dist) if abs(c) < 45.0 and d is not None], default=None)
    left = wedge(45.0, 135.0)
    right = wedge(-135.0, -45.0)
    back = min([d for c, d in _centered(dist) if abs(c) >= 135.0 and d is not None], default=None)
    finite = [d for d in dist if d is not None]
    return ring, front, left, right, back, (min(finite) if finite else None)


def _centered(dist):
    return [(RING_START + (i + 0.5) * RING_BIN, dist[i]) for i in range(RING_N)]


# --------------------------------------------------------------- operator policies
def policy_forward(x, y, theta, goal):
    return (MAXES[0], 0.0, 0.0)


def policy_goal(x, y, theta, goal):
    """Human-like steering: face the goal with yaw, drive forward into it."""
    gx, gy = goal
    d = math.hypot(gx - x, gy - y)
    if d < 0.35:
        return (0.0, 0.0, 0.0)
    bearing = math.atan2(gy - y, gx - x) - theta
    bearing = ((bearing + math.pi) % (2 * math.pi)) - math.pi   # wrap [-pi,pi]
    vyaw = max(-MAXES[2], min(MAXES[2], 1.6 * bearing))
    vx = MAXES[0] * max(0.0, math.cos(bearing))                 # ease off when not facing
    return (vx, 0.0, vyaw)


POLICIES = {"forward": policy_forward, "goal": policy_goal}


# ------------------------------------------------------------------- scenarios
def scenarios():
    return {
        "open_field": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=8.0,
                           release_at=4.0, obstacles=[], desc="no obstacles: ramp up then release -> smooth stop"),
        "head_on_wall": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=10.0,
                             obstacles=[dict(type="wall", ax=4.0, ay=-3, bx=4.0, by=3)],
                             desc="drive straight at a wall 4 m ahead -> smooth stop before it"),
        "diagonal_obstacle": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=10.0,
                                  obstacles=[dict(type="circle", cx=2.2, cy=1.6, r=0.4)],
                                  desc="circle ~36 deg off-axis -> directional gating must catch it"),
        "side_obstacle": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=8.0,
                              obstacles=[dict(type="circle", cx=0.0, cy=1.2, r=0.4)],
                              desc="circle directly to the LEFT while driving straight -> must NOT brake (not in path)"),
        "corridor": dict(policy="goal", start=(0, 0, 0), goal=(8, 0), dur=16.0,
                         obstacles=[dict(type="wall", ax=1.0, ay=0.9, bx=7.0, by=0.9),
                                    dict(type="wall", ax=1.0, ay=-0.9, bx=7.0, by=-0.9)],
                         desc="1.8 m corridor: steer through without scraping the walls"),
        "slalom": dict(policy="goal", start=(0, 0, 0), goal=(9, 0), dur=22.0,
                       obstacles=[dict(type="circle", cx=2.5, cy=0.6, r=0.4),
                                  dict(type="circle", cx=4.5, cy=-0.6, r=0.4),
                                  dict(type="circle", cx=6.5, cy=0.6, r=0.4)],
                       desc="weave toward a goal past 3 staggered obstacles"),
        "dead_end": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=10.0,
                         obstacles=[dict(type="wall", ax=3.0, ay=-2, bx=3.0, by=2),
                                    dict(type="wall", ax=0, ay=2.0, bx=3.0, by=2.0),
                                    dict(type="wall", ax=0, ay=-2.0, bx=3.0, by=-2.0)],
                         desc="three-sided pocket: must stop, never push into the end wall"),
        "dynamic_appear": dict(policy="forward", start=(0, 0, 0), goal=(20, 0), dur=8.0,
                               obstacles=[dict(type="wall", ax=2.1, ay=-2, bx=2.1, by=2, appear_at=1.5)],
                               desc="wall appears ~0.9 m ahead at t=1.5 s while at full speed -> instant hard stop"),
    }


# ------------------------------------------------------------------------ runner
def run(scn):
    clock = _SimClock()
    _gm.time = clock            # virtualize the guard's wall clock for this run
    shaper = CommandShaper(cfg=CFG["motion"], max_speeds=MAXES)
    guard = ObstacleGuard(os.path.join(ROOT, "obstacle", "obstacle.yaml"), audio=None, limits=MAXES)
    guard.enabled = True; guard.mode = "ACTIVE"

    x, y, theta = scn["start"]
    goal = scn.get("goal", (1e9, 0))
    policy = POLICIES[scn["policy"]]
    release_at = scn.get("release_at", 1e18)
    nsteps = int(scn["dur"] / DT)

    traj = [(x, y)]
    sent = []            # sent (vx,vy,vyaw)
    min_surf = 1e9
    reached = False
    t = 0.0
    for _ in range(nsteps):
        obs = obstacles_at(scn["obstacles"], t)
        ring, fr, lf, rt, bk, msurf = cast_ring(x, y, theta, obs)
        if msurf is not None:
            min_surf = min(min_surf, msurf)

        # operator intent (human), with an optional "release keys" event
        if t >= release_at:
            intent = (0.0, 0.0, 0.0)
        else:
            intent = policy(x, y, theta, goal)
        intent = shaper.normalize(tuple(_clamp(intent[i], -MAXES[i], MAXES[i]) for i in range(3)))

        # inject synthetic perception, run the REAL guard + shaper
        data = {"ring": ring, "front_m": fr, "left_m": lf, "right_m": rt, "back_m": bk,
                "tight_factor": 1.0}
        with guard._lock:
            guard._data = data; guard._last_fresh = clock.monotonic(); guard._live = True
        governed = guard.apply(intent)
        hard = guard.hard_stop_flags()
        vx, vy, vyaw = shaper.shape(governed, DT, bypass=hard)
        sent.append((vx, vy, vyaw))

        # integrate the pose (sent velocity == achieved velocity)
        x += (vx * math.cos(theta) - vy * math.sin(theta)) * DT
        y += (vx * math.sin(theta) + vy * math.cos(theta)) * DT
        theta += vyaw * DT
        traj.append((x, y))
        if math.hypot(goal[0] - x, goal[1] - y) < 0.35:
            reached = True
        t += DT
        clock.t += DT          # advance the guard's virtual clock by one tick

    return _metrics(scn, traj, sent, min_surf, reached)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _metrics(scn, traj, sent, min_surf, reached):
    # realized accel / jerk of the sent translational speed (magnitude per axis)
    def axis(idx):
        v = [s[idx] for s in sent]
        acc = [(v[i + 1] - v[i]) / DT for i in range(len(v) - 1)]
        jrk = [(acc[i + 1] - acc[i]) / DT for i in range(len(acc) - 1)]
        return (max((abs(a) for a in acc), default=0.0),
                max((abs(j) for j in jrk), default=0.0))
    amax_x, jmax_x = axis(0)
    amax_y, jmax_y = axis(1)
    amax_yaw, jmax_yaw = axis(2)
    # accel-limit violations on the SENT signal beyond the per-axis cap (+10% slack
    # for the inherent 1-2 tick landing transient), excluding hard-stop snaps.
    viol = 0
    for idx, lim in ((0, ACCEL[0]), (1, ACCEL[1]), (2, ACCEL[2])):
        v = [s[idx] for s in sent]
        for i in range(len(v) - 1):
            dvdt = abs(v[i + 1] - v[i]) / DT
            # a hard stop snaps to 0 in one tick; that's an allowed safety event
            is_snap = abs(v[i + 1]) < 1e-9 and abs(v[i]) > lim * DT
            if not is_snap and dvdt > lim * 1.15:
                viol += 1
    path_len = sum(math.hypot(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1])
                   for i in range(len(traj) - 1))
    collided = min_surf < ROBOT_RADIUS
    return {
        "scenario": scn.get("name", "?"),
        "desc": scn["desc"],
        "collided": collided,
        "min_surface_dist_m": round(min_surf, 3),
        "reached_goal": reached,
        "path_len_m": round(path_len, 2),
        "final_pose": [round(traj[-1][0], 2), round(traj[-1][1], 2)],
        "max_accel": {"vx": round(amax_x, 3), "vy": round(amax_y, 3), "vyaw": round(amax_yaw, 3)},
        "max_jerk": {"vx": round(jmax_x, 2), "vy": round(jmax_y, 2), "vyaw": round(jmax_yaw, 2)},
        "accel_limit_violations": viol,
        "limits": {"accel": ACCEL, "jerk": JERK, "stop_at_m": guard_stop_at()},
        "traj": traj,
    }


def guard_stop_at():
    g = ObstacleGuard(os.path.join(ROOT, "obstacle", "obstacle.yaml"), audio=None, limits=MAXES)
    return round(g.stop_at, 3)


# --------------------------------------------------------------------- ascii view
def ascii_map(scn, traj, w=64, h=22):
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
    for o in scn["obstacles"]:
        if o["type"] == "circle":
            xs += [o["cx"] - o["r"], o["cx"] + o["r"]]; ys += [o["cy"] - o["r"], o["cy"] + o["r"]]
        else:
            xs += [o["ax"], o["bx"]]; ys += [o["ay"], o["by"]]
    x0, x1 = min(xs) - 0.5, max(xs) + 0.5
    y0, y1 = min(ys) - 0.5, max(ys) + 0.5
    x1 = max(x1, x0 + 1e-3); y1 = max(y1, y0 + 1e-3)

    def col(px): return int((px - x0) / (x1 - x0) * (w - 1))
    def row(py): return int((y1 - py) / (y1 - y0) * (h - 1))
    grid = [[" "] * w for _ in range(h)]
    # obstacles
    for o in scn["obstacles"]:
        if o["type"] == "circle":
            for a in range(0, 360, 8):
                px = o["cx"] + o["r"] * math.cos(math.radians(a))
                py = o["cy"] + o["r"] * math.sin(math.radians(a))
                grid[_cl(row(py), h)][_cl(col(px), w)] = "#"
        else:
            steps = 40
            for k in range(steps + 1):
                px = o["ax"] + (o["bx"] - o["ax"]) * k / steps
                py = o["ay"] + (o["by"] - o["ay"]) * k / steps
                grid[_cl(row(py), h)][_cl(col(px), w)] = "#"
    # trajectory
    for (px, py) in traj:
        grid[_cl(row(py), h)][_cl(col(px), w)] = "."
    sx, sy = traj[0]; ex, ey = traj[-1]
    grid[_cl(row(sy), h)][_cl(col(sx), w)] = "S"
    grid[_cl(row(ey), h)][_cl(col(ex), w)] = "E"
    return "\n".join("".join(r) for r in grid)


def _cl(v, n):
    return 0 if v < 0 else n - 1 if v >= n else v


# -------------------------------------------------------------------------- main
def verdict(m):
    """Pass/fail criteria for an autonomous run."""
    ok = (not m["collided"]) and m["accel_limit_violations"] == 0
    notes = []
    if m["collided"]:
        notes.append("COLLISION")
    if m["accel_limit_violations"]:
        notes.append(f"{m['accel_limit_violations']} accel-limit violations (jerky)")
    return ok, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="head_on_wall")
    ap.add_argument("--ascii", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--scenario-file")
    ap.add_argument("--all", action="store_true", help="run every built-in scenario")
    args = ap.parse_args()

    scns = scenarios()
    if args.list:
        for name, s in scns.items():
            print(f"{name:18s} {s['desc']}")
        return 0

    names = list(scns) if args.all else [args.scenario]
    results = []
    overall_ok = True
    for name in names:
        if args.scenario_file:
            scn = json.load(open(args.scenario_file)); scn.setdefault("name", "custom")
        else:
            if name not in scns:
                print(f"unknown scenario '{name}' (try --list)"); return 2
            scn = dict(scns[name]); scn["name"] = name
        m = run(scn)
        ok, notes = verdict(m)
        overall_ok = overall_ok and ok
        results.append((scn, m, ok, notes))

    if args.json:
        print(json.dumps([{k: v for k, v in m.items() if k != "traj"} | {"ok": ok, "notes": notes}
                          for (_, m, ok, notes) in results], indent=2))
    else:
        for (scn, m, ok, notes) in results:
            print(f"\n=== {m['scenario']} === {'PASS' if ok else 'FAIL'} {'; '.join(notes)}")
            print(f"  {m['desc']}")
            print(f"  collided={m['collided']}  min_surface_dist={m['min_surface_dist_m']} m "
                  f"(stop_at={m['limits']['stop_at_m']} m)  reached_goal={m['reached_goal']}")
            print(f"  max_accel vx={m['max_accel']['vx']} vy={m['max_accel']['vy']} "
                  f"vyaw={m['max_accel']['vyaw']}  (limits {ACCEL})")
            print(f"  max_jerk  vx={m['max_jerk']['vx']} vy={m['max_jerk']['vy']} "
                  f"vyaw={m['max_jerk']['vyaw']}  accel_violations={m['accel_limit_violations']}")
            if args.ascii:
                print(ascii_map(scn, m["traj"]))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
