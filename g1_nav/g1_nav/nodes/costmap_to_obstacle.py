#!/usr/bin/env python3
"""costmap_to_obstacle -- Nav2 local costmap -> /dev/shm/g1_obstacle.json.

WHAT THIS IS
------------
A Foxy-native rclpy bridge that turns the Nav2 LOCAL COSTMAP into the EXACT same
shared-memory obstacle contract the original raw-lidar node wrote
(/home/unitree/projects/g1/obstacle/obstacle_node.py), so the EXISTING dashboard
ObstacleGuard (/home/unitree/projects/g1/obstacle/guard.py), the UI overlay and the
voice feedback are REUSED UNCHANGED -- we only swap the obstacle *source* from the
raw Mid-360 cloud to the costmap.

WHY THE COSTMAP INSTEAD OF RAW LIDAR
------------------------------------
The costmap is built from the GROUND-REMOVED cloud (/patchwork/nonground) with STVL
voxel MEMORY (blind-spot persistence) and footprint-clearing of the robot's own legs
(update_footprint_enabled). It therefore sees thin/low obstacles, remembers them when
they leave the live FOV, and does not self-mark the robot's stance -- all things the
raw single-frame wedge scan could not do. This node simply reads that costmap.

PIPELINE
--------
  driver (Mid-360 CustomMsg) -> FAST-LIO -> /cloud_registered (frame camera_init)
    -> ground_seg -> /patchwork/nonground (frame camera_init, floor removed)
    -> Nav2 local costmap (STVL; global_frame camera_init, robot_base_frame body)
    -> THIS NODE: /local_costmap/costmap + TF camera_init->body
                  -> per-direction nearest obstacle -> /dev/shm/g1_obstacle.json

FRAMES
------
FAST-LIO's NATIVE frames are used directly: camera_init (world/global) and body
(IMU/robot). NO odom/base_link rename, NO livox_frame. The costmap is published in
camera_init; we look up the robot pose via TF camera_init->body, rotate each lethal
cell into the ROBOT frame (x fwd, y LEFT -- matching the contract), and bin it into
front/back/left/right wedges (~+/-45 deg each).

ROBUSTNESS
----------
The node must come up and KEEP WRITING even before any costmap or TF arrives: until
data is present it writes an explicit "all clear" frame (all distances null, all
scales 1.0, zone CLEAR). It never raises on a missing costmap, missing TF, or a
costmap that is momentarily empty. The shm write is atomic (tmp + os.replace) so the
guard never reads a partial file.

SCALE / TIGHT LOGIC is copied VERBATIM (same constants, same maths) from
obstacle_node.py so the guard's braking behaviour is identical: a smoothstep
distance->scale over [stop_at_m, slow_at_m] with a slow_min_scale floor, plus a
tight_factor = average of the four directional closenesses floored at tight_min.
Constants are read from /home/unitree/projects/g1/obstacle/obstacle.yaml with baked
defaults so a missing file/key never crashes the node.

GAP (steering) is intentionally NEUTRAL here -- this is a teleop guard, gap-follow is
off. We keep the gap keys present (state FOLLOW, all commands 0, passable true,
turn_factor 1.0) so the overlay/guard find them. profile is an empty list (tolerated).
"""

import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from nav_msgs.msg import OccupancyGrid
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)


SHM_PATH = "/dev/shm/g1_obstacle.json"
# obstacle.yaml is the SHARED source of tuning (also read by guard.py). Pin to the
# original obstacle/ dir so scale/tight behaviour matches the raw-lidar node exactly.
YAML_PATH = "/home/unitree/projects/g1/obstacle/obstacle.yaml"

# Baked defaults -- a missing yaml file or key must NEVER crash the node. These mirror
# obstacle.yaml; only the subset this node actually uses (distance->scale + tight).
DEFAULTS = {
    "obstacle_range_m": 3.0,
    "slow_at_m": 1.6,
    "stop_at_m": 0.6,
    "slow_min_scale": 0.5,
    "filter_alpha": 0.25,
    "ease_scale": True,
    "tight_open_m": 1.2,
    "tight_min": 0.7,
    "min_cluster_points": 10,   # k for the robust k-th-nearest cell pick
}

# Frames + topics (parameters so the launch can override without code edits).
DEFAULT_GLOBAL_FRAME = "camera_init"   # FAST-LIO world frame (costmap global_frame)
DEFAULT_ROBOT_FRAME = "body"           # FAST-LIO robot frame (costmap robot_base_frame)
DEFAULT_COSTMAP_TOPIC = "/local_costmap/costmap"
LETHAL_THRESHOLD = 99                  # OccupancyGrid value >= this == a lethal cell
PUBLISH_HZ = 10.0                      # shm-write rate (independent of costmap rate)


def load_cfg():
    """Load obstacle.yaml over the baked defaults; never raise."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except (OSError, yaml.YAMLError):
        pass  # defaults are fine -- the node must still come up
    return cfg


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _r(v):
    """Round a distance to 3 dp, preserving None (the 'clear' sentinel)."""
    return None if v is None else round(float(v), 3)


def _yaw_from_quat(qx, qy, qz, qw):
    """Yaw (rotation about z) from a quaternion -- enough for a planar pose."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class CostmapToObstacle(Node):
    def __init__(self):
        super().__init__("costmap_to_obstacle")

        # --- tuning (copied logic + constants from obstacle_node.py) ----------
        c = load_cfg()
        self.range_m = float(c["obstacle_range_m"])
        self.slow_at = float(c["slow_at_m"])
        self.stop_at = float(c["stop_at_m"])
        self.slow_min = float(c["slow_min_scale"])
        self.alpha = float(c["filter_alpha"])
        self.ease = bool(c["ease_scale"])
        self.tight_open = float(c["tight_open_m"])
        self.tight_min = float(c["tight_min"])
        self.k_cluster = max(1, int(c["min_cluster_points"]))

        # --- frames / topic (overridable params) ------------------------------
        self.global_frame = str(
            self.declare_parameter("global_frame", DEFAULT_GLOBAL_FRAME).value)
        self.robot_frame = str(
            self.declare_parameter("robot_base_frame", DEFAULT_ROBOT_FRAME).value)
        costmap_topic = str(
            self.declare_parameter("costmap_topic", DEFAULT_COSTMAP_TOPIC).value)
        self.lethal = int(self.declare_parameter("lethal_threshold", LETHAL_THRESHOLD).value)
        pub_hz = float(self.declare_parameter("publish_hz", PUBLISH_HZ).value)

        # --- per-direction EMA state (None until a first reading per dir) ------
        self.seq = 0
        self.front_filt = None
        self.back_filt = None
        self.left_filt = None
        self.right_filt = None

        # --- latest costmap (kept; we process on a timer, not on every update) -
        self._grid = None

        # --- TF: camera_init -> body ------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- subscribe to the local costmap -----------------------------------
        # Nav2 publishes the costmap with a latched (transient_local) QoS so a late
        # subscriber still gets the current grid. Match it: reliable + transient_local.
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            OccupancyGrid, costmap_topic, self._on_costmap, costmap_qos)

        # --- process timer (~10 Hz, independent of costmap update rate) --------
        period = 1.0 / pub_hz if pub_hz > 0 else 0.1
        self.create_timer(period, self._tick)

        # write one 'all clear' frame immediately so the guard sees a live file
        # even before the first costmap / TF arrives.
        self._write_clear()

        self.get_logger().info(
            f"costmap_to_obstacle: {costmap_topic} (frame {self.global_frame}) + "
            f"TF {self.global_frame}->{self.robot_frame} -> {SHM_PATH} | "
            f"range={self.range_m}m slow={self.slow_at} stop={self.stop_at} "
            f"lethal>={self.lethal} @ {pub_hz:.0f}Hz")

    # ------------------------------------------------------------------ inputs
    def _on_costmap(self, msg):
        """Cache the latest costmap. Heavy work happens on the timer."""
        self._grid = msg

    def _lookup_pose(self):
        """Robot pose (x, y, yaw) in the global frame from TF, or None.

        Uses the latest available transform (Time() = 0) so we never extrapolate;
        tolerant of TF not being up yet (returns None -> 'all clear' this tick).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.0))
        except (LookupException, ConnectivityException,
                ExtrapolationException, Exception):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return float(t.x), float(t.y), _yaw_from_quat(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------- tick
    def _tick(self):
        """Once per period: costmap + TF -> per-direction nearest -> shm.

        On any missing input (no costmap yet, no TF yet) we still write a fully
        valid 'all clear' frame so the dashboard guard stays fresh (never faults
        for staleness just because the costmap is warming up).
        """
        grid = self._grid
        if grid is None:
            self._write_clear()
            return

        pose = self._lookup_pose()
        if pose is None:
            self._write_clear()
            return

        front_m, back_m, left_m, right_m = self._directional_nearest(grid, pose)
        self._write_frame(front_m, back_m, left_m, right_m)

    # ----------------------------------------------------- costmap -> distances
    def _directional_nearest(self, grid, pose):
        """Per-direction robust nearest LETHAL-cell distance in the ROBOT frame.

        Convert every lethal cell's world (x, y) into the robot frame (rotate by
        -yaw about the robot, translate by -robot_xy), keep those within
        obstacle_range_m, bin by atan2(ry, rx) into the four ~+/-45 deg wedges, and
        take the robust k-th-smallest hypot per wedge (or plain min when a wedge has
        fewer than k cells). Returns (front, back, left, right) metres or None.
        Sign convention matches the contract: x fwd, +y LEFT (front=+x, left=+y,
        right=-y, back=-x).
        """
        info = grid.info
        res = float(info.resolution)
        w = int(info.width)
        h = int(info.height)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        if res <= 0.0 or w == 0 or h == 0:
            return None, None, None, None

        data = np.asarray(grid.data, dtype=np.int16)
        if data.size != w * h:
            return None, None, None, None

        # indices of lethal cells
        lethal_idx = np.nonzero(data >= self.lethal)[0]
        if lethal_idx.size == 0:
            return None, None, None, None

        # cell (row, col) -> world centre (x, y) in the global (costmap) frame
        col = (lethal_idx % w).astype(np.float64)
        row = (lethal_idx // w).astype(np.float64)
        wx = ox + (col + 0.5) * res
        wy = oy + (row + 0.5) * res

        rx0, ry0, yaw = pose
        # translate into a robot-centred frame, then rotate by -yaw so +x is robot
        # forward and +y is robot LEFT.
        dx = wx - rx0
        dy = wy - ry0
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        rx = cy * dx + sy * dy          # forward (+x)
        ry = -sy * dx + cy * dy         # left (+y)

        dist = np.hypot(rx, ry)
        in_range = dist <= self.range_m
        if not np.any(in_range):
            return None, None, None, None
        rx = rx[in_range]
        ry = ry[in_range]
        dist = dist[in_range]

        ang = np.degrees(np.arctan2(ry, rx))   # 0 = ahead, + = left, - = right
        front = self._robust_nearest(dist[np.abs(ang) < 45.0])
        left = self._robust_nearest(dist[(ang >= 45.0) & (ang < 135.0)])
        right = self._robust_nearest(dist[(ang <= -45.0) & (ang > -135.0)])
        back = self._robust_nearest(dist[np.abs(ang) >= 135.0])
        return front, back, left, right

    def _robust_nearest(self, dists):
        """The k-th smallest distance (k = min_cluster_points) for robustness, or
        the plain min when a wedge has fewer than k cells, or None when empty.

        (Costmap cells are already inflation/voxel-filtered, so requiring k full
        cells like the raw-lidar node would too often drop a real thin obstacle;
        we therefore fall back to min for sparse wedges instead of returning None.)
        """
        n = dists.size
        if n == 0:
            return None
        if n < self.k_cluster:
            return float(np.min(dists))
        return float(np.partition(dists, self.k_cluster - 1)[self.k_cluster - 1])

    # ------------------------------------------------------- scale / tight maths
    # (copied VERBATIM from obstacle_node.py so the guard behaves identically)
    def _ema(self, state, raw):
        """One EMA step. A null (clear) reading feeds obstacle_range_m so the filter
        relaxes toward 'open' instead of sticking at the last close reading."""
        feed = raw if raw is not None else self.range_m
        if state is None:
            return feed
        return self.alpha * feed + (1.0 - self.alpha) * state

    def _dist_scale(self, d):
        """Filtered distance -> 0..1 speed scale. Smoothstep over [stop_at, slow_at]
        with a slow_min floor above stop_at; 0 at/under stop_at. Null relaxes to
        obstacle_range_m (clear)."""
        if d is None:
            d = self.range_m
        if d <= self.stop_at:
            return 0.0
        span = self.slow_at - self.stop_at
        t = 1.0 if span <= 0 else clamp((d - self.stop_at) / span, 0.0, 1.0)
        e = t * t * (3.0 - 2.0 * t) if self.ease else t
        return self.slow_min + (1.0 - self.slow_min) * e

    def _tight_factor(self, *dists):
        """1.0 when open, down to tight_min only when closed-in on MULTIPLE sides.
        closeness per direction is 0 (open)..1 (very close); average the four so a
        single near wall barely moves it."""
        span = self.tight_open - self.stop_at
        if span <= 0:
            return 1.0
        total = 0.0
        for d in dists:
            if d is None:
                d = self.range_m
            total += clamp((self.tight_open - d) / span, 0.0, 1.0)
        tightness = total / len(dists)
        return 1.0 - tightness * (1.0 - self.tight_min)

    # ------------------------------------------------------------- shm writers
    def _neutral_gap(self):
        """Minimal/neutral gap dict -- gap-follow steering is OFF for this teleop
        guard; we keep the keys present so the overlay/guard find them."""
        return {
            "state": "FOLLOW",
            "center_deg": 0.0,
            "passable": True,
            "yaw_cmd": 0.0,
            "vy_cmd": 0.0,
            "turn_factor": 1.0,
        }

    def _write_clear(self):
        """Write a fully valid 'all clear' frame (no costmap/TF yet, or empty).

        Distances null, EMA relaxed to range, all scales 1.0, zone CLEAR. Still
        advances the EMA toward 'open' so the first real reading isn't a step."""
        self.front_filt = self._ema(self.front_filt, None)
        self.back_filt = self._ema(self.back_filt, None)
        self.left_filt = self._ema(self.left_filt, None)
        self.right_filt = self._ema(self.right_filt, None)
        self._emit(None, None, None, None,
                   self.front_filt, self.back_filt, self.left_filt, self.right_filt)

    def _write_frame(self, front_m, back_m, left_m, right_m):
        """Real frame: EMA the four raw distances, then emit the full contract."""
        self.front_filt = self._ema(self.front_filt, front_m)
        self.back_filt = self._ema(self.back_filt, back_m)
        self.left_filt = self._ema(self.left_filt, left_m)
        self.right_filt = self._ema(self.right_filt, right_m)
        self._emit(front_m, back_m, left_m, right_m,
                   self.front_filt, self.back_filt, self.left_filt, self.right_filt)

    def _emit(self, front_m, back_m, left_m, right_m,
              front_filt, back_filt, left_filt, right_filt):
        """Assemble + atomically write the JSON contract the dashboard guard reads."""
        scale_front = self._dist_scale(front_filt)
        scale_back = self._dist_scale(back_filt)
        scale_left = self._dist_scale(left_filt)
        scale_right = self._dist_scale(right_filt)
        speed_scale = scale_front                # back-compat: speed_scale == scale_front

        # zone is driven by the (filtered) FRONT distance, as in obstacle_node.py.
        ff = front_filt if front_filt is not None else self.range_m
        if ff >= self.slow_at:
            zone = "CLEAR"
        elif ff <= self.stop_at:
            zone = "STOP"
        else:
            zone = "SLOW"

        tight_factor = self._tight_factor(
            front_filt, back_filt, left_filt, right_filt)

        self.seq += 1
        out = {
            "seq": self.seq,
            "stamp": round(time.time(), 3),
            "front_m": _r(front_m),
            "front_filt_m": _r(front_filt),
            "left_m": _r(left_m),
            "right_m": _r(right_m),
            "back_m": _r(back_m),
            "back_filt_m": _r(back_filt),
            "left_filt_m": _r(left_filt),
            "right_filt_m": _r(right_filt),
            "zone": zone,
            "speed_scale": round(speed_scale, 3),
            "scale_front": round(scale_front, 3),
            "scale_back": round(scale_back, 3),
            "scale_left": round(scale_left, 3),
            "scale_right": round(scale_right, 3),
            "tight_factor": round(tight_factor, 3),
            "side": {"left": left_m is not None, "right": right_m is not None},
            "gap": self._neutral_gap(),
            "profile": [],
        }
        self._write_shm(out)

    def _write_shm(self, obj):
        """Atomic write: tmp file then os.replace (guard never reads a partial)."""
        tmp = SHM_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, SHM_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CostmapToObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
