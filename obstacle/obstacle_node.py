#!/usr/bin/env python3
"""G1 obstacle perception node (per-direction distances + 360 deg ring).

Reads the Livox Mid-360 cloud (/livox/lidar), already x-fwd / y-left / z-up with
the origin at the sensor (the driver applies the 180 deg mount roll, so we DO NOT
re-rotate here -- the floor simply sits at z ~= -sensor_height_m), and turns each
frame into the perception contract the dashboard guard consumes:

  * per-direction robust nearest distance (front / left / right / back),
  * an EMA-smoothed front distance -> zone (CLEAR/SLOW/STOP) + speed_scale,
  * a full-circle 360 deg ring of per-sector distances (directional gating + viz).

Everything is published atomically to a single shared-memory JSON file
(/dev/shm/g1_obstacle.json) every frame; the guard reads it by mtime. Mirrors the
shm + atomic-write pattern of g1_mapping/map_bridge.py.

Runs under ROS2 Foxy (domain 99) with env.sh sourced. Subscribes with best-effort
sensor QoS and auto-detects PointCloud2 vs livox CustomMsg at startup.

Sign convention (the UI must match): angle = atan2(y, x): 0 = straight ahead,
+ = LEFT, - = RIGHT. The ring's sector i center = start_deg + (i+0.5)*bin_deg.
"""

import json
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

SHM_PATH = "/dev/shm/g1_obstacle.json"
YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacle.yaml")

# CONTRACT defaults -- baked in so a missing yaml file/key never crashes the node.
DEFAULTS = {
    "topic": "/livox/lidar",
    "sensor_height_m": 1.10,
    "obstacle_range_m": 3.0,
    "slow_at_m": 1.6,
    "stop_at_m": 0.6,
    "safety_bubble_m": 0.35,   # arm-clearance bubble added to slow_at + stop_at (all directions)
    "slow_min_scale": 0.5,
    "min_height_m": -0.4,
    "max_height_m": 0.50,
    "blind_radius_m": 0.45,
    "ground_removal": True,        # tilt-aware floor cut + footprint self-mask (false = old band)
    "ground_clearance_m": 0.08,    # keep points sticking up MORE than this above the fitted floor
    "ground_band_m": 0.5,          # floor-candidate points are below sensor_height+this (plane fit)
    "ground_min_points": 80,       # need >= this many candidates to fit a plane (else assume level)
    "floor_seed_percentile": 25.0, # robust floor fit: seed plane offset from this z-percentile (low=floor)
    "floor_inlier_band_m": 0.12,   # robust floor fit: refit inliers within +/- this of the current plane
    "floor_fit_iters": 3,          # robust floor fit: number of seed-then-refit iterations
    "obstacle_max_above_m": 1.9,   # cut points higher than this above the floor (ceiling/overhead)
    "self_front_m": 0.35,          # footprint self-mask: forward extent (covers leg swing)
    "self_back_m": 0.35,           # footprint self-mask: rearward extent
    "self_half_width_m": 0.30,     # footprint self-mask: half-width (covers the leg stance)
    "self_mask_max_height_m": 0.6, # self-mask ONLY below this height above floor (legs/feet); overhead kept
    "min_cluster_points": 10,
    "filter_alpha": 0.25,
    "ease_scale": True,
    "tight_open_m": 1.2,
    "tight_min": 0.7,
    "scale_slew_per_s": 1.8,
    "announce_cooldown_s": 3.0,
    "speaker_id": 0,
    "led_warnings": True,
    "stale_timeout_s": 1.0,
    "startup_grace_s": 8.0,
    "fault_stop_s": 3.0,
    "ring_bin_deg": 5.0,           # full-circle ring: 360/this = sector count
    "ring_min_points": 3,          # points needed before a ring sector reads a distance
    "tripwire_range_m": 1.6,       # NEAR-FIELD tripwire: a sector with >=tripwire_min_points within
                                   #   this range reports the nearest one even below ring_min_points
    "tripwire_min_points": 2,      # min returns inside tripwire_range to fire the near-field tripwire
    "enabled_default": False,
    "publish_hz": 10,
    # --- obstacle-point export for the dashboard 3D sphere -------------------
    # Ship the ACTUAL kept obstacle points (post ground/self/range filter) so the
    # operator sees exactly what the guard perceives -- including low things on the
    # ground -- and, by their absence, what it is blind to. viz frame: x fwd, y left,
    # z = height ABOVE the fitted floor (floor = 0), so ground-level returns sit near 0.
    "viz_points": True,            # write the "points" array into the shm contract
    "viz_max_points": 3000,        # cap (strided decimation) -> bounded JSON/WS payload
                                   #   (~2.5k kept/frame here, so 3000 sends them all = densest)
    # --- rolling accumulation for the RING + 3D sphere -----------------------
    # The Mid-360 scans NON-REPETITIVELY: one ~100 ms frame paints only a thin
    # "petal", so per-frame the 2D ring + 3D sphere show a sparse arc that SWEEPS
    # the circle like radar (and flickers as near-empty petals arrive). Merge the
    # last viz_accum_s of frames before computing the ring + viz points so both
    # show a full, steady cloud. The 4 safety wedges + zone stay PER-FRAME (front
    # stop latency unchanged); only the ring + points use the merged set. The ring
    # feeds the guard's 360 deg gating, so this ALSO densifies that gating --
    # fail-safe: a cleared sector lingers up to viz_accum_s (conservative, never
    # blind). 0 = off (old per-frame sweeping behaviour).
    "viz_accum_s": 0.6,            # rolling merge window in seconds (0 disables)
    "viz_accum_max_frames": 12,    # hard cap on buffered frames (memory/CPU bound)
}


def load_cfg():
    """Load obstacle.yaml over the baked-in defaults; never raise."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if v is not None:
                cfg[k] = v
    except (OSError, yaml.YAMLError):
        pass  # defaults are fine -- the node must still come up
    return cfg


def pc2_xyz(msg):
    """Parse x,y,z (float32) out of a PointCloud2 without sensor_msgs_py.

    Verbatim from g1_mapping/map_bridge.py (the validated offset-driven parser).
    """
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ("x", "y", "z")):
        return np.zeros((0, 3), np.float32)
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

    def col(name):
        o = offs[name]
        return raw[:, o:o + 4].copy().view("<f4").reshape(-1)

    return np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)


def custom_xyz(msg, decimate):
    """Build (N,3) from a livox CustomMsg's .points list (the SLOW fallback)."""
    pts = msg.points
    if decimate > 1:
        pts = pts[::decimate]
    if not pts:
        return np.zeros((0, 3), np.float32)
    return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class ObstacleNode(Node):
    def __init__(self):
        super().__init__("g1_obstacle")
        self.cfg = load_cfg()
        c = self.cfg

        # --- tuning, pulled out as plain floats (cheap to read each frame) ----
        self.topic = str(c["topic"])
        self.range_m = float(c["obstacle_range_m"])
        # safety bubble (arm clearance): slow/stop this much EARLIER in every direction.
        self.bubble = float(c.get("safety_bubble_m", 0.0))
        self.slow_at = float(c["slow_at_m"]) + self.bubble
        self.stop_at = float(c["stop_at_m"]) + self.bubble
        self.slow_min = float(c["slow_min_scale"])
        self.min_h = float(c["min_height_m"])
        self.max_h = float(c["max_height_m"])
        self.blind = float(c["blind_radius_m"])
        self.sensor_h = float(c["sensor_height_m"])
        self.ground_removal = bool(c["ground_removal"])
        self.ground_clear = float(c["ground_clearance_m"])
        self.ground_band = float(c["ground_band_m"])
        self.ground_min_pts = max(3, int(c["ground_min_points"]))
        self.floor_seed_pct = float(c.get("floor_seed_percentile", 25.0))
        self.floor_inlier_band = float(c.get("floor_inlier_band_m", 0.12))
        self.floor_fit_iters = max(1, int(c.get("floor_fit_iters", 3)))
        self.max_above = float(c["obstacle_max_above_m"])
        self.self_front = float(c["self_front_m"])
        self.self_back = float(c["self_back_m"])
        self.self_half_w = float(c["self_half_width_m"])
        self.self_mask_max_h = float(c.get("self_mask_max_height_m", 0.6))
        self.k_cluster = max(1, int(c["min_cluster_points"]))
        self.alpha = float(c["filter_alpha"])
        self.ease = bool(c["ease_scale"])
        self.tight_open = float(c["tight_open_m"])
        self.tight_min = float(c["tight_min"])
        # full-circle ring geometry (360 deg sectors)
        self.ring_bin_deg = float(c["ring_bin_deg"])
        self.ring_min = max(1, int(c["ring_min_points"]))
        self.tripwire_range = float(c.get("tripwire_range_m", 1.6))
        self.tripwire_min = max(1, int(c.get("tripwire_min_points", 1)))
        self.ring_n = max(1, int(round(360.0 / self.ring_bin_deg)))
        self.ring_start_deg = -180.0   # bin 0 center = start + 0.5*bin_deg; shared by node + UI
        # obstacle-point export for the dashboard 3D sphere
        self.viz_points = bool(c.get("viz_points", True))
        self.viz_max_points = max(0, int(c.get("viz_max_points", 1500)))
        # rolling accumulation window (RING + viz points only; see DEFAULTS). The
        # deque holds (stamp, x, y, z, ang, horiz) per frame; maxlen bounds memory.
        self.accum_s = max(0.0, float(c.get("viz_accum_s", 0.0)))
        self.accum_max = max(1, int(c.get("viz_accum_max_frames", 12)))
        self._accum = deque(maxlen=self.accum_max)

        # --- per-frame state --------------------------------------------------
        self.seq = 0
        self.front_filt = None        # EMA of the front distance (None until first reading)
        self.back_filt = None         # EMA of the back distance
        self.left_filt = None         # EMA of the left distance
        self.right_filt = None        # EMA of the right distance
        self.first_frame = True       # gate the one-shot mount-frame sanity check
        self.floor_abc = None         # last fitted floor plane (a, b, c), for the sanity print

        # --- front-cone diagnostic (opt-in; G1_OBS_FRONT_DIAG=1) ---------------
        # Traces front-cone (|ang|<45 deg) points through each filter stage so we
        # can see WHY the front reads clear: never arriving (blind), cut as
        # overhead (up-tilt sees the wall above obstacle_max_above_m), cut as
        # floor/self, or present-but-too-sparse for the k-point wedge. Off by
        # default (zero hot-path cost); when on, a compact 'front_diag' is added to
        # the SHM JSON AND logged ~1 Hz. Changes NO perception/safety logic.
        self.front_diag = os.environ.get("G1_OBS_FRONT_DIAG", "0") \
            not in ("0", "", "false", "False", "no")
        self._front_diag_last = None
        self._front_diag_log_seq = 0

        # --- subscribe (auto-detect message type) -----------------------------
        msg_type, kind = self._resolve_msg_type()
        self.kind = kind  # "pc2" or "custom"
        self.create_subscription(
            msg_type, self.topic, self._on_cloud, qos_profile_sensor_data)

        self.get_logger().info(
            f"g1_obstacle: {self.topic} ({kind}) -> {SHM_PATH} | range={self.range_m}m "
            f"slow={self.slow_at} stop={self.stop_at} ring={self.ring_n}sectors")

    # ------------------------------------------------------------------ setup
    def _resolve_msg_type(self):
        """Poll the graph until 'topic' shows up, then pick the message class.

        Returns (msg_class, kind) where kind is 'pc2' or 'custom'. Falls back to
        PointCloud2 if the type never resolves (the common case on this robot).
        """
        from sensor_msgs.msg import PointCloud2

        type_name = None
        deadline = time.time() + 15.0
        while time.time() < deadline:
            for name, types in self.get_topic_names_and_types():
                if name == self.topic and types:
                    type_name = types[0]
                    break
            if type_name is not None:
                break
            time.sleep(0.25)

        if type_name == "livox_ros_driver2/msg/CustomMsg":
            from livox_ros_driver2.msg import CustomMsg
            self.get_logger().warn(
                "using livox CustomMsg SLOW path (per-point Python parse); "
                "will decimate large frames")
            return CustomMsg, "custom"

        if type_name is None:
            self.get_logger().warn(
                f"{self.topic} not advertised yet; assuming PointCloud2")
        return PointCloud2, "pc2"

    # --------------------------------------------------------------- callback
    def _on_cloud(self, msg):
        # ---- 0. message -> (N,3) -------------------------------------------
        if self.kind == "custom":
            # CustomMsg has no zero-copy numpy view, so the per-point Python parse
            # is the slow path. Decimate to ~6000 points -- ample for obstacle
            # distances/gaps and keeps the callback well under the 10 Hz budget on
            # the Jetson. (We use CustomMsg because this driver build's PointCloud2
            # path stalls after one frame.)
            pn = int(getattr(msg, "point_num", 0) or len(msg.points))
            decimate = max(1, pn // 6000)
            pts = custom_xyz(msg, decimate)
        else:
            pts = pc2_xyz(msg)
        n_raw = len(pts)
        if n_raw == 0:
            return

        # ---- 1. finite + spatial filter ------------------------------------
        # Two paths: the tilt-aware ground-plane removal + footprint self-mask
        # (default), or the old [min_height_m, max_height_m] band when
        # ground_removal is false. Both end with x/y/z/horiz = the kept points.
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        horiz = np.hypot(x, y)

        if self.ground_removal:
            keep = self._ground_mask(x, y, z, horiz)
        else:
            # legacy height-band path (kept intact for easy revert)
            keep = (
                (horiz >= self.blind) & (horiz <= self.range_m)
                & (z >= self.min_h) & (z <= self.max_h)
            )

        # opt-in diagnostic: trace the front cone through each filter stage BEFORE
        # the kept-subset reassignment below (needs the full arrays + the floor
        # plane that _ground_mask just fitted into self.floor_abc).
        if self.front_diag:
            self._front_cone_diag(x, y, z, horiz, keep)

        x = x[keep]
        y = y[keep]
        z = z[keep]
        horiz = horiz[keep]
        ang = np.degrees(np.arctan2(y, x))  # 0=ahead, + = left, - = right
        n_keep = len(horiz)

        # ---- 2. first-frame mount-frame sanity check -----------------------
        if self.first_frame:
            self.first_frame = False
            self._mount_sanity_check(z, horiz)

        # ---- 3. wedge robust-nearest distances -----------------------------
        front_m = self._robust_nearest(horiz[np.abs(ang) < 45.0])
        left_m = self._robust_nearest(horiz[(ang >= 45.0) & (ang < 135.0)])
        right_m = self._robust_nearest(horiz[(ang <= -45.0) & (ang > -135.0)])
        back_m = self._robust_nearest(horiz[np.abs(ang) >= 135.0])

        # ---- 4. per-direction EMA distances -> smooth scales + tight_factor -
        # When a direction is clear, feed obstacle_range_m so its filter relaxes
        # back toward "open" instead of sticking at the last close reading.
        self.front_filt = self._ema(self.front_filt, front_m)
        self.back_filt = self._ema(self.back_filt, back_m)
        self.left_filt = self._ema(self.left_filt, left_m)
        self.right_filt = self._ema(self.right_filt, right_m)
        front_filt = self.front_filt
        back_filt = self.back_filt
        left_filt = self.left_filt
        right_filt = self.right_filt

        # smoothstep (or linear) speed scales, one per governed direction.
        scale_front = self._dist_scale(front_filt)
        scale_back = self._dist_scale(back_filt)
        scale_left = self._dist_scale(left_filt)
        scale_right = self._dist_scale(right_filt)
        speed_scale = scale_front  # back-compat: speed_scale == scale_front

        # zone is still driven by the (filtered) FRONT distance, as before.
        if front_filt >= self.slow_at:
            zone = "CLEAR"
        elif front_filt <= self.stop_at:
            zone = "STOP"
        else:
            zone = "SLOW"

        # tight_factor: only drops hard when CLOSED-IN on several sides at once.
        tight_factor = self._tight_factor(
            front_filt, back_filt, left_filt, right_filt)

        # ---- 5a. rolling accumulation for the RING + 3D-sphere viz ----------
        # Merge the last viz_accum_s of frames so the non-repetitive Mid-360 scan
        # fills the whole circle instead of one sweeping petal. ONLY the ring +
        # viz points below use these merged arrays; the wedges/zone above are
        # per-frame, so the safety stop keeps its per-frame latency.
        acc_x, acc_y, acc_z, acc_ang, acc_horiz = self._accumulate(
            x, y, z, ang, horiz)

        # ---- 5b. full-circle 360 deg ring (additive; merged-window points) --
        ring_dist = self._ring_distances(acc_ang, acc_horiz)

        # ---- 6. write the contract -----------------------------------------
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
            "ring": {
                "n": self.ring_n,
                "bin_deg": self.ring_bin_deg,
                "start_deg": self.ring_start_deg,
                "dist": [_r(d) for d in ring_dist],
            },
        }
        # Actual kept obstacle points (what the guard perceives) for the 3D sphere,
        # merged over the accumulation window (matches the ring above).
        if self.viz_points:
            out["points"] = self._viz_points(acc_x, acc_y, acc_z)
        # opt-in front-cone filter-stage trace (G1_OBS_FRONT_DIAG=1).
        if self.front_diag and self._front_diag_last is not None:
            out["front_diag"] = self._front_diag_last
        self._write_shm(out)
        self._print_status(n_raw, n_keep, front_m, left_m, right_m, back_m,
                           zone, speed_scale, tight_factor)

    # ------------------------------------------------------------- helpers
    def _ema(self, state, raw):
        """One EMA step for a direction. A null (clear) reading feeds
        obstacle_range_m so the filter relaxes toward 'open'. Seeds on first use."""
        feed = raw if raw is not None else self.range_m
        if state is None:
            return feed
        return self.alpha * feed + (1.0 - self.alpha) * state

    def _dist_scale(self, d):
        """Filtered distance -> 0..1 speed scale. Above stop_at the scale never drops
        below slow_min, so the robot KEEPS MOVING (slowly) when it has space and only
        fully stops via the hard stop inside stop_at. Eased to a smoothstep S-curve
        when ease_scale is on. A null distance relaxes to obstacle_range_m (clear)."""
        if d is None:
            d = self.range_m
        if d <= self.stop_at:
            return 0.0                       # hard-stop zone (the guard stops anyway)
        span = self.slow_at - self.stop_at
        t = 1.0 if span <= 0 else clamp((d - self.stop_at) / span, 0.0, 1.0)
        e = t * t * (3.0 - 2.0 * t) if self.ease else t
        return self.slow_min + (1.0 - self.slow_min) * e

    def _tight_factor(self, *dists):
        """Global multiplier: 1.0 when open, down to tight_min only when the robot
        is closed-in on MULTIPLE sides. closeness per direction is 0 (open) .. 1
        (very close); we AVERAGE the four so a single near wall barely moves it."""
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

    def _robust_nearest(self, dists):
        """The k-th smallest distance (k = min_cluster_points), or None if a
        wedge has fewer than k points -- robust to a single noisy near return."""
        if len(dists) < self.k_cluster:
            return None
        # np.partition puts the k-th smallest at index k-1 in O(n)
        return float(np.partition(dists, self.k_cluster - 1)[self.k_cluster - 1])

    def _ring_distances(self, ang, horiz):
        """Robust-nearest horizontal distance per full-circle sector.

        Bins ALL kept points into self.ring_n sectors over [-180, 180) and takes
        the ring_min-th smallest hypot per sector (robust to one noisy near
        return), reusing the same np.partition idiom as the direction wedges.
        Returns a length-ring_n list, distance (m) or None per sector. Sector i
        center angle = ring_start_deg + (i + 0.5) * ring_bin_deg, same sign
        convention as everything else (atan2(y, x): 0=ahead, +=LEFT, -=RIGHT).
        """
        dist = [None] * self.ring_n
        if len(ang) == 0:
            return dist
        idx = np.floor((ang - self.ring_start_deg) / self.ring_bin_deg).astype(np.int64)
        idx %= self.ring_n                  # wrap the atan2 +180 edge (-> bin 0)
        for b in range(self.ring_n):
            d = horiz[idx == b]
            val = None
            if len(d) >= self.ring_min:
                # robust k-th-smallest: rejects 1-2 lone noisy near returns.
                val = float(np.partition(d, self.ring_min - 1)[self.ring_min - 1])
            # NEAR-FIELD TRIPWIRE: ALWAYS also take the nearest of any cluster of
            # >=tripwire_min returns within tripwire_range, and report the CLOSER of the
            # two. This catches (a) a wide+sparse obstacle giving <ring_min returns
            # (taut cable, barrier tape, thin railing) AND (b) a near hazard sharing a
            # sector with a FARTHER dense surface -- the k-th-smallest would otherwise
            # mask the near points behind the farther crowd (monotonicity bug). Range-
            # gated to near-field so far noisy returns don't add false positives;
            # tripwire_min=2 keeps lone floor-noise spikes from tripping it.
            near = d[d <= self.tripwire_range]
            if len(near) >= self.tripwire_min:
                nm = float(near.min())
                val = nm if val is None else min(val, nm)
            dist[b] = val
        return dist

    def _fit_floor(self, x, y, z):
        """Robustly fit a floor plane z = a*x + b*y + c to the candidate points.

        Returns (a, b, c). The floor is the LOWEST surface, so a plain (unweighted)
        least-squares fit over all low candidates is dangerous: a dense flat surface
        only just ABOVE the floor (a low overhang / table underside that slipped into
        the candidate band) drags the fitted plane UP and tilts it, after which the
        ground cut deletes the very surface that should have been flagged as a hazard
        (BUG 1). We therefore fit ROBUSTLY toward the lowest points:

          1) Seed the offset from a LOW percentile of z (the floor, not the overhang).
          2) Iterate: keep only points within a thin band ABOVE the current plane and
             at/below it (reject points clearly above -- those belong to the overhang
             or to real obstacles, never to the floor), then refit. A competing
             surface above the floor is rejected after the first iteration instead of
             capturing the plane.

        If the fit is degenerate, NaN, or implausibly tilted/offset, falls back to a
        level floor (a=b=0, c=-sensor_height_m) so a bad fit can never invent or hide
        a floor that clips real obstacles.
        """
        level = (0.0, 0.0, -self.sensor_h)
        if len(z) < self.ground_min_pts:
            return level

        def lstsq_fit(xx, yy, zz):
            A = np.stack([xx, yy, np.ones_like(xx)], axis=1)
            sol, _, _, _ = np.linalg.lstsq(A, zz, rcond=None)
            return float(sol[0]), float(sol[1]), float(sol[2])

        try:
            # --- seed from the LOWEST points so an overhang can't own the plane ---
            # Start level, anchored to a low percentile of z (robust floor height).
            a, b, c = 0.0, 0.0, float(np.percentile(z, self.floor_seed_pct))
            # Iteratively refit only over points near/below the current plane: keep
            # residuals in (-band, +band) but ASYMMETRICALLY reject points clearly
            # above (they are overhang/obstacle, not floor).
            band = self.floor_inlier_band
            for _ in range(self.floor_fit_iters):
                resid = z - (a * x + b * y + c)
                inl = (resid > -band) & (resid < band)
                if int(inl.sum()) < self.ground_min_pts:
                    break
                a, b, c = lstsq_fit(x[inl], y[inl], z[inl])
        except (np.linalg.LinAlgError, ValueError):
            return level

        # sanity-bound the fit: reject implausible tilt or offset.
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            return level
        if abs(a) > 0.9 or abs(b) > 0.9:            # > ~42 deg tilt -> implausible. Bound raised for
            return level                            #   the G1's tilted-up head (~37 deg -> floor slope
                                                    #   a ~= tan(37) ~= 0.75; the old 0.4 (~22deg) wrongly
                                                    #   rejected it and fell back to a level floor).
        if abs(c + self.sensor_h) > 0.8:            # floor offset too far from expected (widened for tilt)
            return level
        return a, b, c

    def _ground_mask(self, x, y, z, horiz):
        """Tilt-aware floor cut + robot footprint self-mask (vectorised).

        1) Fit the local floor as a tilted plane (handles sway) over low,
           in-range candidate points; fall back to a level floor when too few.
        2) Keep points whose height ABOVE that plane is in
           (ground_clearance_m, obstacle_max_above_m) -- the floor itself drops
           but anything sticking up is kept, even if low and close.
        3) Drop points inside the robot's own footprint box (by position, NOT
           height) so its legs/feet don't read as obstacles.
        Stores the fitted plane on self.floor_abc for the sanity print.
        """
        # blind_radius drops the robot's OWN body / harness self-returns at ANY
        # height. The legacy band applied this (horiz >= self.blind); the ground
        # path MUST too, or a harness post / gantry leg taller than
        # self_mask_max_height_m sitting ~0.1 m away reads as a permanent
        # side/back obstacle and pins strafe + reverse to zero forever. Anything
        # within blind_radius of a body-mounted sensor is the robot itself; real
        # obstacles are stopped far outside this (stop_at + bubble ~= 0.95 m).
        in_range = (horiz >= self.blind) & (horiz <= self.range_m)

        # floor candidates: in range AND clearly low (below sensor + band).
        cand = in_range & (z < (-self.sensor_h + self.ground_band))
        a, b, c = self._fit_floor(x[cand], y[cand], z[cand])
        self.floor_abc = (a, b, c)

        above = z - (a * x + b * y + c)   # height of each point above the floor
        keep_ground = (above > self.ground_clear) & (above < self.max_above)

        # robot footprint self-mask: a box around the base. HEIGHT-LIMITED so it only
        # swallows LOW returns (the robot's own legs/feet) -- a real obstacle at
        # chest/head height directly above the footprint column (height above floor
        # >= self_mask_max_h) is a genuine overhead hazard and is KEPT (BUG 2).
        in_self = (
            (x > -self.self_back) & (x < self.self_front)
            & (np.abs(y) < self.self_half_w)
            & (above < self.self_mask_max_h)
        )
        return in_range & keep_ground & (~in_self)

    def _front_cone_diag(self, x, y, z, horiz, keep):
        """DIAGNOSTIC (opt-in via G1_OBS_FRONT_DIAG=1) -- changes NO perception.

        Trace front-cone (|ang| < 45 deg, same wedge as front_m) points through each
        filter stage so we can see WHY the front reads clear. Mirrors the masks in
        _ground_mask, recomputed on the cone subset only (cheap). Writes a compact
        dict to self._front_diag_last (added to the SHM as 'front_diag') and logs it
        ~1 Hz. Reads the floor plane _ground_mask just fitted into self.floor_abc.

        Keys:
          raw         points in the cone (finite, pre-spatial-filter)
          in_range    of those, within [blind_radius, obstacle_range]
          floor_cut   in-range but at/below ground_clearance (floor/legs dropped)
          overhead_cut in-range but >= obstacle_max_above (CEILING cut -- the prime
                       suspect: an up-tilted sensor sees a near wall HIGH)
          self_cut    in-range but inside the footprint self-mask box (low)
          kept        survive ALL stages (what front_m/the wedge actually sees)
          wedge_k     points the wedge needs before it reports (min_cluster_points)
          h_above     [p5, p50, p95] height-above-floor (m) of in-range cone points
          near_horiz  nearest in-range horizontal distance (the wall, if present)
        """
        ang = np.degrees(np.arctan2(y, x))
        cone = np.abs(ang) < 45.0
        n_raw = int(cone.sum())
        hc, zc, xc, yc = horiz[cone], z[cone], x[cone], y[cone]
        a, b, c = self.floor_abc if self.floor_abc else (0.0, 0.0, -self.sensor_h)
        above = zc - (a * xc + b * yc + c)
        in_range = (hc >= self.blind) & (hc <= self.range_m)
        overhead = in_range & (above >= self.max_above)
        floorish = in_range & (above <= self.ground_clear)
        self_box = (
            in_range
            & (xc > -self.self_back) & (xc < self.self_front)
            & (np.abs(yc) < self.self_half_w) & (above < self.self_mask_max_h)
        )
        inr_h = above[in_range]

        def _p(arr, p):
            return round(float(np.percentile(arr, p)), 2) if len(arr) else None

        near = hc[in_range]
        self._front_diag_last = {
            "cone_deg": 45,
            "raw": n_raw,
            "in_range": int(in_range.sum()),
            "floor_cut": int(floorish.sum()),
            "overhead_cut": int(overhead.sum()),
            "self_cut": int(self_box.sum()),
            "kept": int(keep[cone].sum()),
            "wedge_k": self.k_cluster,
            "max_above_m": round(self.max_above, 2),
            "h_above": [_p(inr_h, 5), _p(inr_h, 50), _p(inr_h, 95)],
            "near_horiz": round(float(near.min()), 2) if len(near) else None,
        }
        # throttle the log to ~1 Hz (lidar runs ~10 Hz) so the terminal stays usable.
        if self.seq - self._front_diag_log_seq >= 10:
            self._front_diag_log_seq = self.seq
            d = self._front_diag_last
            self.get_logger().info(
                "FRONT-DIAG cone(+/-45): raw=%d in_range=%d -> kept=%d "
                "(floor_cut=%d overhead_cut=%d self_cut=%d) | wedge needs k=%d | "
                "h_above[p5,p50,p95]=%s (max_above=%.2f) | near=%s m"
                % (d["raw"], d["in_range"], d["kept"], d["floor_cut"],
                   d["overhead_cut"], d["self_cut"], d["wedge_k"],
                   d["h_above"], d["max_above_m"], d["near_horiz"]))

    def _mount_sanity_check(self, z, horiz):
        """Floor returns near the robot should be BELOW the sensor (z<0). If the
        median near-z is strongly positive the cloud may be double-flipped."""
        near = z[horiz < 1.0]
        if len(near) < 20:
            return
        med = float(np.median(near))
        # report the fitted floor tilt so the ground-plane removal is observable.
        if self.floor_abc is not None:
            a, b, c = self.floor_abc
            tilt = f" | floor fit a={a:+.3f} b={b:+.3f} c={c:+.2f} (tilt slope x/y)"
        else:
            tilt = " | floor fit: n/a (ground_removal off)"
        if med > 0.3:
            self.get_logger().warn(
                f"FIRST-FRAME SANITY: median near-z = {med:+.2f} m (expected < 0 "
                f"for floor returns). The mount frame may be DOUBLE-FLIPPED -- "
                f"check the Livox driver roll. Continuing anyway.{tilt}")
        else:
            self.get_logger().info(
                f"first-frame sanity OK: median near-z = {med:+.2f} m{tilt}")

    def _accumulate(self, x, y, z, ang, horiz):
        """Merge the last viz_accum_s of kept-point frames for the RING + 3D
        sphere, so the Mid-360's non-repetitive scan shows a full circle instead
        of one sweeping petal (and stops flickering on near-empty frames).

        Points are stored in the SENSOR frame at capture time, so fast robot
        rotation smears older points by up to ~yaw_rate * viz_accum_s -- this is
        fail-safe (a stale return lingers, it never invents a gap). The 4 safety
        wedges + zone are computed on the CURRENT frame by the caller, so the
        front stop keeps its per-frame latency. viz_accum_s <= 0 disables the
        window (returns the current frame -> old per-frame behaviour).
        """
        if self.accum_s <= 0.0:
            return x, y, z, ang, horiz
        now = time.time()
        self._accum.append((now, x, y, z, ang, horiz))
        # drop frames older than the window (keep >= 1 so we never empty it out)
        cutoff = now - self.accum_s
        while len(self._accum) > 1 and self._accum[0][0] < cutoff:
            self._accum.popleft()
        if len(self._accum) == 1:
            return x, y, z, ang, horiz
        cols = list(zip(*self._accum))   # cols[1..5] = the x/y/z/ang/horiz arrays
        return (np.concatenate(cols[1]), np.concatenate(cols[2]),
                np.concatenate(cols[3]), np.concatenate(cols[4]),
                np.concatenate(cols[5]))

    def _viz_points(self, x, y, z):
        """Decimated kept-obstacle points for the dashboard 3D sphere.

        Returns a FLAT [x0,y0,z0, x1,y1,z1, ...] list in the viz frame: x forward,
        y left, z = height ABOVE the fitted floor (so the floor sits at 0 and a low
        box on the ground reads as a few cm, not ~-1.1 m). Strided to <= viz_max_points
        for a bounded payload, rounded to 2 dp (cm). These are EXACTLY the points that
        survived the ground/self/range filter -- i.e. what the guard actually reacts to.
        """
        n = len(x)
        if n == 0:
            return []
        # height above the fitted tilted floor when we have it (ground_removal path);
        # else fall back to sensor-relative z shifted by sensor_height (floor -> ~0).
        if self.ground_removal and self.floor_abc is not None:
            a, b, c = self.floor_abc
            zf = z - (a * x + b * y + c)
        else:
            zf = z + self.sensor_h
        step = 1 if (self.viz_max_points <= 0 or n <= self.viz_max_points) \
            else (n // self.viz_max_points + 1)
        xs = x[::step]
        ys = y[::step]
        zs = zf[::step]
        out = np.empty(xs.size * 3, dtype=float)
        out[0::3] = np.round(xs, 2)
        out[1::3] = np.round(ys, 2)
        out[2::3] = np.round(zs, 2)
        return out.tolist()

    def _write_shm(self, obj):
        """Atomic write: tmp file then os.replace (guard never reads a partial)."""
        tmp = SHM_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, SHM_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")

    def _print_status(self, n_raw, n_keep, front_m, left_m, right_m, back_m,
                      zone, scale, tight):
        """One cheap human-readable line per frame for staged bring-up."""
        def f(v):
            return "clr" if v is None else f"{v:.2f}"
        print(f"[{self.seq}] pts {n_raw}->{n_keep}  front={f(front_m)} "
              f"left={f(left_m)} right={f(right_m)} back={f(back_m)} "
              f"zone={zone} scale={scale:.2f} tight={tight:.2f}", flush=True)


def _r(v):
    """Round a distance to 3 dp, preserving None (the 'clear' sentinel)."""
    return None if v is None else round(float(v), 3)


def main():
    rclpy.init()
    node = ObstacleNode()
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
