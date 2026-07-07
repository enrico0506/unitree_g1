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
import sys
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# --- optional precision modules (siblings in obstacle/). Guarded so a missing/
#     broken module degrades gracefully to the validated legacy behaviour rather
#     than crashing the perception node. See obstacle-precision-modules memory. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from occupancy import PolarOccupancyGrid   # log-odds ring (stable + 3-state)
except Exception:                              # pragma: no cover
    PolarOccupancyGrid = None
try:
    from deskew import ImuGyroDeskewer          # IMU-gyro motion compensation
except Exception:                              # pragma: no cover
    ImuGyroDeskewer = None
try:
    from filters import custommsg_to_numpy      # fast CustomMsg->numpy (with per-point times)
except Exception:                              # pragma: no cover
    custommsg_to_numpy = None
try:
    import ground                               # per-cell elevation ground seg (QW1)
except Exception:                              # pragma: no cover
    ground = None
try:
    import filters as _filters                  # range-scaled ring + graded decimate (QW2/QW3)
except Exception:                              # pragma: no cover
    _filters = None

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
    # --- QW1: per-cell elevation ground seg (ground.py). Replaces the single global
    #     tilted-plane cut for the KEEP decision so small ground objects aren't absorbed
    #     into a 12 cm inlier band. Global plane still fit for viz/diag/self-mask height. ---
    "ground_elevation": True,      # true = per-20cm-cell floor (ground.segment_obstacles_elevation)
    "elev_cell_m": 0.20,           # elevation-grid cell size (m)
    "elev_floor_percentile": 10.0, # per-cell floor = this z-percentile (low = robust to overhang)
    "elev_min_cell_points": 3,     # cells below this keep points as obstacles (safety-first)
    "self_front_m": 0.35,          # footprint self-mask: forward extent (covers leg swing)
    "self_back_m": 0.35,           # footprint self-mask: rearward extent
    "self_half_width_m": 0.30,     # footprint self-mask: half-width (covers the leg stance)
    "self_mask_max_height_m": 0.6, # self-mask ONLY below this height above floor (legs/feet); overhead kept
    "min_cluster_points": 10,
    "filter_alpha": 0.25,           # legacy symmetric EMA (kept for reference; wedges now use the asym filter)
    # --- asymmetric per-direction wedge filter: "fast to fear, slow to trust" -----
    #   A CLOSER raw reading (a real obstacle approaching) is adopted (near-)instantly so
    #   the stop latency is UNCHANGED; a FARTHER raw reading (a Mid-360 false-CLEAR petal,
    #   e.g. front 1.25 -> 1.75) is trusted only SLOWLY and is first clamped to a short
    #   temporal MEDIAN of recent raws, so a lone single-frame far spike is rejected outright
    #   -- frame-rate independent (the median is over frames, not seconds), so it survives the
    #   CPU throttling that drops the frame rate under battery sag. Applies to all 4 wedges.
    "wedge_near_alpha": 1.0,        # EMA weight when the new reading is CLOSER (1.0 = adopt instantly = fear)
    "wedge_far_alpha": 0.05,        # EMA weight when the new reading is FARTHER (small = trust a far jump slowly)
    "wedge_median_n": 5,           # temporal median window (frames) rejecting a single-frame far false-clear
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
    # --- QW2: build the ring via filters.sector_kth_nearest (range-scaled required-points)
    #     + sector_tripwire, instead of the flat inline per-sector logic. Keeps sparse
    #     thin/far hazards (1-2 pts) while rejecting lone near fliers. ---
    "ring_filter": True,           # true = range-scaled filters ring; false = legacy inline ring
    # --- QW3: range-graded decimation. Replaces the blind `point_num//6000` stride
    #     (which thins already-sparse near objects) with keep-all-near / thin-far under a
    #     point budget, so small near-ground objects retain their few returns. ---
    "range_graded_decimate": True, # true = keep-near/thin-far; false = legacy blind stride
    "decimate_budget": 8000,       # target max points/frame after graded decimation
    "decimate_near_full_m": 2.5,   # keep ALL points within this range; thin only beyond
    "decimate_parse_cap": 45000,   # above this raw point_num, fall back to blind stride (perf valve)
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
    # --- 3D sphere stabilisation (viz-only; SEPARATE from the ring window above) --
    # The sphere used to flicker badly (measured: 46% of voxels vanished frame-to-
    # frame, 27% seen in a single frame) and looked tilted. Two fixes, both cosmetic
    # and isolated from the safety ring/wedges/guard:
    #   * LEVEL by a FIXED mount-tilt correction (the head is bolted nose-up at a
    #     constant angle; the per-frame floor fit is unreliable at that steep up-tilt
    #     -- it barely sees the floor and falls back to "level" -> no correction),
    #     then anchor the floor to z=0.
    #   * accumulate the last viz_win_s of LEVELED frames and VOXEL-downsample to a
    #     fixed grid, so points snap to stable positions and persist across the
    #     non-repetitive scan's petals (Jaccard 0.42 -> ~0.86, one-frame 27% -> 4%).
    "viz_win_s": 1.0,              # 3D-sphere accumulation window (s); ~5 frames @ 10 Hz
    "viz_win_max_frames": 16,      # hard cap on buffered sphere frames
    "viz_voxel_m": 0.10,           # sphere voxel-grid cell (m): stable positions + bounded count
    # --- 3D-sphere voxel TIME-PERSISTENCE (frame-rate independent steadiness) -----
    #   At ~5 Hz the viz_win_s union is only ~5 sparse Mid-360 petals, so boundary voxels
    #   flip on/off ("appear/disappear"). Instead we STAMP each occupied voxel cell with the
    #   time it was last seen and EMIT every cell seen within viz_hold_s -- independent of how
    #   many frames landed this window. Mirrors the log-odds ring's decay, but simpler (one
    #   stamp per cell). A hard cap bounds memory. Cosmetic only (no safety path reads these).
    "viz_hold_s": 0.8,             # emit any voxel cell seen within this many seconds (0 = current frame only)
    "viz_cell_cap": 6000,          # hard cap on persisted cells (memory); over cap drops the OLDEST-seen first
    "viz_mount_pitch_deg": 28.0,   # FIXED head nose-up tilt corrected in the sphere (measured ~28 deg).
                                   #   Raise if the floor still dips toward the FRONT; lower if it dips toward the BACK.
    "viz_mount_roll_deg": 0.0,     # FIXED head roll corrected in the sphere (+ = right side down); usually ~0
    # --- precision pipeline (obstacle-precision-modules; graceful fallback) ---
    "use_occupancy": True,         # fuse the raw ring into a log-odds occupancy grid:
                                   #   stable (no flicker), 3-state (unknown/free/occupied).
                                   #   Emitted ring distance = min(occupancy, raw) so the guard
                                   #   is NEVER less conservative than the current scan.
    "occ_decay_s": 0.7,            # occupancy leaky-decay time constant (s) toward unknown
    "occ_l_high": 1.5,             # log-odds a sector must reach to count as OCCUPIED (~2 returns;
                                   #   l_occ=0.85 so 0.85 == 1 return == noisy -> boxes the robot in)
    "occ_r_bin_m": 0.05,           # occupancy radial bin size (m)
    "use_deskew": True,            # IMU-gyro motion-compensate the cloud (needs /livox/imu +
                                   #   per-point times); identity when either is unavailable.
    "imu_topic": "/livox/imu",     # Livox 200 Hz IMU (same frame as the cloud)
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


def graded_indices(horiz, near_full_m, budget):
    """QW3: range-graded decimation index set.

    Keep EVERY point within ``near_full_m`` (small near-ground objects have only a few
    returns -- a blind ``[::stride]`` erases them), then evenly thin the far points to
    fill the remaining ``budget``. If the near set alone exceeds the budget, the near set
    itself is strided down. Returns a sorted index array into ``horiz``.
    """
    n = int(horiz.shape[0])
    if n <= budget:
        return np.arange(n)
    near_idx = np.nonzero(horiz < near_full_m)[0]
    far_idx = np.nonzero(horiz >= near_full_m)[0]
    remaining = budget - near_idx.size
    if remaining <= 0:                                  # near alone over budget
        step = int(np.ceil(near_idx.size / float(budget)))
        return near_idx[::max(1, step)]
    if far_idx.size > remaining:
        step = int(np.ceil(far_idx.size / float(remaining)))
        far_idx = far_idx[::max(1, step)]
    idx = np.concatenate((near_idx, far_idx))
    idx.sort()
    return idx


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
        # QW1 -- per-cell elevation ground seg (only if ground.py imported)
        self.ground_elevation = bool(c.get("ground_elevation", True)) and ground is not None
        self.elev_cell = float(c.get("elev_cell_m", 0.20))
        self.elev_floor_pct = float(c.get("elev_floor_percentile", 10.0))
        self.elev_min_cell_pts = max(1, int(c.get("elev_min_cell_points", 3)))
        # QW2 -- range-scaled filters ring (only if filters imported)
        self.ring_filter = bool(c.get("ring_filter", True)) and _filters is not None
        # QW3 -- range-graded decimation
        self.range_graded_decimate = bool(c.get("range_graded_decimate", True))
        self.decimate_budget = max(1000, int(c.get("decimate_budget", 8000)))
        self.decimate_near_full_m = float(c.get("decimate_near_full_m", 2.5))
        self.decimate_parse_cap = max(6000, int(c.get("decimate_parse_cap", 45000)))
        self.self_front = float(c["self_front_m"])
        self.self_back = float(c["self_back_m"])
        self.self_half_w = float(c["self_half_width_m"])
        self.self_mask_max_h = float(c.get("self_mask_max_height_m", 0.6))
        self.k_cluster = max(1, int(c["min_cluster_points"]))
        self.alpha = float(c["filter_alpha"])
        # asymmetric wedge filter ("fast to fear, slow to trust"): near/far EMA weights
        # + a per-direction temporal-median window rejecting a single-frame far false-clear.
        self.wedge_near_alpha = clamp(float(c.get("wedge_near_alpha", 1.0)), 0.0, 1.0)
        self.wedge_far_alpha = clamp(float(c.get("wedge_far_alpha", 0.05)), 0.0, 1.0)
        self.wedge_median_n = max(1, int(c.get("wedge_median_n", 5)))
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
        # 3D-sphere stabilisation (viz-only; independent of the ring accum above).
        # Holds (stamp, leveled_xyz) per frame; unioned + voxelised in _viz_points.
        self.viz_win_s = max(0.0, float(c.get("viz_win_s", 1.0)))
        self.viz_win_max = max(1, int(c.get("viz_win_max_frames", 16)))
        self.viz_voxel_m = max(0.01, float(c.get("viz_voxel_m", 0.10)))
        # voxel TIME-PERSISTENCE: cell key (i,j,k) -> (last_seen_stamp, count). Emit every
        # cell seen within viz_hold_s so the sphere is steady at ANY (low/variable) frame rate.
        self.viz_hold_s = max(0.0, float(c.get("viz_hold_s", 0.8)))
        self.viz_cell_cap = max(100, int(c.get("viz_cell_cap", 6000)))
        self._viz_cells = {}
        self.viz_floor_clip_m = max(0.0, float(c.get("viz_floor_clip_m", 0.0)))
        # sphere leveling: rotate the exported cloud by a FIXED mount-tilt plane so the
        # floor sits flat at z=0 (see _viz_level). The head is bolted nose-up at a
        # constant angle; a fixed correction is stable and needs no floor fit.
        self.viz_detilt_pct = float(c.get("viz_detilt_pct", 20.0))   # low percentile = floor band (resid diag)
        pitch = math.radians(float(c.get("viz_mount_pitch_deg", 28.0)))
        roll = math.radians(float(c.get("viz_mount_roll_deg", 0.0)))
        # floor plane z = a*x + b*y + c seen by a head tilted nose-up by `pitch`: the
        # floor slopes DOWN toward +x (front), so a = -tan(pitch); roll tilts it in y.
        self._mount_a = -math.tan(pitch)
        self._mount_b = math.tan(roll)
        self._viz_accum = deque(maxlen=self.viz_win_max)
        self._viz_z0 = None            # EMA of the leveled floor height (floor -> 0 offset)
        self._viz_dbg = None           # last sphere-build diagnostics (added to shm)

        # --- precision pipeline: occupancy ring + IMU de-skew --------------------
        # occ: log-odds polar grid the raw ring feeds each frame -> a stable, 3-state
        # ring (see _finalize_ring). deskewer: IMU-gyro motion comp (fed by _on_imu).
        # Both degrade to legacy behaviour when the module/IMU is unavailable.
        self.use_occupancy = bool(c.get("use_occupancy", True)) and PolarOccupancyGrid is not None
        self.use_deskew = bool(c.get("use_deskew", True)) and ImuGyroDeskewer is not None
        self.occ = None
        if self.use_occupancy:
            self.occ = PolarOccupancyGrid(
                n_sectors=self.ring_n, r_max=self.range_m,
                r_bin=float(c.get("occ_r_bin_m", 0.05)),
                start_deg=self.ring_start_deg,
                tau_s=float(c.get("occ_decay_s", 0.7)),
                l_high=float(c.get("occ_l_high", 1.5)))   # ~2 returns to mark occupied (noise reject)
        self._deskewer = ImuGyroDeskewer(buffer_s=0.5) if self.use_deskew else None
        # QW1 fix 1.5: EMA of the IMU accelerometer = measured gravity in the sensor frame,
        # used to LEVEL the elevation-grid ground seg (previously fed raw z, assumed z-up).
        self.ground_use_gravity = bool(c.get("ground_use_gravity", True))
        self._gravity = None           # (3,) float32 accelerometer EMA, or None until first IMU
        self._last_proc_t = None       # wall time of last processed frame (occupancy dt)

        # --- per-frame state --------------------------------------------------
        self.seq = 0
        self.front_filt = None        # asym-filtered front distance (None until first reading)
        self.back_filt = None         # asym-filtered back distance
        self.left_filt = None         # asym-filtered left distance
        self.right_filt = None        # asym-filtered right distance
        # per-direction recent-raw ring buffers feeding the far-branch temporal median.
        self._front_hist = deque(maxlen=self.wedge_median_n)
        self._back_hist = deque(maxlen=self.wedge_median_n)
        self._left_hist = deque(maxlen=self.wedge_median_n)
        self._right_hist = deque(maxlen=self.wedge_median_n)
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

        # IMU (200 Hz, same frame as the cloud) feeds the de-skewer. Best-effort:
        # if the topic/type is unavailable, de-skew simply stays identity.
        if self._deskewer is not None:
            try:
                from sensor_msgs.msg import Imu
                self.create_subscription(
                    Imu, str(self.cfg.get("imu_topic", "/livox/imu")),
                    self._on_imu, qos_profile_sensor_data)
            except Exception as e:
                self.get_logger().warn(f"IMU sub failed ({e}); de-skew disabled")
                self._deskewer = None

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

    # ------------------------------------------------------------------- IMU
    def _on_imu(self, msg):
        """Buffer the 200 Hz Livox gyro for cloud de-skew. Runs on a different
        callback than _on_cloud; ImuGyroDeskewer.add_gyro is internally locked.
        Also maintains a slow EMA of the accelerometer = measured gravity in the
        sensor frame, used to LEVEL the elevation-grid ground seg (QW1 fix 1.5)."""
        a = msg.linear_acceleration
        g = np.array((float(a.x), float(a.y), float(a.z)), dtype=np.float32)
        if np.isfinite(g).all():
            # heavy smoothing (alpha=0.02) rejects transient motion accel, keeps gravity
            self._gravity = g if self._gravity is None else (0.98 * self._gravity + 0.02 * g)
        if self._deskewer is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = msg.angular_velocity
        self._deskewer.add_gyro(t, float(w.x), float(w.y), float(w.z))

    # --------------------------------------------------------------- callback
    def _on_cloud(self, msg):
        # ---- 0. message -> (N,3) [+ per-point times for de-skew] ------------
        # CustomMsg has no zero-copy numpy view. Decimate to ~6000 points -- ample
        # for obstacle distances/gaps and keeps the callback under the 10 Hz Jetson
        # budget. When de-skewing, parse via custommsg_to_numpy so we ALSO get each
        # point's absolute time (timebase+offset_time); otherwise the xyz-only path.
        times = None
        if self.kind == "custom":
            pn = int(getattr(msg, "point_num", 0) or len(msg.points))
            # QW3: range-graded decimation -- parse FULL, then keep-all-near / thin-far to
            # the budget (blind `pn//6000` stride erases already-sparse near objects). Only
            # on the fast numpy+deskew path, and only under decimate_parse_cap (perf valve);
            # otherwise fall back to the legacy blind stride below.
            graded = (self.range_graded_decimate and custommsg_to_numpy is not None
                      and self._deskewer is not None and 0 < pn <= self.decimate_parse_cap)
            if graded:
                try:
                    pts, times, _ = custommsg_to_numpy(msg, decimate=1)
                    h = np.hypot(pts[:, 0], pts[:, 1])
                    idx = graded_indices(h, self.decimate_near_full_m, self.decimate_budget)
                    pts = pts[idx]
                    times = times[idx] if times is not None else None
                except Exception:
                    graded = False
            if not graded:
                decimate = max(1, pn // 6000)
                if self._deskewer is not None and custommsg_to_numpy is not None:
                    try:
                        pts, times, _ = custommsg_to_numpy(msg, decimate=decimate)
                    except Exception:
                        pts, times = custom_xyz(msg, decimate), None
                else:
                    pts = custom_xyz(msg, decimate)
        else:
            pts = pc2_xyz(msg)
        n_raw = len(pts)
        if n_raw == 0:
            return

        # ---- 0b. IMU-gyro de-skew (motion compensation) --------------------
        # Rotate every point into the sensor orientation at the frame-end time,
        # removing the intra-frame smear from body sway/turn. No-op without a gyro
        # buffer or per-point times (the pc2 path, or before the first IMU frames).
        if (self._deskewer is not None and times is not None and len(pts)
                and self._deskewer.n_samples() > 1):
            try:
                pts = self._deskewer.deskew(pts, times, float(np.max(times)))
            except Exception:
                pass

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

        # ---- 4. per-direction asymmetric-filtered distances -> scales + tight -
        # "Fast to fear, slow to trust": a CLOSER reading is adopted instantly (stop
        # latency preserved); a FARTHER reading (false-CLEAR petal) is trusted slowly
        # and a lone one-frame far spike is median-rejected. When a direction is clear,
        # feed obstacle_range_m so its filter relaxes back toward "open" (slowly).
        self.front_filt = self._asym_filter(self.front_filt, front_m, self._front_hist)
        self.back_filt = self._asym_filter(self.back_filt, back_m, self._back_hist)
        self.left_filt = self._asym_filter(self.left_filt, left_m, self._left_hist)
        self.right_filt = self._asym_filter(self.right_filt, right_m, self._right_hist)
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
        # 3D-sphere ONLY: push THIS frame's kept points, LEVELED to gravity-up, into
        # the separate viz window (voxelised in _viz_points). Kept out of the ring
        # accumulation above so no safety gating is affected.
        if self.viz_points:
            self._viz_push(x, y, z)

        # ---- 5b. full-circle 360 deg ring (additive; merged-window points) --
        raw_ring = self._ring_distances(acc_ang, acc_horiz)
        # 5c. fuse the raw ring into the log-odds occupancy grid -> a stable,
        # 3-state ring. Emitted distance = min(occupancy, raw) so directional
        # gating is NEVER less conservative than the current scan (safety); the
        # state/prob arrays drive the dashboard's honest unknown/free/occupied view.
        now = time.time()
        dt = 0.1 if self._last_proc_t is None \
            else clamp(now - self._last_proc_t, 1e-3, 1.0)
        self._last_proc_t = now
        ring_dist, ring_state, ring_prob = self._finalize_ring(raw_ring, dt)

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
            # the guard's real thresholds, so the dashboard colours match exactly
            # (effective = raw + safety_bubble) instead of the old hardcoded 0.7/2.0.
            "stop_m": round(self.stop_at, 3),
            "slow_m": round(self.slow_at, 3),
            "ring": {
                "n": self.ring_n,
                "bin_deg": self.ring_bin_deg,
                "start_deg": self.ring_start_deg,
                "dist": [_r(d) for d in ring_dist],
            },
        }
        # 3-state occupancy overlay (only when the occupancy grid is active).
        if ring_state is not None:
            out["ring"]["state"] = ring_state
            out["ring"]["prob"] = ring_prob
        # Stabilised 3D-sphere cloud: the kept obstacle points, LEVELED (floor->0),
        # accumulated over viz_win_s and voxel-downsampled to stable cell centres --
        # cosmetic only (the guard gates on the per-frame x/y/z + ring computed above).
        if self.viz_points:
            out["points"] = self._viz_points()
            if self._viz_dbg is not None:
                out["viz_dbg"] = self._viz_dbg
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

    def _asym_filter(self, state, raw, hist):
        """One asymmetric per-direction step -- "fast to fear, slow to trust".

        A CLOSER reading (feed <= state: a real obstacle approaching) is adopted
        (near-)instantly via wedge_near_alpha (default 1.0), so the stop latency is
        UNCHANGED and a direction is never falsely believed clear. A FARTHER reading
        (feed > state: a potential Mid-360 false-CLEAR petal, e.g. front 1.25 -> 1.75)
        is trusted only SLOWLY (wedge_far_alpha) AND first clamped to the temporal
        MEDIAN of the last wedge_median_n raws -- so a lone single-frame far spike is
        rejected outright (the median stays near the close value) no matter the frame
        rate. A null (clear) reading feeds obstacle_range_m. Seeds on first use."""
        feed = raw if raw is not None else self.range_m
        hist.append(feed)
        if state is None:
            return feed
        if feed <= state:
            a = self.wedge_near_alpha                    # fast to fear: adopt the closer reading
            return a * feed + (1.0 - a) * state
        # farther: reject a one-frame far outlier via the median, then creep up slowly.
        med = float(np.median(hist))
        target = med if med < feed else feed             # median never exceeds the raw here
        if target < state:
            target = state                               # far branch must not pull the state DOWN
        a = self.wedge_far_alpha
        return a * target + (1.0 - a) * state

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
        if self.ring_filter:
            # QW2: range-scaled required-points ring. A sector reports its k-th nearest
            # only if it holds >= required_points(range) points -- so a sparse thin/far
            # hazard (few returns) survives while a lone near flier is rejected -- then the
            # same near-field tripwire is merged by nearest. Same sector convention as the
            # legacy path below (n_sectors=ring_n, start=ring_start_deg, bin=ring_bin_deg).
            base, _counts = _filters.sector_kth_nearest(
                ang, horiz, n_sectors=self.ring_n, start_deg=self.ring_start_deg,
                k=self.ring_min, range_scaled_mincount=True,
                k0=self.ring_min, r0=1.0, r_max=self.range_m)
            trip = _filters.sector_tripwire(
                ang, horiz, n_sectors=self.ring_n, start_deg=self.ring_start_deg,
                tripwire_range=self.tripwire_range, tripwire_min=self.tripwire_min)
            merged = np.asarray(_filters.merge_min(base, trip), dtype=float).ravel()
            return [None if not np.isfinite(v) else float(v) for v in merged]

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

    def _finalize_ring(self, raw_dist, dt):
        """Fuse the raw per-sector nearest into the log-odds occupancy grid and
        return (dist, state, prob) for the contract. `dt` = seconds since the last
        finalize (drives the occupancy decay).

        dist[i] = the CLOSER of the occupancy-derived nearest and THIS frame's raw
        reading, so the guard is never LESS conservative than the current scan --
        occupancy only ADDS persistence (an obstacle seen a moment ago still shows)
        plus an honest per-sector state (0=unknown / 1=free / 2=occupied). Returns
        (raw_dist, None, None) when occupancy is disabled (legacy ring behaviour).
        """
        if self.occ is None:
            return raw_dist, None, None
        sr = np.array([np.nan if d is None else d for d in raw_dist], dtype=float)
        self.occ.update(sr, dt)
        occ_dist, occ_state = self.occ.nearest_occupied()
        prob = self.occ.prob_per_sector()
        out = [None] * self.ring_n
        for i in range(self.ring_n):
            raw = raw_dist[i]
            od = float(occ_dist[i])
            od = None if not math.isfinite(od) else od
            if raw is None:
                out[i] = od
            elif od is None:
                out[i] = raw
            else:
                out[i] = raw if raw < od else od
        state = [int(s) for s in occ_state]
        prob_l = [round(float(p), 3) for p in prob]
        return out, state, prob_l

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

        above = z - (a * x + b * y + c)   # height above the GLOBAL plane (viz/diag/self-mask)

        if self.ground_elevation:
            # QW1: per-cell (20 cm) floor instead of one global tilted plane, so a small
            # object is measured against its LOCAL floor and not absorbed into the plane's
            # ~12 cm inlier band. Global plane above/floor_abc kept above for the self-mask
            # height test, viz and the front-cone diagnostic. True == obstacle (keep).
            pts = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            # QW1 fix 1.5: level the elevation-grid input against measured gravity so a
            # tilted/swaying sensor's per-cell floor is computed in a gravity-aligned frame
            # (rotation preserves point order -> the returned mask still aligns to x/y/z).
            g = self._gravity
            if self.ground_use_gravity and g is not None:
                # Orient the accelerometer EMA to point DOWN by its own z sign -- sign-agnostic,
                # so it is correct whether the IMU reports gravity or specific-force-UP (REP-145).
                # Then LEVEL only if it is plausibly vertical (< ~35 deg from straight down): a
                # wrong sign or garbage IMU asks for a large/flipped rotation -> rejected -> safe
                # no-op (falls back to the assumed-z-up behaviour, never rotates the cloud wrong).
                gd = g if g[2] < 0.0 else -g
                gn = float(np.sqrt(gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2]))
                if gn > 1e-3 and (-gd[2] / gn) > 0.82:     # cos(35 deg) ~ 0.82
                    pts = ground.level_by_gravity(pts, gd)
            keep_ground = ground.segment_obstacles_elevation(
                pts, cell=self.elev_cell, ground_clearance=self.ground_clear,
                max_above=self.max_above, floor_pct=self.elev_floor_pct,
                min_cell_pts=self.elev_min_cell_pts)
        else:
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

    def _viz_push(self, x, y, z):
        """Buffer THIS frame's kept points (RAW sensor frame) for the 3D sphere.

        Leveling happens later in _viz_points, on the ACCUMULATED cloud, using its
        own low-percentile plane -- the surface objects actually rest on. That is far
        more reliable than the node's per-frame floor fit: the Mid-360 is ~30 deg
        nose-up so it barely sees the floor, and MEASURED live, leveling by that fit
        removed almost none of the tilt (fe~9 deg in, resid~8 deg out). viz-only."""
        if len(x) == 0:
            return
        self._viz_accum.append(
            (time.time(), np.column_stack((x, y, z)).astype(np.float32)))

    def _viz_level(self, raw):
        """Level the accumulated cloud by the FIXED mount-tilt plane so the floor sits
        flat at z=0. Returns (leveled_pts (N,3) float32, diag).

        The head is bolted nose-up at a CONSTANT angle, so we rotate by a fixed floor
        plane (viz_mount_pitch_deg / roll -> _mount_a/_mount_b) instead of a per-frame
        fit. The fit is unreliable at this steep up-tilt -- it barely sees the floor and
        returns "level" (a=b=0) -> no correction, so the raw tilt showed through (floor
        dipping below in front). A constant is stable and needs no floor visibility.
        Then the floor is anchored to z=0 with an EMA'd low percentile. Cosmetic (viz
        only); no safety path is affected."""
        R = None
        src = "mount"
        if ground is not None and (self._mount_a != 0.0 or self._mount_b != 0.0):
            R = ground.rotation_from_gravity(np.array([self._mount_a, self._mount_b, -1.0]))
        pts = (raw @ R.T.astype(np.float32)) if R is not None else raw.copy()
        # floor -> 0 via an EMA'd low percentile of the leveled z (steady vertical anchor).
        p2 = float(np.percentile(pts[:, 2], 2))
        self._viz_z0 = p2 if self._viz_z0 is None else (0.2 * p2 + 0.8 * self._viz_z0)
        if math.isfinite(self._viz_z0):
            pts[:, 2] -= self._viz_z0
        diag = {"src": src, "tilt_deg": round(float(np.degrees(
            np.arctan(np.hypot(self._mount_a, self._mount_b)))), 1)}
        if pts.shape[0] >= 60:                           # residual tilt of the leveled floor
            thr = np.percentile(pts[:, 2], self.viz_detilt_pct)
            lo = pts[pts[:, 2] <= thr]
            if len(lo) >= 40:
                A = np.c_[lo[:, 0], lo[:, 1], np.ones(len(lo))]
                sol, *_ = np.linalg.lstsq(A, lo[:, 2], rcond=None)
                diag["resid_deg"] = round(float(np.degrees(np.arctan(np.hypot(sol[0], sol[1])))), 1)
        return pts.astype(np.float32), diag

    def _viz_points(self):
        """Stabilised kept-obstacle points for the dashboard 3D sphere.

        LEVELS the last viz_win_s of frames (pushed by _viz_push) to a fixed viz_voxel_m
        grid, then STAMPS each occupied cell with the time it was last seen and EMITS
        every cell seen within viz_hold_s -- so the cloud is steady at ANY (low or
        variable) frame rate, not just when many petals happen to land in one window.
        Returns a FLAT [x0,y0,z0, ...] list, x fwd / y left (both horizontal), z = height
        above the floor (floor 0).

        Voxel time-persistence mirrors the log-odds ring's decay but simpler (one stamp
        per cell); a hard cap (viz_cell_cap) bounds memory. This kills the frame-to-frame
        "appear/disappear" that the plain viz_win_s union suffers at ~5 Hz (only ~5 sparse
        Mid-360 petals -> boundary voxels flip). Purely cosmetic: no safety gating reads
        these points (wedges/ring/guard use the per-frame x/y/z + ring computed earlier).
        Emitted payload capped at viz_max_points (drops the sparsest cells first)."""
        now = time.time()                                # SAME clock as the frame stamps
        cell = self.viz_voxel_m
        # prune the leveling window (feeds _viz_level's fixed-tilt rotation + floor->0 z0).
        if self.viz_win_s > 0.0:
            cutoff = now - self.viz_win_s
            while len(self._viz_accum) > 1 and self._viz_accum[0][0] < cutoff:
                self._viz_accum.popleft()
        # STAMP this window's occupied cells into the persistence store (leveling preserved).
        n_raw = 0
        leveldiag = None
        if self._viz_accum:
            raw = np.concatenate([p for _, p in self._viz_accum], axis=0)
            n_raw = raw.shape[0]
            # LEVEL by the fixed mount tilt + anchor the floor to z=0 (unchanged).
            pts, leveldiag = self._viz_level(raw)
            # Optional below-floor clip (DEFAULT OFF): points below the leveled floor are
            # REAL small floor objects -- they only dip below z=0 via RESIDUAL leveling
            # tilt, so do NOT clip them (would hide real obstacles); level better instead.
            if self.viz_floor_clip_m > 0.0:
                pts = pts[pts[:, 2] > -self.viz_floor_clip_m]
            if pts.shape[0]:
                keys = np.floor(pts / cell).astype(np.int64)
                uk, counts = np.unique(keys, axis=0, return_counts=True)
                for k, cnt in zip(map(tuple, uk.tolist()), counts.tolist()):
                    self._viz_cells[k] = (now, cnt)      # (last-seen stamp, this-frame count)
        # TIME-PERSISTENCE decay: keep only cells seen within viz_hold_s (steady @ any rate).
        hold_cut = now - self.viz_hold_s
        if self._viz_cells:
            self._viz_cells = {k: v for k, v in self._viz_cells.items() if v[0] >= hold_cut}
        # hard memory cap: if still oversized, keep the most-recently-seen cells.
        if len(self._viz_cells) > self.viz_cell_cap:
            kept = sorted(self._viz_cells.items(), key=lambda kv: kv[1][0],
                          reverse=True)[:self.viz_cell_cap]
            self._viz_cells = dict(kept)
        if not self._viz_cells:
            self._viz_dbg = None
            return []
        items = list(self._viz_cells.items())
        # bound the payload: keep the DENSEST cells (most returns = most real).
        if self.viz_max_points > 0 and len(items) > self.viz_max_points:
            items.sort(key=lambda kv: kv[1][1], reverse=True)
            items = items[:self.viz_max_points]
        centres = (np.array([k for k, _ in items], dtype=np.float64) + 0.5) * cell
        self._viz_dbg = {
            "n_raw": int(n_raw), "n_out": int(centres.shape[0]),
            "cells": len(self._viz_cells),               # persisted-cell count (time-held)
            "frames": len(self._viz_accum), "z0": round(float(self._viz_z0 or 0.0), 2),
        }
        self._viz_dbg.update(leveldiag or {})
        out = np.empty(centres.shape[0] * 3, dtype=float)
        out[0::3] = np.round(centres[:, 0], 2)
        out[1::3] = np.round(centres[:, 1], 2)
        out[2::3] = np.round(centres[:, 2], 2)
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
