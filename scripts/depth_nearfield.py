"""DepthNearField -- RealSense D435i depth -> nearest forward near-GROUND obstacle.

Fills the band the Mid-360 cannot see. The Mid-360's vertical FOV is mostly UP
(~ -7 deg down), so the floor and low objects within a few metres in front are a
blind wedge. The head D435i is tilted ~47 deg DOWN (URDF d435_joint pitch
0.83 rad), so it images exactly that near-ground band -- roughly 0.5-2 m in front.

This runs INSIDE the dashboard process (domain 0): /dev/video0 is single-open and
LidarSource already holds it, so we reuse its decoded cloud (LidarSource.get_cloud)
rather than opening the camera again. The cloud comes back in a LEVEL world frame
(x fwd, y left, z up, camera height baked into z) -- _depth_to_cloud does NOT apply
the mount pitch -- so we undo the baked height, rotate by the real camera pitch,
re-add the real camera height, then take the nearest non-ground point in a forward
corridor.

Exposed: front_near_m() -> float metres or None (clear / not enough points). The
ObstacleGuard mixes this into its forward distance (min) so the robot slows/stops
for low things ahead the lidar missed. Always ON -- no UI/runtime toggle disables
it; the frame-sanity self-gate (frame_check) still auto-skips a frame it judges
untrustworthy (wrong pitch/height), which is a safety guard, not a user toggle.
"""

import math
import threading
import time

import numpy as np
import yaml


_DEFAULTS = {
    "enabled": True,           # always on -- kept for cfg back-compat; DepthNearField ignores it
    "pitch_deg": 47.6,         # D435i downward mount tilt (URDF 0.83078 rad). + = looks DOWN.
    "camera_height_m": 1.30,   # real D435i height above the floor (MEASURE; ~pelvis + 0.47)
    "corridor_half_m": 0.35,   # forward corridor half-width (robot half-width + margin)
    "min_range_m": 0.35,       # ignore closer than the D435i min range (~0.3 m)
    "max_range_m": 2.5,        # only fuse the NEAR band (beyond this the lidar sees fine)
    "ground_clearance_m": 0.08,# keep points more than this above the floor (drop the floor)
    "max_height_m": 1.8,       # drop overhead / ceiling
    "min_points": 8,           # need >= this many near points before reporting (robust to noise)
    "kth": 4,                  # report the k-th nearest forward distance (reject a single flier)
    "poll_hz": 15.0,           # sampling rate of the depth cloud
    # --- per-sector ring (fills the front blind wedge across ALL forward sectors, not
    #     just the single forward corridor). Merged into the guard ring by NaN-aware min,
    #     so depth can only make a sector CLOSER, never clear a lidar reading (UNKNOWN-safe:
    #     a depth hole yields no points -> NaN -> no override). ---
    "ring_bin_deg": 5.0,       # MUST match the lidar ring (obstacle.yaml ring_bin_deg) to merge
    "ring_fov_half_deg": 43.0, # D435i horizontal half-FOV (~87 deg HFOV) the camera actually sees
    "ring_min_points": 2,      # min near points in a 5 deg sector before it reports (per-sector)
    # SMALL-OBJECT floor: the ring uses a LOWER clearance than the scalar corridor so a
    # low CABLE / cord / thin bar on the floor (a few cm tall) clears the floor and is seen.
    # Bounded by the D435i near-field noise (~1-3 cm) -- below ~3 cm objects blend into floor
    # noise (physics floor). Raise if the flat floor false-triggers on your surface.
    "ring_ground_clearance": 0.03,  # ring obstacle floor (m); scalar corridor keeps 0.08
    # --- frame self-validation: a wrong mount pitch/height tilts the derotated floor so its
    #     height ramps with range. With the correct frame a flat floor sits at height ~0 at
    #     ALL ranges. If the floor ramps more than the tolerance, the frame is UNTRUSTWORTHY
    #     -> emit NO ring/scalar this frame (fail to lidar-only), never a false near-stop.
    #     This is what makes always-on depth fusion safe: watch telemetry frame_ok. ---
    "frame_check": True,
    "frame_floor_ramp_m": 0.15,  # max floor-height DRIFT across the 0.5-2 m band (wrong pitch)
    "frame_floor_offset_m": 0.12, # max |median floor height| from 0 (wrong camera_height)
    "frame_min_floor_pts": 30,   # need this many candidate floor points to judge the frame
    # --- full-cloud export for the mapping fusion (map_cloud()) -- SEPARATE band from the
    #     scalar/ring obstacle bands above: wider range (mapping wants more context than the
    #     near-field guard), floor dropped by a small clearance (not the cable-catching ring
    #     clearance), ceiling capped so overhead clutter doesn't pollute the map. ---
    "map_min_range_m": 0.30,     # ignore closer than the D435i min range (~0.3 m)
    "map_max_range_m": 4.0,      # far enough to add real map context beyond the near-field band
    "map_ground_clearance_m": 0.03,  # drop the floor, keep low cables/objects
    "map_max_height_m": 2.0,     # drop overhead / ceiling
    # --- "Full Depth" 3D-sphere toggle -- unlike map_cloud() (which strips the floor
    #     and caps range/height for the SLAM map export), this keeps almost everything
    #     the sensor returns, floor included, so the user can see literally what the
    #     D435i sees and compare it against the narrow ground-only safety band above.
    #     Still only ever the forward ~87 deg wedge the camera is physically mounted to
    #     see -- that's the sensor's real FOV, not a software restriction. Demand-gated
    #     by the client toggle (see robot_web_controller.py state.depth_full_view) so it
    #     costs nothing while off. ---
    "full_min_range_m": 0.30,    # D435i physical min range
    "full_max_range_m": 6.0,     # generous -- depth gets noisy/sparse well before this
    "full_min_height_m": -0.30,  # BELOW the floor plane on purpose: shows real floor + noise
    "full_max_height_m": 2.5,    # generous ceiling; still bounded to reject wild outliers
    "full_cap": 4000,            # point cap for the WS payload when the toggle is on
}


class DepthNearField:
    def __init__(self, lidar_source, cfg_path=None, ls_mount_height=1.30):
        self.lidar = lidar_source
        self.ls_mount_height = float(ls_mount_height)   # height LidarSource baked into z

        cfg = dict(_DEFAULTS)
        if cfg_path:
            try:
                with open(cfg_path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                sub = loaded.get("depth_fusion") if isinstance(loaded, dict) else None
                if isinstance(sub, dict):
                    cfg.update(sub)
            except Exception as e:
                print(f"[DEPTH] config load failed ({e}); using defaults", flush=True)
        self.cfg = cfg

        self.enabled = True   # always on -- no UI/runtime toggle can disable this (see set_enabled)
        theta = math.radians(float(cfg["pitch_deg"]))
        self._ct, self._st = math.cos(theta), math.sin(theta)
        self.cam_h = float(cfg["camera_height_m"])
        self.corridor = float(cfg["corridor_half_m"])
        self.min_range = float(cfg["min_range_m"])
        self.max_range = float(cfg["max_range_m"])
        self.ground_clear = float(cfg["ground_clearance_m"])
        self.max_height = float(cfg["max_height_m"])
        self.min_points = max(1, int(cfg["min_points"]))
        self.kth = max(1, int(cfg["kth"]))
        self.period = 1.0 / max(1.0, float(cfg["poll_hz"]))
        # per-sector ring geometry (must match the lidar ring to merge)
        self.ring_bin_deg = float(cfg.get("ring_bin_deg", 5.0))
        self.ring_n = max(1, int(round(360.0 / self.ring_bin_deg)))
        self.ring_start_deg = -180.0                         # shared convention with the node/guard
        self.ring_fov_half = float(cfg.get("ring_fov_half_deg", 43.0))
        self.ring_min_pts = max(1, int(cfg.get("ring_min_points", 2)))
        self.ring_ground_clear = float(cfg.get("ring_ground_clearance", 0.03))  # low: catch cables
        # frame self-validation
        self.frame_check = bool(cfg.get("frame_check", True))
        self.frame_ramp_tol = float(cfg.get("frame_floor_ramp_m", 0.15))
        self.frame_offset_tol = float(cfg.get("frame_floor_offset_m", 0.12))
        self.frame_min_floor_pts = int(cfg.get("frame_min_floor_pts", 30))
        # full-cloud export band for mapping fusion (see map_cloud())
        self.map_min_range = float(cfg.get("map_min_range_m", 0.30))
        self.map_max_range = float(cfg.get("map_max_range_m", 4.0))
        self.map_ground_clear = float(cfg.get("map_ground_clearance_m", 0.03))
        self.map_max_height = float(cfg.get("map_max_height_m", 2.0))
        self.full_min_range = float(cfg.get("full_min_range_m", 0.30))
        self.full_max_range = float(cfg.get("full_max_range_m", 6.0))
        self.full_min_height = float(cfg.get("full_min_height_m", -0.30))
        self.full_max_height = float(cfg.get("full_max_height_m", 2.5))
        self.full_cap = max(1, int(cfg.get("full_cap", 4000)))

        self._lock = threading.Lock()
        self._front = None         # last computed nearest forward distance (m) or None
        self._ring = None          # last per-sector ring (np (ring_n,), NaN = no reading) or None
        self._points = None        # last kept near-ground points (M,3) for the 3D viz, or None
        self._n_near = 0           # qualifying point count (for telemetry / validation)
        self._frame_ok = True      # last frame passed the floor-ramp sanity check
        self._floor_ramp = 0.0     # last measured floor-height ramp (m) across the forward band
        self._floor_offset = 0.0   # last measured |median floor height| (m)
        self._floor_nf = 0         # last measured candidate floor-point count
        self._warn_ct = 0          # throttle for the frame-rejected warning
        self._live = False
        self._stop = threading.Event()
        self._thread = None

    # --- lifecycle --------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="depth-nearfield",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_enabled(self, on):
        # Depth fusion is permanently ON -- there is no UI toggle to disable it, and
        # this must not become a disable path. Kept as a no-op (rather than removed)
        # so any caller that still invokes it (e.g. forced-enable at startup) is safe.
        self.enabled = True

    # --- core -------------------------------------------------------------
    def _compute(self, cloud):
        """cloud: (N,3) LidarSource level-frame points -> nearest forward near-ground
        distance (m) or None. Vectorised; called off the hot path."""
        if cloud is None or len(cloud) == 0:
            return None, 0
        X = cloud[:, 0]
        Y = cloud[:, 1]
        Zc = cloud[:, 2] - self.ls_mount_height      # undo baked height -> camera-relative up (level)

        # rotate the LEVEL cloud DOWN by the real mount pitch about the left (Y) axis,
        # then re-add the real camera height -> true height above the floor.
        fwd = X * self._ct + Zc * self._st
        up = -X * self._st + Zc * self._ct
        left = Y
        height = up + self.cam_h

        keep = (
            (height > self.ground_clear) & (height < self.max_height)
            & (np.abs(left) < self.corridor)
            & (fwd > self.min_range) & (fwd < self.max_range)
        )
        n = int(np.count_nonzero(keep))
        if n < self.min_points:
            return None, n
        f = fwd[keep]
        # robust nearest: k-th smallest forward distance (reject a lone flier).
        k = min(self.kth, f.size)
        nearest = float(np.partition(f, k - 1)[k - 1])
        return nearest, n

    def _compute_ring(self, cloud):
        """cloud -> per-sector nearest HORIZONTAL distance over the camera's forward FOV.

        Same validated derotation as ``_compute`` but, instead of one forward corridor,
        bins the near-ground points into the lidar ring's sectors and reports each
        sector's k-th nearest horizontal range. Returns (dist, kept_points, n_near); ``dist`` is
        (ring_n,) float64 with NaN == "no reading" (never emitted as clear -- absence of a
        near obstacle, or a depth hole, leaves NaN so the guard merge keeps the lidar value).
        """
        out = np.full(self.ring_n, np.nan, dtype=np.float64)
        empty = np.zeros((0, 3), dtype=np.float32)
        if cloud is None or len(cloud) == 0:
            return out, empty, 0
        X = cloud[:, 0]
        Y = cloud[:, 1]
        Zc = cloud[:, 2] - self.ls_mount_height
        fwd = X * self._ct + Zc * self._st
        up = -X * self._st + Zc * self._ct
        left = Y
        height = up + self.cam_h
        ang = np.degrees(np.arctan2(left, fwd))          # 0 ahead, + left (node/guard convention)
        horiz = np.hypot(fwd, left)
        keep = (
            (height > self.ring_ground_clear) & (height < self.max_height)  # LOW floor: catch cables
            & (fwd > self.min_range) & (fwd < self.max_range)
            & (np.abs(ang) < self.ring_fov_half)         # only where the camera actually sees
        )
        n = int(np.count_nonzero(keep))
        if n == 0:
            return out, empty, 0
        # kept near-ground points for the viz, in the SAME frame as the node's msg.points
        # (x-fwd, y-left, z = height above floor) -- so the 3D sphere shows exactly the
        # near-floor obstacles the depth fusion reacts to. Strided to a bounded count.
        kpts = np.column_stack((fwd[keep], left[keep], height[keep])).astype(np.float32)
        if kpts.shape[0] > 800:
            kpts = kpts[:: (kpts.shape[0] // 800 + 1)]
        a = ang[keep]
        h = horiz[keep]
        sector = (np.floor((a - self.ring_start_deg) / self.ring_bin_deg)
                  .astype(np.int64) % self.ring_n)
        counts = np.bincount(sector, minlength=self.ring_n)
        populated = counts >= self.ring_min_pts
        if not populated.any():
            return out, kpts, n
        # k-th nearest per sector via one lexsort (sector, then range) + block starts.
        order = np.lexsort((h, sector))
        h_sorted = h[order]
        starts = np.zeros(self.ring_n, dtype=np.int64)
        starts[1:] = np.cumsum(counts)[:-1]
        kth_rank = np.minimum(self.kth, counts)          # clamp k to the sector's count
        kth_idx = starts + (kth_rank - 1)
        out[populated] = h_sorted[kth_idx[populated]]
        return out, kpts, n

    def _frame_sanity(self, cloud):
        """Frame-sanity metrics. With the CORRECT mount frame the derotated floor sits at
        height ~0 at every forward range. Returns (ramp_m, offset_m, n_floor):
          - offset = |median floor height|  -> catches a wrong camera_height (uniform shift)
          - ramp   = spread of per-range-bin floor height -> catches a wrong mount pitch (tilt)
        A large ramp/offset (or too few floor points) => untrustworthy frame."""
        if cloud is None or len(cloud) == 0:
            return 0.0, 0.0, 0
        X = cloud[:, 0]
        Y = cloud[:, 1]
        Zc = cloud[:, 2] - self.ls_mount_height
        fwd = X * self._ct + Zc * self._st
        up = -X * self._st + Zc * self._ct
        height = up + self.cam_h
        ang = np.degrees(np.arctan2(Y, fwd))
        m = ((fwd > 0.5) & (fwd < 2.0) & (np.abs(ang) < self.ring_fov_half)
             & (np.abs(height) < 0.6))               # plausible floor band only
        nf = int(np.count_nonzero(m))
        if nf < self.frame_min_floor_pts:
            return 0.0, 0.0, nf
        fb, hb = fwd[m], height[m]
        offset = abs(float(np.median(hb)))
        floors = []
        for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0)):
            sel = (fb >= lo) & (fb < hi)
            if np.count_nonzero(sel) >= 5:
                floors.append(float(np.percentile(hb[sel], 10.0)))   # 10th pct = floor
        ramp = float(max(floors) - min(floors)) if len(floors) >= 2 else 0.0
        return ramp, offset, nf

    def _run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.enabled and self.lidar is not None:
                try:
                    cloud = self.lidar.get_cloud()
                    ok, ramp, offset, nf = True, 0.0, 0.0, 0
                    if self.frame_check:
                        ramp, offset, nf = self._frame_sanity(cloud)
                        ok = (nf >= self.frame_min_floor_pts
                              and ramp <= self.frame_ramp_tol
                              and offset <= self.frame_offset_tol)
                    if ok:
                        front, n = self._compute(cloud)
                        ring, dpts, _nr = self._compute_ring(cloud)
                    else:
                        front, ring, dpts, n = None, None, None, 0  # bad frame -> fail to lidar-only
                        if cloud is not None and (self._warn_ct % 30 == 0):
                            print(f"[DEPTH] frame rejected (floor ramp {ramp:.2f} m / offset "
                                  f"{offset:.2f} m, floor pts {nf}); depth OFF this frame -- "
                                  f"CHECK D435i pitch/height.", flush=True)
                        self._warn_ct += 1
                    with self._lock:
                        self._front = front
                        self._ring = ring
                        self._points = dpts
                        self._n_near = n
                        self._frame_ok = ok
                        self._floor_ramp = ramp
                        self._floor_offset = offset
                        self._floor_nf = nf
                        self._live = cloud is not None
                except Exception as e:
                    with self._lock:
                        self._front = None
                        self._ring = None
                        self._points = None
                        self._n_near = 0
                        self._frame_ok = False
                    print(f"[DEPTH] compute error: {e}", flush=True)
            else:
                with self._lock:
                    self._front = None
                    self._ring = None
                    self._points = None
            dt = time.monotonic() - t0
            if dt < self.period:
                self._stop.wait(self.period - dt)

    # --- consumers --------------------------------------------------------
    def front_near_m(self):
        """Nearest forward near-ground obstacle distance (m), or None. Cheap: the
        ObstacleGuard calls this on its hot path."""
        if not self.enabled:
            return None
        with self._lock:
            return self._front

    def front_ring(self):
        """Per-sector near-ground depth ring, or None. Same schema/geometry as the lidar
        ring so the guard can NaN-aware-min-merge it: {n, bin_deg, start_deg, dist:[...]}
        with dist[i]=None where the camera saw no near obstacle (never emitted as clear).
        The ObstacleGuard calls this on its hot path -- keep it cheap."""
        if not self.enabled:
            return None
        with self._lock:
            ring = self._ring
        if ring is None:
            return None
        return {
            "n": self.ring_n,
            "bin_deg": self.ring_bin_deg,
            "start_deg": self.ring_start_deg,
            "dist": [None if not np.isfinite(v) else round(float(v), 3) for v in ring],
        }

    def front_points(self):
        """Kept near-ground DEPTH points as a flat [x0,y0,z0, ...] list (x-fwd, y-left,
        z = height above floor) -- same frame/units as the node's msg.points, so the 3D
        sphere can render the near-floor obstacles (e.g. a cable) the depth fusion sees
        but the up-tilted Mid-360 misses. None when disabled / frame rejected."""
        if not self.enabled:
            return None
        with self._lock:
            pts = self._points
        if pts is None or len(pts) == 0:
            return None
        return [round(float(v), 2) for v in np.asarray(pts, dtype=np.float32).ravel()]

    def map_cloud(self):
        """FULL leveled depth cloud (M,3) float32 in BODY floor-referenced frame (x-fwd,
        y-left, z = height above floor) for the mapping fusion -- NOT the narrow forward
        corridor/ring used by the obstacle guard, and not FOV-restricted. Reuses the same
        validated derotation as ``_compute`` (pitch + camera height) and the same
        frame-sanity gate, so a bad-pitch/height frame is never fed into the map. Pulls a
        FRESH cloud straight from the lidar source (not the cached near-field state) since
        callers (the map dumper) run on their own cadence. Returns None when disabled, no
        camera, no points, or the frame fails sanity."""
        if not self.enabled or self.lidar is None:
            return None
        cloud = self.lidar.get_cloud()
        if cloud is None or len(cloud) == 0:
            return None
        if self.frame_check:
            ramp, offset, nf = self._frame_sanity(cloud)
            if not (nf >= self.frame_min_floor_pts
                    and ramp <= self.frame_ramp_tol
                    and offset <= self.frame_offset_tol):
                return None
        X = cloud[:, 0]
        Y = cloud[:, 1]
        Zc = cloud[:, 2] - self.ls_mount_height
        fwd = X * self._ct + Zc * self._st
        up = -X * self._st + Zc * self._ct
        left = Y
        height = up + self.cam_h
        rng = np.hypot(fwd, left)
        keep = (
            (height > self.map_ground_clear) & (height < self.map_max_height)
            & (rng > self.map_min_range) & (rng < self.map_max_range)
        )
        if not np.any(keep):
            return None
        return np.column_stack((fwd[keep], left[keep], height[keep])).astype(np.float32)

    def full_points(self):
        """MINIMALLY-FILTERED DEPTH cloud (floor included, generous range/height) as a flat
        [x0,y0,z0, ...] list -- same frame as front_points() (x-fwd, y-left, z = height
        above floor). Backs the 3D sphere's "Full Depth" toggle, so the user can see
        literally what the D435i returns and compare it against the narrow ground-only
        safety band front_points() exposes. Unlike map_cloud() (which strips the floor and
        caps range/height for the SLAM export), this keeps the floor plane -- reuses the
        same validated derotation + frame-sanity gate, different mask. Still only ever the
        camera's actual forward ~87 deg wedge, not a software-imposed limit. Demand-gated
        by the caller (only invoke while the toggle is on). None when disabled / frame
        rejected / no points."""
        if not self.enabled or self.lidar is None:
            return None
        cloud = self.lidar.get_cloud()
        if cloud is None or len(cloud) == 0:
            return None
        if self.frame_check:
            ramp, offset, nf = self._frame_sanity(cloud)
            if not (nf >= self.frame_min_floor_pts
                    and ramp <= self.frame_ramp_tol
                    and offset <= self.frame_offset_tol):
                return None
        X = cloud[:, 0]
        Y = cloud[:, 1]
        Zc = cloud[:, 2] - self.ls_mount_height
        fwd = X * self._ct + Zc * self._st
        up = -X * self._st + Zc * self._ct
        left = Y
        height = up + self.cam_h
        rng = np.hypot(fwd, left)
        keep = (
            (height > self.full_min_height) & (height < self.full_max_height)
            & (rng > self.full_min_range) & (rng < self.full_max_range)
        )
        if not np.any(keep):
            return None
        out = np.column_stack((fwd[keep], left[keep], height[keep])).astype(np.float32)
        if out.shape[0] > self.full_cap:
            idx = np.random.choice(out.shape[0], self.full_cap, replace=False)
            out = out[idx]
        return [round(float(v), 2) for v in out.ravel()]

    def telemetry(self):
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "front_near_m": (round(self._front, 3) if self._front is not None else None),
                "n_near": int(self._n_near),
                "live": bool(self._live),
                "frame_ok": bool(self._frame_ok),        # watch this on-robot to validate the frame
                "floor_ramp_m": round(float(self._floor_ramp), 3),
                # Which frame_ok sub-check is failing, if any -- ramp>tol (bad pitch), or
                # offset>tol (bad height), or n_floor<min (no clean floor patch in view).
                "floor_offset_m": round(float(self._floor_offset), 3),
                "n_floor": int(self._floor_nf),
                "ramp_tol_m": self.frame_ramp_tol,
                "offset_tol_m": self.frame_offset_tol,
                "min_floor_pts": self.frame_min_floor_pts,
            }
