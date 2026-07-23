#!/usr/bin/env python3
"""person_fusion.py -- fuses person detections with depth into a bearing/distance
track, written to /dev/shm/g1_person_track.json for patrol_supervisor.py.

INPUT 1 -- person boxes (perception/detect/detect_service.py)
---------------------------------------------------------------
Reads /dev/shm/g1_detect_tracks.json: {"w": int, "h": int, "items": [{"cls":
str, "conf": float, "box": [x1,y1,x2,y2]}]} -- box is SOURCE-FRAME PIXELS (see
detect_service.py/detector.py: OWL-ViT boxes clamped to the frame, x1<x2,
y1<y2). Filtered to cls == "person" and conf >= min_conf. detect_service.py
does NOT track identity across frames (no tracker, just per-frame boxes), so
`track_id` in this node's output is always null -- there is no cross-frame
person identity anywhere in this perception stack yet; documented here rather
than silently invented.

BEARING -- reuses depth_nearfield.py's FOV convention, not its pixel math
---------------------------------------------------------------------------
scripts/depth_nearfield.py never actually does pixel-column -> bearing
arithmetic (it works entirely on the D435i's own pre-unprojected 3D cloud); it
DOES consistently use the D435i's datasheet horizontal FOV (87 deg / 1.5184
rad -- see its `ring_fov_half_deg: 43.0` default, identical to
g1_nav/config/nav2_params.yaml's `horizontal_fov_angle: 1.5184` for the same
camera) as its one angular reference. g1_nav.patrol_logic.pixel_x_to_bearing_rad
reuses that EXACT constant (D435I_HFOV_RAD) with a pinhole-model mapping from a
detection box's pixel center to a bearing (rad, 0 = boresight, + = LEFT, the
same sign convention as depth_nearfield's/obstacle/guard.py's ring `ang`).
Bearing is independent of the camera's ~47 deg DOWNWARD mount pitch (pitch only
changes the vertical FOV/appearance, not the horizontal boresight-to-target
angle for a camera with no roll), so no pitch correction is needed here.

DISTANCE -- an HONEST gap, flagged rather than faked
-------------------------------------------------------
Getting a real per-person RANGE needs the D435i depth image, but /dev/video0 is
SINGLE-OPEN and already held by robot_web_controller.py's LidarSource/
DepthNearField (see the "Sensor split + FOV" / single-open memory notes) --
this node runs in a SEPARATE process (g1_nav, ROS2) and cannot reopen it.
No continuously-available depth-cloud shm contract exists today. The ONE
depth cloud already exported to shm is robot_web_controller.py's
DepthMapDumper -> /dev/shm/g1_depth_for_map.bin, and it is written ONLY WHILE
A MAPPING SESSION IS ACTIVE (see that class's docstring) -- everywhere else
(i.e. normal patrol operation) it does not exist. This node reads it
OPPORTUNISTICALLY and BEST-EFFORT: when fresh, it derives a per-person range by
taking the MEDIAN horizontal range of the cloud points falling in a NARROWED
("central-region") angular window around the person's bearing, guarding
against too few points (NaN/sparse) exactly like depth_nearfield.py's own
min-points idiom; when the file is absent/stale (the common case), distance_m
is null and only bearing is reported. FOLLOW-UP NEEDED (flagged, not built
here, out of scope for this task): wire a small always-on (not
mapping-gated) leveled-depth-cloud shm export in robot_web_controller.py so
this node can range people during normal (non-mapping) patrol operation.

OUTPUT CONTRACT (/dev/shm/g1_person_track.json), atomic write:
    {"t": float, "frame_w": int, "frame_h": int,
     "persons": [{"bearing_rad": float, "distance_m": float|null,
                  "box": [x1,y1,x2,y2], "track_id": null}]}
`frame_w`/`frame_h` are ADDITIVE beyond the plan's minimum {t, persons:[...]}
shape (harmless -- JSON consumers ignore unknown keys) so patrol_supervisor.py
can compute a box-height-fraction "looks close" fallback when distance_m is
null (see that node's docstring: this reuses the SAME box-size proxy pattern
the plan itself already endorses for Track C's voice-command person-nearby
gate, "approximate via POSE_TRACKS box heuristic until Track A's real
bearing/distance fusion lands, then swap").
"""

import json
import math
import os
import struct
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node

from g1_nav.patrol_logic import pixel_x_to_bearing_rad, D435I_HFOV_RAD


# NOTE: this node deliberately does NOT import scripts/depth_nearfield.py or
# re-derive its pitch/mount-height derotation. The one depth cloud it reads
# (DEPTH_MAP_SHM below) is DepthNearField.map_cloud()'s OUTPUT -- already
# leveled (x-fwd, y-left, z=height-above-floor) by that module's own pitch
# correction upstream, inside robot_web_controller.py, before it ever reaches
# this shm file. Re-applying a pitch rotation here would double-correct it.
# We do NOT instantiate DepthNearField itself here either way (it needs a live
# LidarSource / open camera, which this process does not and must not hold --
# single-open /dev/video0).

DETECT_TRACKS = "/dev/shm/g1_detect_tracks.json"     # perception/detect/detect_service.py
DETECT_DEMAND = "/dev/shm/g1_detect_demand"           # heartbeat: keep the detector inferring
DEPTH_MAP_SHM = "/dev/shm/g1_depth_for_map.bin"       # robot_web_controller.py's DepthMapDumper
                                                        # -- ONLY present while mapping is active
OUT_PATH = "/dev/shm/g1_person_track.json"
YAML_PATH = "/home/unitree/projects/g1/g1_nav/config/patrol.yaml"

DEFAULTS = {
    "close_distance_m": 1.6,
    "min_conf": 0.35,
    "depth_min_points": 5,
    "depth_shm_stale_s": 1.0,
    "detect_demand_hz": 2.0,
}
DETECT_TRACKS_STALE_S = 1.0     # a stale/frozen detector feed reports zero persons, not stale ones
PUBLISH_HZ = 10.0               # this node's own tick rate (independent of detector's INFER_HZ)
CENTRAL_FRACTION = 0.5          # narrow the box's angular window to its central 50% before
                                 # taking the median range -- the closest available analogue to
                                 # "median over the box's central region" now that only a
                                 # bearing-space (not per-pixel image) depth source exists;
                                 # see module docstring's DISTANCE section.


def load_cfg():
    """Load config/patrol.yaml's `person_fusion:` block over the baked
    defaults; never raise (missing file / bad YAML / missing keys fall back)."""
    cfg = dict(DEFAULTS)
    try:
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        sub = data.get("person_fusion") if isinstance(data, dict) else None
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k in cfg and v is not None:
                    cfg[k] = v
    except (OSError, yaml.YAMLError):
        pass
    return cfg


def _read_depth_map_cloud(stale_s):
    """Best-effort read of robot_web_controller.py's DepthMapDumper cloud (see
    module docstring's DISTANCE section) -- an (N,3) float32 ndarray in
    body-relative (x-fwd, y-left, z=height-above-floor) metres, or None if the
    file is absent/stale/malformed (the common case outside of mapping).

    Wire format copied VERBATIM from DepthMapDumper.run() in
    scripts/robot_web_controller.py: struct.pack("<dI", t, n) then n*3 float32
    -- not reinvented here.
    """
    try:
        age = time.time() - os.path.getmtime(DEPTH_MAP_SHM)
    except OSError:
        return None
    if age > stale_s:
        return None
    try:
        with open(DEPTH_MAP_SHM, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if len(raw) < 12:
        return None
    try:
        _t, n = struct.unpack_from("<dI", raw, 0)
    except struct.error:
        return None
    if n <= 0 or len(raw) < 12 + n * 3 * 4:
        return None
    try:
        arr = np.frombuffer(raw, dtype="<f4", count=n * 3, offset=12).reshape(n, 3)
    except ValueError:
        return None
    return arr


def _bearing_window_distance(cloud, bearing_rad, half_width_rad, min_points):
    """Median horizontal range (m) of `cloud` points whose bearing falls within
    +-half_width_rad of `bearing_rad`, or None if fewer than min_points qualify
    (the NaN/sparse guard, matching depth_nearfield.py's own min-points idiom
    for exactly this kind of "not enough real data -> report absence, not a
    wrong number" failure mode)."""
    if cloud is None or len(cloud) == 0:
        return None
    fwd = cloud[:, 0].astype(np.float64)
    left = cloud[:, 1].astype(np.float64)
    finite = np.isfinite(fwd) & np.isfinite(left)
    if not np.any(finite):
        return None
    fwd = fwd[finite]
    left = left[finite]
    ang = np.arctan2(left, fwd)
    diff = np.abs(np.arctan2(np.sin(ang - bearing_rad), np.cos(ang - bearing_rad)))
    keep = diff <= half_width_rad
    n = int(np.count_nonzero(keep))
    if n < min_points:
        return None
    rng = np.hypot(fwd[keep], left[keep])
    return float(np.median(rng))


class PersonFusion(Node):
    def __init__(self):
        super().__init__("person_fusion")
        cfg = load_cfg()
        self.min_conf = float(self.declare_parameter("min_conf", cfg["min_conf"]).value)
        self.depth_min_points = int(
            self.declare_parameter("depth_min_points", cfg["depth_min_points"]).value)
        self.depth_stale_s = float(
            self.declare_parameter("depth_shm_stale_s", cfg["depth_shm_stale_s"]).value)
        demand_hz = float(
            self.declare_parameter("detect_demand_hz", cfg["detect_demand_hz"]).value)

        self._last_tracks_mtime = None

        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.create_timer(1.0 / max(0.1, demand_hz), self._touch_demand)

        self._write([], 0, 0)   # come up writing immediately (empty persons, never a missing file)
        self.get_logger().info(
            f"person_fusion: min_conf={self.min_conf} depth_min_points={self.depth_min_points} "
            f"hfov_rad={D435I_HFOV_RAD} -> {OUT_PATH}")

    # ------------------------------------------------------------- upstream
    def _touch_demand(self):
        """Keep detect_service.py inferring while this node is running (same
        idiom as GreetingService._touch_demand -- a heartbeat mtime touch)."""
        try:
            with open(DETECT_DEMAND, "wb") as f:
                f.write(b"1")
        except OSError:
            pass

    def _read_tracks(self):
        """Latest DETECT_TRACKS frame, or None if missing/stale/malformed. Does
        NOT gate on mtime-changed-since-last-read (unlike GreetingService) --
        this node runs on its OWN tick clock (PUBLISH_HZ) rather than reacting
        to each new detector frame, so re-reading an unchanged file is fine
        (and simpler): we always want to emit a fresh bearing/distance frame at
        our own rate even if the detector itself is momentarily slower."""
        try:
            age = time.time() - os.path.getmtime(DETECT_TRACKS)
        except OSError:
            return None
        if age > DETECT_TRACKS_STALE_S:
            return None
        try:
            with open(DETECT_TRACKS) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    # ------------------------------------------------------------------ tick
    def _tick(self):
        data = self._read_tracks()
        if data is None:
            self._write([], 0, 0)
            return

        w = data.get("w")
        h = data.get("h")
        items = data.get("items")
        if not isinstance(w, (int, float)) or w <= 0 or not isinstance(items, list):
            self._write([], 0, 0)
            return
        h = h if isinstance(h, (int, float)) and h > 0 else 0

        persons = [it for it in items
                   if isinstance(it, dict) and it.get("cls") == "person"
                   and isinstance(it.get("box"), (list, tuple)) and len(it["box"]) == 4
                   and float(it.get("conf", 0.0)) >= self.min_conf]
        if not persons:
            self._write([], int(w), int(h))
            return

        # Depth cloud is read ONCE per tick (best-effort; may be None -- see
        # module docstring's DISTANCE section) and shared across all persons
        # this frame, not re-read per-person.
        cloud = _read_depth_map_cloud(self.depth_stale_s)

        out = []
        for it in persons:
            x1, y1, x2, y2 = (float(v) for v in it["box"])
            if x2 <= x1 or y2 <= y1:
                continue                      # degenerate box -- skip (matches detector.py's own guard)
            cx = (x1 + x2) / 2.0
            bearing = pixel_x_to_bearing_rad(cx, float(w))

            # Angular half-width the box itself subtends, narrowed to its
            # CENTRAL fraction before sampling the depth cloud (see
            # CENTRAL_FRACTION comment above).
            b1 = pixel_x_to_bearing_rad(x1, float(w))
            b2 = pixel_x_to_bearing_rad(x2, float(w))
            half_width = abs(b1 - b2) / 2.0 * CENTRAL_FRACTION
            half_width = max(half_width, math.radians(1.0))   # floor: never a zero-width window

            distance = _bearing_window_distance(
                cloud, bearing, half_width, self.depth_min_points)

            out.append({
                "bearing_rad": round(bearing, 4),
                "distance_m": (None if distance is None else round(distance, 3)),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "track_id": None,   # no cross-frame identity in detect_service.py yet
            })

        self._write(out, int(w), int(h))

    # ------------------------------------------------------------- shm write
    def _write(self, persons, frame_w, frame_h):
        obj = {"t": round(time.time(), 3), "frame_w": int(frame_w), "frame_h": int(frame_h),
               "persons": persons}
        tmp = OUT_PATH + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, OUT_PATH)
        except OSError as e:
            self.get_logger().warn(f"shm write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PersonFusion()
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
