#!/usr/bin/env python3
"""Standalone people-pose grabber + annotator (runs INSIDE the jetson docker image).

Mirrors scripts/camera_service.py: its own process, reads the latest head-camera
JPEG that camera_service.py already wrote to shared memory, runs Ultralytics
YOLO11-pose with ByteTrack, and writes the per-person skeleton GEOMETRY (keypoints
+ box + stable track id) to shared memory as JSON. The browser draws the skeleton
on a <canvas> over the raw feed -- so it can show at the same time as the object-
detection overlay, and no annotated video stream is needed. We NEVER open the
camera (single-consumer over DDS) -- we only read the JPEG the existing grabber
produces.

Data flow (all via /dev/shm, bind-mounted into the container):
    read   CAMERA_SHM     latest raw JPEG from camera_service.py
    write  POSE_TRACKS    {w,h,items:[{id,name,box,kpts}]} geometry -> /camera/pose/tracks
    read   POSE_LABELS    {"<id>": "name"} operator labels     <- /camera/pose/label
    read   POSE_DEMAND    heartbeat touched by the tracks poll -> only infer while watched

IR fallback: the RGB head camera has a narrow ~42 deg vertical FOV, so a person
standing close has their head cropped out of frame -- no head keypoints, and
gesture_reactor's wave-back logic starves for confident keypoints. The D435i's
IR/stereo node (scripts/ir_service.py) has a wider ~58 deg FOV. When the RGB
frame looks head-cropped, this process opportunistically decodes and infers on
the IR JPEG instead (same already-loaded model, no second GPU-resident copy),
publishing IR-native w/h to POSE_TRACKS for that tick -- the browser overlay and
gesture_reactor already read w/h per-payload, so no downstream changes needed.
    read   IR_CAMERA_SHM  latest raw IR JPEG from ir_service.py
    write  IR_DEMAND      heartbeat asking robot_web_controller to keep ir_service.py alive

Config via environment variables (see run_pose.sh):
    MODEL        model file (default yolo11n-pose.pt; swap to .engine after TensorRT export)
    MODELS_DIR   working dir for weights/exports (default /models, a mounted volume)
    INFER_HZ     max inference rate (default 12) -- caps GPU load vs. the control loop
    CONF         detection confidence threshold (default 0.5)
    IMGSZ        inference image size (default 640)
    ALWAYS_ON    "1" to ignore demand-gating (use for manual testing)
    IR_FALLBACK          "1" to enable the IR fallback above (default on)
    IR_EDGE_MARGIN_PX    box.y1 must be within this many px of the top edge (default 6)
    IR_MIN_BOX_FRAC_H    box height must exceed this fraction of frame height (default 0.25)
    IR_FALLBACK_DWELL_S  min seconds to stay on IR once switched (default 2.0)
    IR_RECOVER_STABLE_N  consecutive clean RGB frames needed to switch back (default 3)
    IR_FRESH_S           ignore the IR frame if older than this (default 1.0)
"""

import json
import os
import time

import cv2
import numpy as np

# --- Paths (shared memory; bind-mounted from the host) ---
CAMERA_SHM = os.environ.get("CAMERA_SHM", "/dev/shm/g1_camera.jpg")
POSE_SHM = os.environ.get("POSE_SHM", "/dev/shm/g1_pose.jpg")
POSE_TRACKS = os.environ.get("POSE_TRACKS", "/dev/shm/g1_pose_tracks.json")
POSE_LABELS = os.environ.get("POSE_LABELS", "/dev/shm/g1_pose_labels.json")
POSE_DEMAND = os.environ.get("POSE_DEMAND", "/dev/shm/g1_pose_demand")
IR_CAMERA_SHM = os.environ.get("IR_CAMERA_SHM", "/dev/shm/g1_camera_ir.jpg")
IR_DEMAND = os.environ.get("IR_DEMAND", "/dev/shm/g1_ir_demand")

# --- Tunables ---
MODEL = os.environ.get("MODEL", "yolo11n-pose.pt")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
INFER_HZ = float(os.environ.get("INFER_HZ", "12"))
CONF = float(os.environ.get("CONF", "0.5"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
ALWAYS_ON = os.environ.get("ALWAYS_ON", "0") == "1"
IR_FALLBACK = os.environ.get("IR_FALLBACK", "1") == "1"
IR_EDGE_MARGIN_PX = float(os.environ.get("IR_EDGE_MARGIN_PX", "6"))
IR_MIN_BOX_FRAC_H = float(os.environ.get("IR_MIN_BOX_FRAC_H", "0.25"))
IR_FALLBACK_DWELL_S = float(os.environ.get("IR_FALLBACK_DWELL_S", "2.0"))
IR_RECOVER_STABLE_N = int(os.environ.get("IR_RECOVER_STABLE_N", "3"))
IR_FRESH_S = float(os.environ.get("IR_FRESH_S", "1.0"))

DT = 1.0 / INFER_HZ
DEMAND_TTL = 3.0          # infer only if POSE_DEMAND was touched within this many seconds
CAMERA_FRESH_S = 2.0      # ignore the raw frame if it's older than this (camera down)
KP_MIN_CONF = 0.3         # keypoints below this confidence are reported but flagged
HEAD_KPT_IDXS = (0, 1, 2, 3, 4)   # COCO order: nose, l_eye, r_eye, l_ear, r_ear


def atomic_write(path, data, mode="wb"):
    """Write then os.replace -- a reader never sees a partial file."""
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def demand_fresh():
    if ALWAYS_ON:
        return True
    try:
        return (time.time() - os.path.getmtime(POSE_DEMAND)) < DEMAND_TTL
    except OSError:
        return False


def read_frame(path, fresh_s):
    """Latest raw JPEG at `path` -> BGR ndarray, or None if missing/stale/undecodable."""
    try:
        if (time.time() - os.path.getmtime(path)) > fresh_s:
            return None
        with open(path, "rb") as f:
            buf = f.read()
    except OSError:
        return None
    if not buf:
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def read_camera_frame():
    return read_frame(CAMERA_SHM, CAMERA_FRESH_S)


def touch_ir_demand():
    """Heartbeat asking robot_web_controller.py to keep ir_service.py running."""
    try:
        with open(IR_DEMAND, "wb") as f:
            f.write(b"1")
    except OSError:
        pass


def head_cropped(items, frame_h):
    """True if a sizeable track has no confident head keypoint AND its box touches
    the top edge -- a proxy for "the FOV cropped the head off", not "the person
    turned away" (which wouldn't touch the edge, so shouldn't trigger a camera switch).
    """
    for it in items:
        kpts = it.get("kpts") or []
        if len(kpts) <= max(HEAD_KPT_IDXS):
            continue
        if any(kpts[i][2] >= KP_MIN_CONF for i in HEAD_KPT_IDXS):
            continue
        box = it.get("box") or [0, 0, 0, 0]
        if box[1] > IR_EDGE_MARGIN_PX:
            continue
        if (box[3] - box[1]) <= IR_MIN_BOX_FRAC_H * frame_h:
            continue
        return True
    return False


def step_source(source, cropped, t0, switched_at, good_rgb_streak):
    """Advance the rgb/ir fallback state machine one tick.

    source: "rgb" or "ir" going in. cropped: this tick's head_cropped() verdict on the
    RGB frame (always evaluated, even while sourcing from IR, so recovery can be
    detected). Returns (new_source, new_switched_at, new_good_rgb_streak). Switching
    is debounced both ways: IR_FALLBACK_DWELL_S before we'll consider recovering, and
    IR_RECOVER_STABLE_N consecutive clean RGB reads before we actually do.
    """
    if source == "rgb":
        if cropped:
            return "ir", t0, 0
        return "rgb", switched_at, good_rgb_streak
    good_rgb_streak = 0 if cropped else good_rgb_streak + 1
    if (t0 - switched_at) >= IR_FALLBACK_DWELL_S and good_rgb_streak >= IR_RECOVER_STABLE_N:
        return "rgb", switched_at, good_rgb_streak
    return "ir", switched_at, good_rgb_streak


class Labels:
    """operator id->name map, reloaded from POSE_LABELS only when it changes."""

    def __init__(self):
        self._map = {}
        self._mtime = 0.0

    def get(self, tid):
        try:
            mt = os.path.getmtime(POSE_LABELS)
            if mt != self._mtime:
                with open(POSE_LABELS) as f:
                    self._map = json.load(f) or {}
                self._mtime = mt
        except (OSError, ValueError):
            pass
        return self._map.get(str(tid), "")


def _iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class SimpleTracker:
    """Tiny greedy-IoU tracker for STABLE person ids across frames.

    We drive detection with model.predict (stateless and reliable) instead of
    model.track -- Ultralytics' built-in ByteTrack, run persistently in this
    long-lived demand-gated process, would intermittently stop emitting any
    detections at all. This assigns ids ourselves so the People naming feature
    keeps working, while detection stays rock-solid.
    """

    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}        # id -> [box, frames_since_seen]
        self.next_id = 1

    def update(self, boxes):
        """boxes: list of [x1,y1,x2,y2]. Returns a parallel list of ids."""
        ids = [None] * len(boxes)
        used = set()
        for bi, box in enumerate(boxes):
            best_id, best = None, self.iou_thresh
            for tid, (tbox, _age) in self.tracks.items():
                if tid in used:
                    continue
                v = _iou(box, tbox)
                if v >= best:
                    best, best_id = v, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            ids[bi] = best_id
            self.tracks[best_id] = [box, 0]
        for tid in list(self.tracks):          # age out tracks not seen this frame
            if tid not in used:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_age:
                    del self.tracks[tid]
        return ids


def build_items(result, labels, tracker):
    """Per-person geometry for the browser canvas: id, name, box, 17 keypoints.

    box  = [x1, y1, x2, y2] in source-frame pixels.
    kpts = [[x, y, conf], ...17]  (COCO order; conf 0-1). The browser hides joints
           and bones below KP_MIN_CONF. Stable ids come from `tracker`.
    """
    items = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        tracker.update([])                     # age out existing tracks
        return items
    xyxy = boxes.xyxy.cpu().numpy()
    n = len(xyxy)
    box_list = [[int(v) for v in xyxy[i]] for i in range(n)]
    ids = tracker.update(box_list)
    kp = result.keypoints
    kdata = kp.data.cpu().numpy() if (kp is not None and kp.data is not None) else None
    for i in range(n):
        tid = ids[i]
        kpts = []
        if kdata is not None and i < len(kdata):
            for kx, ky, kc in kdata[i]:
                kpts.append([int(kx), int(ky), round(float(kc), 2)])
        items.append({"id": tid, "name": labels.get(tid) if tid is not None else "",
                      "box": box_list[i], "kpts": kpts})
    return items


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.chdir(MODELS_DIR)   # weights/exports auto-download here (persisted volume)

    import torch
    from ultralytics import YOLO

    print(f"[pose_service] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={'cuda:0' if torch.cuda.is_available() else 'cpu'}", flush=True)
    # GPU quirk on this Jetson image (debugged at length 2026-06-09): the FIRST
    # model instance's CUDA inference in a process returns ZERO detections until
    # CUDA is initialized, AND warming a model's OWN predictor poisons it (it then
    # returns nothing on the real camera frames -> the skeleton silently never
    # appears). Fix: warm CUDA with a THROWAWAY instance we discard, then create the
    # real model and send it straight to the loop -- its first real predict sets up
    # its own clean predictor on the already-initialized CUDA context. (Verified: a
    # second, never-self-warmed instance detected reliably while the first returned
    # nothing on the identical frame.)
    try:
        _warm = YOLO(MODEL)
        _warm.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, verbose=False)
        del _warm
        print("[pose_service] cuda warmed (throwaway instance)", flush=True)
    except Exception as e:
        print(f"[pose_service] cuda warm skipped: {e}", flush=True)

    model = YOLO(MODEL)
    print(f"[pose_service] model={MODEL} infer_hz={INFER_HZ} conf={CONF} imgsz={IMGSZ} "
          f"always_on={ALWAYS_ON} ir_fallback={IR_FALLBACK}", flush=True)

    labels = Labels()
    tracker = SimpleTracker()
    ir_tracker = SimpleTracker()
    idle_logged = False
    source = "rgb"          # or "ir", while a close person's head is cropped from RGB
    switched_at = 0.0
    good_rgb_streak = 0
    while True:
        t0 = time.time()

        if not demand_fresh():
            if not idle_logged:
                print("[pose_service] no viewers -> idling (GPU free)", flush=True)
                idle_logged = True
            time.sleep(0.25)
            continue
        idle_logged = False

        frame = read_camera_frame()
        if frame is None:
            time.sleep(DT)
            continue

        try:
            result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            rgb_items = build_items(result, labels, tracker)
        except Exception as e:
            print(f"[pose_service] inference error: {e}", flush=True)
            time.sleep(DT)
            continue

        rgb_h, rgb_w = frame.shape[:2]

        if IR_FALLBACK:
            cropped = head_cropped(rgb_items, rgb_h)
            prev_source = source
            source, switched_at, good_rgb_streak = step_source(
                source, cropped, t0, switched_at, good_rgb_streak)
            if source != prev_source:
                print(f"[pose_service] {prev_source} -> {source} fallback switch", flush=True)
        else:
            source = "rgb"

        if source == "ir":
            touch_ir_demand()
            ir_frame = read_frame(IR_CAMERA_SHM, IR_FRESH_S)
            if ir_frame is None:
                # ir_service.py may still be starting up (or stopped) -- nothing
                # fresh to publish this tick; try again next tick.
                dt = time.time() - t0
                if dt < DT:
                    time.sleep(DT - dt)
                continue
            try:
                ir_result = model.predict(ir_frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
                items = build_items(ir_result, labels, ir_tracker)
            except Exception as e:
                print(f"[pose_service] IR inference error: {e}", flush=True)
                items = []
            h, w = ir_frame.shape[:2]
        else:
            items, h, w = rgb_items, rgb_h, rgb_w

        atomic_write(POSE_TRACKS,
                     json.dumps({"w": w, "h": h, "items": items}).encode(), "wb")

        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
