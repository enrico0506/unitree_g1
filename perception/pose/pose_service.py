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

Config via environment variables (see run_pose.sh):
    MODEL        model file (default yolo11n-pose.pt; swap to .engine after TensorRT export)
    MODELS_DIR   working dir for weights/exports (default /models, a mounted volume)
    INFER_HZ     max inference rate (default 12) -- caps GPU load vs. the control loop
    CONF         detection confidence threshold (default 0.5)
    IMGSZ        inference image size (default 640)
    ALWAYS_ON    "1" to ignore demand-gating (use for manual testing)
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

# --- Tunables ---
MODEL = os.environ.get("MODEL", "yolo11n-pose.pt")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
INFER_HZ = float(os.environ.get("INFER_HZ", "12"))
CONF = float(os.environ.get("CONF", "0.5"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
ALWAYS_ON = os.environ.get("ALWAYS_ON", "0") == "1"

DT = 1.0 / INFER_HZ
DEMAND_TTL = 3.0          # infer only if POSE_DEMAND was touched within this many seconds
CAMERA_FRESH_S = 2.0      # ignore the raw frame if it's older than this (camera down)
KP_MIN_CONF = 0.3         # keypoints below this confidence are reported but flagged


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


def read_camera_frame():
    """Latest raw JPEG -> BGR ndarray, or None if missing/stale/undecodable."""
    try:
        if (time.time() - os.path.getmtime(CAMERA_SHM)) > CAMERA_FRESH_S:
            return None
        with open(CAMERA_SHM, "rb") as f:
            buf = f.read()
    except OSError:
        return None
    if not buf:
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


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
          f"always_on={ALWAYS_ON}", flush=True)

    labels = Labels()
    tracker = SimpleTracker()
    idle_logged = False
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
            items = build_items(result, labels, tracker)
        except Exception as e:
            print(f"[pose_service] inference error: {e}", flush=True)
            time.sleep(DT)
            continue

        h, w = frame.shape[:2]
        atomic_write(POSE_TRACKS,
                     json.dumps({"w": w, "h": h, "items": items}).encode(), "wb")

        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
