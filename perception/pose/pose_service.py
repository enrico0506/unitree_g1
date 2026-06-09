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


def build_items(result, labels):
    """Per-person geometry for the browser canvas: id, name, box, 17 keypoints.

    box  = [x1, y1, x2, y2] in source-frame pixels.
    kpts = [[x, y, conf], ...17]  (COCO order; conf 0-1). The browser hides joints
           and bones below KP_MIN_CONF.
    """
    items = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return items
    xyxy = boxes.xyxy.cpu().numpy()
    n = len(xyxy)
    # Track ids are OPTIONAL: ByteTrack may not assign one every frame. The skeleton
    # is drawn from keypoints regardless; the id (and operator name) is just an extra
    # when present. Gating the whole person on a track id would hide skeletons.
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * n
    kp = result.keypoints
    kdata = kp.data.cpu().numpy() if (kp is not None and kp.data is not None) else None
    for i in range(n):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        tid = ids[i] if i < len(ids) else None
        kpts = []
        if kdata is not None and i < len(kdata):
            for kx, ky, kc in kdata[i]:
                kpts.append([int(kx), int(ky), round(float(kc), 2)])
        items.append({"id": tid, "name": labels.get(tid) if tid is not None else "",
                      "box": [x1, y1, x2, y2], "kpts": kpts})
    return items


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.chdir(MODELS_DIR)   # weights/exports auto-download here (persisted volume)

    import torch
    from ultralytics import YOLO

    print(f"[pose_service] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={'cuda:0' if torch.cuda.is_available() else 'cpu'}", flush=True)
    model = YOLO(MODEL)
    print(f"[pose_service] model={MODEL} infer_hz={INFER_HZ} conf={CONF} imgsz={IMGSZ} "
          f"always_on={ALWAYS_ON}", flush=True)

    # Warm up the model + tracker now: the cold first inference does slow CUDA/JIT
    # and tracker init (tens of seconds on Jetson). Doing it at startup means the
    # first frame after the Skeleton toggle is fast, not a 30 s stall. Mirrors
    # detect_service's warmup.
    try:
        model.track(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), persist=True,
                    tracker="bytetrack.yaml", verbose=False)
        print("[pose_service] warmup done", flush=True)
    except Exception as e:
        print(f"[pose_service] warmup skipped: {e}", flush=True)

    labels = Labels()
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
            result = model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                                 tracker="bytetrack.yaml", verbose=False)[0]
            items = build_items(result, labels)
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
