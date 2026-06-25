#!/usr/bin/env python3
"""Standalone object-detection grabber + annotator (runs INSIDE the jetson docker image).

Mirrors perception/pose/pose_service.py: its own process, reads the latest head-
camera JPEG that camera_service.py already wrote to shared memory, runs a swappable
object detector (see detector.py), and atomically writes the detection GEOMETRY
(boxes + class + confidence) to shared memory as JSON. The browser draws the boxes
on a <canvas> over the raw feed -- so detection can show at the same time as the
skeleton overlay. We NEVER open the camera (single-consumer over DDS) -- we only
read the JPEG the existing grabber produces.

Data flow (all via /dev/shm, bind-mounted into the container):
    read   CAMERA_SHM     latest raw JPEG from camera_service.py
    write  DETECT_TRACKS  {w,h,items:[{cls,conf,box}]} geometry  -> /camera/detect/objects
                          (the browser draws boxes on a canvas over the raw feed)
    read   DETECT_DEMAND  heartbeat touched by the objects poll  -> only infer while watched

Config via environment variables (see run_detect.sh):
    DETECTOR_IMPL  which detector ("yoloworld" | "nanoowlsam" | "passthrough"; default yoloworld)
    MODEL          model passed to the detector (default yolov8s-worldv2.pt; for
                   nanoowlsam this is the OWL-ViT HF id, e.g. google/owlvit-base-patch32)
    MODELS_DIR     working dir for weights/engines (default /models, a mounted volume)
    DETECT_PROMPTS open-vocab class prompt (e.g. "door . person . chair")
    INFER_HZ       max inference rate (default 10) -- caps GPU load vs. the control loop
    CONF           detection confidence threshold (default 0.35; YOLO-World only)
    OWL_THRESHOLD  NanoOWL score threshold (default 0.1; OWL scores run lower than YOLO's)
    SEG_EVERYTHING "1" to add a costly class-agnostic mask pass (nanoowlsam only; default 0)
    IMGSZ          inference image size (default 640; YOLO-World only)
    ALWAYS_ON      "1" to ignore demand-gating (use for manual testing)
"""

import json
import os
import sys
import time

import cv2
import numpy as np

# The seam (detector.py) is mounted alongside this file at /app; make sure it is
# importable even after we os.chdir(MODELS_DIR) below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Paths (shared memory; bind-mounted from the host) ---
CAMERA_SHM = os.environ.get("CAMERA_SHM", "/dev/shm/g1_camera.jpg")
DETECT_SHM = os.environ.get("DETECT_SHM", "/dev/shm/g1_detect.jpg")
DETECT_TRACKS = os.environ.get("DETECT_TRACKS", "/dev/shm/g1_detect_tracks.json")
DETECT_DEMAND = os.environ.get("DETECT_DEMAND", "/dev/shm/g1_detect_demand")

# --- Tunables ---
DETECTOR_IMPL = os.environ.get("DETECTOR_IMPL", "yoloworld")
MODEL = os.environ.get("MODEL", "yolov8s-worldv2.pt")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DETECT_PROMPTS = os.environ.get(
    "DETECT_PROMPTS",
    "person . door . chair . table . monitor . laptop . keyboard . bottle . cup . backpack")
INFER_HZ = float(os.environ.get("INFER_HZ", "10"))
CONF = float(os.environ.get("CONF", "0.35"))
OWL_THRESHOLD = float(os.environ.get("OWL_THRESHOLD", "0.1"))
SEG_EVERYTHING = os.environ.get("SEG_EVERYTHING", "0") == "1"
IMGSZ = int(os.environ.get("IMGSZ", "640"))
ALWAYS_ON = os.environ.get("ALWAYS_ON", "0") == "1"

DT = 1.0 / INFER_HZ
DEMAND_TTL = 3.0          # infer only if DETECT_DEMAND was touched within this many seconds
CAMERA_FRESH_S = 2.0      # ignore the raw frame if it's older than this (camera down)


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
        return (time.time() - os.path.getmtime(DETECT_DEMAND)) < DEMAND_TTL
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


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.chdir(MODELS_DIR)   # weights/exports live here (persisted volume)

    from detector import make_detector
    detector = make_detector(DETECTOR_IMPL, MODEL, CONF, IMGSZ, DETECT_PROMPTS,
                             models_dir=MODELS_DIR, owl_threshold=OWL_THRESHOLD,
                             seg_everything=SEG_EVERYTHING)
    print(f"[detect_service] impl={DETECTOR_IMPL} model={MODEL} infer_hz={INFER_HZ} "
          f"conf={CONF} owl_threshold={OWL_THRESHOLD} seg_everything={SEG_EVERYTHING} "
          f"imgsz={IMGSZ} prompts={DETECT_PROMPTS!r} always_on={ALWAYS_ON}",
          flush=True)

    # Warm up now: the first inference does slow CUDA/JIT init (tens of seconds on
    # Jetson). Doing it at startup means the first frame after a viewer connects is
    # fast, instead of stalling the toggle for a minute.
    try:
        detector.detect(np.zeros((IMGSZ, IMGSZ, 3), np.uint8))
        print("[detect_service] warmup done", flush=True)
    except Exception as e:
        print(f"[detect_service] warmup skipped: {e}", flush=True)

    idle_logged = False
    while True:
        t0 = time.time()

        if not demand_fresh():
            if not idle_logged:
                print("[detect_service] no viewers -> idling (GPU free)", flush=True)
                idle_logged = True
            time.sleep(0.25)
            continue
        idle_logged = False

        frame = read_camera_frame()
        if frame is None:
            time.sleep(DT)
            continue

        try:
            dets = detector.detect(frame)
        except Exception as e:
            print(f"[detect_service] inference error: {e}", flush=True)
            time.sleep(DT)
            continue

        h, w = frame.shape[:2]
        atomic_write(DETECT_TRACKS,
                     json.dumps({"w": w, "h": h, "items": dets}).encode(), "wb")

        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
