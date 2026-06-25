#!/usr/bin/env python3
"""Hand-landmark service (runs INSIDE the mediapipe jetson docker image).

Mirrors perception/pose/pose_service.py exactly: its own process, reads the
latest head-camera JPEG that camera_service.py already wrote to shared memory,
runs the MediaPipe Hand Landmarker, and writes the per-hand GEOMETRY (21 finger
landmarks each) to shared memory as JSON. The dashboard draws the fingers on the
same <canvas> as the skeleton, tied to the SAME "Skeleton" toggle -- so when you
turn on Skeleton and a person is in frame, their hands get landmarked too.

We NEVER open the camera: /dev/video0 is single-consumer and camera_service.py
already holds it. We only re-read the JPEG it produces in /dev/shm.

Data flow (all via /dev/shm, bind-mounted into the container):
    read   CAMERA_SHM     latest raw JPEG from camera_service.py
    write  HANDS_TRACKS   {w,h,items:[{hand,score,landmarks:[[x,y,z]x21]}]} -> /camera/hands/tracks
    read   HANDS_DEMAND   heartbeat touched by the tracks poll -> only infer while watched

Config via environment variables (see run_hands.sh):
    MODEL_PATH   hand_landmarker.task bundle (default /app/models/hand_landmarker.task)
    INFER_HZ     max inference rate (default 10) -- caps GPU load (shared with pose/detect)
    MAX_HANDS    most hands to landmark per frame (default 2)
    MIN_DET_CONF min hand-detection confidence (default 0.6)
    MIN_TRK_CONF min tracking confidence (default 0.6)
    ALWAYS_ON    "1" to ignore demand-gating (use for manual testing)
"""

import json
import os
import time

import cv2
import numpy as np

from hand_detector import HandDetector

# --- Paths (shared memory; bind-mounted from the host) ---
CAMERA_SHM = os.environ.get("CAMERA_SHM", "/dev/shm/g1_camera.jpg")
HANDS_TRACKS = os.environ.get("HANDS_TRACKS", "/dev/shm/g1_hands_tracks.json")
HANDS_DEMAND = os.environ.get("HANDS_DEMAND", "/dev/shm/g1_hands_demand")

# --- Tunables ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/hand_landmarker.task")
INFER_HZ = float(os.environ.get("INFER_HZ", "10"))
MAX_HANDS = int(os.environ.get("MAX_HANDS", "2"))
MIN_DET_CONF = float(os.environ.get("MIN_DET_CONF", "0.6"))
MIN_TRK_CONF = float(os.environ.get("MIN_TRK_CONF", "0.6"))
ALWAYS_ON = os.environ.get("ALWAYS_ON", "0") == "1"

DT = 1.0 / INFER_HZ
DEMAND_TTL = 3.0          # infer only if HANDS_DEMAND was touched within this many seconds
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
        return (time.time() - os.path.getmtime(HANDS_DEMAND)) < DEMAND_TTL
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
    print(f"[hands_service] model={MODEL_PATH} infer_hz={INFER_HZ} max_hands={MAX_HANDS} "
          f"det_conf={MIN_DET_CONF} trk_conf={MIN_TRK_CONF} always_on={ALWAYS_ON}", flush=True)

    # VIDEO mode tracks hands between frames (faster + steadier than re-detecting).
    detector = HandDetector(
        model_path=MODEL_PATH,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_presence_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
        running_mode="video",
    )

    # VIDEO mode rejects out-of-order timestamps. Use a monotonic millisecond
    # clock (immune to wall-clock changes) and force it strictly increasing, so
    # the long demand-gated idles between bursts never feed a stale timestamp.
    def now_ms():
        return int(time.monotonic() * 1000)

    last_ts = -1

    # Warm the GPU delegate once on a black frame so the first REAL frame isn't
    # slow. (MediaPipe doesn't have YOLO's warmup-poison quirk; a plain warm is
    # enough -- see ../pose/pose_service.py for the YOLO-specific dance.)
    try:
        warm = np.zeros((480, 640, 3), np.uint8)
        last_ts = now_ms()
        detector.detect(warm, timestamp_ms=last_ts)
        print("[hands_service] gpu delegate warmed", flush=True)
    except Exception as e:
        print(f"[hands_service] warm skipped: {e}", flush=True)

    idle_logged = False
    while True:
        t0 = time.time()

        if not demand_fresh():
            if not idle_logged:
                print("[hands_service] no viewers -> idling (GPU free)", flush=True)
                idle_logged = True
            time.sleep(0.25)
            continue
        idle_logged = False

        frame = read_camera_frame()
        if frame is None:
            time.sleep(DT)
            continue

        try:
            ts = now_ms()
            if ts <= last_ts:        # keep timestamps strictly increasing for VIDEO mode
                ts = last_ts + 1
            last_ts = ts
            hands = detector.detect(frame, timestamp_ms=ts)
        except Exception as e:
            print(f"[hands_service] inference error: {e}", flush=True)
            time.sleep(DT)
            continue

        h, w = frame.shape[:2]
        atomic_write(HANDS_TRACKS,
                     json.dumps({"w": w, "h": h, "items": hands}).encode(), "wb")

        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
