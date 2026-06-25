#!/usr/bin/env python3
"""Hand-landmark service (runs INSIDE the mediapipe jetson docker image).

Mirrors perception/pose/pose_service.py: its own process, reads the latest
head-camera JPEG that camera_service.py wrote to shared memory, runs the
MediaPipe Hand Landmarker, and writes the per-hand GEOMETRY (21 finger landmarks
each) to shared memory as JSON. The dashboard draws the fingers on the same
<canvas> as the skeleton, tied to the SAME "Skeleton" toggle.

We NEVER open the camera: /dev/video0 is single-consumer and camera_service.py
already holds it. We only re-read the JPEG it produces in /dev/shm.

WRIST-ROI for distance
----------------------
MediaPipe resizes the whole input frame down to a small model input, so a hand
that is far away (only ~30-40 px in the 1920x1080 frame) shrinks to nothing and
is missed. Fix: use the POSE service's wrist keypoints (it shares the same camera
and rides the same Skeleton toggle) to crop a box around each wrist and run the
hand detector on just that crop -- now the hand fills the model input and detects
even across a room. Landmarks are then mapped back to full-frame pixels so they
still line up with the skeleton on the canvas. If pose has no usable wrist (or is
stale), we fall back to a single full-frame pass (good for close-up hands).

Data flow (all via /dev/shm, bind-mounted into the container):
    read   CAMERA_SHM     latest raw JPEG from camera_service.py
    read   POSE_TRACKS    pose geometry {items:[{kpts:[[x,y,c]x17]}]} -> wrist ROIs
    write  HANDS_TRACKS   {w,h,items:[{hand,score,landmarks:[[x,y,z]x21]}]} -> /camera/hands/tracks
    read   HANDS_DEMAND   heartbeat touched by the tracks poll -> only infer while watched

Config via environment variables (see run_hands.sh):
    MODEL_PATH      hand_landmarker.task bundle (default /app/models/hand_landmarker.task)
    INFER_HZ        max inference rate (default 10)
    MAX_HANDS       most hands per detect() call (default 2)
    MIN_DET_CONF    min hand-detection confidence (default 0.5)
    MIN_TRK_CONF    min tracking confidence (default 0.5)
    WRIST_MIN_CONF  min pose-keypoint confidence to trust a wrist (default 0.3)
    BOX_SCALE       wrist crop half-size as a multiple of forearm length (default 1.6)
    MIN_BOX_HALF    floor for the crop half-size in px (default 70)
    MAX_CROPS       cap on wrist crops per frame (default 4) -- bounds GPU load
    ALWAYS_ON       "1" to ignore demand-gating (manual testing)
"""

import json
import math
import os
import time

import cv2
import numpy as np

from hand_detector import HandDetector

# --- Paths (shared memory; bind-mounted from the host) ---
CAMERA_SHM = os.environ.get("CAMERA_SHM", "/dev/shm/g1_camera.jpg")
POSE_TRACKS = os.environ.get("POSE_TRACKS", "/dev/shm/g1_pose_tracks.json")
HANDS_TRACKS = os.environ.get("HANDS_TRACKS", "/dev/shm/g1_hands_tracks.json")
HANDS_DEMAND = os.environ.get("HANDS_DEMAND", "/dev/shm/g1_hands_demand")

# --- Tunables ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/hand_landmarker.task")
INFER_HZ = float(os.environ.get("INFER_HZ", "10"))
MAX_HANDS = int(os.environ.get("MAX_HANDS", "2"))
MIN_DET_CONF = float(os.environ.get("MIN_DET_CONF", "0.5"))
MIN_TRK_CONF = float(os.environ.get("MIN_TRK_CONF", "0.5"))
WRIST_MIN_CONF = float(os.environ.get("WRIST_MIN_CONF", "0.3"))
BOX_SCALE = float(os.environ.get("BOX_SCALE", "1.6"))
MIN_BOX_HALF = int(os.environ.get("MIN_BOX_HALF", "70"))
MAX_CROPS = int(os.environ.get("MAX_CROPS", "4"))
ALWAYS_ON = os.environ.get("ALWAYS_ON", "0") == "1"

DT = 1.0 / INFER_HZ
DEMAND_TTL = 3.0          # infer only if HANDS_DEMAND was touched within this many seconds
CAMERA_FRESH_S = 2.0      # ignore the raw frame if it's older than this (camera down)
POSE_FRESH_S = 2.0        # ignore pose tracks older than this (pose not running)
DEDUP_PX = 40             # two hands whose wrists land within this many px are the same hand

# COCO-17 pairs: (wrist index, elbow index) for the left and right arm. The wrist
# anchors the crop; the elbow gives the forearm length + direction so the crop is
# sized to the person and nudged toward where the hand actually is.
ARM_PAIRS = [(9, 7), (10, 8)]


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


def read_pose_tracks():
    """Latest pose geometry dict, or None if missing/stale (pose not running)."""
    try:
        if (time.time() - os.path.getmtime(POSE_TRACKS)) > POSE_FRESH_S:
            return None
        with open(POSE_TRACKS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def wrist_crops(pose, frame_w, frame_h):
    """Crop boxes (x0,y0,x1,y1) around each confident wrist in the pose tracks.

    The hand is distal to the wrist, so we centre the box a little past the wrist
    (away from the elbow) and size it to the forearm -- a far, small person gets a
    small tight crop, a close person a big one, but either way the hand fills it.
    """
    boxes = []
    for person in (pose.get("items", []) if pose else []):
        kpts = person.get("kpts", [])
        if len(kpts) < 17:
            continue
        for wrist_idx, elbow_idx in ARM_PAIRS:
            wx, wy, wc = kpts[wrist_idx]
            if wc < WRIST_MIN_CONF:
                continue
            ex, ey, ec = kpts[elbow_idx]
            forearm = 0.0
            if ec >= WRIST_MIN_CONF:
                forearm = math.hypot(wx - ex, wy - ey)

            half = max(forearm * BOX_SCALE, MIN_BOX_HALF)
            center_x, center_y = float(wx), float(wy)
            if forearm > 1.0:
                # push the centre half a forearm beyond the wrist, toward the hand
                center_x = wx + (wx - ex) / forearm * forearm * 0.5
                center_y = wy + (wy - ey) / forearm * forearm * 0.5

            x0 = max(0, int(center_x - half))
            y0 = max(0, int(center_y - half))
            x1 = min(frame_w, int(center_x + half))
            y1 = min(frame_h, int(center_y + half))
            if x1 - x0 > 10 and y1 - y0 > 10:
                boxes.append((x0, y0, x1, y1))
    return boxes[:MAX_CROPS]


def detect_hands(detector, frame):
    """All hands in `frame` as full-frame-pixel landmark dicts.

    Uses wrist-ROI crops from the pose tracks when available (detects far hands),
    else one full-frame pass (close-up hands when pose has no wrist).
    """
    frame_h, frame_w = frame.shape[:2]
    pose = read_pose_tracks()
    boxes = wrist_crops(pose, frame_w, frame_h)

    if not boxes:
        # Fallback: no usable wrist -> scan the whole frame (works close up).
        return detector.detect(frame)

    hands = []
    for (x0, y0, x1, y1) in boxes:
        crop = frame[y0:y1, x0:x1]
        for hand in detector.detect(crop):
            # crop-pixel landmarks -> full-frame pixels (just shift by the origin)
            shifted = []
            for lx, ly, lz in hand["landmarks"]:
                shifted.append([x0 + lx, y0 + ly, lz])
            hand["landmarks"] = shifted
            if not _is_duplicate(hand, hands):
                hands.append(hand)
    return hands


def _is_duplicate(hand, existing):
    """True if `hand`'s wrist sits on top of an already-found hand (overlapping crops)."""
    wx, wy = hand["landmarks"][0][0], hand["landmarks"][0][1]
    for other in existing:
        ox, oy = other["landmarks"][0][0], other["landmarks"][0][1]
        if math.hypot(wx - ox, wy - oy) < DEDUP_PX:
            return True
    return False


def main():
    print(f"[hands_service] model={MODEL_PATH} infer_hz={INFER_HZ} max_hands={MAX_HANDS} "
          f"det_conf={MIN_DET_CONF} wrist_roi=on box_scale={BOX_SCALE} max_crops={MAX_CROPS} "
          f"always_on={ALWAYS_ON}", flush=True)

    # IMAGE mode: we run the detector on independent wrist crops (and an occasional
    # full frame), so per-frame VIDEO tracking across changing crops doesn't apply.
    detector = HandDetector(
        model_path=MODEL_PATH,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_presence_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
        running_mode="image",
    )

    # Warm the GPU delegate once so the first real frame isn't slow.
    try:
        detector.detect(np.zeros((256, 256, 3), np.uint8))
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
            hands = detect_hands(detector, frame)
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
