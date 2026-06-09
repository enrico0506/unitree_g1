#!/usr/bin/env bash
# Launch the object-detection service inside the jetson docker image.
# Reads the head-camera JPEG from /dev/shm (written by camera_service.py) and
# writes the annotated JPEG back to /dev/shm for the web server to serve.
#
# The detector model is swappable via DETECTOR_IMPL (see detector.py):
#   yoloworld    -- Ultralytics YOLO-World, OPEN-VOCABULARY (default; detects the
#                   DETECT_PROMPTS classes, e.g. doors). Open-source, runs offline.
#   passthrough  -- echoes the frame, no boxes (validate the pipeline only)
#
# The CLIP text-encoder weight is mounted read-only from the project folder
# (models/clip/ViT-B-32.pt) to where Ultralytics expects it, so YOLO-World runs
# fully offline without baking the 354 MB weight into the image.
#
# Env overrides (with defaults):
#   IMAGE=g1-detect:latest        # built from ./Dockerfile (public ultralytics base + clip)
#   DETECTOR_IMPL=yoloworld
#   MODEL=yolov8s-worldv2.pt
#   DETECT_PROMPTS="person . door . ..."     # open-vocab classes (period/comma separated)
#   INFER_HZ=10  CONF=0.35  IMGSZ=640
#   ALWAYS_ON=0                   # set 1 to ignore demand-gating (manual testing)
#   RESTART=unless-stopped        # set "no" for a throwaway run that dies with the host
set -euo pipefail

IMAGE="${IMAGE:-g1-detect:latest}"
DETECTOR_IMPL="${DETECTOR_IMPL:-yoloworld}"
MODEL="${MODEL:-yolov8s-worldv2.pt}"
DETECT_PROMPTS="${DETECT_PROMPTS:-person . door . chair . table . monitor . laptop . keyboard . bottle . cup . backpack}"
INFER_HZ="${INFER_HZ:-10}"
CONF="${CONF:-0.35}"
IMGSZ="${IMGSZ:-640}"
ALWAYS_ON="${ALWAYS_ON:-0}"
RESTART="${RESTART:-unless-stopped}"
DETECT_DIR="/home/unitree/projects/g1/perception/detect"

# Idempotent: drop any prior instance first. We no longer use --rm (it is mutually
# exclusive with --restart), so a leftover container with this name would otherwise
# make `docker run --name g1-detect` fail.
sudo docker rm -f g1-detect >/dev/null 2>&1 || true

# Detached with a restart policy so the detector survives crashes AND a full power
# cycle: the docker daemon is enabled on boot and re-launches any container with a
# restart policy that wasn't explicitly stopped. The GPU stays demand-gated inside
# detect_service.py, so an always-present container still idles at ~0% GPU until
# someone watches the Object Detection view.
sudo docker run -d --name g1-detect \
  --restart "${RESTART}" \
  --runtime nvidia --network host \
  -v /dev/shm:/dev/shm \
  -v "${DETECT_DIR}:/app" \
  -v "${DETECT_DIR}/models:/models" \
  -v "${DETECT_DIR}/models/clip/ViT-B-32.pt:/ultralytics/weights/clip/ViT-B-32.pt:ro" \
  -e DETECTOR_IMPL="${DETECTOR_IMPL}" \
  -e MODEL="${MODEL}" \
  -e MODELS_DIR=/models \
  -e DETECT_PROMPTS="${DETECT_PROMPTS}" \
  -e INFER_HZ="${INFER_HZ}" \
  -e CONF="${CONF}" \
  -e IMGSZ="${IMGSZ}" \
  -e ALWAYS_ON="${ALWAYS_ON}" \
  "${IMAGE}" \
  python3 /app/detect_service.py

echo "[run_detect] g1-detect started detached with --restart=${RESTART} (survives reboot)."
echo "[run_detect] Following logs below — Ctrl-C only stops the log tail; the container keeps running."
exec sudo docker logs -f g1-detect
