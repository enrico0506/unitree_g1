#!/usr/bin/env bash
# Launch the people-pose service inside the Ultralytics jetson docker image.
# Reads the head-camera JPEG from /dev/shm (written by camera_service.py) and
# writes the annotated JPEG back to /dev/shm for the web server to serve.
#
# Runs DETACHED with a restart policy so the pose engine survives crashes AND a
# full power cycle: the docker daemon is enabled on boot and re-launches any
# container with a restart policy that wasn't explicitly stopped. The GPU stays
# demand-gated inside pose_service.py, so an always-present container still idles
# at ~0% GPU until someone watches the skeleton view.
#
# Env overrides (with defaults):
#   IMAGE=g1-pose:latest
#   MODEL=yolo11n-pose.pt      # swap to yolo11n-pose.engine after TensorRT export (Phase 4)
#   INFER_HZ=12  CONF=0.5  IMGSZ=640
#   ALWAYS_ON=0                # set 1 to ignore demand-gating (manual testing)
#   RESTART=unless-stopped     # set "no" for a throwaway run that dies with the host
set -euo pipefail

IMAGE="${IMAGE:-g1-pose:latest}"
MODEL="${MODEL:-yolo11n-pose.pt}"
INFER_HZ="${INFER_HZ:-12}"
CONF="${CONF:-0.5}"
IMGSZ="${IMGSZ:-640}"
ALWAYS_ON="${ALWAYS_ON:-0}"
RESTART="${RESTART:-unless-stopped}"
POSE_DIR="/home/unitree/projects/g1/perception/pose"

# Idempotent: drop any prior instance first. We no longer use --rm (it is
# mutually exclusive with --restart), so a leftover container with this name
# would otherwise make `docker run --name g1-pose` fail.
sudo docker rm -f g1-pose >/dev/null 2>&1 || true

sudo docker run -d --name g1-pose \
  --restart "${RESTART}" \
  --runtime nvidia --network host \
  -v /dev/shm:/dev/shm \
  -v "${POSE_DIR}:/app" \
  -v "${POSE_DIR}/models:/models" \
  -e MODEL="${MODEL}" \
  -e MODELS_DIR=/models \
  -e INFER_HZ="${INFER_HZ}" \
  -e CONF="${CONF}" \
  -e IMGSZ="${IMGSZ}" \
  -e ALWAYS_ON="${ALWAYS_ON}" \
  "${IMAGE}" \
  python3 /app/pose_service.py

echo "[run_pose] g1-pose started detached with --restart=${RESTART} (survives reboot)."
echo "[run_pose] Following logs below — Ctrl-C only stops the log tail; the container keeps running."
exec sudo docker logs -f g1-pose
