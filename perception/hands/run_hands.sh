#!/usr/bin/env bash
# Launch the hand-landmark service inside the MediaPipe jetson docker image.
# Reads the head-camera JPEG from /dev/shm (written by camera_service.py) and
# writes the 21-finger-landmark geometry back to /dev/shm for the dashboard to
# draw on the skeleton canvas.
#
# Like run_pose.sh it runs DETACHED with a restart policy so it survives crashes
# AND a full power cycle. The GPU stays demand-gated inside hands_service.py, so
# an always-present container still idles at ~0% GPU until the Skeleton overlay
# is on (which is what heartbeats HANDS_DEMAND).
#
# REQUIRED runtime bits (both baked into the image / set here): --runtime nvidia
# (the MediaPipe graph needs an EGL/GPU context even with the CPU delegate, and
# the GPU delegate is the only one that detects on this build) and the
# LD_PRELOAD set by the Dockerfile.
#
# Env overrides (with defaults):
#   IMAGE=g1-hands:latest
#   INFER_HZ=10  MAX_HANDS=2  MIN_DET_CONF=0.6  MIN_TRK_CONF=0.6
#   ALWAYS_ON=0                # set 1 to ignore demand-gating (manual testing)
#   RESTART=unless-stopped     # set "no" for a throwaway run that dies with the host
set -euo pipefail

IMAGE="${IMAGE:-g1-hands:latest}"
INFER_HZ="${INFER_HZ:-10}"
MAX_HANDS="${MAX_HANDS:-2}"
MIN_DET_CONF="${MIN_DET_CONF:-0.6}"
MIN_TRK_CONF="${MIN_TRK_CONF:-0.6}"
ALWAYS_ON="${ALWAYS_ON:-0}"
RESTART="${RESTART:-unless-stopped}"
HANDS_DIR="/home/unitree/projects/g1/perception/hands"

# Idempotent: drop any prior instance first (no --rm; it conflicts with --restart).
sudo docker rm -f g1-hands >/dev/null 2>&1 || true

sudo docker run -d --name g1-hands \
  --restart "${RESTART}" \
  --runtime nvidia --network host \
  -v /dev/shm:/dev/shm \
  -v "${HANDS_DIR}:/app" \
  -e MODEL_PATH=/app/models/hand_landmarker.task \
  -e INFER_HZ="${INFER_HZ}" \
  -e MAX_HANDS="${MAX_HANDS}" \
  -e MIN_DET_CONF="${MIN_DET_CONF}" \
  -e MIN_TRK_CONF="${MIN_TRK_CONF}" \
  -e ALWAYS_ON="${ALWAYS_ON}" \
  "${IMAGE}" \
  python3 /app/hands_service.py

echo "[run_hands] g1-hands started detached with --restart=${RESTART} (survives reboot)."
echo "[run_hands] Following logs below — Ctrl-C only stops the log tail; the container keeps running."
exec sudo docker logs -f g1-hands
