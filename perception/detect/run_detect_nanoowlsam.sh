#!/usr/bin/env bash
# Launch the NanoOWL + NanoSAM object-detection service inside the jetson docker image.
#
# Drop-in replacement for run_detect.sh: same /dev/shm contract (reads g1_camera.jpg,
# writes g1_detect_tracks.json), same container name "g1-detect" -> running this REPLACES
# the YOLO-World detect container. The detector adds a segmentation "mask" polygon to every
# detection; the dashboard renders the masks automatically (cam-overlay.js).
#
# Prereqs (one-time): build the image and populate the TensorRT engines --
#   sudo docker build -f Dockerfile.nanoowlsam -t g1-detect-nanoowlsam:latest .
#   ./prepare_nanoowlsam_engines.sh
# See BUILD_NANOOWLSAM.md for the full runbook and on-device verification.
#
# Env overrides (with defaults):
#   IMAGE=g1-detect-nanoowlsam:latest
#   MODEL=google/owlvit-base-patch32        # OWL-ViT HF id (text encoder + processor)
#   DETECT_PROMPTS="person . door . ..."    # open-vocab classes (period/comma separated)
#   INFER_HZ=6                              # OWL dominates the budget; 8-10 AGX Orin, 3-5 Orin Nano
#   OWL_THRESHOLD=0.1                       # OWL scores run lower than YOLO's CONF=0.35
#   SEG_EVERYTHING=0                        # set 1 for the costly class-agnostic mask pass
#   ALWAYS_ON=0                             # set 1 to ignore demand-gating (manual testing)
#   RESTART=unless-stopped                  # set "no" for a throwaway run that dies with the host
set -euo pipefail

IMAGE="${IMAGE:-g1-detect-nanoowlsam:latest}"
MODEL="${MODEL:-google/owlvit-base-patch32}"
DETECT_PROMPTS="${DETECT_PROMPTS:-person . door . chair . table . monitor . laptop . keyboard . bottle . cup . backpack}"
INFER_HZ="${INFER_HZ:-6}"
OWL_THRESHOLD="${OWL_THRESHOLD:-0.1}"
SEG_EVERYTHING="${SEG_EVERYTHING:-0}"
ALWAYS_ON="${ALWAYS_ON:-0}"
RESTART="${RESTART:-unless-stopped}"
DETECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All THREE engines are required (detector.py raises if any is missing). The OWL one is
# built locally and least likely to be absent; the SAM ones depend on a fragile extract,
# so check each so a partial prepare fails on the host, not as a container crash.
for eng in owl_image_encoder_patch32.engine resnet18_image_encoder.engine mobile_sam_mask_decoder.engine; do
  if [ ! -f "${DETECT_DIR}/models/data/${eng}" ]; then
    echo "[run_detect_nanoowlsam] ERROR: missing TensorRT engine models/data/${eng}." >&2
    echo "  Run ./prepare_nanoowlsam_engines.sh first (see BUILD_NANOOWLSAM.md)." >&2
    exit 1
  fi
done

# Idempotent: drop any prior instance (YOLO or nanoowlsam) using the same name.
sudo docker rm -f g1-detect >/dev/null 2>&1 || true

# Detached + restart policy so the detector survives crashes and a power cycle. The GPU
# stays demand-gated inside detect_service.py, so an always-present container still idles
# at ~0% GPU until someone watches the Object Detection view.
sudo docker run -d --name g1-detect \
  --restart "${RESTART}" \
  --runtime nvidia --network host \
  -v /dev/shm:/dev/shm \
  -v "${DETECT_DIR}:/app" \
  -v "${DETECT_DIR}/models:/models" \
  -e DETECTOR_IMPL=nanoowlsam \
  -e MODEL="${MODEL}" \
  -e MODELS_DIR=/models \
  -e DETECT_PROMPTS="${DETECT_PROMPTS}" \
  -e INFER_HZ="${INFER_HZ}" \
  -e OWL_THRESHOLD="${OWL_THRESHOLD}" \
  -e SEG_EVERYTHING="${SEG_EVERYTHING}" \
  -e ALWAYS_ON="${ALWAYS_ON}" \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  "${IMAGE}" \
  python3 /app/detect_service.py

echo "[run_detect_nanoowlsam] g1-detect (nanoowlsam) started detached with --restart=${RESTART}."
echo "[run_detect_nanoowlsam] Following logs below — Ctrl-C only stops the log tail; the container keeps running."
exec sudo docker logs -f g1-detect
