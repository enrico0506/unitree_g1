#!/usr/bin/env python3
"""Phase-1 smoke test: confirm YOLO11-pose runs on the GPU inside the container.

Loads yolo11n-pose, runs it on one frame (default: the saved head-camera frame),
prints torch/CUDA status + inference time, and saves an annotated frame.
Run inside the jetson docker image:

    docker run --rm --runtime nvidia -v /dev/shm:/dev/shm \
      -v /home/unitree/perception/pose:/app -v /home/unitree/perception/pose/models:/models \
      ultralytics/ultralytics:latest-jetson-jetpack5 \
      python3 /app/smoke_test.py /app/test_frame.jpg
"""

import os
import sys
import time

import cv2
import torch
from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else "/app/test_frame.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/app/test_frame_pose.jpg"
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

os.makedirs(MODELS_DIR, exist_ok=True)
os.chdir(MODELS_DIR)   # weights download here (persisted volume)

frame = cv2.imread(SRC)
if frame is None:
    raise SystemExit(f"could not read {SRC}")
print(f"frame: {frame.shape}", flush=True)

model = YOLO("yolo11n-pose.pt")

# Warm-up (first call includes CUDA init + graph build), then a timed run.
model.predict(frame, conf=0.5, verbose=False)
t0 = time.time()
result = model.predict(frame, conf=0.5, verbose=False)[0]
dt = (time.time() - t0) * 1000.0

n = 0 if result.boxes is None else len(result.boxes)
print(f"people detected: {n}", flush=True)
print(f"inference time (warmed): {dt:.1f} ms  (~{1000.0/dt:.1f} FPS single-frame)", flush=True)

annotated = result.plot()
cv2.imwrite(OUT, annotated)
print(f"saved annotated frame -> {OUT}", flush=True)
