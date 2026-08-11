#!/usr/bin/env python3
"""Standalone infrared-camera grabber.

Mirrors camera_service.py, but reads the D435i's Infrared/stereo-module node
(/dev/video2, GREY format) directly via V4L2 instead of going through the
vendor `videohub` RPC, which only serves the RGB color stream. The IR node has
a wider vertical FOV (~58°, same stereo pair the depth stream is derived from)
than the RGB stream (~42°), which keeps a close-standing person in frame for
gesture/wave detection.

Usage: ir_service.py [device] [fps] [out_path]
"""

import os
import sys
import time

import cv2

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"
HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "/dev/shm/g1_camera_ir.jpg"
DT = 1.0 / HZ
TMP = OUT + ".tmp"


def main():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[ir_service] failed to open {DEVICE}", flush=True)
        sys.exit(1)
    print(f"[ir_service] {DEVICE} opened; writing {OUT} at {HZ:.0f} Hz", flush=True)
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if ok:
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                with open(TMP, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(TMP, OUT)   # atomic swap -> reader never sees a partial frame
        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
