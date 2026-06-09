#!/usr/bin/env python3
"""Standalone head-camera grabber.

Runs as its OWN process so it has its own DDS participant + RPC channels +
Python interpreter (own GIL), completely isolated from the main server's
locomotion RPC. A busy/blocked Move() can no longer starve the camera.

It pulls JPEG frames from the on-robot `videohub` service and writes the latest
one to a shared-memory file (tmpfs); the web server just reads that file to
serve the MJPEG stream.

Usage: camera_service.py [interface] [fps] [out_path]
"""

import os
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

IFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"
HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "/dev/shm/g1_camera.jpg"
DT = 1.0 / HZ
TMP = OUT + ".tmp"


def main():
    ChannelFactoryInitialize(0, IFACE)
    vc = VideoClient()
    vc.SetTimeout(1.0)
    vc.Init()
    print(f"[camera_service] videohub connected; writing {OUT} at {HZ:.0f} Hz",
          flush=True)
    while True:
        t0 = time.time()
        try:
            code, data = vc.GetImageSample()
        except Exception:
            code, data = -1, None
        if code == 0 and data:
            with open(TMP, "wb") as f:
                f.write(bytes(data))
            os.replace(TMP, OUT)   # atomic swap -> reader never sees a partial frame
        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)


if __name__ == "__main__":
    main()
