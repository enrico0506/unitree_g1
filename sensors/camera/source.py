"""Camera frame provider for the web server.

The actual capture runs in a SEPARATE process (camera_service.py) with its own
DDS participant, so the camera RPC is fully isolated from locomotion RPC. That
process writes the latest JPEG to a shared-memory file; here we just read it.
"""

import os
import time

import cv2
import numpy as np

SHM_PATH = "/dev/shm/g1_camera.jpg"
FRESH_S = 2.0   # frame considered live if written within this many seconds


def _placeholder():
    img = np.full((720, 1280, 3), (21, 18, 14), np.uint8)
    cv2.putText(img, "NO CAMERA", (470, 375), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (77, 183, 255), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class CameraSource:
    def __init__(self, path=SHM_PATH):
        self.path = path
        self.backend = "videohub"
        self._placeholder = _placeholder()

    # The capture process is launched/stopped by the server; nothing to do here.
    def start(self):
        pass

    def stop(self):
        pass

    def get_jpeg(self):
        try:
            with open(self.path, "rb") as f:
                return f.read() or self._placeholder
        except OSError:
            return self._placeholder

    def is_live(self):
        try:
            return (time.time() - os.path.getmtime(self.path)) < FRESH_S
        except OSError:
            return False
