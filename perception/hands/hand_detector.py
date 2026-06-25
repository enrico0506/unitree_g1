#!/usr/bin/env python3
"""Importable hand + finger landmark detector (MediaPipe Hand Landmarker).

This is the FOUNDATION layer only: given a camera frame, return 21 finger
landmarks per hand. There is deliberately NO gesture recognition and NO
command/action logic here -- that is a later phase. Keep this module a pure
detector so the later gesture code can import it without dragging in the
service loop or any display code.

Why the MediaPipe Tasks API (not the older mp.solutions.hands):
  On this Jetson image (ghcr.io/lanzani/mediapipe:l4t35.4.1-...-mp0.10.7) the
  legacy mp.solutions.hands graph is broken ("Unable to find the type for
  stream 'image'"), but the modern Tasks HandLandmarker works. Tasks also needs
  a model file (hand_landmarker.task) which we ship in ./models.

Why the GPU delegate:
  On this build the CPU delegate loads but detects ZERO hands (the float16 model
  does not run correctly on the XNNPACK CPU path here), while the GPU delegate
  detects reliably at ~34 ms/frame. The container must therefore run with
  `--runtime nvidia` and `LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0`
  (see ../hands/run_hands.sh and Dockerfile).

The 21 landmarks use MediaPipe's standard order (same as any future gesture
model would expect):
    0 wrist
    1-4   thumb  (cmc, mcp, ip, tip)
    5-8   index  (mcp, pip, dip, tip)
    9-12  middle (mcp, pip, dip, tip)
    13-16 ring   (mcp, pip, dip, tip)
    17-20 pinky  (mcp, pip, dip, tip)
"""

from typing import List, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# A "hand" returned by detect() is a plain dict so it serialises straight to the
# shm JSON the dashboard reads. Shape:
#   {
#     "hand": "Left" | "Right" | "Unknown",   # handedness label
#     "score": float,                          # handedness confidence 0..1
#     "landmarks": [[x_px, y_px, z], ...21]    # x,y in SOURCE-frame pixels
#   }
# x,y are pixels in the ORIGINAL frame (normalised landmark * frame size) so they
# map through the exact same canvas projector the skeleton uses -> the fingers
# line up pixel-for-pixel with the body skeleton. z is MediaPipe's relative depth
# (smaller = closer to the camera), kept for later use; the overlay ignores it.
Hand = Dict


class HandDetector:
    """Wraps a MediaPipe Tasks HandLandmarker. One instance per process."""

    def __init__(
        self,
        model_path: str,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        running_mode: str = "video",
    ) -> None:
        # running_mode "video" tracks the hand between frames (faster + far less
        # jittery than re-detecting every frame); "image" re-detects each call and
        # is only used by the still-image test. Mirrors the spec's
        # static_image_mode=False default.
        if running_mode not in ("video", "image"):
            raise ValueError("running_mode must be 'video' or 'image'")
        self.running_mode = running_mode

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.GPU,
        )
        if running_mode == "video":
            mp_mode = mp_vision.RunningMode.VIDEO
        else:
            mp_mode = mp_vision.RunningMode.IMAGE
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> List[Hand]:
        """Run the landmarker on one BGR frame, return a list of hands.

        timestamp_ms is REQUIRED in video mode and must increase every call
        (MediaPipe rejects out-of-order timestamps). It is ignored in image mode.
        """
        # MediaPipe wants RGB; OpenCV frames are BGR. Convert before every call.
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError("video mode requires an increasing timestamp_ms")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        frame_h, frame_w = frame_bgr.shape[:2]
        return self._to_hands(result, frame_w, frame_h)

    def _to_hands(self, result, frame_w: int, frame_h: int) -> List[Hand]:
        hands: List[Hand] = []
        if not result.hand_landmarks:
            return hands

        for hand_index, landmarks in enumerate(result.hand_landmarks):
            # Handedness is reported from the IMAGE's point of view. Our camera
            # feed is not mirrored, so "Left"/"Right" may read swapped vs. the
            # person -- fine for Phase 1 (we only need the 21 points); a later
            # phase can correct it if a gesture needs true handedness.
            hand_label = "Unknown"
            hand_score = 0.0
            if result.handedness and hand_index < len(result.handedness):
                top = result.handedness[hand_index][0]
                hand_label = top.category_name
                hand_score = float(top.score)

            points = []
            for landmark in landmarks:
                x_px = int(round(landmark.x * frame_w))
                y_px = int(round(landmark.y * frame_h))
                z = round(float(landmark.z), 3)
                points.append([x_px, y_px, z])

            hands.append({
                "hand": hand_label,
                "score": round(hand_score, 2),
                "landmarks": points,
            })
        return hands

    def close(self) -> None:
        self._landmarker.close()
