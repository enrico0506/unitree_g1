#!/usr/bin/env python3
"""Diagnostic: visualise the wrist-ROI hand detection on the LIVE frame.

Reads the live camera JPEG + live pose tracks straight from /dev/shm, runs the
same wrist-ROI logic the service uses, and draws the crop boxes (green) and any
detected 21-point hands (blue) so we can see whether the crop lands on the hand
and whether detection fired. Writes /app/test_images/roi_now.jpg.
"""

import cv2
import numpy as np

from hand_detector import HandDetector
import hands_service as hs

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def main():
    frame = hs.read_camera_frame()
    if frame is None:
        print("no live camera frame"); return
    h, w = frame.shape[:2]

    pose = hs.read_pose_tracks()
    n_people = len(pose.get("items", [])) if pose else 0
    print(f"frame {w}x{h}  pose: {n_people} person(s)" + ("" if pose else " (stale/none)"))

    boxes = hs.wrist_crops(pose, w, h)
    print(f"wrist crop boxes: {boxes}")

    detector = HandDetector(
        model_path=hs.MODEL_PATH, max_num_hands=2,
        min_detection_confidence=0.4, min_presence_confidence=0.4,
        min_tracking_confidence=0.4, running_mode="image",
    )
    hands = hs.detect_hands(detector, frame)
    print(f"hands detected: {len(hands)}")
    for i, hand in enumerate(hands):
        print(f"  hand{i} {hand['hand']} score={hand['score']} wrist={tuple(hand['landmarks'][0][:2])}")

    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for hand in hands:
        lm = hand["landmarks"]
        for a, b in HAND_BONES:
            cv2.line(frame, (lm[a][0], lm[a][1]), (lm[b][0], lm[b][1]), (250, 200, 90), 2)
        for x, y, _z in lm:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    out = "/app/test_images/roi_now.jpg"
    cv2.imwrite(out, frame)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
