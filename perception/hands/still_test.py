#!/usr/bin/env python3
"""Still-image hand-landmark test (M1) -- the dev/verification visualiser.

Kept SEPARATE from hand_detector.py on purpose: this draws and saves annotated
images for eyeballing, while hand_detector.py stays a clean importable detector.
The real LIVE visualisation is the dashboard canvas overlay, not an on-screen
window (the robot is headless), so this tool writes annotated JPEGs to disk.

Run inside the mediapipe container, e.g.:
    LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 \
      python3 /app/still_test.py /app/test_images/*.jpg
Outputs <name>_hands.jpg next to each input.
"""

import os
import sys
import time

import cv2

from hand_detector import HandDetector


MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/hand_landmarker.task")

# Standard MediaPipe 21-landmark hand connections (same list the dashboard uses).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


def draw_hand(image, hand) -> None:
    points = hand["landmarks"]
    for a, b in HAND_BONES:
        ax, ay = points[a][0], points[a][1]
        bx, by = points[b][0], points[b][1]
        cv2.line(image, (ax, ay), (bx, by), (90, 200, 250), 2)
    for x, y, _z in points:
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    wrist = points[0]
    label = hand["hand"] + " " + str(hand["score"])
    cv2.putText(image, label, (wrist[0], wrist[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    image_paths = sys.argv[1:]
    if not image_paths:
        print("usage: still_test.py IMAGE [IMAGE ...]")
        sys.exit(1)

    # Permissive thresholds + up to 4 hands so we can see what the model finds.
    detector = HandDetector(
        model_path=MODEL_PATH,
        max_num_hands=4,
        min_detection_confidence=0.4,
        min_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode="image",
    )

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: could not read")
            continue
        h, w = image.shape[:2]

        start = time.time()
        hands = detector.detect(image)
        elapsed_ms = (time.time() - start) * 1000.0

        print(f"{path} ({w}x{h}): {len(hands)} hand(s) in {elapsed_ms:.0f}ms")
        for i, hand in enumerate(hands):
            tip = hand["landmarks"][8]   # index fingertip
            print(f"  hand{i} {hand['hand']} score={hand['score']} "
                  f"wrist={tuple(hand['landmarks'][0][:2])} index_tip={tuple(tip[:2])}")
            draw_hand(image, hand)

        out_path = os.path.splitext(path)[0] + "_hands.jpg"
        cv2.imwrite(out_path, image)
        print(f"  -> {out_path}")

    detector.close()


if __name__ == "__main__":
    main()
