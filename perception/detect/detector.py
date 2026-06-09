#!/usr/bin/env python3
"""Swappable object-detector seam for the G1 dashboard detect lane.

detect_service.py only ever calls `detector.detect(frame)` -- it never imports a
model library directly. To change models you write a new class implementing the
same contract and select it with the DETECTOR_IMPL env var; the service loop and
the whole dashboard stay untouched.

Contract:
    detect(frame) -> detections
        frame       : BGR np.ndarray (HxWx3), as returned by read_camera_frame()
        detections  : list[dict] {"cls": str, "conf": float, "box": [x1,y1,x2,y2]}
                      cls = human-readable class name; box = source-frame pixels.
                      The browser draws the boxes on a <canvas> over the raw feed,
                      so the detector returns geometry only -- it no longer renders
                      anything onto the image (that lets the detection + skeleton
                      overlays show at the same time).

Implementations:
    YoloWorldDetector   -- the live default: Ultralytics YOLO-World, OPEN-VOCABULARY
                           (classes set from a text prompt, e.g. "door . person").
                           Open-source and runs in the existing jetson image.
    PassthroughDetector -- dev aid, NOT a model: returns an empty detection list.
                           Lets the toggle/routes/coordinator/demand-gating be
                           validated end-to-end without any model.
"""

import re


class Detector:
    """Abstract detector seam. See module docstring for the contract."""

    def detect(self, frame):
        raise NotImplementedError


class PassthroughDetector(Detector):
    """Returns an empty detection list.

    Not a model -- purely to prove the plumbing (toggle, routes, demand-gating,
    canvas overlay) works before a real detector is wired in.
    """

    def detect(self, frame):
        return []


def _parse_prompts(prompts):
    """'door . person, chair' -> ['door', 'person', 'chair'] (period/comma separated)."""
    return [c.strip() for c in re.split(r"[.,]", prompts or "") if c.strip()]


class YoloWorldDetector(Detector):
    """Open-vocabulary detector via Ultralytics YOLO-World.

    Detects arbitrary classes given as a text prompt (DETECT_PROMPTS), e.g.
    "door . person . chair" -> set_classes(["door", "person", "chair"]). Runs in
    the jetson ultralytics image; the CLIP text-encoder weight is mounted from the
    project (models/clip/ViT-B-32.pt), so it runs fully offline.
    """

    def __init__(self, model_path, conf, imgsz, prompts):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(model_path)
        self.classes = _parse_prompts(prompts)
        if self.classes:
            self.model.set_classes(self.classes)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                    verbose=False)[0]
        names = result.names
        dets = []
        boxes = result.boxes
        if boxes is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.int().tolist()
            conf = boxes.conf.cpu().numpy()
            for i, c in enumerate(cls):
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                name = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
                dets.append({"cls": name, "conf": round(float(conf[i]), 2),
                             "box": [x1, y1, x2, y2]})
        return dets


def make_detector(impl, model_path, conf, imgsz, prompts):
    """Factory selected by the DETECTOR_IMPL env var."""
    impl = (impl or "").lower()
    if impl == "passthrough":
        return PassthroughDetector()
    if impl in ("yoloworld", "yolo-world", "yolo_world"):
        return YoloWorldDetector(model_path, conf, imgsz, prompts)
    raise ValueError(
        f"unknown DETECTOR_IMPL={impl!r} (expected 'yoloworld' or 'passthrough')")
