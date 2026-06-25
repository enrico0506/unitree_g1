#!/usr/bin/env python3
"""Swappable object-detector seam for the G1 dashboard detect lane.

detect_service.py only ever calls `detector.detect(frame)` -- it never imports a
model library directly. To change models you write a new class implementing the
same contract and select it with the DETECTOR_IMPL env var; the service loop and
the whole dashboard stay untouched.

Contract:
    detect(frame) -> detections
        frame       : BGR np.ndarray (HxWx3), as returned by read_camera_frame()
        detections  : list[dict] {"cls": str, "conf": float, "box": [x1,y1,x2,y2],
                                   "mask": [poly, ...]   # OPTIONAL}
                      cls  = human-readable class name
                      box  = source-frame pixels
                      mask = OPTIONAL list of polygons; each polygon is a FLAT
                             [x0,y0,x1,y1,...] list of source-frame pixel ints
                             (same coordinate space as box). Detectors that don't
                             segment simply omit it.
                      The browser draws boxes + filled mask polygons on a <canvas>
                      over the raw feed, so the detector returns geometry only -- it
                      never renders onto the image (that lets the detection + skeleton
                      overlays show at the same time).

Implementations:
    NanoOwlSamDetector  -- the live default: NanoOWL (TensorRT OWL-ViT) open-vocab
                           labeled boxes + NanoSAM (TensorRT ResNet18 encoder +
                           MobileSAM decoder) masks: every OWL box is segmented into a
                           polygon, so you get BOTH labels and full per-object coverage.
                           Class-agnostic NMS dedupes OWL's overlapping boxes. Optional
                           "segment everything" pass (SEG_EVERYTHING). Needs the
                           g1-detect-nanoowlsam image + prebuilt TensorRT engines
                           (see BUILD_NANOOWLSAM.md).
    PassthroughDetector -- dev aid, NOT a model: returns an empty detection list.
                           Lets the toggle/routes/coordinator/demand-gating be
                           validated end-to-end without any model.
"""

import os
import re

import cv2
import numpy as np


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


# --- mask -> compact polygon helpers (shared by the NanoSAM paths) -----------

def mask_to_polygons(mask, epsilon_frac=0.01, min_area_px=200.0,
                     max_points=40, max_polys=2):
    """Boolean/uint8 HxW mask -> list of polygons.

    Each polygon is a FLAT [x0,y0,x1,y1,...] list of SOURCE-frame pixel ints -- the
    SAME coordinate space as the detection "box" -- so the browser maps masks and
    boxes through one identical letterbox projector. Contours are simplified with
    approxPolyDP to keep the JSON small (~<1 KB/poly); specks below min_area_px are
    dropped, and point count is capped so one blob can't bloat the payload. Returns
    [] when the mask yields no usable contour (caller then omits the "mask" field).
    """
    m = np.ascontiguousarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # largest blobs first so a per-box mask keeps the object, not a speck
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polys = []
    for cnt in contours:
        if len(polys) >= max_polys:
            break
        if cv2.contourArea(cnt) < min_area_px:
            continue
        peri = cv2.arcLength(cnt, True)
        frac = epsilon_frac
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        while len(approx) > max_points and frac < 0.2:   # bound JSON size
            frac *= 1.5
            approx = cv2.approxPolyDP(cnt, frac * peri, True)
        if len(approx) < 3:
            continue
        polys.append([int(v) for v in approx.reshape(-1).tolist()])
    return polys


def _to_numpy(x):
    """Torch tensor OR array-like -> np.ndarray (no device assumptions)."""
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _scalar(x):
    arr = _to_numpy(x).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _mask_to_bool(mask, iou=None):
    """NanoSAM logits (1,num_masks,H,W) -> boolean HxW, thresholded at 0.

    The MobileSAM decoder returns several mask candidates UNSORTED, so slot 0 is not
    the best one. When the predicted-IoU vector is supplied we pick its argmax; else
    we fall back to slot 0 (matching nanosam basic_usage).
    """
    arr = np.squeeze(_to_numpy(mask))        # drop batch + singleton dims
    if arr.ndim == 3:                        # (num_masks, H, W) -> choose one mask
        idx = 0
        if iou is not None:
            iarr = _to_numpy(iou).reshape(-1)
            if iarr.size == arr.shape[0]:
                idx = int(np.argmax(iarr))
        arr = arr[idx]
    if arr.ndim != 2:
        return None
    return arr > 0


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _box_iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets, iou_thresh):
    """Greedy class-agnostic NMS over (x1,y1,x2,y2,cls,score) tuples.

    OWL-ViT emits many overlapping boxes per region (often one surface labelled several
    ways or at several scales) and has NO built-in NMS, which shows up as a stack of
    near-duplicate detections (e.g. five "table" boxes on one desk). Keep the highest-
    scoring box, then drop any later box overlapping a kept one by more than iou_thresh
    -> roughly one detection per region.
    """
    keep = []
    for d in sorted(dets, key=lambda t: t[5], reverse=True):
        if all(_box_iou(d[:4], k[:4]) <= iou_thresh for k in keep):
            keep.append(d)
    return keep


class NanoOwlSamDetector(Detector):
    """Open-vocabulary labeled segmentation: NanoOWL boxes + NanoSAM masks.

    NanoOWL (TensorRT OWL-ViT) detects the DETECT_PROMPTS classes as labeled boxes;
    NanoSAM (ResNet18 image encoder + MobileSAM mask decoder, both TensorRT) then
    segments EACH box into a mask via the box-corner point prompt (corners as points
    with labels [2,3]). The image is encoded once per frame (set_image) and the cheap
    decoder runs per box, so adding masks costs little beyond OWL itself.

    OWL has no built-in NMS, so its overlapping boxes are deduped with class-agnostic
    NMS (nms_iou) before segmenting -- this also avoids wasting SAM calls on duplicates.

    Each item keeps the {cls,conf,box} shape and adds a compact "mask" polygon list,
    so detect_service.py serialization and the browser overlay only need the new
    optional field. With SEG_EVERYTHING=1, a class-agnostic point-grid pass appends
    {"cls":"object"} masked items for full coverage -- this is COSTLY (seconds/frame),
    intended as a debug/exploration toggle, not the real-time default.

    Engines are loaded relative to models_dir/data (mounted, device-specific TensorRT
    files); see perception/detect/BUILD_NANOOWLSAM.md for how to produce them.
    """

    def __init__(self, models_dir, owl_model, owl_threshold, prompts,
                 seg_everything=False, nms_iou=0.5):
        import PIL.Image  # noqa: F401 -- kept lazy so the module loads without the heavy deps
        from nanoowl.owl_predictor import OwlPredictor
        from nanosam.utils.predictor import Predictor as SamPredictor

        data = os.path.join(models_dir, "data")
        owl_engine = os.environ.get(
            "OWL_ENGINE", os.path.join(data, "owl_image_encoder_patch32.engine"))
        sam_encoder = os.environ.get(
            "SAM_ENCODER", os.path.join(data, "resnet18_image_encoder.engine"))
        sam_decoder = os.environ.get(
            "SAM_DECODER", os.path.join(data, "mobile_sam_mask_decoder.engine"))
        for p in (owl_engine, sam_encoder, sam_decoder):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"nanoowlsam: missing TensorRT engine {p!r}. Build/extract the "
                    f"engines into {data} (see perception/detect/BUILD_NANOOWLSAM.md).")

        self.prompts = _parse_prompts(prompts) or ["object"]
        self.owl = OwlPredictor(owl_model, image_encoder_engine=owl_engine)
        # text encodings depend only on the prompts -> precompute once (the heavy text pass)
        self.text_encodings = self.owl.encode_text(self.prompts)
        self.sam = SamPredictor(sam_encoder, sam_decoder)   # positional: (encoder, decoder) engine paths
        self.owl_threshold = owl_threshold
        self.seg_everything = seg_everything
        self.nms_iou = nms_iou
        self._sam_warned = False

    def detect(self, frame):
        import PIL.Image
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # nanoowl/nanosam expect RGB
        pil = PIL.Image.fromarray(rgb)

        out = self.owl.predict(image=pil, text=self.prompts,
                               text_encodings=self.text_encodings,
                               threshold=self.owl_threshold, pad_square=True)
        boxes = _to_numpy(out.boxes).reshape(-1, 4)
        labels = _to_numpy(out.labels).astype(int).reshape(-1)
        scores = _to_numpy(out.scores).reshape(-1)

        # clamp + drop degenerate boxes
        kept = []
        for i in range(len(labels)):
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, min(int(round(x1)), w - 1))
            x2 = max(0, min(int(round(x2)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            y2 = max(0, min(int(round(y2)), h - 1))
            if x2 <= x1 or y2 <= y1:               # OWL can emit out-of-frame/degenerate boxes
                continue
            li = labels[i]
            cls = self.prompts[li] if 0 <= li < len(self.prompts) else "object"
            kept.append((x1, y1, x2, y2, cls, float(scores[i])))

        # OWL has no NMS -> collapse the overlapping near-duplicates to one box/region
        kept = _nms(kept, self.nms_iou)

        # encode the frame ONCE -- and only if there is something to segment, so an empty
        # scene (no boxes, seg-everything off) skips the heavy ResNet18 image-encode pass
        if kept or self.seg_everything:
            self.sam.set_image(pil)

        dets = []
        for x1, y1, x2, y2, cls, score in kept:
            item = {"cls": cls, "conf": round(score, 2), "box": [x1, y1, x2, y2]}
            poly = self._segment_box(x1, y1, x2, y2)   # reuses the cached encoding above
            if poly:
                item["mask"] = poly
            dets.append(item)

        if self.seg_everything:
            dets.extend(self._segment_everything(w, h))
        return dets

    def _warn_sam(self, exc):
        # Log the FIRST NanoSAM failure once. A systematic decoder/shape/engine error
        # would otherwise present as "labeled boxes but never any masks" with no trace
        # in `docker logs g1-detect` -- the exact bring-up failure to watch for.
        if not self._sam_warned:
            print(f"[detector] NanoSAM predict failed; masks disabled this frame: {exc!r}",
                  flush=True)
            self._sam_warned = True

    def _segment_box(self, x1, y1, x2, y2):
        # box prompt = the two corners as points, labels 2=top-left, 3=bottom-right
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        point_labels = np.array([2, 3], dtype=np.float32)
        try:
            mask, iou, _low = self.sam.predict(points, point_labels)
        except Exception as e:
            self._warn_sam(e)
            return []
        mbool = _mask_to_bool(mask, iou)
        if mbool is None:
            return []
        return mask_to_polygons(mbool)

    def _segment_everything(self, w, h, grid=16, iou_thresh=0.85,
                            min_area_frac=0.0008, max_area_frac=0.5, nms_iou=0.7):
        """Class-agnostic coverage: a coarse foreground point grid -> mask per point,
        deduped by greedy mask-IoU NMS. COSTLY (grid*grid decoder calls + O(n^2) NMS);
        gated by SEG_EVERYTHING and meant for interactive exploration, not real time.
        """
        xs = np.linspace(w / (grid + 1), w - w / (grid + 1), grid)
        ys = np.linspace(h / (grid + 1), h - h / (grid + 1), grid)
        min_area = min_area_frac * w * h
        max_area = max_area_frac * w * h
        cands = []   # (quality_iou, bool_mask)
        for gy in ys:
            for gx in xs:
                pts = np.array([[gx, gy]], dtype=np.float32)
                lbl = np.array([1], dtype=np.float32)   # 1 = foreground point
                try:
                    mask, iou, _ = self.sam.predict(pts, lbl)
                except Exception as e:
                    self._warn_sam(e)
                    continue
                q = float(_to_numpy(iou).max())     # best candidate's predicted IoU
                if q < iou_thresh:
                    continue
                mb = _mask_to_bool(mask, iou)
                if mb is None:
                    continue
                area = int(mb.sum())
                if area < min_area or area > max_area:
                    continue
                cands.append((q, mb))

        cands.sort(key=lambda c: c[0], reverse=True)   # keep higher-quality masks first
        kept = []
        for q, mb in cands:
            if all(_mask_iou(mb, k) < nms_iou for _, k in kept):
                kept.append((q, mb))

        out = []
        for q, mb in kept:
            polys = mask_to_polygons(mb)
            if not polys:
                continue
            ys_idx, xs_idx = np.where(mb)
            box = [int(xs_idx.min()), int(ys_idx.min()),
                   int(xs_idx.max()), int(ys_idx.max())]
            out.append({"cls": "object", "conf": round(float(q), 2),
                        "box": box, "mask": polys})
        return out


def make_detector(impl, model_path, conf, imgsz, prompts,
                  models_dir=None, owl_threshold=None, seg_everything=False,
                  nms_iou=0.5):
    """Factory selected by the DETECTOR_IMPL env var.

    conf/imgsz are kept in the signature for seam stability but are unused by the
    current impls; nanoowlsam uses owl_threshold/nms_iou/models_dir/seg_everything.
    """
    impl = (impl or "").lower()
    if impl == "passthrough":
        return PassthroughDetector()
    if impl in ("nanoowlsam", "nanoowl-sam", "nanoowl_nanosam", "owlsam"):
        if not models_dir:
            raise ValueError("nanoowlsam requires models_dir (the TensorRT engine location)")
        owl_model = model_path or "google/owlvit-base-patch32"
        thr = owl_threshold if owl_threshold is not None else 0.12
        return NanoOwlSamDetector(models_dir, owl_model, thr, prompts,
                                  seg_everything=seg_everything, nms_iou=nms_iou)
    raise ValueError(
        f"unknown DETECTOR_IMPL={impl!r} (expected 'nanoowlsam' or 'passthrough')")
