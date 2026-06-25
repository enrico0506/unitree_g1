#!/usr/bin/env python3
"""Offline tests for the detector seam — everything that does NOT need the GPU engines.

Covers mask->polygon conversion, the make_detector factory routing/validation, the
mask helpers, and the JSON contract shape the browser overlay expects. The NanoOWL /
NanoSAM inference path itself is verified on-device (see BUILD_NANOOWLSAM.md), since it
needs the TensorRT engines + torch; here we prove the plumbing around it.

Run:  python3 perception/detect/test_detector.py
"""

import json
import sys

import numpy as np

import detector as D


def test_mask_to_polygons_rectangle():
    # a filled rectangle should reduce to a small polygon in SOURCE-pixel coords
    m = np.zeros((200, 300), dtype=bool)
    m[40:160, 60:240] = True
    polys = D.mask_to_polygons(m)
    assert len(polys) == 1, polys
    poly = polys[0]
    assert len(poly) % 2 == 0 and len(poly) >= 6, poly      # flat [x,y,...], >=3 pts
    assert 4 <= len(poly) // 2 <= 8, len(poly)              # a rect simplifies to ~4 pts
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) <= 65 and max(xs) >= 235, (min(xs), max(xs))   # spans the rect in x
    assert min(ys) <= 45 and max(ys) >= 155, (min(ys), max(ys))   # spans the rect in y
    assert all(isinstance(v, int) for v in poly)           # JSON-friendly ints


def test_mask_to_polygons_drops_specks_and_empty():
    speck = np.zeros((100, 100), dtype=bool)
    speck[1:3, 1:3] = True                                  # 4 px << min_area_px
    assert D.mask_to_polygons(speck) == []
    assert D.mask_to_polygons(np.zeros((50, 50), dtype=bool)) == []


def test_mask_to_polygons_point_cap():
    # a noisy blob must still respect the max_points cap (bounded JSON size)
    rng = np.random.default_rng(0)
    m = rng.random((256, 256)) > 0.2
    for poly in D.mask_to_polygons(m, max_points=20, max_polys=5):
        assert len(poly) // 2 <= 20, len(poly) // 2


def test_factory_routing():
    assert isinstance(D.make_detector("passthrough", None, 0.3, 640, ""),
                      D.PassthroughDetector)
    assert D.make_detector("passthrough", None, 0.3, 640, "").detect(None) == []

    # nanoowlsam without models_dir is rejected BEFORE importing the heavy deps
    try:
        D.make_detector("nanoowlsam", None, 0.3, 640, "person")
        assert False, "expected ValueError for missing models_dir"
    except ValueError as e:
        assert "models_dir" in str(e)

    for bad in ("", "bogus", "yolo", "yoloworld"):   # yoloworld removed -> now invalid
        try:
            D.make_detector(bad, None, 0.3, 640, "")
            assert False, f"expected ValueError for impl={bad!r}"
        except ValueError:
            pass


def test_nms():
    # class-agnostic NMS keeps the highest-scoring box, drops overlapping near-dupes
    a = (0, 0, 100, 100, "table", 0.30)        # winner
    b = (5, 5, 105, 105, "chair", 0.20)        # ~IoU 0.85 with a -> dropped
    c = (200, 200, 300, 300, "person", 0.10)   # disjoint -> kept
    keep = D._nms([b, a, c], 0.5)
    assert {k[4] for k in keep} == {"table", "person"}, keep
    assert keep[0][5] >= keep[-1][5]           # sorted by score desc
    assert abs(D._box_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert D._box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_mask_helpers():
    assert D._mask_iou(np.array([[1, 1], [0, 0]], bool),
                       np.array([[1, 0], [0, 0]], bool)) == 0.5
    assert D._mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0.0
    # single-mask logits (1, 1, H, W) -> thresholded at 0
    logits = np.array([[[[-1.0, 2.0], [0.5, -3.0]]]])      # shape (1,1,2,2)
    mb = D._mask_to_bool(logits)
    assert mb.shape == (2, 2) and mb.tolist() == [[False, True], [True, False]]
    # multi-mask (1, num_masks, H, W): pick the highest predicted-IoU slot, not slot 0
    multi = np.array([[[[5.0, 5.0], [5.0, 5.0]],          # slot 0: all fg (low iou)
                       [[-1.0, 9.0], [-1.0, -1.0]]]])      # slot 1: one px  (high iou)
    assert D._mask_to_bool(multi, iou=np.array([0.10, 0.95])).tolist() == [[False, True], [False, False]]
    assert D._mask_to_bool(multi).sum() == 4               # no iou -> slot 0 (all fg)
    assert D._scalar(np.array([0.87])) == 0.87
    assert abs(D._scalar([[0.5]]) - 0.5) < 1e-9


def test_json_contract_shape():
    # an item carrying a mask must serialize and match what cam-overlay.js reads
    m = np.zeros((120, 120), dtype=bool)
    m[20:100, 20:100] = True
    item = {"cls": "person", "conf": 0.34, "box": [20, 20, 100, 100],
            "mask": D.mask_to_polygons(m)}
    payload = {"w": 120, "h": 120, "items": [item]}
    round_trip = json.loads(json.dumps(payload))            # must be JSON-clean
    it = round_trip["items"][0]
    assert set(["cls", "conf", "box", "mask"]).issubset(it)
    assert isinstance(it["mask"], list) and isinstance(it["mask"][0], list)
    assert len(it["box"]) == 4


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nPASS — {len(tests)} tests")


if __name__ == "__main__":
    sys.exit(main())
