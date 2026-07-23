#!/usr/bin/env python3
"""Unit tests for exhibition_conductor.py's pure decision-logic helpers
(g1_nav/g1_nav/exhibition_conductor_logic.py): pending-dance-request parsing,
the `activity` label mapping, and the inter-segment gap check. Pure Python --
NO ROS, NO robot: the node itself (exhibition_conductor.py) imports rclpy and
websockets and is intentionally NOT imported here, same reasoning
test_circle_patrol.py gives for testing sim_circle.py instead of
circle_patrol.py directly.

    pytest scripts/test_exhibition_conductor_logic.py
    python3 scripts/test_exhibition_conductor_logic.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "g1_nav"))   # mirrors test_exhibition_logic.py's own import setup

from g1_nav.exhibition_conductor_logic import (   # noqa: E402
    build_dance_segment_from_request, activity_label, gap_elapsed,
)
from g1_nav.patrol_logic import ExhibitionSegment  # noqa: E402


# ===================================================== build_dance_segment_from_request
def test_build_dance_segment_from_valid_request():
    data = {"fsm_id": 503, "name": "Dance Mode", "space_m": 2.0, "requested_t": 123.0}
    seg = build_dance_segment_from_request(data, default_duration_s=8.0)
    assert seg is not None
    assert seg.kind == "dance"
    assert seg.dance_fsm_id == 503
    assert seg.dance_name == "Dance Mode"
    assert seg.dance_space_m == 2.0
    assert seg.duration_s == 8.0


def test_build_dance_segment_uses_configured_default_duration_not_the_request():
    # duration_s is NEVER taken from the request payload -- see the function's
    # own docstring: no authoritative per-dance duration exists yet.
    data = {"fsm_id": 503, "duration_s": 999.0}
    seg = build_dance_segment_from_request(data, default_duration_s=8.0)
    assert seg.duration_s == 8.0


def test_build_dance_segment_missing_fsm_id_is_rejected():
    assert build_dance_segment_from_request({"name": "Dance Mode"}, 8.0) is None


def test_build_dance_segment_non_dict_payload_is_rejected():
    for bad in (None, "not a dict", 42, [1, 2, 3]):
        assert build_dance_segment_from_request(bad, 8.0) is None


def test_build_dance_segment_tolerates_missing_optional_fields():
    seg = build_dance_segment_from_request({"fsm_id": 503}, 8.0)
    assert seg is not None
    assert seg.dance_name is None
    assert seg.dance_space_m is None


# ===================================================================== activity_label
def test_activity_label_none_segment():
    assert activity_label(None) is None


def test_activity_label_roam():
    seg = ExhibitionSegment(kind="roam", target=(1.0, 2.0))
    assert activity_label(seg) == "roam"


def test_activity_label_gesture_includes_gesture_name():
    seg = ExhibitionSegment(kind="gesture", gesture="high_wave")
    assert activity_label(seg) == "gesture:high_wave"


def test_activity_label_dance_includes_fsm_id():
    seg = ExhibitionSegment(kind="dance", dance_fsm_id=503)
    assert activity_label(seg) == "dance:503"


def test_activity_label_idle():
    seg = ExhibitionSegment(kind="idle")
    assert activity_label(seg) == "idle"


# ======================================================================= gap_elapsed
def test_gap_elapsed_true_when_no_prior_segment():
    assert gap_elapsed(None, now=100.0, min_segment_gap_s=1.5) is True


def test_gap_elapsed_false_within_the_gap_window():
    assert gap_elapsed(last_segment_end_t=100.0, now=100.5, min_segment_gap_s=1.5) is False


def test_gap_elapsed_true_once_the_window_passes():
    assert gap_elapsed(last_segment_end_t=100.0, now=101.6, min_segment_gap_s=1.5) is True


def test_gap_elapsed_boundary_is_inclusive():
    assert gap_elapsed(last_segment_end_t=100.0, now=101.5, min_segment_gap_s=1.5) is True


if __name__ == "__main__":
    import inspect
    failures = 0
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
