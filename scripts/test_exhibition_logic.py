#!/usr/bin/env python3
"""Unit tests for exhibition-mode logic (g1_nav/g1_nav/patrol_logic.py's
random_point_in_disc / ExhibitionSegment / predict_segment_end /
plan_next_segment). Pure Python -- no ROS, no robot, no sim clock: just the
disc geometry and the lookahead oracle's accept/reject/fallback behavior.

The bias here matches this repo's usual one: proving the REJECTIONS (an
unsafe roam target, an under-clearance dance, an exhausted-retries fallback)
matters as much as proving the happy path, because the whole point of the
oracle is to keep a randomized exhibition inside the 3 m safety disc.

    pytest scripts/test_exhibition_logic.py
    python3 scripts/test_exhibition_logic.py
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "g1_nav"))   # mirrors scripts/sim_runner.py's own import setup

from g1_nav.patrol_logic import (         # noqa: E402
    ExhibitionSegment, random_point_in_disc, predict_segment_end, plan_next_segment,
)

CENTER = (0.0, 0.0)
RADIUS = 3.0
EDGE = 0.5
SAFE_R = RADIUS - EDGE


def base_cfg(**overrides):
    cfg = {
        "center": CENTER, "radius_m": RADIUS, "edge_clearance_m": EDGE,
        "roam_speed_mps": 0.5, "roam_min_dist_m": 0.3,
        "segments": [
            {"kind": "roam", "weight": 0.5},
            {"kind": "gesture", "weight": 0.5, "choices": ["high_wave", "heart"]},
            {"kind": "idle", "weight": 0.0},
        ],
        "lookahead_segments": 5, "max_reroll_attempts": 8,
    }
    cfg.update(overrides)
    return cfg


# =============================================================== random_point_in_disc
def test_random_point_in_disc_always_within_safety_margin():
    rng = random.Random(1)
    for _ in range(500):
        x, y = random_point_in_disc(CENTER, RADIUS, EDGE, rng)
        assert math.hypot(x, y) <= SAFE_R + 1e-9


def test_random_point_in_disc_respects_min_distance():
    rng = random.Random(2)
    here = (1.0, 0.0)
    for _ in range(200):
        pt = random_point_in_disc(CENTER, RADIUS, EDGE, rng,
                                    min_dist_from=here, min_dist_m=1.0, max_attempts=50)
        # either satisfies the constraint, or (rare/never at this radius) fell back to center
        assert (math.hypot(pt[0] - here[0], pt[1] - here[1]) >= 1.0 - 1e-9
                or pt == CENTER)


def test_random_point_in_disc_falls_back_to_center_when_impossible():
    rng = random.Random(3)
    # min_dist_m far larger than the disc itself -- no point can ever satisfy it.
    pt = random_point_in_disc(CENTER, RADIUS, EDGE, rng,
                                min_dist_from=(0.0, 0.0), min_dist_m=1000.0, max_attempts=10)
    assert pt == CENTER


def test_random_point_in_disc_is_not_degenerate_center_bias():
    # Uniform-disc sampling (r = R*sqrt(u)) should NOT cluster near the center --
    # a plain r = R*u would put ~50% of points within 0.5*R rather than the
    # correct ~25% (area scales as r^2).
    rng = random.Random(4)
    near_center = sum(
        1 for _ in range(2000)
        if math.hypot(*random_point_in_disc(CENTER, RADIUS, EDGE, rng)) < SAFE_R * 0.5
    )
    frac = near_center / 2000.0
    assert 0.15 < frac < 0.35, f"expected ~0.25 (uniform-area), got {frac}"


# =============================================================== predict_segment_end
def test_roam_endpoint_inside_margin_is_ok():
    seg = ExhibitionSegment(kind="roam", target=(1.0, 1.0))
    pose, ok, _ = predict_segment_end((0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert ok
    assert math.isclose(pose[0], 1.0) and math.isclose(pose[1], 1.0)


def test_roam_endpoint_outside_margin_is_rejected():
    seg = ExhibitionSegment(kind="roam", target=(2.9, 0.0))   # outside RADIUS-EDGE=2.5
    pose, ok, reason = predict_segment_end((0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert not ok
    assert "outside safety margin" in reason
    assert pose == (0.0, 0.0, 0.0)   # unchanged on rejection


def test_roam_heading_points_toward_target():
    seg = ExhibitionSegment(kind="roam", target=(0.0, 2.0))
    (x, y, yaw), ok, _ = predict_segment_end((0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert ok
    assert math.isclose(yaw, math.pi / 2.0, abs_tol=1e-9)


def test_gesture_and_idle_check_current_position_not_target():
    ok_pose = (0.0, 0.0, 0.0)          # well inside margin
    bad_pose = (2.9, 0.0, 0.0)         # outside RADIUS-EDGE
    for kind in ("gesture", "idle"):
        seg = ExhibitionSegment(kind=kind, gesture="high_wave")
        _, ok, _ = predict_segment_end(ok_pose, seg, CENTER, RADIUS, EDGE)
        assert ok
        _, ok, reason = predict_segment_end(bad_pose, seg, CENTER, RADIUS, EDGE)
        assert not ok


def test_dance_requires_its_own_space_m_not_just_edge_clearance():
    # At radius 2.0 from center: inside the generic EDGE=0.5 margin (2.0 <= 2.5)
    # but NOT inside a dance needing space_m=2.0 (2.0 <= 3.0-2.0=1.0 is false).
    pose = (2.0, 0.0, 0.0)
    seg = ExhibitionSegment(kind="dance", dance_fsm_id=503, dance_space_m=2.0)
    _, ok, reason = predict_segment_end(pose, seg, CENTER, RADIUS, EDGE)
    assert not ok
    assert "clearance" in reason
    # Same dance, robot near center: plenty of room.
    _, ok, _ = predict_segment_end((0.2, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert ok


def test_dance_without_space_m_falls_back_to_edge_clearance():
    seg = ExhibitionSegment(kind="dance", dance_fsm_id=503, dance_space_m=None)
    _, ok, _ = predict_segment_end((0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert ok


def test_unknown_segment_kind_fails_safe():
    seg = ExhibitionSegment(kind="cartwheel")
    _, ok, reason = predict_segment_end((0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE)
    assert not ok
    assert "unknown segment kind" in reason


# =============================================================== plan_next_segment (the oracle)
def test_plan_next_segment_returns_a_safe_segment_from_center():
    rng = random.Random(10)
    cfg = base_cfg()
    for _ in range(100):
        seg = plan_next_segment((0.0, 0.0, 0.0), cfg, rng)
        pose, ok, reason = predict_segment_end(
            (0.0, 0.0, 0.0), seg, CENTER, RADIUS, EDGE, cfg["roam_speed_mps"])
        assert ok, f"oracle returned an unsafe segment: {seg} ({reason})"


def test_plan_next_segment_never_draws_dance_from_the_weighted_pool():
    # Even if a config mistakenly includes a "dance" entry with weight, the
    # oracle must never fire an UNREQUESTED dance -- see the module's safety-net
    # comment on _draw_candidate_segment.
    rng = random.Random(11)
    cfg = base_cfg(segments=[{"kind": "dance", "weight": 1.0}])
    for _ in range(50):
        seg = plan_next_segment((0.0, 0.0, 0.0), cfg, rng)
        assert seg.kind != "dance"


def test_plan_next_segment_fires_a_pending_dance_when_safe():
    rng = random.Random(12)
    cfg = base_cfg()
    pending = ExhibitionSegment(kind="dance", dance_fsm_id=503,
                                 dance_name="Dance Mode", dance_space_m=2.0,
                                 duration_s=8.0)
    # Robot at the exact center: max clearance, request should be accepted.
    seg = plan_next_segment((0.0, 0.0, 0.0), cfg, rng, pending_dance=pending)
    assert seg.kind == "dance" and seg.dance_fsm_id == 503


def test_plan_next_segment_defers_a_pending_dance_when_unsafe_right_now():
    # Robot too close to the edge for the dance's space_m -- the oracle must
    # NOT force an unsafe dance; it should fall through to a normal (or
    # fallback) segment instead, leaving the request for the caller to re-offer
    # once the robot has moved somewhere safer.
    rng = random.Random(13)
    cfg = base_cfg()
    pending = ExhibitionSegment(kind="dance", dance_fsm_id=503,
                                 dance_name="Dance Mode", dance_space_m=2.0,
                                 duration_s=8.0)
    seg = plan_next_segment((2.9, 0.0, 0.0), cfg, rng, pending_dance=pending)
    assert seg.kind != "dance"
    pose, ok, _ = predict_segment_end(
        (2.9, 0.0, 0.0), seg, CENTER, RADIUS, EDGE, cfg["roam_speed_mps"])
    assert ok


def test_plan_next_segment_falls_back_to_center_when_every_retry_fails():
    rng = random.Random(14)
    # An impossible config (every roam target request will be rejected: the
    # only "segments" entry is roam, but min_dist_m is absurdly large so
    # random_point_in_disc always degrades to returning `center` itself --
    # which IS safe, so this actually exercises the "always converges to the
    # guaranteed-safe fallback" property from a different angle). To force a
    # genuine multi-attempt exhaustion, shrink radius_m below edge_clearance_m
    # so NOTHING but the exact center ever satisfies _in_safe_disc, and rely on
    # the fallback path explicitly.
    cfg = base_cfg(radius_m=0.1, edge_clearance_m=0.5, max_reroll_attempts=3,
                    segments=[{"kind": "roam", "weight": 1.0}])
    seg = plan_next_segment((0.0, 0.0, 0.0), cfg, rng)
    assert seg.kind == "roam"
    assert seg.target == cfg["center"]


def test_plan_next_segment_is_deterministic_for_a_seeded_rng():
    cfg = base_cfg()
    seg_a = plan_next_segment((0.5, 0.5, 0.1), cfg, random.Random(42))
    seg_b = plan_next_segment((0.5, 0.5, 0.1), cfg, random.Random(42))
    assert seg_a == seg_b


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
