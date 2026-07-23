#!/usr/bin/env python3
"""Unit tests for the circle-patrol geometry + interaction state machine
(scripts/sim_circle.py). Pure Python -- NO ROS, NO robot, no sim clock: just the
math (carrot-point pursuit geometry, the re-anchor-on-drift threshold) and the
PatrolFSM transition table.

    pytest scripts/test_circle_patrol.py
    python3 scripts/test_circle_patrol.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim_circle import (          # noqa: E402
    Circle, carrot_point, pursuit_command, wrap, PatrolFSM, STATES, TRANSITIONS,
    REANCHOR_DRIFT_M,
)

CIRCLE = Circle(0.0, 0.0, 3.0)
LOOKAHEAD = 1.25


# =============================================================== carrot-point math
def test_carrot_on_circle_at_angle_zero():
    # exactly on the circle, at angle 0, CCW travel -> carrot walked LOOKAHEAD ahead.
    r = carrot_point(CIRCLE.r, 0.0, CIRCLE, LOOKAHEAD, direction=1.0)
    expected_ang = LOOKAHEAD / CIRCLE.r
    assert math.isclose(r.x, CIRCLE.r * math.cos(expected_ang), abs_tol=1e-9)
    assert math.isclose(r.y, CIRCLE.r * math.sin(expected_ang), abs_tol=1e-9)
    assert math.isclose(r.radial_dev, 0.0, abs_tol=1e-9)
    assert not r.reanchored


def test_carrot_direction_reversed_walks_the_other_way():
    r_ccw = carrot_point(CIRCLE.r, 0.0, CIRCLE, LOOKAHEAD, direction=1.0)
    r_cw = carrot_point(CIRCLE.r, 0.0, CIRCLE, LOOKAHEAD, direction=-1.0)
    assert r_ccw.y > 0.0        # CCW from (r,0) moves to +y
    assert r_cw.y < 0.0         # CW from (r,0) moves to -y
    assert math.isclose(r_ccw.x, r_cw.x, abs_tol=1e-9)   # symmetric about the x axis


def test_carrot_angular_positions_around_the_circle():
    # a range of starting angles, exactly ON the circle: the carrot must always be
    # exactly LOOKAHEAD (arc-length) further around, in radians = LOOKAHEAD_M / r.
    for deg in (0, 37, 90, 135, 180, 225, 270, 359):
        ang = math.radians(deg)
        x, y = CIRCLE.r * math.cos(ang), CIRCLE.r * math.sin(ang)
        res = carrot_point(x, y, CIRCLE, LOOKAHEAD, direction=1.0)
        expected_ang = wrap(ang + LOOKAHEAD / CIRCLE.r)
        got_ang = math.atan2(res.y - CIRCLE.cy, res.x - CIRCLE.cx)
        assert math.isclose(wrap(got_ang - expected_ang), 0.0, abs_tol=1e-9), deg


def test_carrot_always_lands_exactly_on_the_circle():
    # regardless of how far off the circle the input point is (inside/outside),
    # the carrot itself must land exactly ON the circle (radius r from center) --
    # this is what makes the projection well-defined at any drift magnitude.
    for (x, y) in [(3.0, 0.0), (1.5, 0.0), (5.0, 0.0), (0.0, 3.0), (2.1, 2.1),
                   (-2.0, -1.0), (0.01, 0.01)]:
        for direction in (1.0, -1.0):
            res = carrot_point(x, y, CIRCLE, LOOKAHEAD, direction=direction)
            d = math.hypot(res.x - CIRCLE.cx, res.y - CIRCLE.cy)
            assert math.isclose(d, CIRCLE.r, abs_tol=1e-9)


def test_carrot_inside_circle_radial_dev_negative():
    x, y = 2.0, 0.0    # inside (r=3)
    res = carrot_point(x, y, CIRCLE, LOOKAHEAD)
    assert res.radial_dev < 0.0
    assert math.isclose(res.radial_dev, 2.0 - CIRCLE.r, abs_tol=1e-9)


def test_carrot_outside_circle_radial_dev_positive():
    x, y = 3.9, 0.0    # outside (r=3)
    res = carrot_point(x, y, CIRCLE, LOOKAHEAD)
    assert res.radial_dev > 0.0
    assert math.isclose(res.radial_dev, 3.9 - CIRCLE.r, abs_tol=1e-9)


# =============================================================== re-anchor threshold
# DESIGN (documented in sim_circle.py next to REANCHOR_DRIFT_M): small radial drift
# (<= REANCHOR_DRIFT_M) is tolerated by normal tangential pure-pursuit (project, then
# walk LOOKAHEAD_M further around the arc). Past REANCHOR_DRIFT_M the lookahead
# COLLAPSES TO ZERO -- the carrot aims straight at the nearest point on the circle
# (pure radial recovery) instead of compounding a large angular jump with a further
# tangential offset. These tests pin that threshold behaviour down exactly.
def test_small_drift_uses_full_lookahead_not_reanchored():
    dev = REANCHOR_DRIFT_M - 0.05
    x = CIRCLE.r + dev
    res = carrot_point(x, 0.0, CIRCLE, LOOKAHEAD)
    assert not res.reanchored
    expected_ang = LOOKAHEAD / CIRCLE.r     # full lookahead still applied
    got_ang = math.atan2(res.y, res.x)
    assert math.isclose(got_ang, expected_ang, abs_tol=1e-9)


def test_large_drift_reanchors_to_zero_lookahead():
    dev = REANCHOR_DRIFT_M + 0.20
    x = CIRCLE.r + dev
    res = carrot_point(x, 0.0, CIRCLE, LOOKAHEAD)
    assert res.reanchored
    # lookahead collapsed to 0 -> carrot sits at the SAME angle as the (projected)
    # current position, i.e. angle 0 here, NOT LOOKAHEAD/r further around.
    got_ang = math.atan2(res.y, res.x)
    assert math.isclose(got_ang, 0.0, abs_tol=1e-9)


def test_reanchor_threshold_boundary():
    # strict '>' means the boundary itself is NOT reanchored (tolerated smoothly);
    # a hair beyond it is. Straddle the threshold rather than testing exactly AT it,
    # since float arithmetic on the composed value (r + REANCHOR_DRIFT_M) is not
    # guaranteed bit-exact for the subtraction back out.
    x_under = CIRCLE.r + REANCHOR_DRIFT_M - 1e-6
    assert not carrot_point(x_under, 0.0, CIRCLE, LOOKAHEAD).reanchored
    x_over = CIRCLE.r + REANCHOR_DRIFT_M + 1e-6
    assert carrot_point(x_over, 0.0, CIRCLE, LOOKAHEAD).reanchored


def test_reanchor_symmetric_inside_and_outside():
    # the threshold gates on |radial_dev|, so drifting INSIDE the circle by more
    # than REANCHOR_DRIFT_M re-anchors exactly like drifting outside.
    x_in = CIRCLE.r - (REANCHOR_DRIFT_M + 0.2)
    assert x_in > 0.0
    assert carrot_point(x_in, 0.0, CIRCLE, LOOKAHEAD).reanchored


# =============================================================== pursuit_command
def test_pursuit_faces_carrot_and_slows_when_misaligned():
    # carrot directly ahead (theta already facing it) -> full vx, ~0 vyaw.
    vx, vy, vyaw = pursuit_command(0.0, 0.0, 0.0, (5.0, 0.0), max_vx=1.5, max_vyaw=2.0)
    assert math.isclose(vx, 1.5, abs_tol=1e-6)
    assert math.isclose(vyaw, 0.0, abs_tol=1e-6)
    assert vy == 0.0


def test_pursuit_carrot_behind_eases_off_forward_speed():
    # carrot directly BEHIND (bearing = +/-pi) -> vx eases to ~0 (cos(bearing) <= 0),
    # vyaw saturates turning toward it.
    vx, vy, vyaw = pursuit_command(0.0, 0.0, 0.0, (-5.0, 0.0), max_vx=1.5, max_vyaw=2.0)
    assert vx < 1e-6
    assert abs(vyaw) <= 2.0 + 1e-9


def test_pursuit_carrot_to_the_left_turns_left_positive_vyaw():
    vx, vy, vyaw = pursuit_command(0.0, 0.0, 0.0, (1.0, 1.0), max_vx=1.5, max_vyaw=2.0)
    assert vyaw > 0.0    # bearing to (1,1) from heading 0 is +45 deg -> +vyaw (left/CCW)


# =============================================================== interaction FSM
HAPPY_PATH = ["circling", "pausing", "facing", "waving", "waiting", "resuming", "circling"]


def test_all_mandated_states_present():
    assert set(STATES) == {
        "circling", "pausing", "facing", "waving", "waiting", "resuming",
        "fault", "manual_override", "paused_odom_degraded",
    }


def test_manual_override_reachable_from_every_other_state():
    for s in STATES:
        if s == "manual_override":
            continue
        assert "manual_override" in TRANSITIONS[s], f"{s} cannot reach manual_override"


def test_paused_odom_degraded_reachable_from_every_other_state():
    for s in STATES:
        if s == "paused_odom_degraded":
            continue
        assert "paused_odom_degraded" in TRANSITIONS[s], f"{s} cannot reach paused_odom_degraded"


def test_happy_path_sequence_achievable():
    fsm = PatrolFSM("circling")
    for nxt in HAPPY_PATH[1:]:
        fsm.transition(nxt)     # must not raise
    assert fsm.state == "circling"


def test_happy_path_can_repeat_indefinitely():
    # a patrol loops laps forever: circling -> ... -> circling -> ... must be
    # re-enterable, not a one-shot path.
    fsm = PatrolFSM("circling")
    for _ in range(3):
        for nxt in HAPPY_PATH[1:]:
            fsm.transition(nxt)
    assert fsm.state == "circling"


def test_invalid_transitions_rejected():
    invalid_pairs = [
        ("circling", "waving"),      # skip pausing/facing
        ("circling", "waiting"),
        ("circling", "resuming"),
        ("facing", "circling"),      # must go through waving/waiting/resuming
        ("facing", "pausing"),
        ("waving", "circling"),
        ("waving", "facing"),
        ("waiting", "facing"),
        ("waiting", "circling"),
        ("resuming", "waving"),
        ("resuming", "facing"),
        ("resuming", "pausing"),
        ("pausing", "waving"),
        ("pausing", "waiting"),
    ]
    for frm, to in invalid_pairs:
        fsm = PatrolFSM(frm)
        assert not fsm.can(to), f"{frm} -> {to} should be INVALID"
        try:
            fsm.transition(to)
            assert False, f"{frm} -> {to} should have raised"
        except ValueError:
            pass
        assert fsm.state == frm    # a rejected transition must not mutate state


def test_fault_recovers_via_manual_override():
    # fault's only BASE (non-safety) forward edge is manual_override (an operator
    # must take over to clear a fault); paused_odom_degraded is also present only
    # because it is mandated reachable from every OTHER state, not as a "recovery".
    fsm = PatrolFSM("fault")
    assert "manual_override" in TRANSITIONS["fault"]
    assert TRANSITIONS["fault"] - {"paused_odom_degraded"} == {"manual_override"}
    fsm.transition("manual_override")
    fsm.transition("circling")
    assert fsm.state == "circling"


def test_manual_override_returns_to_circling():
    fsm = PatrolFSM("manual_override")
    fsm.transition("circling")
    assert fsm.state == "circling"


def test_paused_odom_degraded_recovers_via_resuming():
    fsm = PatrolFSM("paused_odom_degraded")
    assert "resuming" in TRANSITIONS["paused_odom_degraded"]
    fsm.transition("resuming")
    fsm.transition("circling")
    assert fsm.state == "circling"


def test_no_self_loops_in_safety_states():
    # manual_override/paused_odom_degraded are reachable from every OTHER state,
    # not from themselves (no degenerate self-transition).
    assert "manual_override" not in TRANSITIONS["manual_override"]
    assert "paused_odom_degraded" not in TRANSITIONS["paused_odom_degraded"]


def test_unknown_state_construction_rejected():
    try:
        PatrolFSM("not_a_real_state")
        assert False, "should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    # bare `python3` runner (no pytest dependency), matching this repo's convention
    # of also supporting direct execution (see scripts/test_gesture_reactor.py).
    tests = [(name, fn) for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
