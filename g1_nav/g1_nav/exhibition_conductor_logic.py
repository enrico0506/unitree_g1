#!/usr/bin/env python3
"""exhibition_conductor_logic.py -- pure-Python helpers factored out of
g1_nav/g1_nav/nodes/exhibition_conductor.py SPECIFICALLY so they stay
unit-testable without importing rclpy/websockets (the same reasoning
patrol_logic.py's own module docstring gives for keeping its oracle pure, and
the same pattern scripts/test_circle_patrol.py already uses -- testing
sim_circle.py's duplicate-but-pure geometry rather than importing the real,
rclpy-importing circle_patrol.py node module). Zero ROS / websocket / file-I/O
imports here -- see scripts/test_exhibition_conductor_logic.py.

Everything genuinely ROS-coupled (the NavigateToPose action client, the WS
gesture/dance firing, the /dev/shm reads themselves) stays in
exhibition_conductor.py and is NOT duplicated here -- only the small bits of
decision logic around them that don't need ROS to reason about.
"""
from g1_nav.patrol_logic import ExhibitionSegment


def build_dance_segment_from_request(data, default_duration_s):
    """Pure parsing: given the ALREADY-LOADED dict from
    /dev/shm/g1_exhibition_dance_request.json (freshness already checked by
    the caller via file mtime -- see exhibition_conductor.py's
    _poll_pending_dance(), mirroring circle_patrol.py's _odom_healthy()
    idiom), build the ExhibitionSegment plan_next_segment()'s `pending_dance`
    param expects, or None if the payload is missing/malformed.

    `duration_s` is NOT read from the request -- there is no authoritative
    "how long does dance <fsm_id> take" value anywhere in this codebase yet
    (dances.yaml's own note on fsm_id 503 only says it "plays the classic
    dance and returns to walk on its own"), so the caller's configured
    `dance_default_duration_s` (a documented placeholder) is used instead.
    """
    if not isinstance(data, dict):
        return None
    fsm_id = data.get("fsm_id")
    if fsm_id is None:
        return None
    return ExhibitionSegment(
        kind="dance",
        dance_fsm_id=fsm_id,
        dance_name=data.get("name"),
        dance_space_m=data.get("space_m"),
        duration_s=float(default_duration_s),
    )


def activity_label(segment):
    """Pure mapping: an ExhibitionSegment (or None) -> the additive
    `activity` string g1_patrol_state.json's new field carries (see the plan
    file's design decision #5 -- this rides ALONGSIDE the existing `phase`
    field, which is never extended). None means "not currently executing a
    segment" (not yet anchored, no pose, or yielding to manual_override /
    the supervisor / an odom-health pause)."""
    if segment is None:
        return None
    if segment.kind == "roam":
        return "roam"
    if segment.kind == "gesture":
        return f"gesture:{segment.gesture}"
    if segment.kind == "dance":
        return f"dance:{segment.dance_fsm_id}"
    return "idle"


def gap_elapsed(last_segment_end_t, now, min_segment_gap_s):
    """True iff enough time has passed since the previous segment ended (or no
    segment has EVER run yet, `last_segment_end_t is None`) to let the
    conductor pick a new one. Mirrors patrol_supervisor.py's `pause_settle_s`
    idiom -- a brief, deliberate settle window between segments rather than
    chaining them back-to-back with zero breathing room."""
    if last_segment_end_t is None:
        return True
    return (now - last_segment_end_t) >= min_segment_gap_s
