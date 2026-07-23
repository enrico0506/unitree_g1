#!/usr/bin/env python3
"""Unit tests for the close-range IR fallback in pose_service.py (see its module
docstring). Covers the two pure pieces only -- head_cropped() (the trigger) and
step_source() (the dwell/recover state machine) -- no camera/GPU/robot required.

    pytest perception/pose/test_pose_ir_fallback.py
"""
from pose_service import head_cropped, step_source, KP_MIN_CONF, HEAD_KPT_IDXS

FRAME_H = 480

CONF_HI = KP_MIN_CONF + 0.2
CONF_LO = KP_MIN_CONF - 0.1


def _kpts(head_conf, n=17):
    """17 keypoints, all zero except the 5 head ones set to `head_conf`."""
    kpts = [[0, 0, 0.0] for _ in range(n)]
    for i in HEAD_KPT_IDXS:
        kpts[i] = [10, 10, head_conf]
    return kpts


# --------------------------------------------------------------- head_cropped()
def test_no_tracks_not_cropped():
    assert head_cropped([], FRAME_H) is False


def test_confident_head_not_cropped():
    it = {"box": [0, 0, 100, 200], "kpts": _kpts(CONF_HI)}
    assert head_cropped([it], FRAME_H) is False


def test_missing_head_but_not_at_top_edge_not_cropped():
    # head missing, but the box isn't touching the top -- person likely just
    # turned away, not cropped by the FOV. Shouldn't trigger a camera switch.
    it = {"box": [0, 50, 100, 250], "kpts": _kpts(CONF_LO)}
    assert head_cropped([it], FRAME_H) is False


def test_missing_head_at_top_edge_but_small_box_not_cropped():
    # touches the top edge, but the box is tiny (distant person / noise) --
    # below IR_MIN_BOX_FRAC_H, shouldn't trigger.
    it = {"box": [0, 0, 30, 40], "kpts": _kpts(CONF_LO)}
    assert head_cropped([it], FRAME_H) is False


def test_missing_head_at_top_edge_large_box_is_cropped():
    it = {"box": [0, 0, 200, 300], "kpts": _kpts(CONF_LO)}
    assert head_cropped([it], FRAME_H) is True


def test_short_kpts_list_skipped():
    it = {"box": [0, 0, 200, 300], "kpts": [[0, 0, 0.0]] * 3}
    assert head_cropped([it], FRAME_H) is False


def test_one_confident_head_kpt_is_enough_to_clear():
    kpts = _kpts(CONF_LO)
    kpts[HEAD_KPT_IDXS[0]][2] = CONF_HI   # nose alone is confident
    it = {"box": [0, 0, 200, 300], "kpts": kpts}
    assert head_cropped([it], FRAME_H) is False


# --------------------------------------------------------------- step_source()
DWELL = 2.0
STABLE_N = 3


def test_rgb_stays_rgb_when_clear():
    source, switched_at, streak = step_source("rgb", False, 10.0, 0.0, 0)
    assert source == "rgb"


def test_rgb_switches_to_ir_when_cropped():
    source, switched_at, streak = step_source("rgb", True, 10.0, 0.0, 0)
    assert (source, switched_at, streak) == ("ir", 10.0, 0)


def test_ir_holds_through_dwell_even_if_clear():
    # just switched (switched_at == t0); still within the dwell window.
    source, switched_at, streak = step_source("ir", False, 10.0, 10.0, 0)
    assert source == "ir"
    assert streak == 1


def test_ir_recovers_after_dwell_and_stable_streak():
    t0 = 10.0
    source, switched_at, streak = "ir", t0, 0
    # feed STABLE_N clean frames spaced past the dwell window
    for i in range(STABLE_N):
        t = t0 + DWELL + i
        source, switched_at, streak = step_source(source, False, t, switched_at, streak)
    assert source == "rgb"


def test_ir_does_not_recover_if_cropped_again_resets_streak():
    t0 = 10.0
    source, switched_at, streak = step_source("ir", False, t0 + DWELL, t0, 0)
    assert streak == 1
    source, switched_at, streak = step_source(source, True, t0 + DWELL + 1, switched_at, streak)
    assert source == "ir"
    assert streak == 0
