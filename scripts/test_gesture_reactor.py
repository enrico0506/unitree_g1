#!/usr/bin/env python3
"""Unit tests for the human-gesture reactor (scripts/gesture_reactor.py).

Two layers are covered:
  * the PURE classifier (classify) -- fed hand-built (t, joints, scale) histories, no
    debounce / target logic in the way, so a failure points straight at the heuristic;
  * the stateful GestureReactor -- debounce/cooldown, low-confidence robustness, and
    multi-person target selection, driven by the deterministic synthesizer in
    sim_gestures.py so tests and the demo harness agree on what a gesture looks like.

The bias throughout matches the demo's: proving the NEGATIVES (idle / walking / static
reach never fire) matters as much as proving the positives, because an arm gesture that
auto-fires at the wrong moment in front of a client is the failure we actually fear.

    pytest scripts/test_gesture_reactor.py
"""
import math

import pytest

from gesture_reactor import (
    GestureReactor, GreetingService, ReactorConfig, classify, describe, GESTURE_TO_ROBOT,
    _kp, _frame_scale, R_EAR, R_WRI,
)
import sim_gestures as sg
from sim_gestures import (
    make_skeleton, feed, fired_gestures, wave_sequence, raise_sequence,
    idle_sequence, walking_sequence, reach_sequence,
    DT, CX, CY, SW,
)


# --------------------------------------------------------------- helpers
def history_from(frames, cfg, t0=0.0, dt=DT):
    """Turn a list of synthesized skeletons into the exact (t, joints, scale) history the
    reactor feeds classify(), for ONE track -- so we can exercise the PURE classifier in
    isolation (no debounce, no target scoring)."""
    hist = []
    t = t0
    for sk in frames:
        kpts = sk["kpts"]
        joints = {}
        for i in range(len(kpts)):
            p = _kp(kpts, i, cfg.kp_min_conf)
            if p is not None:
                joints[i] = p
        hist.append((t, joints, _frame_scale(joints, sk["box"])))
        t += dt
    return hist


CFG = ReactorConfig()


# =============================================================== pure classifier
def test_wave_classifies():
    assert classify(history_from(wave_sequence(), CFG), CFG) == "wave"


def test_wave_left_arm_classifies():
    assert classify(history_from(wave_sequence(arm="left"), CFG), CFG) == "wave"


def test_raise_hand_classifies():
    assert classify(history_from(raise_sequence(), CFG), CFG) == "raise_hand"


def test_idle_classifies_none():
    assert classify(history_from(idle_sequence(), CFG), CFG) is None


def test_walking_classifies_none():
    # Horizontal wrist oscillation with the wrists BELOW the shoulders must not read as a
    # wave -- the "wrist above shoulder" gate is what rejects arm-swing while walking.
    assert classify(history_from(walking_sequence(), CFG), CFG) is None


def test_static_reach_classifies_none():
    # Arm up but not oscillating and not above the head: neither wave nor raise_hand.
    assert classify(history_from(reach_sequence(), CFG), CFG) is None


def test_single_frame_never_fires():
    assert classify(history_from(wave_sequence(n=1), CFG), CFG) is None


def test_raise_is_not_a_wave_and_vice_versa():
    # Guards the mutual-exclusion the precedence relies on: a still raised hand is never a
    # wave (no oscillation), and an oscillating wrist below the head is never a raise.
    assert classify(history_from(raise_sequence(), CFG), CFG) != "wave"
    assert classify(history_from(wave_sequence(), CFG), CFG) != "raise_hand"


# =============================================================== reactor: firing
def test_reactor_fires_wave_once():
    f = fired_gestures(feed(GestureReactor(CFG), wave_sequence()))
    assert f.count("wave") == 1


def test_reactor_fires_each_gesture():
    for seq, want in ((wave_sequence(), "wave"),
                      (raise_sequence(), "raise_hand")):
        assert fired_gestures(feed(GestureReactor(CFG), seq)) == [want]


def test_continuous_wave_debounced_to_one():
    # A long, uninterrupted wave (well past the cooldown window) still fires exactly once.
    f = fired_gestures(feed(GestureReactor(CFG), wave_sequence(n=48)))
    assert f == ["wave"]


def test_event_payload_shape():
    ev = feed(GestureReactor(CFG), wave_sequence())[0]
    assert ev["gesture"] == "wave"
    assert ev["robot_gesture"] == GESTURE_TO_ROBOT["wave"]
    assert ev["track_id"] == 1
    assert "human" in ev and "t" in ev


# =============================================================== reactor: debounce timing
def test_cooldown_blocks_quick_refire():
    # Two wave bursts within the per-gesture cooldown -> only the first fires.
    cfg = ReactorConfig(cooldown_s=4.0, refractory_s=2.0)
    r = GestureReactor(cfg)
    frames = wave_sequence(n=20) + idle_sequence(n=6) + wave_sequence(n=20)
    assert fired_gestures(feed(r, frames)).count("wave") == 1


def test_fires_again_after_cooldown_elapses():
    # Same reactor, two separate waves with a clock gap far larger than cooldown +
    # refractory between them -> each burst fires once (debounce is per-episode, not a
    # permanent lockout). n=20 -> ~1.6 s, shorter than the 4 s cooldown, so no double-fire
    # within a burst; the second burst starts well after the cooldown has expired.
    r = GestureReactor(CFG)
    first = fired_gestures(feed(r, wave_sequence(n=20), t0=100.0))
    second = fired_gestures(feed(r, wave_sequence(n=20), t0=120.0))
    assert first == ["wave"] and second == ["wave"]


def test_min_track_age_delays_first_fire():
    # A wave present from the very first frame must not fire until the track has been seen
    # for min_track_age_s (no reacting to a person who just popped into view).
    events = feed(GestureReactor(CFG), wave_sequence(), t0=100.0)
    assert events and (events[0]["t"] - 100.0) >= CFG.min_track_age_s


# =============================================================== reactor: robustness
def test_wave_survives_wrist_dropout():
    # Right wrist drops below KP_MIN_CONF on every 3rd frame; the good frames still carry
    # enough of the ~1 s window to classify the wave.
    seq = wave_sequence(n=30)
    for i, sk in enumerate(seq):
        if i % 3 == 0:
            sk["kpts"][R_WRI][2] = 0.1
    assert fired_gestures(feed(GestureReactor(CFG), seq)).count("wave") == 1


def test_wave_survives_head_joint_dropout():
    assert fired_gestures(feed(GestureReactor(CFG),
                               wave_sequence(low_conf=(R_EAR,)))) == ["wave"]


def test_low_keypoint_count_target_ignored():
    # A person with almost no confident keypoints (below min_conf_frac) is never reacted to.
    sk = make_skeleton(sg.REST_L, (CX + 0.5 * SW, CY - 0.6 * SW))
    for i in range(17):
        if i not in (R_WRI,):
            sk["kpts"][i][2] = 0.05      # only the wrist is confident -> too sparse
    assert feed(GestureReactor(CFG), [sk] * 20) == []


def test_seeded_jitter_wave_still_fires():
    import random
    seq = wave_sequence(rng=random.Random(7), jitter=3.5)
    assert fired_gestures(feed(GestureReactor(CFG), seq)).count("wave") == 1


# =============================================================== reactor: negatives
@pytest.mark.parametrize("seq_fn", [idle_sequence, walking_sequence, reach_sequence])
def test_negative_sequences_never_fire(seq_fn):
    assert feed(GestureReactor(CFG), seq_fn()) == []


def test_items_without_id_ignored():
    seq = wave_sequence()
    for sk in seq:
        sk["id"] = None
    assert feed(GestureReactor(CFG), seq) == []


def test_empty_frames_no_crash():
    r = GestureReactor(CFG)
    assert r.update({"w": 640, "h": 480, "items": []}, 100.0) is None
    assert r.update([], 100.1) is None


# =============================================================== reactor: multi-person target
def _waving_person(i, tid, cx, cy, sw, freq=1.4, amp=0.28):
    t = i * DT
    dx = amp * sw * math.sin(2 * math.pi * freq * t)
    return make_skeleton((cx - 0.5 * sw, cy + 1.15 * sw),
                         (cx + 0.5 * sw + dx, cy - 0.55 * sw),
                         cx=cx, cy=cy, sw=sw, tid=tid)


def _idle_person(tid, cx, cy, sw):
    return make_skeleton((cx - 0.5 * sw, cy + 1.15 * sw),
                         (cx + 0.5 * sw, cy + 1.15 * sw),
                         cx=cx, cy=cy, sw=sw, tid=tid)


def test_target_pick_reacts_to_largest_waver():
    # Two people: a big (near) waver on the left + a small (far) idle person on the right.
    # The reactor greets the nearest/biggest person -> waves back, tagged with THEIR id.
    frames = [[_waving_person(i, 1, cx=210, cy=240, sw=150),
               _idle_person(2, cx=520, cy=240, sw=85)] for i in range(26)]
    events = feed(GestureReactor(CFG), frames)
    assert fired_gestures(events).count("wave") == 1
    assert events[0]["track_id"] == 1


def test_target_pick_ignores_small_waver_behind_big_idler():
    # Flip the sizes: the big (near) person is idle, the waver is small/far. The reactor
    # targets the big idle person, so the far wave is (correctly) NOT reacted to.
    frames = [[_idle_person(1, cx=320, cy=240, sw=150),
               _waving_person(i, 2, cx=560, cy=240, sw=80)] for i in range(26)]
    assert feed(GestureReactor(CFG), frames) == []


# =============================================================== misc API
def test_reset_clears_state():
    r = GestureReactor(CFG)
    feed(r, wave_sequence(), t0=100.0)
    r.reset()
    assert r._hist == {} and r._first_seen == {} and r._cooldown == {}


def test_describe_reads_naturally():
    ev = feed(GestureReactor(CFG), wave_sequence())[0]
    assert describe(ev) == "saw wave -> waving back"
    assert describe(None) == ""


def test_gesture_to_robot_mapping_complete():
    # Every human gesture the classifier can emit must map to a robot response, and every
    # response must be a cmd the controller's apply_cmd already knows (guards silent typos).
    known_cmds = {"wave", "shake", "high_five", "hug", "heart", "clap", "high_wave",
                  "kiss", "hands_up", "release_arm"}
    assert set(GESTURE_TO_ROBOT) == {"wave", "raise_hand"}
    for robot_cmd in GESTURE_TO_ROBOT.values():
        assert robot_cmd in known_cmds


# =============================================================== GreetingService gate
# The safety-gated bridge the controller wires. We drive poll_once() directly with
# synthesized frames + lambda gates -- no thread, no shm -- so the fire/skip logic that
# keeps the arms safe near clients is deterministically tested.
def _greeting(safe, cfg=None):
    fired, skipped, events = [], [], []
    svc = GreetingService(
        tracks_path="/nonexistent", enabled_fn=lambda: True, safe_fn=safe,
        fire_fn=fired.append, on_event=events.append, on_skip=skipped.append, cfg=cfg or CFG)
    return svc, fired, skipped, events


def _drive(svc, frames, t0=100.0):
    t = t0
    for fr in frames:
        items = fr if isinstance(fr, list) else [fr]
        svc.poll_once({"w": 640, "h": 480, "items": items}, t)
        t += DT


def test_greeting_fires_robot_gesture_when_safe():
    svc, fired, skipped, events = _greeting(safe=lambda: True)
    _drive(svc, wave_sequence())
    assert fired == [GESTURE_TO_ROBOT["wave"]]     # WaveHand cmd, once
    assert skipped == [] and len(events) == 1


def test_greeting_never_fires_when_unsafe():
    # e.g. robot is walking / mid-gesture -> classify still happens (dashboard sees it) but
    # the arm MUST NOT move; on_skip is notified instead.
    svc, fired, skipped, events = _greeting(safe=lambda: False)
    _drive(svc, wave_sequence())
    assert fired == []
    assert len(skipped) == 1 and len(events) == 1


def test_greeting_idle_neither_fires_nor_skips():
    svc, fired, skipped, events = _greeting(safe=lambda: True)
    _drive(svc, idle_sequence())
    assert fired == [] and skipped == [] and events == []


def test_greeting_disable_resets_reactor():
    # Mirrors the controller's OFF handler: reset() so a pose held across the OFF window
    # cannot fire the instant greeting mode comes back ON.
    svc, fired, _, _ = _greeting(safe=lambda: True)
    _drive(svc, wave_sequence()[:6])
    svc.reactor.reset()
    assert svc.reactor._hist == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
