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
    _kp, _frame_scale, _hand_open, _swings, R_EAR, R_WRI,
)
import sim_gestures as sg
from sim_gestures import (
    make_skeleton, feed, fired_gestures, wave_sequence, raise_sequence,
    idle_sequence, walking_sequence, reach_sequence,
    make_hand, hands_for_wave, no_shoulders,
    shoulder_level_wave_sequence, subtle_wave_sequence, clap_sequence, gesticulate_sequence,
    held_raised_jitter_sequence, stir_sequence, nod_sequence, drift_sequence, tremor_sequence,
    clear_wave_at_distance_sequence,
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


def test_raise_hand_no_longer_classifies():
    # Auto high-five removed: a still raised hand must classify as nothing now.
    assert classify(history_from(raise_sequence(), CFG), CFG) is None


def test_idle_classifies_none():
    assert classify(history_from(idle_sequence(), CFG), CFG) is None


def test_walking_classifies_none():
    # Horizontal wrist oscillation with the wrists BELOW the shoulders must not read as a
    # wave -- the "wrist above shoulder" gate is what rejects arm-swing while walking.
    assert classify(history_from(walking_sequence(), CFG), CFG) is None


def test_static_reach_classifies_none():
    # Arm up but not oscillating: not a wave.
    assert classify(history_from(reach_sequence(), CFG), CFG) is None


def test_single_frame_never_fires():
    assert classify(history_from(wave_sequence(n=1), CFG), CFG) is None


# =============================================================== reactor: firing
def test_reactor_fires_wave_once():
    f = fired_gestures(feed(GestureReactor(CFG), wave_sequence()))
    assert f.count("wave") == 1


def test_reactor_fires_only_wave():
    # The sole gesture: a wave fires exactly one 'wave' event and nothing else.
    assert fired_gestures(feed(GestureReactor(CFG), wave_sequence())) == ["wave"]


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
    assert set(GESTURE_TO_ROBOT) == {"wave"}
    for robot_cmd in GESTURE_TO_ROBOT.values():
        assert robot_cmd in known_cmds


# =============================================================== wave tiers + palm fusion
# The G1 is short: a normal wave is often at SHOULDER LEVEL (hand not raised high), and a
# close person's head + shoulders drop out of frame. Admission is elbow-based (forearm
# raised); a CLEAR sweep fires on the skeleton alone at any height; a SUBTLE (small) sweep
# needs open-palm corroboration; two raised oscillating arms are vetoed (clap).
def test_full_wave_fires_without_hands():
    # Regression: a clear overhead wave still fires with NO hand feed at all.
    assert fired_gestures(feed(GestureReactor(CFG), wave_sequence())) == ["wave"]


def test_strong_wave_tagged_skeleton():
    ev = feed(GestureReactor(CFG), wave_sequence())[0]
    assert ev["source"] == "skeleton"


def test_shoulder_level_wave_fires_alone():
    # THE reported miss: a normal wave at shoulder height (hand not raised high) now fires on
    # the skeleton alone via the elbow-based raised test.
    events = feed(GestureReactor(CFG), shoulder_level_wave_sequence())
    assert fired_gestures(events).count("wave") == 1
    assert events[0]["source"] == "skeleton"


def test_shoulderless_clear_wave_fires_alone():
    # Partial body (shoulders out of frame) + a clear sweep -> fires alone (elbow reference).
    assert fired_gestures(feed(GestureReactor(CFG), no_shoulders(wave_sequence()))).count("wave") == 1


def test_subtle_wave_alone_does_not_fire():
    # A small-amplitude wave is a WEAK candidate; with no palm it must NOT fire.
    assert feed(GestureReactor(CFG), subtle_wave_sequence()) == []


def test_subtle_wave_with_open_palm_fires():
    seq = subtle_wave_sequence()
    hands = hands_for_wave(subtle_wave_sequence(), is_open=True)
    events = feed(GestureReactor(CFG), seq, hand_frames=hands)
    assert fired_gestures(events).count("wave") == 1
    assert events[0]["source"] == "skeleton+palm"


def test_subtle_wave_with_fist_does_not_fire():
    seq = subtle_wave_sequence()
    fists = hands_for_wave(subtle_wave_sequence(), is_open=False)
    assert feed(GestureReactor(CFG), seq, hand_frames=fists) == []


def test_subtle_wave_stray_hand_does_not_corroborate():
    # An open oscillating palm far from the wrist (assoc gate) can't validate the weak wave.
    seq = subtle_wave_sequence()
    stray = [{"w": 640, "h": 480, "items": [make_hand(CX + 3 * SW + 12 * math.sin(i), CY)]}
             for i in range(len(seq))]
    assert feed(GestureReactor(CFG), seq, hand_frames=stray) == []


def test_subtle_wave_hands_disabled_does_not_fire():
    # With fusion off, a subtle wave can't be corroborated -> never fires.
    cfg = ReactorConfig(use_hands=False)
    seq = subtle_wave_sequence()
    hands = hands_for_wave(subtle_wave_sequence(), is_open=True)
    assert feed(GestureReactor(cfg), seq, hand_frames=hands) == []


def test_clap_is_vetoed():
    # Both forearms raised + oscillating -> bimanual veto -> no wave.
    assert feed(GestureReactor(CFG), clap_sequence()) == []


def test_single_sweep_reach_does_not_fire():
    # One lateral sweep (rev <= 1) is a reach / conversational gesture, not a wave.
    assert feed(GestureReactor(CFG), gesticulate_sequence()) == []


def test_hand_open_detects_open_vs_fist():
    assert _hand_open(make_hand(300, 200, is_open=True)["landmarks"]) is True
    assert _hand_open(make_hand(300, 200, is_open=False)["landmarks"]) is False


# =============================================================== jitter / non-wave robustness
# The reported on-robot false positive: a raised-but-HELD hand (palm down) fired because pose
# keypoints jitter a few px/frame and at distance the body scale is small, so that noise faked
# reversals + amplitude. The hysteresis swing floor (absolute px) + horizontal-dominance +
# rate-cap must reject a whole family of non-wave movements while keeping real waves.
def test_swings_ignores_jitter_but_counts_a_wave():
    import random as _r
    rng = _r.Random(3)
    held = [100.0 + rng.uniform(-5, 5) for _ in range(20)]        # jitter around a held position
    rev_h, sw_h = _swings(held, 14.0)
    assert rev_h == 0 and sw_h < 14.0
    wave = [100.0 + 30 * math.sin(2 * math.pi * 1.3 * i * 0.08) for i in range(20)]
    rev_w, sw_w = _swings(wave, 14.0)
    assert rev_w >= 2 and sw_w > 40


def test_held_raised_jitter_does_not_fire():
    # A raised, HELD hand at distance with pose jitter -> no reversal clears the swing floor.
    assert feed(GestureReactor(CFG), held_raised_jitter_sequence()) == []


def test_held_raised_jitter_with_open_palm_does_not_fire():
    # The reported bug end-to-end: held raised hand + jitter + a corroborating open-palm feed.
    seq = held_raised_jitter_sequence()
    hands = hands_for_wave(held_raised_jitter_sequence(), is_open=True)
    assert feed(GestureReactor(CFG), seq, hand_frames=hands) == []


def test_held_closed_fist_jitter_does_not_fire():
    seq = held_raised_jitter_sequence()
    fists = hands_for_wave(held_raised_jitter_sequence(), is_open=False)
    assert feed(GestureReactor(CFG), seq, hand_frames=fists) == []


def test_circular_stir_does_not_fire():
    # x is a real sine but y co-moves (a circle) -> horizontal-dominance rejects it.
    assert feed(GestureReactor(CFG), stir_sequence()) == []


def test_vertical_nod_does_not_fire():
    assert feed(GestureReactor(CFG), nod_sequence()) == []


def test_monotonic_drift_does_not_fire():
    # A slow one-way drift never retraces past the swing floor -> zero reversals.
    assert feed(GestureReactor(CFG), drift_sequence()) == []


def test_fast_tremor_does_not_fire():
    # A fast (~4 Hz) shake clears the swing floor but exceeds the reversal-rate cap.
    assert feed(GestureReactor(CFG), tremor_sequence()) == []


def test_clear_wave_at_distance_still_fires():
    # Hardening must not kill a clear wave far away (only tiny/subtle far waves are traded off).
    assert fired_gestures(feed(GestureReactor(CFG), clear_wave_at_distance_sequence())).count("wave") == 1


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
    assert fired == [GESTURE_TO_ROBOT["wave"]]     # working wave-back cmd (high_wave), once
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
