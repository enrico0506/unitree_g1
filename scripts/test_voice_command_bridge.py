#!/usr/bin/env python3
"""Unit tests for the voice-command bridge (scripts/voice_command_bridge.py).

Two layers, same split as test_gesture_reactor.py:
  * VoiceCommandClassifier -- the PURE decision logic, fed hand-built events + context,
    no shm/thread/robot in the way.
  * VoiceCommandBridge -- the stateful service wrapper (enabled_fn/safe_fn gating,
    dispatch routing), driven with plain lambdas exactly like test_gesture_reactor.py
    drives GreetingService.

The bias throughout matches gesture_reactor's: proving the NEGATIVE cases (stale, low
confidence, no person nearby, cooldown, disabled) matters as much as the positives,
because an auto-fired robot action at the wrong moment is the failure that actually
matters here -- a voice command that fires when it shouldn't is worse than one that
occasionally fails to fire when it should.

No real audio/robot/mic needed anywhere in this file -- every event is injected
directly as a plain dict, exactly the shape perception/voice/voice_service.py writes to
/dev/shm/g1_voice_cmd.json.

    pytest scripts/test_voice_command_bridge.py -v
"""
import pytest

from voice_command_bridge import (
    VoiceCommandBridge, VoiceCommandClassifier, VoiceBridgeConfig, CommandDef,
    load_voice_config, cfg_from_loaded, person_nearby, VOICE_COMMANDS_PATH,
)


# --------------------------------------------------------------------- fixtures/helpers
def make_commands():
    """A small synthetic catalog covering every action kind, independent of the real
    config/voice_commands.yaml file (so these tests don't break if that file's phrasing
    or cooldowns get retuned)."""
    return [
        CommandDef("stop", ["stop", "halt"], {"kind": "stop"},
                   requires_person_nearby=False, cooldown_s=0.0),
        CommandDef("resume", ["resume"], {"kind": "hook", "name": "resume"},
                   requires_person_nearby=False, cooldown_s=3.0),
        CommandDef("come_here", ["come here"], {"kind": "hook", "name": "come_here"},
                   requires_person_nearby=True, cooldown_s=5.0),
        CommandDef("dance", ["dance"], {"kind": "dance", "fsm_id": 503},
                   requires_person_nearby=False, cooldown_s=10.0),
        CommandDef("wave", ["wave"], {"kind": "cmd", "name": "high_wave"},
                   requires_person_nearby=True, cooldown_s=4.0),
        CommandDef("sit", ["sit"], {"kind": "cmd", "name": "low_stand"},
                   requires_person_nearby=False, cooldown_s=3.0),
        CommandDef("stand", ["stand"], {"kind": "cmd", "name": "high_stand"},
                   requires_person_nearby=False, cooldown_s=3.0),
    ]


def make_cfg(**kw):
    kw.setdefault("max_event_age_s", 1.8)
    kw.setdefault("default_confidence", 0.65)
    return VoiceBridgeConfig(**kw)


def event(cmd, confidence=0.9, t=1000.0, raw_text=None):
    return {"cmd": cmd, "confidence": confidence, "raw_text": raw_text or cmd, "t": t}


def person_frame(near=True):
    """A single-person POSE_TRACKS-shaped frame, centered and either big (near) or
    tiny (far), matching person_nearby()'s box-area-fraction heuristic."""
    if near:
        box = [200, 100, 440, 380]     # big, centered box in a 640x480 frame
    else:
        box = [310, 235, 330, 245]     # tiny box -> far below person_nearby_min_frac
    return {"w": 640, "h": 480, "items": [{"id": 1, "box": box}]}


# =====================================================================================
# VoiceCommandClassifier -- pure logic
# =====================================================================================
class TestClassifierPositive:
    def test_fresh_confident_command_fires(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("dance"), now=1000.2)
        assert d["fire"] is True
        assert d["reason"] == "ok"
        assert d["action"] == {"kind": "dance", "fsm_id": 503}

    def test_person_gated_command_fires_with_person_present(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("wave"), now=1000.2, person_nearby=True)
        assert d["fire"] is True

    def test_hook_action_resolves_but_bridge_routes_it_not_dispatch_fn(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("resume"), now=1000.2)
        assert d["fire"] is True
        assert d["action"]["kind"] == "hook"


class TestClassifierNegative:
    def test_stale_event_rejected(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg(max_event_age_s=1.5))
        # event.t is 3s before `now` -- well past the 1.5s staleness floor.
        d = clf.decide(event("dance", t=1000.0), now=1003.0)
        assert d["fire"] is False
        assert d["reason"] == "stale"

    def test_just_inside_staleness_window_still_fires(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg(max_event_age_s=1.5))
        d = clf.decide(event("dance", t=1000.0), now=1001.0)   # 1.0s old, under 1.5s floor
        assert d["fire"] is True

    def test_low_confidence_event_rejected(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg(default_confidence=0.65))
        d = clf.decide(event("dance", confidence=0.3), now=1000.1)
        assert d["fire"] is False
        assert d["reason"] == "low_confidence"

    def test_confidence_right_at_floor_fires(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg(default_confidence=0.65))
        d = clf.decide(event("dance", confidence=0.65), now=1000.1)
        assert d["fire"] is True

    def test_per_command_confidence_override_wins_over_default(self):
        cmds = make_commands()
        cmds.append(CommandDef("shy", ["shy"], {"kind": "cmd", "name": "high_stand"},
                                confidence_override=0.9))
        clf = VoiceCommandClassifier(cmds, make_cfg(default_confidence=0.3))
        # Would pass the GLOBAL default (0.3) but not this command's override (0.9).
        d = clf.decide(event("shy", confidence=0.5), now=1000.1)
        assert d["fire"] is False
        assert d["reason"] == "low_confidence"

    def test_motion_command_rejected_when_person_required_and_absent(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("come_here"), now=1000.1, person_nearby=False)
        assert d["fire"] is False
        assert d["reason"] == "no_person_nearby"

    def test_wave_rejected_without_person(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("wave"), now=1000.1, person_nearby=False)
        assert d["fire"] is False
        assert d["reason"] == "no_person_nearby"

    def test_cooldown_prevents_rapid_refire(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d1 = clf.decide(event("dance", t=1000.0), now=1000.1)
        assert d1["fire"] is True
        d2 = clf.decide(event("dance", t=1001.0), now=1001.1)   # 1s later, cooldown_s=10
        assert d2["fire"] is False
        assert d2["reason"] == "cooldown"

    def test_cooldown_clears_after_window(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        clf.decide(event("wave", t=1000.0), now=1000.1, person_nearby=True)
        # wave's cooldown_s=4.0 -- 5s later should be clear.
        d2 = clf.decide(event("wave", t=1005.1), now=1005.2, person_nearby=True)
        assert d2["fire"] is True

    def test_unknown_command_rejected(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d = clf.decide(event("moonwalk"), now=1000.1)
        assert d["fire"] is False
        assert d["reason"] == "unknown_command"

    def test_missing_timestamp_rejected(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        e = event("dance")
        del e["t"]
        d = clf.decide(e, now=1000.1)
        assert d["fire"] is False
        assert d["reason"] == "missing_timestamp"

    def test_stop_bypasses_cooldown(self):
        clf = VoiceCommandClassifier(make_commands(), make_cfg())
        d1 = clf.decide(event("stop", t=1000.0), now=1000.0)
        assert d1["fire"] is True
        d2 = clf.decide(event("stop", t=1000.05), now=1000.05)   # 50ms later
        assert d2["fire"] is True     # stop's cooldown_s=0 AND it's exempt from the gate

    def test_stop_bypasses_person_nearby_gate(self):
        # stop has requires_person_nearby=False in the catalog anyway, but assert the
        # exemption explicitly holds even if a future config mistakenly set it True.
        cmds = make_commands()
        cmds[0] = CommandDef("stop", ["stop"], {"kind": "stop"},
                              requires_person_nearby=True, cooldown_s=0.0)
        clf = VoiceCommandClassifier(cmds, make_cfg())
        d = clf.decide(event("stop"), now=1000.1, person_nearby=False)
        assert d["fire"] is True

    def test_stop_still_rejected_when_stale(self):
        # stop bypasses cooldown/person-nearby, NOT staleness/confidence -- a stale or
        # unintelligible "stop" is still meaningless.
        clf = VoiceCommandClassifier(make_commands(), make_cfg(max_event_age_s=1.5))
        d = clf.decide(event("stop", t=1000.0), now=1003.0)
        assert d["fire"] is False
        assert d["reason"] == "stale"


# =====================================================================================
# person_nearby() heuristic
# =====================================================================================
class TestPersonNearby:
    def test_big_centered_box_counts_as_nearby(self):
        assert person_nearby(person_frame(near=True)) is True

    def test_tiny_box_does_not_count_as_nearby(self):
        assert person_nearby(person_frame(near=False)) is False

    def test_missing_frame_is_not_nearby(self):
        assert person_nearby(None) is False

    def test_empty_items_is_not_nearby(self):
        assert person_nearby({"w": 640, "h": 480, "items": []}) is False


# =====================================================================================
# VoiceCommandBridge -- the shm-polling service (driven directly, no threads/shm)
# =====================================================================================
def _bridge(enabled=True, safe=True, on_event=None, on_skip=None, **kw):
    fired = []
    hooks_fired = []
    events = [] if on_event is None else on_event
    skipped = [] if on_skip is None else on_skip

    def dispatch_fn(action):
        fired.append(action)

    def resume_fn():
        hooks_fired.append("resume")

    def come_here_fn():
        hooks_fired.append("come_here")

    bridge = VoiceCommandBridge(
        shm_path="/dev/shm/__unused_in_these_tests__.json",
        commands=make_commands(), cfg=make_cfg(),
        enabled_fn=lambda: enabled, safe_fn=lambda: safe,
        dispatch_fn=dispatch_fn, resume_fn=resume_fn, come_here_fn=come_here_fn,
        on_event=events.append, on_skip=skipped.append,
        **kw)
    return bridge, fired, hooks_fired, events, skipped


class TestVoiceCommandBridge:
    def test_fires_cmd_action_through_dispatch_fn_when_safe(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        ok = bridge.poll_once(event("wave", t=1000.0), now=1000.1,
                               person_frame=person_frame(near=True))
        assert ok is True
        assert fired == [{"kind": "cmd", "name": "high_wave"}]
        assert hooks == []
        assert len(events) == 1 and skipped == []

    def test_hook_action_routes_to_resume_fn_not_dispatch_fn(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        ok = bridge.poll_once(event("resume", t=1000.0), now=1000.1)
        assert ok is True
        assert fired == []
        assert hooks == ["resume"]

    def test_hook_action_come_here_routes_correctly(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        ok = bridge.poll_once(event("come_here", t=1000.0), now=1000.1,
                               person_frame=person_frame(near=True))
        assert ok is True
        assert hooks == ["come_here"]

    def test_disabled_bridge_fires_nothing(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=False, safe=True)
        ok = bridge.poll_once(event("dance", t=1000.0), now=1000.1)
        assert ok is False
        assert fired == [] and hooks == []
        # Disabled means we never even reach the classifier -- no event/skip callback either.
        assert events == [] and skipped == []

    def test_unsafe_gate_blocks_dispatch_but_still_reports_the_event(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=False)
        ok = bridge.poll_once(event("dance", t=1000.0), now=1000.1)
        assert ok is False
        assert fired == [] and hooks == []
        # Classified fine (on_event sees it) but gated out by safe_fn (on_skip notified),
        # mirroring GreetingService's on_event/on_skip split exactly.
        assert len(events) == 1
        assert len(skipped) == 1 and skipped[0]["reason"] == "unsafe"

    def test_classifier_gate_skip_also_reports_on_skip(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        ok = bridge.poll_once(event("dance", confidence=0.1, t=1000.0), now=1000.1)
        assert ok is False
        assert len(skipped) == 1 and skipped[0]["reason"] == "low_confidence"

    def test_stop_dispatches_even_when_a_cooldown_would_block_others(self):
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        bridge.poll_once(event("stop", t=1000.0), now=1000.0)
        ok2 = bridge.poll_once(event("stop", t=1000.05), now=1000.05)
        assert ok2 is True
        assert fired == [{"kind": "stop"}, {"kind": "stop"}]

    def test_re_enabling_resets_cooldown_state(self):
        # Mirrors GreetingService's reset-on-disable: a cooldown started while voice_mode
        # was on shouldn't silently leak past a disable/enable cycle in the _loop path.
        bridge, fired, hooks, events, skipped = _bridge(enabled=True, safe=True)
        bridge.poll_once(event("dance", t=1000.0), now=1000.0)
        bridge.classifier.reset()   # what _loop does on enabled_fn() -> False
        ok = bridge.poll_once(event("dance", t=1000.5), now=1000.5)   # well inside cooldown_s
        assert ok is True


# =====================================================================================
# config/voice_commands.yaml -- sanity-check the REAL file this all loads from
# =====================================================================================
class TestRealConfig:
    def test_real_config_parses_and_has_all_seven_commands(self):
        loaded = load_voice_config(VOICE_COMMANDS_PATH)
        names = {c.name for c in loaded["commands"]}
        assert names == {"stop", "resume", "come_here", "dance", "wave", "sit", "stand"}

    def test_real_config_hook_commands_are_not_real_actions(self):
        loaded = load_voice_config(VOICE_COMMANDS_PATH)
        by_name = {c.name: c for c in loaded["commands"]}
        assert by_name["resume"].action["kind"] == "hook"
        assert by_name["come_here"].action["kind"] == "hook"

    def test_real_config_stop_bypasses_by_kind(self):
        loaded = load_voice_config(VOICE_COMMANDS_PATH)
        by_name = {c.name: c for c in loaded["commands"]}
        assert by_name["stop"].action == {"kind": "stop"}

    def test_real_config_wave_uses_working_high_wave_not_dead_wave(self):
        loaded = load_voice_config(VOICE_COMMANDS_PATH)
        by_name = {c.name: c for c in loaded["commands"]}
        assert by_name["wave"].action == {"kind": "cmd", "name": "high_wave"}

    def test_real_config_dance_uses_verified_fsm_id_503(self):
        loaded = load_voice_config(VOICE_COMMANDS_PATH)
        by_name = {c.name: c for c in loaded["commands"]}
        assert by_name["dance"].action == {"kind": "dance", "fsm_id": 503}

    def test_real_config_loads_into_a_working_bridge(self):
        # End-to-end sanity: construct a VoiceCommandBridge from the REAL config file
        # (not the synthetic make_commands()) and fire one command through it.
        fired = []
        bridge = VoiceCommandBridge(
            shm_path="/dev/shm/__unused_in_these_tests__.json",
            config_path=VOICE_COMMANDS_PATH,
            enabled_fn=lambda: True, safe_fn=lambda: True,
            dispatch_fn=fired.append)
        ok = bridge.poll_once(event("dance", t=1000.0, confidence=0.9), now=1000.1)
        assert ok is True
        assert fired == [{"kind": "dance", "fsm_id": 503}]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
