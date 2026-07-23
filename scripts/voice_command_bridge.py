#!/usr/bin/env python3
"""Voice-command -> robot-action BRIDGE (the safety-gated half of Track C).

Mirrors gesture_reactor.py's split exactly: a PURE classifier (no robot, no I/O, no
shm -- deterministic, fully unit-testable) plus a thin shm-polling *Service* class that
applies a safety gate and fires the resolved action through caller-supplied callbacks.

perception/voice/voice_service.py (a separate, standalone process -- mic capture, wake
word, transcription, phrase matching) writes ONE recognized-command event to
/dev/shm/g1_voice_cmd.json: {"cmd": <name>, "confidence": <0-1>, "raw_text": <str>,
"t": <event wall-clock time>} -- ONLY after a wake-word hit + a phrase match (never a
raw always-on transcript; see that file + config/voice_commands.yaml for why). This
module never touches audio -- it only reads that one small JSON file.

Command definitions (phrases / action / requires_person_nearby / cooldown_s) live in
config/voice_commands.yaml, loaded once at construction (same "config is the source of
truth" convention as config/dances.yaml).

LAYERED GATES, in this order (VoiceCommandClassifier.decide):
    1. staleness      -- event.t more than ~1.5-2s old BY READ TIME -> reject. A stale
                         event usually means the event file wasn't refreshed (voice
                         service died/stalled) and firing a minutes-old "come here" is
                         exactly the kind of surprise motion this gate exists to kill.
    2. confidence     -- event.confidence must clear a floor (config default, optional
                         per-command override) -- a marginal phrase match doesn't fire.
    3. person-nearby  -- ONLY for commands with requires_person_nearby: true (currently
                         come_here + wave). Approximated via a POSE_TRACKS box-size/
                         centrality heuristic (see person_nearby() below), deliberately
                         mirroring gesture_reactor.py's _target_score() -- this is a
                         PLACEHOLDER for real bearing/distance fusion once Track A's
                         g1_person_track.json exists; swap the implementation then, not
                         the gate's position in the pipeline.
    4. cooldown       -- per-command refractory (config cooldown_s), mirrors
                         ReactorConfig's cooldown_s/refractory_s idea: a repeated
                         "dance" can't machine-gun the robot into repeated whole-body
                         routines.
`stop` is a deliberate exception: it still needs a fresh, confident event (a stale or
mis-heard "stop" is meaningless either way), but it BYPASSES the person-nearby gate and
the cooldown/refractory bookkeeping -- a spoken stop must always be able to fire again
immediately, exactly like the WS mtype=="stop" fast path it mirrors.

On top of the classifier, VoiceCommandBridge (the shm-polling service) adds the SAME
safety envelope GreetingService already uses:
    enabled_fn() -- master "voice_mode" toggle, OFF by default (wiring supplies this).
    safe_fn()    -- reuse the EXACT SAME predicate GreetingService's _greeting_safe uses
                    (robot_web_controller.py: upright walk mode, not transitioning, arm
                    not raised, no pending_cmd, arm not busy) -- passed in by the wiring
                    step, not reimplemented here.
    dispatch_fn(action) -- how a resolved stop/cmd/dance action actually reaches the
                    robot (the wiring step connects this to apply_cmd()/state, see the
                    module docstring's wiring notes at the bottom of this file).
    resume_fn() / come_here_fn() -- the explicit hook seam for `kind: hook` actions.
                    Default to a harmless log-only stub so this file never needs
                    editing again once Track A's patrol_supervisor.py lands -- that
                    later workstream just passes real callables in.

    python3 scripts/voice_command_bridge.py --selftest
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_COMMANDS_PATH = BASE_DIR / "config" / "voice_commands.yaml"


# --------------------------------------------------------------------- config
class VoiceBridgeConfig:
    """Tunables not already covered by config/voice_commands.yaml (keyword-overridable,
    mirrors gesture_reactor.ReactorConfig's style so tests can tighten/loosen)."""

    def __init__(self, **kw):
        # staleness / confidence (normally sourced FROM voice_commands.yaml at load
        # time via cfg_from_loaded(); overridable directly for tests).
        self.max_event_age_s = 1.8      # event.t older than this (by read time) -> stale
        self.default_confidence = 0.65  # global confidence floor; per-cmd override wins

        # person-nearby heuristic (placeholder -- see module docstring). A box's
        # (area / frame_area) must clear this to count as "near", with an off-centre
        # penalty of the SAME shape as gesture_reactor._target_score's (up to
        # person_nearby_center_penalty knocked off for a box at the frame edge).
        self.person_nearby_min_frac = 0.12
        self.person_nearby_center_penalty = 0.3

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown VoiceBridgeConfig field: {k}")
            setattr(self, k, v)


DEFAULT_CFG = VoiceBridgeConfig()


# --------------------------------------------------------------------- command catalog
class CommandDef:
    """One config/voice_commands.yaml entry. Plain attribute bag (no methods) -- easy to
    hand-construct in tests without touching the YAML file at all."""

    __slots__ = ("name", "phrases", "action", "requires_person_nearby",
                 "cooldown_s", "confidence_override")

    def __init__(self, name, phrases, action, requires_person_nearby=False,
                 cooldown_s=3.0, confidence_override=None):
        self.name = name
        self.phrases = list(phrases or [])
        self.action = dict(action or {})
        self.requires_person_nearby = bool(requires_person_nearby)
        self.cooldown_s = float(cooldown_s)
        self.confidence_override = (float(confidence_override)
                                     if confidence_override is not None else None)

    def __repr__(self):
        return f"CommandDef({self.name!r}, action={self.action!r})"


def load_voice_config(path=None):
    """Parse config/voice_commands.yaml -> {"wake_word", "default_confidence",
    "max_event_age_s", "commands": [CommandDef, ...]}. Malformed/missing entries are
    skipped defensively (mirrors load_dances()'s tolerance) rather than crashing the
    whole catalog over one bad line."""
    path = path or VOICE_COMMANDS_PATH
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    commands = []
    for c in (data.get("commands") or []):
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        phrases = [str(p).strip().lower() for p in (c.get("phrases") or []) if str(p).strip()]
        action = c.get("action") or {}
        if not isinstance(action, dict) or "kind" not in action:
            print(f"[VOICE] skipping command {name!r} -- action missing/malformed", flush=True)
            continue
        commands.append(CommandDef(
            name=name, phrases=phrases, action=action,
            requires_person_nearby=bool(c.get("requires_person_nearby", False)),
            cooldown_s=float(c.get("cooldown_s", 3.0)),
            confidence_override=c.get("confidence_override"),
        ))
    return {
        "wake_word": data.get("wake_word") or {},
        "default_confidence": float(data.get("default_confidence", 0.65)),
        "max_event_age_s": float(data.get("max_event_age_s", 1.8)),
        "commands": commands,
    }


def cfg_from_loaded(loaded, **overrides):
    """Build a VoiceBridgeConfig from load_voice_config()'s dict, letting callers
    override individual fields (e.g. tests tightening max_event_age_s)."""
    kw = {"max_event_age_s": loaded["max_event_age_s"],
          "default_confidence": loaded["default_confidence"]}
    kw.update(overrides)
    return VoiceBridgeConfig(**kw)


# --------------------------------------------------------------------- person-nearby
def parse_pose_frame(frame):
    """Pose shm dict {w,h,items:[...]} or a bare items list -> (items, w, h). Same
    tolerant shape as gesture_reactor.parse_frame (deliberately duplicated rather than
    imported -- this module must stay importable even if gesture_reactor.py's pose-
    specific COCO constants ever change; the two files agree by convention, not
    coupling)."""
    if isinstance(frame, dict):
        return frame.get("items", []) or [], frame.get("w"), frame.get("h")
    return list(frame or []), None, None


def _box_nearby_score(box, w, h, cfg):
    """One person's box -> a "how near/central is this person" score in [0, ~1].

    Approach mirrors gesture_reactor._target_score(): box AREA as a fraction of the
    frame (bigger box = nearer camera), with an off-centre penalty of the same shape.
    This is a coarse, camera-FOV-only proxy -- no real bearing/distance -- explicitly a
    placeholder until Track A's person_fusion.py -> g1_person_track.json exists."""
    if not box or len(box) < 4 or not w or not h:
        return 0.0
    bw, bh = abs(box[2] - box[0]), abs(box[3] - box[1])
    area_frac = (bw * bh) / float(w * h)
    if area_frac <= 0:
        return 0.0
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    off = math.hypot(cx - w / 2.0, cy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
    return area_frac * (1.0 - cfg.person_nearby_center_penalty * min(1.0, off))


def person_nearby(frame, cfg=DEFAULT_CFG):
    """True iff SOME person in `frame` (a POSE_TRACKS-shaped dict/list, or None) looks
    "near enough" by the coarse box heuristic above. None/empty frame -> False
    (fail-closed: a person-gated command never fires just because we couldn't tell)."""
    if not frame:
        return False
    items, w, h = parse_pose_frame(frame)
    best = 0.0
    for it in items:
        best = max(best, _box_nearby_score(it.get("box"), w, h, cfg))
    return best >= cfg.person_nearby_min_frac


# --------------------------------------------------------------------- pure classifier
class VoiceCommandClassifier:
    """PURE decision logic: one recognized voice event + current context -> fire or not,
    and why. No shm, no robot, no threads -- deterministic and unit-testable straight
    from synthesized events (test_voice_command_bridge.py)."""

    def __init__(self, commands, cfg=None):
        self.commands = {c.name: c for c in commands}
        self.cfg = cfg or DEFAULT_CFG
        self._last_fired = {}   # cmd name -> t of last ACCEPTED firing (cooldown state)

    def reset(self):
        """Forget all cooldown state (call when voice_mode toggles OFF, mirroring
        GreetingService's reset-on-disable so a stale cooldown can't linger across a
        toggle cycle)."""
        self._last_fired.clear()

    def decide(self, event, now, person_nearby=False):
        """event: {"cmd", "confidence", "raw_text", "t"} (whatever voice_service.py just
        wrote). now: the caller's clock (tests pass a synthetic one for determinism).
        person_nearby: precomputed bool (see person_nearby() above) -- the classifier
        itself never touches shm.

        Returns a dict: {cmd, fire, reason, action, raw_text, confidence, t}. `action`
        is the resolved config/voice_commands.yaml action dict iff fire is True, else
        None. `reason` is always populated (dashboard/log feedback, fired or not)."""
        cmd_name = event.get("cmd")
        raw_text = event.get("raw_text", "")
        confidence = float(event.get("confidence") or 0.0)
        t = event.get("t")

        def result(fire, reason, action=None):
            return {"cmd": cmd_name, "fire": fire, "reason": reason, "action": action,
                    "raw_text": raw_text, "confidence": confidence, "t": t}

        # --- gate 1: staleness (by READ time, i.e. `now`, not the event's own clock) ---
        if t is None:
            return result(False, "missing_timestamp")
        age = now - t
        if age > self.cfg.max_event_age_s:
            return result(False, "stale")

        cdef = self.commands.get(cmd_name)
        if cdef is None:
            return result(False, "unknown_command")
        is_stop = cdef.action.get("kind") == "stop"

        # --- gate 2: confidence floor (config default, optional per-command override) ---
        floor = (cdef.confidence_override if cdef.confidence_override is not None
                 else self.cfg.default_confidence)
        if confidence < floor:
            return result(False, "low_confidence")

        # --- gate 3: person-nearby (skipped entirely for `stop`) ---
        if cdef.requires_person_nearby and not is_stop and not person_nearby:
            return result(False, "no_person_nearby")

        # --- gate 4: per-command cooldown (skipped entirely for `stop`) ---
        if not is_stop:
            last = self._last_fired.get(cmd_name, -1e9)
            if (now - last) < cdef.cooldown_s:
                return result(False, "cooldown")
            self._last_fired[cmd_name] = now   # accepted -> starts this command's window

        return result(True, "ok", action=cdef.action)


# =====================================================================================
# VoiceCommandBridge -- the SAFETY-GATED bridge to the robot (shm -> classifier -> callback)
# =====================================================================================
class VoiceCommandBridge:
    """Runs a VoiceCommandClassifier against /dev/shm/g1_voice_cmd.json in a daemon
    thread and dispatches the resolved action through caller-supplied callbacks -- but
    ONLY when BOTH gates pass, exactly like GreetingService:

        enabled_fn()  -- the "voice_mode" master toggle (default: OFF; caller wires the
                         dashboard toggle to flip this, mirroring greeting_mode).
        safe_fn()     -- reuse robot_web_controller.py's EXACT _greeting_safe predicate
                         (upright walk, not transitioning, arm not raised, no
                         pending_cmd, arm not busy) -- supplied by the wiring step.

    All robot coupling is a callback, so this class imports nothing from the controller
    and is exercised in tests with plain lambdas -- same contract as GreetingService.
    """

    def __init__(self, shm_path, commands=None, config_path=None, pose_tracks_path=None,
                 enabled_fn=None, safe_fn=None, dispatch_fn=None,
                 resume_fn=None, come_here_fn=None,
                 on_event=None, on_skip=None, cfg=None,
                 period_s=0.15, idle_period_s=0.3):
        self.shm_path = shm_path
        self.pose_tracks_path = pose_tracks_path

        if commands is None:
            loaded = load_voice_config(config_path)
            commands = loaded["commands"]
            if cfg is None:
                cfg = cfg_from_loaded(loaded)
        self.cfg = cfg or DEFAULT_CFG
        self.classifier = VoiceCommandClassifier(commands, self.cfg)

        self.enabled_fn = enabled_fn or (lambda: False)     # OFF unless wired
        self.safe_fn = safe_fn or (lambda: True)
        self.dispatch_fn = dispatch_fn or (lambda action: None)
        # Hook seams for Track A (patrol_supervisor.py). Harmless log-only stubs until
        # that workstream lands -- this file never needs another edit for that swap.
        self.resume_fn = resume_fn or (
            lambda: print("[VOICE] 'resume' hook not wired yet (Track A pending)", flush=True))
        self.come_here_fn = come_here_fn or (
            lambda: print("[VOICE] 'come_here' hook not wired yet (Track A pending)", flush=True))
        self.on_event = on_event    # every classified event, fired or not (dashboard feedback)
        self.on_skip = on_skip      # gated out (by decide() OR by safe_fn) -- logging hook

        self.period_s = period_s
        self.idle_period_s = idle_period_s
        self._stop = threading.Event()
        self._thread = None
        self._last_mtime = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="voice_bridge", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---- shm reads (mtime-gated freshness, matches detect_service.py's atomic-write
    # idiom on the WRITE side and GreetingService._read_json on the READ side) ----
    def _read_event(self):
        try:
            mt = os.path.getmtime(self.shm_path)
        except OSError:
            return None
        if mt == self._last_mtime:
            return None          # unchanged since last read -> a frozen/dead voice_service
        self._last_mtime = mt
        try:
            with open(self.shm_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _read_pose_frame(self):
        if not self.pose_tracks_path:
            return None
        try:
            with open(self.pose_tracks_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _dispatch(self, decision):
        """Route a FIRED decision to the right callback. `hook` actions never touch
        dispatch_fn -- they go straight to the explicit resume_fn/come_here_fn seam so a
        later workstream only ever has to pass in two callables, never edit this file."""
        action = decision["action"]
        kind = action.get("kind")
        if kind == "hook":
            name = action.get("name")
            if name == "resume":
                self.resume_fn()
            elif name == "come_here":
                self.come_here_fn()
            else:
                print(f"[VOICE] unknown hook action {name!r} -- ignored", flush=True)
        else:
            self.dispatch_fn(action)

    def _handle_event(self, decision):
        """Apply the safety gate to one classifier decision and dispatch (or skip)
        accordingly. Split out of the loop so it's unit-testable with plain lambdas, no
        thread or shm needed (see test_voice_command_bridge.py). on_event always fires
        (dashboard sees every recognized phrase, matched or not); the robot only acts
        when safe_fn() says it's safe RIGHT NOW. Returns True iff dispatched."""
        if self.on_event:
            self.on_event(decision)
        if not decision["fire"]:
            if self.on_skip:
                self.on_skip(decision)
            return False
        if not self.safe_fn():
            if self.on_skip:
                self.on_skip({**decision, "reason": "unsafe"})
            return False
        self._dispatch(decision)
        return True

    def poll_once(self, event, now, person_frame=None):
        """Run the classifier on ONE voice event and dispatch through the safety gate.
        Checks enabled_fn() itself (unlike GreetingService.poll_once, which relies on the
        loop for that) so tests can assert "disabled -> fires nothing" by calling this
        directly, with no thread involved."""
        if not self.enabled_fn():
            return False
        if person_frame is None:
            person_frame = self._read_pose_frame()
        nearby = person_nearby(person_frame, self.cfg)
        decision = self.classifier.decide(event, now, person_nearby=nearby)
        return self._handle_event(decision)

    def _loop(self):
        while not self._stop.is_set():
            if not self.enabled_fn():
                self.classifier.reset()     # clean cooldown slate while voice mode is OFF
                self._stop.wait(self.idle_period_s)
                continue
            event = self._read_event()
            if event is not None:
                self.poll_once(event, time.time())
            self._stop.wait(self.period_s)


# ------------------------------------------------------------------------- selftest
def selftest():
    """Fast smoke test of the PURE classifier (no shm/thread/robot) -- the full negative-
    case matrix lives in test_voice_command_bridge.py; this is just a quick sanity pass
    for `python3 scripts/voice_command_bridge.py --selftest`."""
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and cond

    commands = [
        CommandDef("stop", ["stop"], {"kind": "stop"}, requires_person_nearby=False, cooldown_s=0.0),
        CommandDef("wave", ["wave"], {"kind": "cmd", "name": "high_wave"},
                   requires_person_nearby=True, cooldown_s=4.0),
        CommandDef("dance", ["dance"], {"kind": "dance", "fsm_id": 503},
                   requires_person_nearby=False, cooldown_s=10.0),
        CommandDef("come_here", ["come here"], {"kind": "hook", "name": "come_here"},
                   requires_person_nearby=True, cooldown_s=5.0),
    ]
    cfg = VoiceBridgeConfig(max_event_age_s=1.5, default_confidence=0.65)

    def ev(cmd, conf=0.9, age=0.1, t0=1000.0):
        return {"cmd": cmd, "confidence": conf, "raw_text": cmd, "t": t0 - age}, t0

    clf = VoiceCommandClassifier(commands, cfg)
    e, now = ev("dance")
    d = clf.decide(e, now)
    c("fresh confident dance fires", d["fire"] and d["reason"] == "ok")

    e, now = ev("dance", age=3.0)
    c("stale event rejected", clf.decide(e, now)["reason"] == "stale")

    e, now = ev("dance", conf=0.2)
    c("low-confidence event rejected", clf.decide(e, now)["reason"] == "low_confidence")

    e, now = ev("wave")
    c("person-gated command rejected with no person",
      clf.decide(e, now, person_nearby=False)["reason"] == "no_person_nearby")
    c("person-gated command fires with a person",
      clf.decide(e, now, person_nearby=True)["fire"])

    clf2 = VoiceCommandClassifier(commands, cfg)
    e1, now1 = ev("dance", t0=1000.0)
    clf2.decide(e1, now1)
    e2, now2 = ev("dance", t0=1001.0)   # 1s later, still inside the 10s cooldown
    c("rapid re-fire blocked by cooldown", clf2.decide(e2, now2)["reason"] == "cooldown")

    clf3 = VoiceCommandClassifier(commands, cfg)
    e1, now1 = ev("stop", t0=1000.0)
    clf3.decide(e1, now1)
    e2, now2 = ev("stop", t0=1000.05)   # 50ms later -- stop has cooldown_s=0 and no gate
    c("stop bypasses cooldown", clf3.decide(e2, now2)["fire"])

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)


# =====================================================================================
# WIRING NOTES for robot_web_controller.py (NOT applied in this file -- see the task's
# instruction to keep that file's edits to a separate step). Mirrors the greeting_mode
# wiring block (search "Interactive \"wave back\" greeting demo" in that file) almost
# exactly:
#
# 1. Add `self.voice_mode = False` and `self.voice_status = ""` to ControlState
#    (next to greeting_mode/greeting_status), and add both to make_state_msg()'s dict
#    so the dashboard toggle/status line can reflect them (mirrors greeting_mode).
#
# 2. In lifespan(), after `greeting = GreetingService(...); greeting.start()`:
#
#        from voice_command_bridge import VoiceCommandBridge
#
#        def _voice_enabled():
#            return state.voice_mode
#
#        def _voice_dispatch(action):
#            kind = action.get("kind")
#            if kind == "stop":
#                # Exactly the WS mtype=="stop" fast path (robot_web_controller.py
#                # ~line 2371-2375) -- bypass the command queue entirely.
#                state.vx = state.vy = state.vyaw = 0.0
#                state.is_moving = False
#            elif kind == "cmd":
#                # Exactly the WS mtype=="cmd" handler (~line 2398-2399).
#                state.pending_cmd = action.get("name", "")
#            elif kind == "dance":
#                # Exactly the WS mtype=="dance" handler's lookup+fire (~line 2401-2432),
#                # reusing fire_whole_body/DANCES/fire_child_combo unchanged. Look up the
#                # matching DANCES entry by fsm_id and fire it the SAME way that handler
#                # does (including its parent_fsm/combo_keys branch), off-thread.
#                fsm = action.get("fsm_id")
#                entry = next((d for d in DANCES if d["fsm_id"] == fsm), None)
#                if entry is not None:
#                    threading.Thread(target=fire_whole_body,
#                                      args=(entry["name"], entry["fsm_id"]),
#                                      kwargs={"require_upright": True}, daemon=True).start()
#
#        voice = VoiceCommandBridge(
#            shm_path="/dev/shm/g1_voice_cmd.json",
#            pose_tracks_path=POSE_TRACKS,
#            enabled_fn=_voice_enabled, safe_fn=_greeting_safe,   # SAME predicate, reused
#            dispatch_fn=_voice_dispatch,
#            # resume_fn / come_here_fn: leave as the default log-only stubs until
#            # g1_nav/g1_nav/nodes/patrol_supervisor.py exists (Track A); swap them for
#            # real callables then -- no other change needed here.
#            on_event=..., on_skip=...)   # mirror _greeting_on_event/_greeting_on_skip's
#                                          # shape: update state.voice_status, log, and
#                                          # (optionally) append a dashboard feedback event.
#        voice.start()
#
# 3. In the WS dispatch (near `elif mtype == "greeting":`, ~line 2509), add:
#
#        elif mtype == "voice":
#            on = bool(msg.get("on"))
#            state.voice_mode = on
#            if not on:
#                state.voice_status = ""
#                if voice is not None:
#                    voice.classifier.reset()
#            await broadcast(make_state_msg())
#
# 4. Declare `voice: VoiceCommandBridge = None` alongside the existing
#    `greeting: GreetingService = None` module-level global (~line 624), and add
#    `global ... voice` to lifespan()'s existing `global` statement (~line 1656).
#
# 5. On shutdown (where `if greeting is not None: greeting.stop()` already is, ~line
#    1906), add the matching `if voice is not None: voice.stop()`.
#
# Dashboard-side (web/index.html + web/controller.js): a "🎤 Voice" toggle next to the
# existing "🤖 Greet" button, sending {type:"voice", on:<bool>} and rendering
# voice_mode/voice_status from fsm_state -- out of scope here per the task (dashboard
# wiring is a separate step), but this is the exact shape make_state_msg() should grow.
# =====================================================================================
