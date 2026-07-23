#!/usr/bin/env python3
"""Standalone voice-command listener (Track C, C1 + the C0 wake-phrase bench spike).

Mirrors the shape of perception/detect/detect_service.py and perception/pose/
pose_service.py: its own process, no ROS/robot coupling, reads a live input (here: the
microphone, via ALSA's `arecord` -- NOT camera_service.py's JPEG shm, this is the one
perception service that doesn't touch the camera at all) and atomically writes small
JSON geometry/events to /dev/shm for scripts/voice_command_bridge.py to consume.

UNLIKE detect_service.py, this file is deliberately NOT demand-gated -- there is no
"someone is watching" concept for audio (the task's own framing: listening should just
run whenever enabled). It writes two files:

    /dev/shm/g1_voice_cmd.json    {"cmd", "confidence", "raw_text", "t"} -- ONLY after a
                                  wake-word hit AND a phrase match against
                                  config/voice_commands.yaml. NEVER a raw always-on
                                  transcript (privacy/safety requirement) -- if nothing
                                  matched, NOTHING is written, ever.
    /dev/shm/g1_voice_state.json {"listening": bool, "last_wake_t": float|None} --
                                  low-rate heartbeat for a dashboard "listening..." dot.

PIPELINE (see perception/voice/RESEARCH.md for the full bench-spike writeup + what's
actually verified vs. not):
    arecord (subprocess, raw PCM -- no pyaudio/sounddevice/PortAudio dependency at all,
    since ALSA userland is already installed and this sidesteps a real dependency-risk
    library on this old py3.8 Jetson image)
        -> openWakeWord (lazy-imported; cheap CPU keyword spotting, always-on)
        -> on a wake-word hit, ~2.5s of audio -> a WAV file -> whisper.cpp's COMPILED CLI
           BINARY (subprocess, NOT the pywhispercpp Python binding -- that has no cp38
           wheel; the binary needs no Python binding at all)
        -> the transcript is fuzzy-matched (stdlib difflib, zero extra dependency) against
           config/voice_commands.yaml's phrases
        -> ONLY on a match, write g1_voice_cmd.json.

Both the openWakeWord model and the whisper.cpp binary are OPTIONAL at runtime: if either
is missing, this process logs a clear reason and disables just that stage rather than
crashing -- see WakeWordDetector/WhisperCppTranscriber below. With NO microphone attached
(the current state of this Jetson), find_capture_device() returns None and the live loop
retries periodically while reporting {"listening": false, "error": ...} on the heartbeat,
rather than crashing.

The ONE part of this whole pipeline that needs neither a mic, a trained wake-word model,
nor a compiled binary is the matching logic itself (normalize/fuzzy_score/match_wake_word/
match_command) -- exercise it right now with:

    python3 perception/voice/voice_service.py --selftest
    python3 perception/voice/voice_service.py --dry-run "hey zephro dance for me"
    echo "zephro stop" | python3 perception/voice/voice_service.py --dry-run
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOICE_COMMANDS_PATH = Path(
    os.environ.get("VOICE_COMMANDS_PATH", str(REPO_ROOT / "config" / "voice_commands.yaml")))

# --- Paths (shared memory; matches the /dev/shm/g1_* naming convention used by every
# other perception service in this repo) ---
VOICE_CMD_PATH = os.environ.get("VOICE_CMD_PATH", "/dev/shm/g1_voice_cmd.json")
VOICE_STATE_PATH = os.environ.get("VOICE_STATE_PATH", "/dev/shm/g1_voice_state.json")

# --- Tunables (env-overridable, mirrors detect_service.py's convention) ---
WAKE_MODEL_NAME = os.environ.get("VOICE_WAKE_MODEL", "hey_jarvis")  # see RESEARCH.md (b):
    # no ready-made openWakeWord model exists for a custom "zephro"-style phrase yet
    # (that needs offline training); hey_jarvis is the practical stand-in to bench the
    # PIPELINE MECHANICS today. The TEXT-DOMAIN wake phrase (what --dry-run/--selftest
    # actually match against) is config/voice_commands.yaml's wake_word.phrase ("zephro"),
    # independent of which acoustic model is loaded.
COMMAND_WINDOW_S = float(os.environ.get("VOICE_COMMAND_WINDOW_S", "2.5"))  # audio captured
    # after a wake-word hit, handed to STT for the command phrase
SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "16000"))
STATE_PERIOD_S = 1.0    # heartbeat write rate (low-rate, per the task's "listening" dot)


def atomic_write(path, data, mode="wb"):
    """Write then os.replace -- a reader never sees a partial file. Matches
    detect_service.py/pose_service.py/hands_service.py's identical helper exactly."""
    tmp = str(path) + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, str(path))


def write_state(listening, last_wake_t, error=None):
    payload = {"listening": bool(listening), "last_wake_t": last_wake_t}
    if error:
        payload["error"] = error   # extra field; harmless to consumers that ignore it
    atomic_write(VOICE_STATE_PATH, json.dumps(payload).encode(), "wb")


def write_voice_cmd_event(cmd, confidence, raw_text, t=None):
    """The ONLY place a recognized command is written to shm -- called ONLY after both
    a wake-word hit and a phrase match. Never called with a raw always-on transcript."""
    payload = {"cmd": cmd, "confidence": round(float(confidence), 3),
               "raw_text": raw_text, "t": t if t is not None else time.time()}
    atomic_write(VOICE_CMD_PATH, json.dumps(payload).encode(), "wb")
    print(f"[voice_service] WROTE event: {payload}", flush=True)


# --------------------------------------------------------------------- config loading
def load_local_voice_config(path=None):
    """Self-contained YAML load -- deliberately does NOT import
    scripts/voice_command_bridge.py. This process is meant to run standalone (its own
    venv/container per RESEARCH.md's isolation recommendation, exactly like
    perception/hands/'s Dockerfile precedent), so it must not assume it shares a Python
    environment or sys.path with robot_web_controller.py's process. Only the phrase
    lists + wake-word block are needed here; the full safety-gating CommandDef machinery
    lives entirely on the other (bridge) side of the shm boundary."""
    path = path or VOICE_COMMANDS_PATH
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    wake_cfg = data.get("wake_word") or {}
    commands = []   # [(name, [phrases])]
    for c in (data.get("commands") or []):
        name = str(c.get("name") or "").strip()
        phrases = [str(p).strip() for p in (c.get("phrases") or []) if str(p).strip()]
        if name and phrases:
            commands.append((name, phrases))
    return {
        "wake_word": wake_cfg,
        "default_confidence": float(data.get("default_confidence", 0.65)),
        "commands": commands,
    }


# --------------------------------------------------------------------- pure text matching
# Everything below is PURE (no I/O) and stdlib-only (difflib) -- exercisable right now,
# with no mic/model/binary, via --selftest / --dry-run.

def normalize(text):
    return " ".join((text or "").lower().strip().split())


def fuzzy_score(a, b):
    """difflib ratio in [0, 1] -- adequate for a tiny, fixed phrase list (7 commands, a
    handful of phrasings each). See RESEARCH.md (b): rapidfuzz is a fine drop-in upgrade
    later (aarch64/cp38 wheels exist) if difflib's looseness proves an issue in real
    testing; not needed to ship this pass, and keeping this stdlib-only means the
    matching logic needs ZERO installs to bench-test today."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_wake_word(text, wake_cfg):
    """Does `text` contain the configured wake phrase (or its backup)? Slides a same-
    length word-window across `text` so a wake phrase embedded in a longer utterance
    ("hey zephro, come here") still matches, rather than requiring an exact whole-string
    match. Returns (heard: bool, score: float, which: "phrase"|"backup_phrase"|None)."""
    text_n = normalize(text)
    words = text_n.split()
    threshold = float(wake_cfg.get("fuzzy_threshold", 0.72))
    best_heard, best_score, best_which = False, 0.0, None
    for key in ("phrase", "backup_phrase"):
        phrase = normalize(str(wake_cfg.get(key) or ""))
        if not phrase:
            continue
        n = max(1, len(phrase.split()))
        if not words:
            score = fuzzy_score(text_n, phrase)
        else:
            score = max((fuzzy_score(" ".join(words[i:i + n]), phrase)
                         for i in range(max(1, len(words) - n + 1))), default=0.0)
        if score > best_score:
            best_score, best_which = score, key
        if score >= threshold:
            best_heard = True
    return best_heard, best_score, (best_which if best_heard else None)


def match_command(text, commands):
    """Best-matching command phrase in `text` -> (name, confidence, matched_phrase) or
    (None, 0.0, None) if nothing scores above 0. `commands`: [(name, [phrases])].
    A phrase that appears verbatim as a substring scores a high floor (0.9) even if the
    surrounding sentence hurts the whole-string ratio -- so "hey zephro can you dance
    please" still matches "dance" strongly."""
    text_n = normalize(text)
    best_name, best_score, best_phrase = None, 0.0, None
    for name, phrases in commands:
        for phrase in phrases:
            phrase_n = normalize(phrase)
            score = fuzzy_score(text_n, phrase_n)
            if phrase_n and phrase_n in text_n:
                score = max(score, 0.9)
            if score > best_score:
                best_name, best_score, best_phrase = name, score, phrase
    return best_name, best_score, best_phrase


def run_match_pipeline(text, loaded):
    """The full text-domain pipeline (wake word -> command), pure and dependency-free.
    Returns a result dict; `would_fire` is what decides whether a real run would call
    write_voice_cmd_event(). Shared by --dry-run, --selftest, and (conceptually) the real
    audio path once STT hands it a transcript."""
    wake_cfg = loaded["wake_word"]
    heard, wake_score, which = match_wake_word(text, wake_cfg)
    result = {"input": text, "wake_word_heard": heard,
              "wake_score": round(wake_score, 3), "wake_match": which,
              "cmd": None, "cmd_score": 0.0, "matched_phrase": None, "would_fire": False}
    if not heard:
        return result   # never even look for a command without the wake word -- privacy gate
    name, score, phrase = match_command(text, loaded["commands"])
    result.update({"cmd": name, "cmd_score": round(score, 3), "matched_phrase": phrase})
    if name is not None and score >= loaded["default_confidence"]:
        result["would_fire"] = True
    return result


# --------------------------------------------------------------------- audio capture (ALSA)
def find_capture_device():
    """Return an ALSA device string to record from (e.g. "plughw:2,0"), or None if no
    usable microphone is attached. Prefers a device whose `arecord -l` line looks like a
    USB mic over the Jetson's onboard APE audio path (not a physical microphone input).
    VOICE_ALSA_DEVICE overrides this entirely (set it once a mic's card/device number is
    known, to skip the name-sniffing heuristic)."""
    override = os.environ.get("VOICE_ALSA_DEVICE")
    if override:
        return override
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None   # arecord itself missing/hung -- treat as "no device", not a crash
    for line in out.splitlines():
        m = re.match(r"card (\d+):\s*\S+\s*\[(.*?)\],\s*device (\d+):", line)
        if m and "usb" in line.lower():
            return f"plughw:{m.group(1)},{m.group(3)}"
    return None   # ALSA userland is present (that's not the gap) but nothing mic-shaped is


class ArecordCapture:
    """Wraps `arecord` as a subprocess emitting raw S16_LE mono PCM on stdout -- no
    pyaudio/sounddevice/PortAudio pip dependency at all, since ALSA userland
    (arecord/aplay) is already installed on this Jetson. This is the "minimal dependency
    risk" audio layer called for in the task."""

    def __init__(self, device, rate=SAMPLE_RATE):
        self.device = device
        self.rate = rate
        self._proc = None

    def start(self):
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
             "-c", "1", "-t", "raw", "-q", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def read_bytes(self, n):
        """Blocking read of exactly-ish n bytes (short reads possible near EOF/error)."""
        if self._proc is None or self._proc.stdout is None:
            return b""
        return self._proc.stdout.read(n)

    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None


def pcm_to_wav_bytes(pcm_bytes, rate=SAMPLE_RATE):
    """Wrap raw S16_LE mono PCM in a minimal WAV container (stdlib `wave`) so the
    whisper.cpp CLI binary (which wants a WAV file, not a raw stream) can read it."""
    buf = tempfile.SpooledTemporaryFile()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------- wake word (openWakeWord)
class WakeWordDetector:
    """Thin wrapper around openWakeWord, lazy-imported so this whole file stays
    importable (and --selftest/--dry-run stay runnable) even when openwakeword isn't
    installed in the current interpreter -- see RESEARCH.md: it should NOT be installed
    into system site-packages (numpy conflict), so this constructor is expected to fail
    on system python3 until a dedicated venv/container exists, and that failure must be
    a clear log line, not a crash."""

    def __init__(self, model_name=WAKE_MODEL_NAME):
        self.model = None
        self.model_name = model_name
        try:
            from openwakeword.model import Model   # deferred import, see docstring
            self.model = Model(wakeword_models=[model_name])
            print(f"[voice_service] openWakeWord loaded (model={model_name!r})", flush=True)
        except Exception as e:
            print(f"[voice_service] openWakeWord unavailable ({e!r}) -- acoustic "
                  f"wake-word detection disabled. This is expected on system python3 "
                  f"per RESEARCH.md (install into an isolated venv to enable it); "
                  f"--selftest/--dry-run still exercise the text-matching logic fully.",
                  flush=True)

    def available(self):
        return self.model is not None

    def feed(self, pcm_chunk):
        """pcm_chunk: 1-D int16 numpy array (openWakeWord's expected input). Returns True
        on a wake-word hit for this chunk (score > 0.5, openWakeWord's usual convention)."""
        if self.model is None:
            return False
        scores = self.model.predict(pcm_chunk)
        return any(v > 0.5 for v in scores.values())


# --------------------------------------------------------------------- STT (whisper.cpp CLI)
class WhisperCppTranscriber:
    """Shells out to the COMPILED whisper.cpp CLI binary -- deliberately NOT the
    `pywhispercpp` Python binding (no cp38 wheel exists for it, see RESEARCH.md). The
    binary itself has no Python-version coupling at all. Not built/verified in this pass
    (would need a real mic to validate transcript quality against); this class fails
    gracefully (available()==False) until WHISPER_CPP_BIN/WHISPER_CPP_MODEL point at a
    real build."""

    def __init__(self, binary_path=None, model_path=None):
        self.binary_path = binary_path or os.environ.get("WHISPER_CPP_BIN", "")
        self.model_path = model_path or os.environ.get("WHISPER_CPP_MODEL", "")
        self._available = bool(self.binary_path and os.path.exists(self.binary_path)
                                and self.model_path and os.path.exists(self.model_path))
        if not self._available:
            print(f"[voice_service] whisper.cpp binary/model not configured "
                  f"(WHISPER_CPP_BIN={self.binary_path!r} WHISPER_CPP_MODEL="
                  f"{self.model_path!r}) -- command transcription disabled; build "
                  f"whisper.cpp and set both env vars to enable it (see RESEARCH.md).",
                  flush=True)

    def available(self):
        return self._available

    def transcribe_wav_bytes(self, wav_bytes, timeout=6.0):
        if not self._available:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav_bytes)
            f.flush()
            try:
                out = subprocess.run(
                    [self.binary_path, "-m", self.model_path, "-f", f.name, "-nt", "-otxt",
                     "-of", f.name],
                    capture_output=True, text=True, timeout=timeout)
                txt_path = f.name + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path) as tf:
                        text = tf.read().strip()
                    os.remove(txt_path)
                    return text
                return out.stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"[voice_service] whisper.cpp transcription failed: {e}", flush=True)
                return ""


# --------------------------------------------------------------------- live loop (real audio)
def run_live(loaded, config_path=None):
    """The real capture -> wake-word -> STT -> match -> shm loop. Bench-untestable right
    now (no mic attached) -- structured so it fails LOUDLY-BUT-GRACEFULLY at each optional
    stage rather than crashing, per the task's explicit requirement."""
    print("[voice_service] starting live loop", flush=True)
    device = find_capture_device()
    if device is None:
        print("[voice_service] NO MICROPHONE DETECTED via `arecord -l` -- ALSA userland "
              "is present, so a plug-and-play USB mic should work the moment one is "
              "attached (or set VOICE_ALSA_DEVICE to force a card). Waiting, retrying "
              "every 5s, heartbeat reports listening=false in the meantime.", flush=True)
    while device is None:
        write_state(listening=False, last_wake_t=None, error="no capture device attached")
        time.sleep(5.0)
        device = find_capture_device()
    print(f"[voice_service] capture device: {device}", flush=True)

    ww = WakeWordDetector(WAKE_MODEL_NAME)
    stt = WhisperCppTranscriber()
    cap = ArecordCapture(device).start()

    last_wake_t = None
    last_state_write = 0.0
    chunk_bytes = SAMPLE_RATE * 2 // 5   # ~0.2s of S16_LE mono per read
    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        while True:
            if not cap.alive():
                print("[voice_service] arecord died mid-stream -- restarting capture",
                      flush=True)
                cap = ArecordCapture(device).start()
                time.sleep(0.5)

            raw = cap.read_bytes(chunk_bytes)
            now = time.time()
            if now - last_state_write >= STATE_PERIOD_S:
                write_state(listening=True, last_wake_t=last_wake_t)
                last_state_write = now

            if not raw:
                continue
            if not (ww.available() and np is not None):
                # No acoustic wake-word model loaded (expected on system python3 today,
                # see WakeWordDetector) -- nothing more this loop can do until one is
                # installed; keep draining audio so arecord's pipe never backs up.
                continue

            pcm = np.frombuffer(raw, dtype=np.int16)
            if not ww.feed(pcm):
                continue

            last_wake_t = now
            print("[voice_service] wake word hit -- capturing command window", flush=True)
            write_state(listening=True, last_wake_t=last_wake_t)
            cmd_bytes = cap.read_bytes(int(SAMPLE_RATE * 2 * COMMAND_WINDOW_S))
            if not stt.available():
                print("[voice_service] wake word heard but no STT backend configured -- "
                      "cannot resolve a command phrase; see WhisperCppTranscriber",
                      flush=True)
                continue
            wav_bytes = pcm_to_wav_bytes(cmd_bytes)
            text = stt.transcribe_wav_bytes(wav_bytes)
            if not text:
                continue
            name, score, phrase = match_command(text, loaded["commands"])
            if name is not None and score >= loaded["default_confidence"]:
                write_voice_cmd_event(name, score, text, t=now)
            else:
                print(f"[voice_service] wake word heard but no command phrase matched "
                      f"(heard: {text!r})", flush=True)
    finally:
        cap.stop()


# --------------------------------------------------------------------- dry-run / selftest
def run_dry_run(text, config_path=None, write_shm=False):
    loaded = load_local_voice_config(config_path)
    result = run_match_pipeline(text, loaded)
    print(json.dumps(result, indent=2))
    if write_shm and result["would_fire"]:
        write_voice_cmd_event(result["cmd"], result["cmd_score"], text)
    return result


def selftest(config_path=None):
    """Exercises the ENTIRE text-domain matching pipeline (wake word + command,
    positive AND near-miss cases) with zero audio/model/binary dependencies -- the one
    part of this whole file that's fully verifiable without a microphone. Run with
    `python3 perception/voice/voice_service.py --selftest`."""
    loaded = load_local_voice_config(config_path)
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and cond

    r = run_match_pipeline("zephro dance", loaded)
    c("wake word + 'dance' fires", r["would_fire"] and r["cmd"] == "dance")

    r = run_match_pipeline("hey zephro can you come here please", loaded)
    c("wake word embedded in a longer sentence + 'come here' fires",
      r["would_fire"] and r["cmd"] == "come_here")

    r = run_match_pipeline("zefro stop", loaded)   # deliberate misspelling/mis-hearing
    c("near-miss wake-phrase spelling ('zefro') still fuzzy-matches", r["wake_word_heard"])

    r = run_match_pipeline("dance", loaded)   # command with NO wake word at all
    c("command without the wake word never fires (privacy/safety gate)",
      not r["would_fire"] and not r["wake_word_heard"])

    r = run_match_pipeline("hey zephro please dance for me", loaded)
    c("wake word + phrase inside a full sentence fires", r["would_fire"])

    r = run_match_pipeline("just talking about the weather and some dancing", loaded)
    c("ordinary background speech containing a command word but no wake word never fires",
      not r["would_fire"])

    r = run_match_pipeline("zephro banana", loaded)   # wake word + gibberish command
    c("wake word + no matching command phrase does not fire", not r["would_fire"])

    r = run_match_pipeline("zephro stop", loaded)
    c("wake word + 'stop' fires", r["would_fire"] and r["cmd"] == "stop")

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                     help="Run the built-in text-domain matching smoke test (no audio).")
    ap.add_argument("--dry-run", nargs="?", const="", default=None, metavar="TEXT",
                     help="Match TEXT through the wake+command logic without touching "
                          "audio hardware (reads stdin if TEXT is omitted).")
    ap.add_argument("--write-shm", action="store_true",
                     help="With --dry-run, also write the resulting event to "
                          "/dev/shm/g1_voice_cmd.json if it would fire (lets you test "
                          "voice_command_bridge.py end-to-end without a mic).")
    ap.add_argument("--config", default=None, help="Override config/voice_commands.yaml path.")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(args.config))

    if args.dry_run is not None:
        text = args.dry_run if args.dry_run else sys.stdin.readline()
        run_dry_run(text.strip(), args.config, write_shm=args.write_shm)
        return

    loaded = load_local_voice_config(args.config)
    run_live(loaded, args.config)


if __name__ == "__main__":
    main()
