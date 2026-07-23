# Voice command bench spike (Track C0)

Status: **software-only research, NOT a live audio test.** No USB microphone is attached to
this Jetson yet (confirmed via `lsusb`/`arecord -l` in prior research on this box) -- ALSA
userland (`arecord`/`aplay`) is already installed, so this is the one physical blocker on
this track. Everything below was checked by actually installing/importing packages and
querying PyPI on 2026-07-23; nothing here is assumed.

## (a) `onnxruntime` wheel availability for this JetPack build

Platform: Jetson Orin NX, JetPack R35.3.1, Ubuntu 20.04.6 (`glibc 2.31`), system
`python3.8.10`, `pip 20.0.2`, `aarch64`.

Checked the real PyPI JSON index (`https://pypi.org/pypi/onnxruntime/json`) for
`aarch64`+`cp38` wheels, then **actually downloaded and imported** rather than trusting the
listing:

- Wheels exist for `cp38`+`aarch64` from onnxruntime `1.7.0` all the way to the current
  `1.19.2` (the `manylinux_2_27`/`manylinux_2_28` tagged builds from `1.17.0` onward).
- BUT this system's `pip 20.0.2` does not understand the newer `manylinux_2_27`/`_2_28`
  platform-compatibility tags (it's a 2020-era pip), so `pip3 download onnxruntime==1.19.2`
  fails to resolve anything past **`1.16.3`** (last release still tagged plain
  `manylinux_2_17`/`manylinux2014`, which old pip's tag list does recognize). This is a pip
  bug/age issue, not a hardware/glibc one -- glibc 2.31 on this box is new enough for
  2_27/2_28 wheels; a newer pip (or `pip install --upgrade pip` into a venv first) would
  unlock up to 1.19.2. For this pass I verified against the version old-pip actually
  resolves.
- Actually installed (`pip install --target <scratch dir> openwakeword`, which pulls
  `onnxruntime<2,>=1.10.0`) and it resolved + installed **`onnxruntime==1.19.2`** fine when
  installed as a *dependency* (pip's dependency resolver for openwakeword pulled a fresh
  enough tag list than the bare `download` command above -- inconsistent old-pip behavior,
  but the practical upshot is the same: it installs and imports).
- Confirmed by import: `onnxruntime 1.19.2`, providers
  `['AzureExecutionProvider', 'CPUExecutionProvider']` (CPU-only, as expected -- no CUDA
  execution provider wheel for this combo, which is fine, keyword-spotting models are tiny).

**Conclusion: yes, `onnxruntime` is installable on this JetPack/py3.8 combination.** No
blocker there.

### The real dependency risk: numpy

Installing `openwakeword` (which needs `onnxruntime`, `scipy`, `scikit-learn`,
`tflite-runtime`) pulled in **`numpy 1.24.4`**. The *system* `python3` already has
**`numpy 1.17.4`** pinned, and other scripts in this repo (`fused_odometry.py` per prior
memory) depend on that exact ancient version. Installing openWakeWord (or anything in its
dependency tree) into system site-packages **would silently upgrade numpy repo-wide and
risk breaking those scripts** -- this is the same trap `sim_runner.py`/mujoco's B0 stage
flagged and solved with a dedicated `sim/.venv`. **Recommendation: install the voice stack
into its own isolated environment** (a `perception/voice/.venv`, or a container following the
`perception/hands/Dockerfile` precedent), never into system site-packages. I did not build
that venv in this pass (no mic yet to justify running the real pipeline end-to-end) -- noted
as a follow-up.

Verified versions that actually installed together cleanly (aarch64, cp38, into a throwaway
`--target` dir, nothing touched system site-packages):

```
onnxruntime     1.19.2
tflite-runtime  2.13.0
scipy           1.10.1
scikit-learn    1.3.2
numpy           1.24.4   <- NOT the system 1.17.4; isolate this
openwakeword    0.6.0
```

## (b) openWakeWord vs whisper.cpp+fuzzy-match vs Porcupine

### openWakeWord (checked live)
- Installs cleanly (`pip install openwakeword` into an isolated target, see above) --
  `import openwakeword` works, `onnxruntime` CPU inference confirmed.
- **Gap found that the plan didn't anticipate:** `openwakeword.MODELS` ships exactly six
  pretrained wake words: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`,
  `weather`. There is **no ready-made model for an arbitrary custom phrase** -- getting a
  distinctive, not-a-dictionary-word wake phrase (what this plan explicitly wants) means
  *training* a custom openWakeWord model offline (their pipeline: synthesize samples with
  `piper-sample-generator` TTS, augment with noise/room-impulse-response, train a small
  classifier head) -- a real, non-trivial offline step that needs neither a mic nor the
  robot, but does need time/compute I did not spend in this pass, and can't be
  false-accept-validated without real room audio anyway (the mic gap applies here too).
  Structurally this is future work, not a blocker for *building* the pipeline now.
- CPU cost is small (it's a tiny keyword classifier over mel features, designed to run
  continuously on a Raspberry Pi) -- important since the Jetson's GPU is already shared
  with the camera/perception containers (see prior "INFER_HZ 3 -- GPU shared w/ camera"
  finding) and this must not add GPU contention. CPU-only keyword spotting is the right
  shape for an always-on listener.

### whisper.cpp + fuzzy match (checked live)
- The Python binding `pywhispercpp` has **no `cp38` wheel at all** on PyPI (checked the full
  release list: wheels start at `cp39` and go up to `cp314`; only a source `.tar.gz` exists
  for older interpreters). Building it from source against system python3.8 would need a
  C++ toolchain + cmake and is extra friction the "zero Python-version coupling" pitch in
  the plan was trying to avoid.
- BUT whisper.cpp's actual value proposition doesn't need the Python binding at all: the
  project's C++ CLI binary (`whisper-cli`/`main`) is a **standalone executable** with no
  Python dependency whatsoever -- you `git clone` + `make` it once (aarch64 build is
  well-trodden on Jetson/RPi) and then any process just `subprocess.Popen`s it with a WAV
  buffer and reads the transcript off stdout. That *is* the "zero Python-version coupling"
  promise, just via a different door than the pip package. I did not build the binary in
  this pass (would need to verify a working aarch64 build, which is a real chunk of work
  better spent once a mic exists to test the actual transcript quality) -- flagged as
  unverified/follow-up.
- Fuzzy phrase matching itself needs **no new dependency at all** -- Python's stdlib
  `difflib.SequenceMatcher` is sufficient for matching a short transcript against a small,
  fixed phrase list (7 commands, a handful of phrasings each). I used exactly this in
  `voice_command_bridge.py`/`voice_service.py` so the matching logic is testable today with
  zero installs. (`rapidfuzz` was checked too -- aarch64/cp38 wheels exist from `3.9.6`
  onward -- and would be a fine drop-in upgrade later if `difflib` proves too loose, but
  isn't needed to ship this pass.)

### Porcupine (not installed/tested; API-key fallback only)
Needs a free Picovoice account + API key, which is an account-creation step outside this
sandbox and not worth doing preemptively. Its wake-word accuracy is generally excellent and
it *does* support custom wake words trained via their web console (no local GPU training
needed, unlike openWakeWord) -- genuinely the easiest path to a real custom wake phrase, at
the cost of an external account dependency and periodic key/model expiry on the free tier.
Keeping this as the stated fallback per the plan; not pursued further this pass.

### Recommendation

**Primary: a hybrid, not a single pick** -- because the plan's two options solve two
different halves of the problem:

- **openWakeWord** for the always-on **wake-word spotter** (cheap, CPU-only, no GPU
  contention with the camera/perception pipeline; this is the "preferred default" the plan
  already named, and it's the one piece that must run continuously so it needs to be
  cheap).
- **whisper.cpp's compiled CLI binary** (not the Python binding -- see above) invoked
  **only for the ~2-3 s window right after a wake-word hit**, to transcribe the following
  command phrase and fuzzy-match it (stdlib `difflib`) against
  `config/voice_commands.yaml`. This is what actually satisfies "better for multi-word
  phrases like 'come here'" -- openWakeWord alone can only ever recognize the *one* phrase
  it was trained on, so it cannot by itself distinguish "stop" from "come here" from
  "dance". Something has to transcribe the command; whisper.cpp is the safest dependency
  for that per the checks above.

Until the custom wake-word model is trained (a follow-up needing GPU time + a mic to
validate false-accept/reject, neither available this pass) and until whisper.cpp's binary is
actually built and bench-tested, `voice_service.py` (deliverable 5) implements this
architecture structurally with the acoustic model loads wrapped in try/except that fail
loudly-but-gracefully, and ships a **pure text-domain matcher** (wake-phrase fuzzy match +
command fuzzy match, stdlib only) that IS fully exercisable right now via `--selftest`/
`--dry-run`, since that's the one part of this whole pipeline that doesn't need a
microphone, a trained model, or a compiled binary to prove correct.

## (c) Suggested wake phrase

**"Zephro"** (`ZEF-roh`, 2 syllables). Reasoning:
- Not an English dictionary word (close to "zephyr" but not it -- low collision risk with
  normal speech/TV background audio, which is exactly the false-accept risk the plan calls
  out).
- Clean, distinct phonemes (no overlap with the 7 command words: stop/resume/come/here/
  dance/wave/sit/stand) so a fuzzy match against the wake phrase is unlikely to be
  triggered by a command phrase alone, or vice versa.
- Short enough (2 syllables) to say naturally as a wake word (compare "hey jarvis" /
  "alexa" -- similar length), but still needs the full false-accept-against-real-background-
  noise test once a mic exists, per the plan's own C0 exit criteria -- **this suggestion is
  unvalidated against real audio and must be bench-tested before being trusted.**
- Backup candidate if "Zephro" turns out to collide with something in testing:
  **"Talvora"** (`tal-VOR-ah`, 3 syllables, same reasoning).

Both are configured as placeholders in `config/voice_commands.yaml`'s `wake_word` block (see
that file) -- swapping the phrase later is a config change, not a code change.

## What's verified vs. not

| Item | Status |
|---|---|
| `onnxruntime` aarch64+py3.8 wheel exists and imports | **Verified** (installed, imported, ran) |
| `openwakeword` installs cleanly on this box | **Verified** (installed into isolated target dir) |
| numpy version conflict risk with system scripts | **Verified as a real risk** (pulls numpy 1.24.4 vs system's pinned 1.17.4) |
| openWakeWord's 6 pretrained wake words (no custom phrase out of the box) | **Verified** (enumerated `openwakeword.MODELS`) |
| `pywhispercpp` has no cp38 wheel | **Verified** (checked full PyPI release list) |
| whisper.cpp CLI binary builds on this Jetson | **NOT verified** (did not attempt the build this pass) |
| Real-room false-accept/false-reject rate for any wake word | **NOT verified** (no mic attached -- the plan's own C0 exit criteria explicitly requires real audio, which is unavailable) |
| Phrase-matching logic (stdlib `difflib`, wake + command) | **Verified** (see `scripts/voice_command_bridge.py`'s and `perception/voice/voice_service.py`'s `--selftest`/`--dry-run` runs, both pass) |
