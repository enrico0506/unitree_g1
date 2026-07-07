"""pytest for the Phase-1 clip recorder — runs under the workstation base python.

No robot/camera here, so the "camera" is SYNTHETIC: a helper daemon thread
overwrites a temp JPEG with distinct PIL images at a known rate. The recorder
samples it exactly as it would sample ``/dev/shm/g1_camera.jpg`` on the Orin, so
these tests exercise the real sampling loop, real ffmpeg encode, and the real
clip.json — end to end, no mocks.

Requires: PIL + pytest (base env has both). ffmpeg-dependent assertions are
gated on ``shutil.which('ffmpeg')`` so the suite SKIPS (never fails) on a box
without ffmpeg. NO module-level ``sys.exit`` — proper ``def test_*`` functions.

Run: ``pytest motion/tests/test_recorder.py`` from the repo root.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

# Put the repo root (parent of motion/) on sys.path so ``motion.app.recorder``
# resolves regardless of where pytest is invoked from. motion/ has no
# __init__.py (implicit namespace package); motion/app does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motion.app.recorder import (  # noqa: E402  (after the sys.path insert)
    Recorder,
    RecorderError,
    encode_mp4,
    jpeg_size,
)

# PIL is required to synthesise the frames; skip the whole module if absent.
Image = pytest.importorskip("PIL.Image", reason="PIL needed to synthesise frames")

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None
_HAVE_FFPROBE = shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# Synthetic frame source: a thread that overwrites one JPEG at a fixed rate
# ---------------------------------------------------------------------------
class _SyntheticCamera:
    """Overwrite ``path`` with a DISTINCT jpeg at ``hz`` for ``duration_s``.

    Writes atomically (tmp + os.replace) so the recorder never reads a
    half-written file — the same guarantee the real ``camera_service.py`` gives.
    Each frame is a solid colour that changes per tick, so consecutive frames
    differ in both mtime and bytes (the recorder counts them as new).
    """

    def __init__(self, path: Path, hz: float, duration_s: float,
                 size: tuple[int, int] = (320, 240)) -> None:
        self.path = Path(path)
        self.hz = hz
        self.duration_s = duration_s
        self.size = size
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_SyntheticCamera":
        self._thread.start()
        return self

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        interval = 1.0 / self.hz
        deadline = time.time() + self.duration_s
        i = 0
        tmp = Path(str(self.path) + ".tmp")
        while time.time() < deadline:
            # distinct content each tick -> a genuinely new frame every time
            colour = (i * 7 % 256, (i * 13) % 256, (i * 29) % 256)
            Image.new("RGB", self.size, colour).save(tmp, "JPEG")
            os.replace(tmp, self.path)   # atomic publish
            i += 1
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_records_clip_and_measures_fps(tmp_path: Path) -> None:
    """A ~20 Hz synthetic source for ~2 s -> clip.mp4 + clip.json exist and the
    MEASURED fps lands within ~1 (tolerance 1.5) of the synthetic rate."""
    synthetic_hz = 20.0
    src = tmp_path / "g1_camera.jpg"
    out_dir = tmp_path / "job"

    cam = _SyntheticCamera(src, hz=synthetic_hz, duration_s=2.2).start()
    # let the first frame land so the recorder doesn't open before it exists
    time.sleep(0.15)

    rec = Recorder(src, out_dir, record_hz=30.0, max_seconds=5.0)
    rec.start()
    assert rec.is_recording is True
    time.sleep(2.0)
    res = rec.stop()
    cam.join()

    assert rec.is_recording is False

    # artifacts produced and non-empty
    assert res.clip_path.exists() and res.clip_path.stat().st_size > 0
    assert res.clip_path == out_dir / "clip.mp4"
    assert (out_dir / "clip.json").exists()
    assert res.clip_json_path == out_dir / "clip.json"

    # measured fps ~ synthetic rate ("within ~1")
    assert abs(res.measured_fps - synthetic_hz) <= 1.5, (
        f"measured {res.measured_fps} vs synthetic {synthetic_hz}"
    )

    # sane counters + dims
    assert res.frame_count >= 1
    assert res.dup_samples >= 0
    assert res.missing_samples >= 0
    assert res.width == 320 and res.height == 240

    # clip.json content mirrors the ClipResult
    import json

    doc = json.loads((out_dir / "clip.json").read_text())
    assert doc["job_id"] == out_dir.name
    assert doc["frame_source"] == str(src)
    assert doc["requested_hz"] == 30.0
    assert doc["frame_count"] == res.frame_count
    assert doc["clip_path"] == "clip.mp4"
    assert abs(doc["measured_fps"] - synthetic_hz) <= 1.5


def test_no_frames_raises_cleanly(tmp_path: Path) -> None:
    """A nonexistent source never crashes the loop; stop() raises RecorderError
    and every tick was booked as a missing sample."""
    out_dir = tmp_path / "job"
    rec = Recorder("/nonexistent/g1_camera.jpg", out_dir,
                   record_hz=30.0, max_seconds=5.0)
    rec.start()
    time.sleep(0.4)
    with pytest.raises(RecorderError):
        rec.stop()
    assert rec.is_recording is False
    assert rec._missing_samples > 0  # noqa: SLF001 (asserting the internal count)
    # no clip artifacts on a failed take
    assert not (out_dir / "clip.mp4").exists()


def test_stop_without_start_raises(tmp_path: Path) -> None:
    """Calling stop() before start() is a clean RecorderError, not an
    AttributeError on the (never-spawned) thread."""
    rec = Recorder("/nonexistent/g1_camera.jpg", tmp_path / "job")
    with pytest.raises(RecorderError):
        rec.stop()


def test_double_start_raises(tmp_path: Path) -> None:
    """A second start() while recording is rejected."""
    rec = Recorder("/nonexistent/g1_camera.jpg", tmp_path / "job",
                   record_hz=30.0, max_seconds=5.0)
    rec.start()
    try:
        with pytest.raises(RecorderError):
            rec.start()
    finally:
        # tear down the sampling thread (no frames -> stop() would raise)
        rec._stop_evt.set()  # noqa: SLF001
        if rec._thread is not None:  # noqa: SLF001
            rec._thread.join()  # noqa: SLF001


def test_jpeg_size_stdlib_and_pil(tmp_path: Path) -> None:
    """jpeg_size returns the true dimensions (via PIL, and its stdlib fallback
    must agree)."""
    p = tmp_path / "probe.jpg"
    Image.new("RGB", (176, 144), (10, 20, 30)).save(p, "JPEG")

    w, h = jpeg_size(p)
    assert (w, h) == (176, 144)

    # force the pure-stdlib SOF parser by hiding PIL from the import machinery
    import builtins

    real_import = builtins.__import__

    def _no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("PIL hidden for test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_pil
    try:
        assert jpeg_size(p) == (176, 144)
    finally:
        builtins.__import__ = real_import


@pytest.mark.skipif(not (_HAVE_FFMPEG and _HAVE_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_encode_mp4_ffprobe_fps(tmp_path: Path) -> None:
    """encode_mp4 on a fixed 0-indexed sequence yields the exact framerate that
    ffprobe reports back (avg_frame_rate == '20/1')."""
    import subprocess

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(20):
        Image.new("RGB", (320, 240), (i * 10 % 256, 0, 0)).save(
            frames_dir / ("frame_%06d.jpg" % i), "JPEG"
        )
    out = tmp_path / "clip.mp4"
    encode_mp4(frames_dir, out, fps=20)

    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True,
    )
    lines = [ln.strip() for ln in probe.stdout.splitlines() if ln.strip()]
    assert "20/1" in lines
    assert "20" in lines  # nb_frames


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_dup_frames_counted(tmp_path: Path) -> None:
    """A source that stops changing is counted as dup ticks (the camera dropped
    frames), not as new frames."""
    src = tmp_path / "g1_camera.jpg"
    out_dir = tmp_path / "job"
    # publish exactly one frame, then leave it frozen
    Image.new("RGB", (320, 240), (1, 2, 3)).save(src, "JPEG")

    rec = Recorder(src, out_dir, record_hz=30.0, max_seconds=5.0)
    rec.start()
    time.sleep(0.6)
    res = rec.stop()

    # exactly one unique frame captured; the rest of the ticks were dups
    assert res.frame_count == 1
    assert res.dup_samples > 0
    assert res.clip_path.exists()
