"""Clean-feed clip recorder for the Motion tab (Phase-1, no ML).

Samples the single overwritten camera file (``pipeline.yaml`` ->
``record.frame_source`` = ``/dev/shm/g1_camera.jpg``, the clean feed
``camera_service.py`` already writes) at ``record_hz`` and encodes the captured
JPEG sequence into ``<job>/clip.mp4``. The *true* frame rate is MEASURED over
wall-clock — the source (a shared-memory file) rewrites at the camera's own,
jittery cadence, so the requested ``record_hz`` is only an upper bound — and the
measured value plus dropped/dup accounting is stamped into ``<job>/clip.json``.

Off-robot (this workstation) ``/dev/shm/g1_camera.jpg`` is simply absent; the
recorder does NOT crash — every tick with a missing/unreadable source bumps
``missing_samples`` and writes no frame. Only ``stop()`` surfaces the failure,
raising :class:`RecorderError` when not a single frame was captured. That is
exactly what makes it testable with a synthetic source: point the recorder at a
temp JPEG a helper thread overwrites at a known rate and ``measured_fps`` lands
within ~1 of that rate (see ``motion/tests/test_recorder.py``).

Deliberately **stdlib + subprocess (ffmpeg) only** — NO ``cv2``, NO ``fastapi``
imports — so it (and its pytest) run under the workstation base python. ``cv2``
lives only in the separate ``g1`` conda env; ffmpeg (``/usr/bin/ffmpeg`` 6.1.1,
libx264) is what the encoder shells out to, yielding browser-playable H.264 for
the ``<video>`` result view. PIL is a soft dependency of :func:`jpeg_size` only,
with a pure-stdlib JPEG-header fallback.
"""

from __future__ import annotations

import os
import json
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


# printf-style pattern the frames are written with AND ffmpeg's image2 demuxer
# consumes. 0-indexed, ascending: frame_000000.jpg, frame_000001.jpg, ...
FRAME_GLOB = "frame_%06d.jpg"


class RecorderError(RuntimeError):
    """Raised on a recorder-level failure (already recording, no frames
    captured, ffmpeg/encode failure)."""


# ---------------------------------------------------------------------------
# Result of one recording (also the source of the clip.json fields)
# ---------------------------------------------------------------------------
@dataclass
class ClipResult:
    """Everything ``stop()`` learned about the take. Mirrors ``clip.json``."""

    clip_path: Path        # <job>/clip.mp4
    clip_json_path: Path   # <job>/clip.json
    measured_fps: float    # frames_written/(t_last-t_first); requested_hz if <2 frames
    frame_count: int       # == frames_written (unique frames encoded)
    duration_s: float      # t_last_write - t_first_write
    dup_samples: int       # ticks where source bytes+mtime unchanged (source dropped a frame)
    missing_samples: int   # ticks where frame_source absent/unreadable/empty
    width: int
    height: int
    started_at: float      # epoch seconds (start() wall-clock)
    stopped_at: float      # epoch seconds (sampling loop exit)


# ---------------------------------------------------------------------------
# Module helpers (unit-testable without the Recorder class)
# ---------------------------------------------------------------------------
def jpeg_size(jpeg_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of a JPEG.

    Uses PIL when importable (fast, robust); falls back to a tiny pure-stdlib
    SOF-marker scan so the recorder keeps working in a bare base python.
    Raises :class:`RecorderError` if the dimensions can't be determined.
    """
    jpeg_path = Path(jpeg_path)
    # -- preferred: PIL ---------------------------------------------------- #
    try:
        from PIL import Image  # soft dependency

        with Image.open(jpeg_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass  # fall through to the stdlib parser

    # -- fallback: walk the JPEG marker segments for a Start-Of-Frame ------- #
    try:
        with open(jpeg_path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise RecorderError(f"cannot read jpeg {jpeg_path}: {e}") from e

    # A JPEG is 0xFFD8 (SOI) then a chain of 0xFF<marker><2-byte length><...>.
    # Any SOFn marker (0xC0..0xCF, excluding the non-SOF 0xC4/0xC8/0xCC) carries
    # [precision(1)][height(2)][width(2)] right after the length field.
    i = 2  # skip SOI
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # standalone markers (no payload): padding 0xFF, SOI/EOI, RSTn
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            i += 2
            continue
        if i + 3 >= n:
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            # SOFn payload: precision, height, width
            height = struct.unpack(">H", data[i + 5:i + 7])[0]
            width = struct.unpack(">H", data[i + 7:i + 9])[0]
            return int(width), int(height)
        i += 2 + seg_len  # skip this segment's payload
    raise RecorderError(f"no SOF marker found in {jpeg_path}")


def encode_mp4(frames_dir: Path, out_path: Path, fps: float) -> None:
    """Encode a 0-indexed ``frame_%06d.jpg`` sequence into an H.264 mp4.

    Primary path shells out to ffmpeg (verified on this box: base env, no cv2).
    Fallback: if ffmpeg is absent but cv2 is importable, use ``cv2.VideoWriter``;
    otherwise raise :class:`RecorderError`.
    """
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-framerate", f"{fps}",
            "-i", str(frames_dir / FRAME_GLOB),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RecorderError(
                f"ffmpeg failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return

    # -- fallback: cv2 (only ever present in the separate g1 env) ---------- #
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RecorderError(
            "no ffmpeg on PATH and cv2 not importable — cannot encode mp4"
        ) from e

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise RecorderError(f"no frames to encode in {frames_dir}")
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )
    try:
        for fp in frames:
            writer.write(cv2.imread(str(fp)))
    finally:
        writer.release()
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RecorderError("cv2 VideoWriter produced no output")


# ---------------------------------------------------------------------------
# Recorder — one take, one sampling thread
# ---------------------------------------------------------------------------
class Recorder:
    """Sample ``frame_source`` into ``out_dir/frames/`` on a daemon thread, then
    (on :meth:`stop`) encode ``out_dir/clip.mp4`` + write ``out_dir/clip.json``.

    New-frame test each tick = ``os.stat().st_mtime_ns`` changed since the last
    written frame, with raw-bytes inequality as a confirmation fallback (guards
    coarse-mtime filesystems). Unchanged -> the source dropped a frame
    (``dup_samples``). Absent/unreadable/empty -> ``missing_samples``.
    """

    def __init__(self, frame_source: str | Path, out_dir: str | Path,
                 record_hz: float = 30.0, max_seconds: float = 30.0) -> None:
        self.frame_source = Path(frame_source)
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.record_hz = float(record_hz)
        self.max_seconds = float(max_seconds)

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._recording = False

        # accumulated by the sampling loop (single writer, so plain attrs are ok)
        self._frames_written = 0
        self._dup_samples = 0
        self._missing_samples = 0
        self._t_first: float | None = None   # wall-clock of first frame write
        self._t_last: float | None = None     # wall-clock of last frame write
        self._started_at: float = 0.0
        self._stopped_at: float = 0.0
        self._width = 0
        self._height = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        """Spawn the daemon sampling thread. Raise if already recording."""
        if self._recording:
            raise RecorderError("recorder is already recording")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # reset counters so a Recorder instance could be reused after stop()
        self._frames_written = 0
        self._dup_samples = 0
        self._missing_samples = 0
        self._t_first = None
        self._t_last = None
        self._width = 0
        self._height = 0

        self._stop_evt.clear()
        self._recording = True
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._run, name="motion-recorder", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        """Sampling loop: one read per tick until stop() or max_seconds."""
        interval = 1.0 / self.record_hz if self.record_hz > 0 else 0.0
        deadline = self._started_at + self.max_seconds
        last_mtime_ns: int | None = None
        last_bytes: bytes | None = None
        idx = 0

        while not self._stop_evt.is_set():
            tick_start = time.time()
            if tick_start >= deadline:
                break

            # -- sample the source once ----------------------------------- #
            data: bytes | None
            mtime_ns = None
            try:
                st = os.stat(self.frame_source)
                mtime_ns = st.st_mtime_ns
                with open(self.frame_source, "rb") as f:
                    data = f.read()
            except OSError:
                data = None

            if not data:
                # absent, unreadable, or a zero-length (mid-write) file
                self._missing_samples += 1
            else:
                # new frame if EITHER mtime or bytes changed; dup only when both
                # are unchanged (== the camera re-published nothing this tick)
                is_new = (mtime_ns != last_mtime_ns) or (data != last_bytes)
                if is_new:
                    frame_path = self.frames_dir / (FRAME_GLOB % idx)
                    with open(frame_path, "wb") as f:
                        f.write(data)
                    now = time.time()
                    if self._t_first is None:
                        self._t_first = now
                        try:
                            self._width, self._height = jpeg_size(frame_path)
                        except Exception:
                            self._width, self._height = 0, 0
                    self._t_last = now
                    self._frames_written += 1
                    idx += 1
                    last_mtime_ns = mtime_ns
                    last_bytes = data
                else:
                    self._dup_samples += 1

            # -- sleep the remainder of the tick (interruptible by stop()) - #
            remaining = interval - (time.time() - tick_start)
            if remaining > 0:
                self._stop_evt.wait(remaining)

        self._stopped_at = time.time()
        self._recording = False

    def stop(self) -> ClipResult:
        """Join the sampling thread, encode clip.mp4, write clip.json.

        Raise :class:`RecorderError` if it was never started, or
        ``'no frames captured from <source>'`` when ``frame_count == 0``.
        """
        if self._thread is None:
            raise RecorderError("recorder was never started")
        self._stop_evt.set()
        self._thread.join()
        self._recording = False

        frame_count = self._frames_written
        if frame_count == 0:
            raise RecorderError(f"no frames captured from {self.frame_source}")

        # duration is the span between the first and last frame writes; with a
        # single frame the span is 0 and measured_fps falls back to requested_hz
        if self._t_first is not None and self._t_last is not None:
            duration_s = self._t_last - self._t_first
        else:
            duration_s = 0.0
        if frame_count >= 2 and duration_s > 0:
            measured_fps = frame_count / duration_s
        else:
            measured_fps = self.record_hz

        measured_fps = round(float(measured_fps), 2)
        duration_s = round(float(duration_s), 3)

        clip_path = self.out_dir / "clip.mp4"
        clip_json_path = self.out_dir / "clip.json"

        # encode the captured sequence into browser-playable H.264
        encode_mp4(self.frames_dir, clip_path, measured_fps)

        result = ClipResult(
            clip_path=clip_path,
            clip_json_path=clip_json_path,
            measured_fps=measured_fps,
            frame_count=frame_count,
            duration_s=duration_s,
            dup_samples=self._dup_samples,
            missing_samples=self._missing_samples,
            width=self._width,
            height=self._height,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
        )
        self._write_clip_json(result)
        return result

    # -- persistence ------------------------------------------------------- #
    def _write_clip_json(self, result: ClipResult) -> None:
        """Atomic write (tmp + os.replace) so a concurrent reader never sees a
        half-written clip.json — same posture as ``pose_label`` in the
        controller and ``status.json`` in jobs.py."""
        doc = {
            "job_id": self.out_dir.name,
            "frame_source": str(self.frame_source),
            "requested_hz": self.record_hz,
            "measured_fps": result.measured_fps,
            "frame_count": result.frame_count,
            "duration_s": result.duration_s,
            "dup_samples": result.dup_samples,
            "missing_samples": result.missing_samples,
            "width": result.width,
            "height": result.height,
            "started_at": result.started_at,
            "stopped_at": result.stopped_at,
            "clip_path": "clip.mp4",
        }
        path = result.clip_json_path
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
