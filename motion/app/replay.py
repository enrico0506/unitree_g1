"""Replay providers — turn a recorded clip into a G1 replay video.

Phase 1 is STUB-only: no ML. A provider's job is to advance a recorded take
through the fixed processing stages POSE -> GMR -> SONIC and, when done, drop a
``replay.mp4`` next to the original ``clip.mp4`` in the job directory.

Design contract (load-bearing):
  * A provider does PURE work + callbacks. It NEVER touches ``status.json`` —
    the route layer owns job state (transition / set_stage / patch). The
    provider only reports progress through the ``on_stage`` callback and writes
    its output artifact (``job.replay_path``).
  * ``recreate`` is BLOCKING. Routes run it in a daemon thread so the HTTP call
    returns immediately; the provider itself stays dumb and synchronous.
  * stdlib only (abc / shutil / time) — no fastapi, no cv2 — so this module
    imports cleanly under the workstation base python used by the test suite.

The real ML provider (Phase 5) will subclass ``ReplayProvider`` the same way,
swapping the fake per-stage ``sleep`` for actual ROMP -> GMR -> SONIC work and
rendering a real MuJoCo replay instead of copying the source clip.
"""

from __future__ import annotations

import abc
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # avoid a runtime import cycle with jobs.py; only for type hints
    from motion.app.jobs import Job

# Repo motion/ root (…/motion) — this file is motion/app/replay.py.
_MOTION_ROOT = Path(__file__).resolve().parent.parent

# Fixed processing order. Mirrors jobs.STEPPER_STAGES — kept as a local literal so
# replay.py has zero runtime dependency on jobs.py (the type hint above is enough).
STAGES: tuple[str, ...] = ("POSE", "GMR", "SONIC")


class ReplayProvider(abc.ABC):
    """Interface every replay backend implements (stub now, SONIC later)."""

    #: short machine name, stamped into status.json (processing.provider) by routes
    name: str = "base"

    @abc.abstractmethod
    def recreate(
        self,
        job: "Job",
        on_stage: Callable[[str, str], None] | None = None,
    ) -> None:
        """Advance POSE -> GMR -> SONIC and produce ``job.replay_path``. BLOCKING.

        Args:
            job: the RECORDED job to process; ``job.clip_path`` is the input,
                ``job.replay_path`` is where the output replay must land.
            on_stage: optional progress callback invoked ``on_stage(stage, status)``
                where ``stage`` is one of ``STAGES`` and ``status`` is ``"active"``
                (stage started) or ``"done"`` (stage finished).

        Raises:
            Exception: any failure (e.g. missing clip) propagates to the caller,
                which transitions the job to ERROR. Providers do not swallow it.
        """
        raise NotImplementedError


class StubProvider(ReplayProvider):
    """No-ML stand-in: walks the stepper with a fake delay, then fakes the replay.

    The "replay" is simply a copy of the original clip — enough to exercise the
    full record -> recreate -> result UX (button morph, stepper animation,
    original-vs-replay view) before any pose/retarget/tracking model exists.
    """

    name = "stub"

    def __init__(self, stage_delay_s: float = 1.0) -> None:
        """Args: stage_delay_s — seconds to linger on each stage (fakes ML work)."""
        self.stage_delay_s = stage_delay_s

    def recreate(
        self,
        job: "Job",
        on_stage: Callable[[str, str], None] | None = None,
    ) -> None:
        # Step through Pose -> GMR -> SONIC, pinging the callback on entry/exit so
        # the Motion tab's stepper lights up in order.
        for stage in STAGES:
            if on_stage is not None:
                on_stage(stage, "active")
            time.sleep(self.stage_delay_s)
            if on_stage is not None:
                on_stage(stage, "done")

        # Fake replay = the original clip. copyfile raises (FileNotFoundError) if
        # clip.mp4 is missing; we let that propagate so routes flip the job to ERROR.
        shutil.copyfile(job.clip_path, job.replay_path)


class ReplayError(RuntimeError):
    """The pipeline ran but produced no usable replay.mp4."""


class SonicStageError(RuntimeError):
    """A pipeline stage (ROMP / GMR / SONIC / render) failed on-device."""


class SonicProvider(ReplayProvider):
    """On-device backend: ROMP -> GMR -> SONIC -> MuJoCo render -> replay.mp4.

    ON-DEVICE ONLY (the heavy tools live on the Orin). This class owns the SEQUENCE, the
    stepper callbacks, the free-RAM pause/restart of the live perception containers, and
    the SONIC -> GMR-kinematic FALLBACK (so the feature still ships if SONIC can't cleanly
    track the reference). The actual per-stage tool invocation is delegated to
    ``pipeline/run_pipeline.sh`` (dispatched by stage), so every heavy dependency stays out
    of this process -- which also makes the whole orchestration unit-testable off-robot by
    injecting ``stage_runner`` + ``container_pauser``.

    Off-robot the default runner shells out to run_pipeline.sh, which no-op-exits when the
    tools are absent -> the stage raises -> routes flips the job to ERROR. So SonicProvider
    is OPT-IN (mount via MOTION_PROVIDER=sonic); StubProvider stays the default.
    """

    name = "sonic"

    def __init__(
        self,
        *,
        motion_root: Path = _MOTION_ROOT,
        stage_runner: Callable[[str, "Job", Path], None] | None = None,
        container_pauser: Callable[[bool], None] | None = None,
        pause_containers: tuple[str, ...] = ("g1-pose", "g1-hands", "g1-detect"),
        fallback_to_gmr: bool = True,
    ) -> None:
        self.motion_root = Path(motion_root)
        self.pause_containers = tuple(pause_containers)
        self.fallback_to_gmr = fallback_to_gmr
        self._run_stage = stage_runner or self._default_stage_runner
        self._pause = container_pauser or self._default_pause

    # -- the sequence: pure orchestration, unit-testable ---------------------
    def recreate(
        self,
        job: "Job",
        on_stage: Callable[[str, str], None] | None = None,
    ) -> None:
        clip = Path(job.clip_path)
        if not clip.exists():
            raise ReplayError(f"clip not found: {clip}")
        jobdir = Path(job.replay_path).parent

        self._pause(True)                        # free the 16 GB for the pipeline
        try:
            self._stage(on_stage, "POSE", job, jobdir)   # ROMP -> smpl.npz
            self._stage(on_stage, "GMR", job, jobdir)    # -> gmr.pkl
            if on_stage is not None:
                on_stage("SONIC", "active")
            try:
                self._run_stage("SONIC", job, jobdir)    # csv + SONIC + render -> replay.mp4
            except SonicStageError:
                if not self.fallback_to_gmr:
                    raise
                # SONIC couldn't track -> render GMR's kinematic playback instead.
                self._run_stage("SONIC_FALLBACK", job, jobdir)
            if on_stage is not None:
                on_stage("SONIC", "done")
        finally:
            self._pause(False)                   # ALWAYS restart perception, even on error

        if not Path(job.replay_path).exists():
            raise ReplayError("pipeline finished but produced no replay.mp4")

    def _stage(self, on_stage, stage, job, jobdir):
        if on_stage is not None:
            on_stage(stage, "active")
        self._run_stage(stage, job, jobdir)
        if on_stage is not None:
            on_stage(stage, "done")

    # -- default on-device implementations (not exercised off-robot) ---------
    def _default_stage_runner(self, stage: str, job: "Job", jobdir: Path) -> None:
        script = self.motion_root / "pipeline" / "run_pipeline.sh"
        cmd = ["bash", str(script), "--stage", stage, "--job", str(jobdir),
               "--clip", str(job.clip_path), "--motion-root", str(self.motion_root)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SonicStageError(
                f"{stage} failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[-500:]}")

    def _default_pause(self, paused: bool) -> None:
        # Stop (free RAM) / restart the live perception containers. Best-effort: a docker
        # error here must NOT abort the pipeline or leave perception down silently.
        action = "stop" if paused else "start"
        for c in self.pause_containers:
            try:
                subprocess.run(["docker", action, c], capture_output=True,
                               text=True, timeout=30)
            except Exception:
                pass
