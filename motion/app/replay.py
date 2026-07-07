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
import shutil
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # avoid a runtime import cycle with jobs.py; only for type hints
    from motion.app.jobs import Job

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
