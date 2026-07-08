"""pytest for SonicProvider's pure orchestration (motion/app/replay.py).

SonicProvider runs the on-device pipeline (ROMP -> GMR -> SONIC -> render), but the heavy
tool calls are delegated to an injectable ``stage_runner`` and the container pause/restart
to an injectable ``container_pauser``. So the SEQUENCE, the stepper callbacks, the free-RAM
pause/restart, and the SONIC->GMR fallback are all unit-testable off-robot with fakes.

Run:  ``pytest motion/tests/test_sonic_provider.py``  from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Resolve ``motion.app.replay`` regardless of the invocation cwd.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from motion.app.replay import (  # noqa: E402  (after sys.path insert)
    ReplayError,
    SonicProvider,
    SonicStageError,
)


class FakeJob:
    """Minimal stand-in for jobs.Job — SonicProvider only touches clip_path/replay_path."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.clip_path = self.dir / "clip.mp4"
        self.replay_path = self.dir / "replay.mp4"


def _mk_clip(job: FakeJob):
    job.clip_path.write_bytes(b"\x00\x00")   # a non-empty clip on disk


def _runner(record, *, produce_on=("SONIC",), fail_on=()):
    """Fake stage_runner: records each stage, fails on `fail_on`, and writes replay.mp4
    on any stage in `produce_on` (mimicking the render step)."""
    def run(stage, job, jobdir):
        record.append(stage)
        if stage in fail_on:
            raise SonicStageError(f"{stage} boom")
        if stage in produce_on:
            Path(job.replay_path).write_bytes(b"\x00")
    return run


def _pauser(record):
    def pause(paused):
        record.append(("pause", paused))
    return pause


def test_happy_path_sequence_pause_and_callbacks(tmp_path):
    job = FakeJob(tmp_path); _mk_clip(job)
    calls, pauses, steps = [], [], []
    p = SonicProvider(stage_runner=_runner(calls), container_pauser=_pauser(pauses))
    p.recreate(job, on_stage=lambda s, st: steps.append((s, st)))

    assert calls == ["POSE", "GMR", "SONIC"]                 # in order, no fallback
    assert pauses == [("pause", True), ("pause", False)]     # free RAM, then restart
    assert job.replay_path.exists()
    for stage in ("POSE", "GMR", "SONIC"):                    # each active -> done
        assert (stage, "active") in steps and (stage, "done") in steps


def test_sonic_failure_triggers_gmr_fallback(tmp_path):
    job = FakeJob(tmp_path); _mk_clip(job)
    calls = []
    p = SonicProvider(
        stage_runner=_runner(calls, produce_on=("SONIC_FALLBACK",), fail_on=("SONIC",)),
        container_pauser=lambda x: None)
    p.recreate(job)
    assert calls == ["POSE", "GMR", "SONIC", "SONIC_FALLBACK"]
    assert job.replay_path.exists()                          # fallback still shipped a replay


def test_sonic_failure_without_fallback_raises_but_restarts(tmp_path):
    job = FakeJob(tmp_path); _mk_clip(job)
    pauses = []
    p = SonicProvider(stage_runner=_runner([], fail_on=("SONIC",)),
                      container_pauser=_pauser(pauses), fallback_to_gmr=False)
    with pytest.raises(SonicStageError):
        p.recreate(job)
    assert ("pause", False) in pauses                        # perception restarted despite error


def test_pose_failure_propagates_and_restarts(tmp_path):
    job = FakeJob(tmp_path); _mk_clip(job)
    pauses = []
    p = SonicProvider(stage_runner=_runner([], fail_on=("POSE",)),
                      container_pauser=_pauser(pauses))
    with pytest.raises(SonicStageError):
        p.recreate(job)
    assert ("pause", False) in pauses


def test_missing_clip_raises_replayerror(tmp_path):
    job = FakeJob(tmp_path)                                   # no clip on disk
    p = SonicProvider(stage_runner=lambda *a: None, container_pauser=lambda x: None)
    with pytest.raises(ReplayError):
        p.recreate(job)


def test_pipeline_producing_no_replay_raises(tmp_path):
    job = FakeJob(tmp_path); _mk_clip(job)
    pauses = []
    p = SonicProvider(stage_runner=_runner([], produce_on=()),   # never writes replay
                      container_pauser=_pauser(pauses))
    with pytest.raises(ReplayError):
        p.recreate(job)
    assert ("pause", False) in pauses                        # still restarted perception
