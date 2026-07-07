"""On-disk job store + record→recreate state machine for the Motion tab.

Phase-1 (STUB, no ML). Pure stdlib — NO fastapi / cv2 imports here, so this module
(and the recorder it sits beside) import cleanly under the workstation base python
that runs ``pytest motion/tests/test_recorder.py``.

One job == one directory under ``motion/data/clips/<job_id>/`` holding::

    frames/frame_%06d.jpg   clip.mp4   clip.json   replay.mp4   status.json

``status.json`` is the single source of truth for a job's lifecycle and IS the body
of ``GET /motion/jobs/<id>/status`` (schema in the frozen contract §3). Every write is
atomic (tmp file + ``os.replace``) — the same posture as ``pose_label`` /
``save_dances`` in ``scripts/robot_web_controller.py`` — so a concurrent reader (the
~700 ms status poller) never sees a half-written file.

State machine (IDLE→RECORDING→RECORDED→PROCESSING→READY→ERROR)::

    IDLE      → RECORDING
    RECORDING → RECORDED | ERROR
    RECORDED  → PROCESSING
    PROCESSING→ READY | ERROR
    READY     → PROCESSING     (re-recreate)
    ERROR     → PROCESSING     (retry recreate)

Anything else raises ``InvalidTransition``.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from enum import Enum
from pathlib import Path


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class JobState(str, Enum):
    """Lifecycle states. ``str`` mixin -> JSON-serialises as the bare value."""

    IDLE = "IDLE"
    RECORDING = "RECORDING"
    RECORDED = "RECORDED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    ERROR = "ERROR"


# Fixed processing-stepper order (Pose→GMR→SONIC). The stub advances through these;
# later phases run the real ROMP → GMR → SONIC stages behind the same names.
STEPPER_STAGES = ("POSE", "GMR", "SONIC")

# The ONLY legal transitions. Keys/values are plain strings so lookups work whether
# the caller passes a str or a JobState (JobState is a str subclass -> compares equal).
ALLOWED = {
    "IDLE":       {"RECORDING"},
    "RECORDING":  {"RECORDED", "ERROR"},
    "RECORDED":   {"PROCESSING"},
    "PROCESSING": {"READY", "ERROR"},
    "READY":      {"PROCESSING"},     # re-recreate an already-finished take
    "ERROR":      {"PROCESSING"},     # retry a recreate that blew up
}


class InvalidTransition(RuntimeError):
    """Raised when ``Job.transition`` is asked to make a move not in ``ALLOWED``."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _state_str(state) -> str:
    """Normalise a JobState|str to its bare string value ('RECORDING', not
    'JobState.RECORDING'). Guards against ``str(JobState.X)`` returning the repr."""
    if isinstance(state, JobState):
        return state.value
    return str(state)


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` in place.

    Nested dicts are merged (so ``patch(processing={"stage": "GMR"})`` updates only
    ``processing.stage`` and keeps ``processing.stages`` / provider / timestamps);
    every other value replaces.
    """
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


# --------------------------------------------------------------------------- #
# Job — one take on disk
# --------------------------------------------------------------------------- #
class Job:
    """A single take. Owns its directory and its ``status.json``.

    Instances are cached one-per-id by the :class:`JobStore`, so every reference to a
    given job shares the same :class:`threading.RLock` — that is what actually makes
    the mutations (transition / patch / set_stage) serialise across the recreate
    daemon thread and the request handlers.
    """

    def __init__(self, job_id: str, job_dir: Path) -> None:
        self.id = job_id
        self.dir = Path(job_dir)
        self._lock = threading.RLock()

    # -- convenience paths -------------------------------------------------- #
    @property
    def clip_path(self) -> Path:
        return self.dir / "clip.mp4"

    @property
    def clip_json_path(self) -> Path:
        return self.dir / "clip.json"

    @property
    def replay_path(self) -> Path:
        return self.dir / "replay.mp4"

    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    @property
    def frames_dir(self) -> Path:
        return self.dir / "frames"

    # -- persistence internals --------------------------------------------- #
    def _artifacts(self) -> dict:
        """Live filesystem truth for the two Phase-1 artifacts. Recomputed on every
        read and before every write so ``status.json`` never lies about what exists."""
        return {"clip": self.clip_path.exists(), "replay": self.replay_path.exists()}

    def _default_status(self) -> dict:
        """A fresh IDLE skeleton (contract §3). Written by ``JobStore.new_job`` and
        used as a defensive fallback if ``status.json`` is missing/corrupt."""
        now = time.time()
        return {
            "id": self.id,
            "state": JobState.IDLE.value,
            "created_at": now,
            "updated_at": now,
            "recording": None,     # filled once RECORDING begins
            "processing": None,    # filled once recreate starts
            "error": None,         # non-null ONLY while state == ERROR
            "artifacts": self._artifacts(),
        }

    def _read(self) -> dict:
        """Load ``status.json``. Falls back to a fresh IDLE skeleton if the file is
        absent or unreadable (never raises on a normal read)."""
        try:
            with open(self.status_path) as f:
                return json.load(f) or self._default_status()
        except (OSError, ValueError):
            return self._default_status()

    def _write(self, data: dict) -> None:
        """Atomic write: tmp + ``os.replace`` (same pattern as ``pose_label``)."""
        tmp = Path(str(self.status_path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.status_path)

    def _initialize(self) -> None:
        """Create the on-disk skeleton (dir + frames/) and the initial IDLE
        ``status.json``. Called once by :meth:`JobStore.new_job`. IDLE is not a target
        of any transition, so the first write is direct rather than via ``transition``.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)   # ready for the recorder
        with self._lock:
            self._write(self._default_status())

    # -- public read ------------------------------------------------------- #
    def status(self) -> dict:
        """Return the full ``status.json`` body with ``artifacts`` refreshed from disk.

        Lock-free: atomic writes guarantee a reader sees a complete old-or-new file,
        so the hot status poller never blocks on a mutation.
        """
        data = self._read()
        data["artifacts"] = self._artifacts()
        return data

    def summary(self) -> dict:
        """Compact list-item view (contract §5 ``<summary>``): id, state, created_at,
        measured_fps|null, has_replay. Convenience for ``GET /motion/jobs``."""
        data = self.status()
        rec = data.get("recording") or {}
        return {
            "id": data.get("id", self.id),
            "state": data.get("state", JobState.IDLE.value),
            "created_at": data.get("created_at"),
            "measured_fps": rec.get("measured_fps"),
            "has_replay": bool(data.get("artifacts", {}).get("replay")),
        }

    # -- public mutations (all serialised on self._lock) ------------------- #
    def transition(self, new_state: str, **patch) -> dict:
        """Validate ``current → new_state`` against ``ALLOWED``, apply the state,
        shallow-merge ``patch`` (top-level keys replace), bump ``updated_at``,
        atomic-write and return the new status.

        Enforces the invariant "``error`` is non-null only in the ERROR state":
        moving to ERROR keeps the supplied/existing error; any other move clears it
        (so a retry PROCESSING/READY starts clean).

        Raises :class:`InvalidTransition` if the move is not allowed.
        """
        target = _state_str(new_state)
        with self._lock:
            data = self._read()
            current = _state_str(data.get("state", JobState.IDLE.value))
            if target not in ALLOWED.get(current, set()):
                raise InvalidTransition(
                    f"{current} -> {target} is not an allowed transition"
                )
            data["state"] = target
            # shallow merge: each provided top-level key replaces that key wholesale
            for key, value in patch.items():
                data[key] = value
            # error invariant
            if target == JobState.ERROR.value:
                data["error"] = patch.get("error", data.get("error"))
            else:
                data["error"] = None
            data["updated_at"] = time.time()
            data["artifacts"] = self._artifacts()
            self._write(data)
            return data

    def patch(self, **fields) -> dict:
        """Deep-merge ``fields`` into the status (e.g. an incremental
        ``processing={"stage": "GMR"}`` that preserves ``processing.stages``), bump
        ``updated_at``, atomic-write and return the new status. No state change."""
        with self._lock:
            data = self._read()
            _deep_merge(data, fields)
            data["updated_at"] = time.time()
            data["artifacts"] = self._artifacts()
            self._write(data)
            return data

    def set_stage(self, stage: str, status: str) -> dict:
        """Update one processing stage's status (``pending|active|done|error``) and
        stamp its ``t``; set ``processing.stage`` to the touched stage. No state change.

        Bootstraps ``processing.stages`` in fixed POSE→GMR→SONIC order if the recreate
        route hasn't seeded it yet, so an out-of-order call still writes a valid shape.
        """
        stage = _state_str(stage)
        with self._lock:
            data = self._read()
            proc = data.get("processing") or {}
            stages = proc.get("stages")
            if not stages:
                stages = [{"name": n, "status": "pending", "t": None}
                          for n in STEPPER_STAGES]
            for entry in stages:
                if entry.get("name") == stage:
                    entry["status"] = status
                    entry["t"] = time.time()
                    break
            proc["stages"] = stages
            proc["stage"] = stage          # currently-active (or last-touched) stage
            data["processing"] = proc
            data["updated_at"] = time.time()
            data["artifacts"] = self._artifacts()
            self._write(data)
            return data


# --------------------------------------------------------------------------- #
# JobStore — the directory of takes
# --------------------------------------------------------------------------- #
class JobStore:
    """Directory-backed collection of :class:`Job` s rooted at ``motion/data/clips``.

    Caches one :class:`Job` object per id so all callers share that job's lock; a
    store-level lock guards the cache itself.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, Job] = {}

    def _job(self, job_id: str) -> Job:
        """Return the cached Job for ``job_id`` (constructing/caching one if needed).
        Does NOT create the on-disk directory."""
        with self._lock:
            job = self._cache.get(job_id)
            if job is None:
                job = Job(job_id, self.root / job_id)
                self._cache[job_id] = job
            return job

    def new_job(self) -> Job:
        """Mint a new take: create its directory skeleton, write an initial IDLE
        ``status.json`` and return the :class:`Job`.

        ``job_id = "%Y%m%d-%H%M%S" + "-" + secrets.token_hex(2)`` — lexicographically
        sortable (newest sorts last) with a short random suffix to avoid same-second
        collisions.
        """
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        job = self._job(job_id)
        job._initialize()
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the Job for ``job_id`` or ``None`` if unknown / directory missing.
        Rejects path-traversal ids defensively (the id arrives from a URL path)."""
        if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
            return None
        if not (self.root / job_id).is_dir():
            return None
        return self._job(job_id)

    def list(self) -> list[Job]:
        """All real takes (dirs holding a ``status.json``), newest-first
        (reverse-sorted by id — which is time-ordered by construction)."""
        jobs: list[Job] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_dir() and (entry / "status.json").exists():
                jobs.append(self._job(entry.name))
        jobs.sort(key=lambda j: j.id, reverse=True)
        return jobs
