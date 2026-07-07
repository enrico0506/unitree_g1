"""Motion tab — FastAPI router for the record -> pause -> recreate flow (Phase 1).

This module owns the HTTP surface only; the actual work is delegated to the
sibling stdlib-only modules so `pytest motion/tests/test_recorder.py` never has
to import fastapi:

    recorder.Recorder / RecorderError   sample the frame source -> clip.mp4
    jobs.JobStore / Job                 on-disk state machine + status.json
    replay.ReplayProvider               POSE->GMR->SONIC stub that makes replay.mp4

The router is built by `build_motion_router(...)` (a factory) and mounted by
`scripts/robot_web_controller.py` right after `app.mount("/static", ...)`. It is
GATED on the single-controller lock: the mutating routes (record start/stop,
recreate) require the caller to be the current control-lock owner. The owner is
read LIVE every request via the injected `get_control_owner` callable because the
controller REASSIGNS its `control_owner_id` module global on every handoff — we
must never capture the value at import time.

Endpoints (prefix `/motion`):
    POST /motion/record/start          gated  -> 201 {"id","state":"RECORDING"}
    POST /motion/record/stop           gated  -> 200 full status.json (RECORDED)
    GET  /motion/jobs                  open   -> 200 {"jobs":[<summary>,...]}
    POST /motion/jobs/{id}/recreate    gated  -> 202 {"id","state":"PROCESSING"}
    GET  /motion/jobs/{id}/status      open   -> 200 full status.json
    GET  /motion/jobs/{id}/replay.mp4  open   -> 200 FileResponse(video/mp4)
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# Sibling modules — stdlib/subprocess only (no fastapi, no cv2). If any of these
# fail to import the whole router is disabled by the defensive try/except at the
# mount site in robot_web_controller.py, so the dashboard never goes down with it.
from motion.app.jobs import JobStore, STEPPER_STAGES, InvalidTransition
from motion.app.recorder import Recorder, RecorderError
from motion.app.replay import ReplayProvider


def build_motion_router(
    get_control_owner: Callable[[], "str | None"],
    store: JobStore,
    provider: ReplayProvider,
    frame_source: str = "/dev/shm/g1_camera.jpg",
    record_hz: float = 30.0,
    max_seconds: float = 30.0,
) -> APIRouter:
    """Build the `/motion` router.

    Args:
        get_control_owner: returns the current `control_owner_id` (or None if the
            lock is free). Called live per request — pass `lambda: control_owner_id`.
        store:        the on-disk JobStore rooted at motion/data/clips.
        provider:     the ReplayProvider used by recreate (StubProvider in Phase 1).
        frame_source: file the camera service overwrites with the clean feed.
        record_hz:    sampling rate; the TRUE fps is measured over wall-clock.
        max_seconds:  hard cap on a single take.
    """
    router = APIRouter(prefix="/motion", tags=["motion"])

    # Single active recorder/job for the whole box (one operator, one take at a
    # time). Guarded by `_lock` so two racing start/stop requests (FastAPI runs
    # sync handlers in a threadpool) can't corrupt it.
    _active: dict = {"recorder": None, "job": None}
    _lock = threading.Lock()

    # ------------------------------------------------------------------ gate --
    def _require_owner(
        x_client_id: "str | None" = Header(default=None, alias="X-Client-Id"),
    ) -> str:
        """Dependency: allow only the current control-lock owner.

        409 if nobody holds the lock, 403 if the header is missing or does not
        match the live owner. The client id travels in a header (never a query
        string) so it never lands in a URL / log line.
        """
        owner = get_control_owner()
        if owner is None:
            raise HTTPException(status_code=409, detail="no controller holds the lock")
        if not x_client_id or x_client_id != owner:
            raise HTTPException(status_code=403, detail="not the control-lock owner")
        return x_client_id

    # ---------------------------------------------------------- recreate glue --
    def _make_on_stage(job) -> Callable[[str, str], None]:
        """Bind a job to an `on_stage(stage, status)` callback for the provider.

        The provider calls this with status in {'active','done'} per stage; we
        write the per-stage status into `stages[]` and advance the current-stage
        pointer. The provider never touches status.json directly — routes own
        state (contract §4).
        """
        def _on_stage(stage: str, status: str) -> None:
            job.set_stage(stage, status)                 # stages[].status
            job.patch(processing={"stage": stage})       # current stage pointer
        return _on_stage

    def _recreate_worker(job) -> None:
        """Daemon-thread body: run the (blocking) provider, then settle state."""
        try:
            provider.recreate(job, on_stage=_make_on_stage(job))
            job.patch(processing={"finished_at": time.time()})   # preserve stages (deep-merge)
            job.transition("READY")                      # PROCESSING -> READY
        except Exception as e:                            # noqa: BLE001 — record any failure
            job.transition("ERROR", error=str(e))        # PROCESSING -> ERROR

    # ----------------------------------------------------------- summaries -----
    def _summary(job) -> dict:
        """Compact list item for GET /motion/jobs (contract §5 <summary>)."""
        st = job.status()
        rec = st.get("recording") or {}
        arts = st.get("artifacts") or {}
        return {
            "id": st.get("id", job.id),
            "state": st.get("state"),
            "created_at": st.get("created_at"),
            "measured_fps": rec.get("measured_fps"),
            "has_replay": bool(arts.get("replay")),
        }

    # ============================================================= routes ======
    # Handlers are plain `def` (not async): `recorder.stop()` runs ffmpeg and
    # `store`/`job` do disk I/O, so FastAPI dispatches them on its threadpool and
    # never blocks the event loop that also serves the MJPEG camera stream.

    @router.post("/record/start")
    def record_start(_owner: str = Depends(_require_owner)):
        """Begin a take. 409 if one is already recording."""
        with _lock:
            rec = _active["recorder"]
            if rec is not None and rec.is_recording:
                raise HTTPException(status_code=409, detail="already recording")
            job = store.new_job()                         # IDLE
            job.transition(
                "RECORDING",
                recording={"started_at": time.time(), "stopped_at": None},
            )
            recorder = Recorder(frame_source, job.dir, record_hz, max_seconds)
            recorder.start()                              # spawns the sampling thread
            _active["recorder"] = recorder
            _active["job"] = job
        return JSONResponse({"id": job.id, "state": "RECORDING"}, status_code=201)

    @router.post("/record/stop")
    def record_stop(_owner: str = Depends(_require_owner)):
        """End the active take: encode clip.mp4, stamp clip.json, return status."""
        with _lock:
            recorder = _active["recorder"]
            job = _active["job"]
            if recorder is None or job is None:
                raise HTTPException(status_code=409, detail="not recording")
            # The recording is over either way — release the slot before the (slow)
            # ffmpeg encode so a fresh /record/start isn't blocked by a stuck take.
            _active["recorder"] = None
            _active["job"] = None

        try:
            res = recorder.stop()                         # joins thread, encodes clip
        except RecorderError as e:
            job.transition("ERROR", error=str(e))         # RECORDING -> ERROR
            return JSONResponse(
                {"id": job.id, "state": "ERROR", "error": str(e)},
                status_code=500,
            )

        status = job.transition(                          # RECORDING -> RECORDED
            "RECORDED",
            recording={
                "started_at": res.started_at,
                "stopped_at": res.stopped_at,
                "measured_fps": res.measured_fps,
                "frame_count": res.frame_count,
                "duration_s": res.duration_s,
            },
        )
        return JSONResponse(status, status_code=200)

    @router.get("/jobs")
    def list_jobs():
        """All takes, newest-first. Open read (mirrors the ungated /camera/* reads)."""
        out = []
        for job in store.list():                          # already newest-first
            try:
                out.append(_summary(job))
            except Exception:                             # noqa: BLE001 — a corrupt job
                continue                                  #   dir must not sink the list
        return JSONResponse({"jobs": out}, status_code=200)

    @router.post("/jobs/{job_id}/recreate")
    def recreate(job_id: str, _owner: str = Depends(_require_owner)):
        """Kick off the stub POSE->GMR->SONIC pipeline in a daemon thread (202)."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")

        state = job.status().get("state")
        if state in ("RECORDING", "PROCESSING"):
            raise HTTPException(status_code=409, detail=f"job is {state}")

        try:
            job.transition(                               # -> PROCESSING
                "PROCESSING",
                processing={
                    "provider": provider.name,
                    "started_at": time.time(),
                    "stage": STEPPER_STAGES[0],           # "POSE"
                    "stages": [
                        {"name": s, "status": "pending", "t": None} for s in STEPPER_STAGES
                    ],
                },
            )
        except InvalidTransition:
            # e.g. a transient IDLE job with no PROCESSING edge -> clean 409, not a 500.
            raise HTTPException(status_code=409, detail=f"cannot recreate from {state}")
        threading.Thread(
            target=_recreate_worker, args=(job,), daemon=True,
        ).start()
        return JSONResponse({"id": job.id, "state": "PROCESSING"}, status_code=202)

    @router.get("/jobs/{job_id}/status")
    def job_status(job_id: str):
        """Full status.json for one take. Open read; 404 on unknown id."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return JSONResponse(job.status(), status_code=200)

    @router.get("/jobs/{job_id}/replay.mp4")
    def job_replay(job_id: str):
        """Serve the recreated replay clip. 404 on unknown id or missing file."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        replay = job.replay_path
        if not replay.exists():
            raise HTTPException(status_code=404, detail="no replay yet")
        return FileResponse(str(replay), media_type="video/mp4")

    return router


# Convenience alias — the integration snippet in the contract calls
# `build_motion_router`, but `get_router` reads well at the mount site too.
get_router = build_motion_router
