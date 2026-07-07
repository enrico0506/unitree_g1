# motion — record a human, recreate it on the G1

Record a person on the robot's camera, run their motion through **ROMP → GMR → SONIC**, and replay it
on a (simulated, then real) Unitree G1 — **all on the robot's onboard Jetson Orin NX 16GB, no workstation.**
Lives as its own **Motion** tab on the dashboard with a clean camera feed (no object detection).

👉 **Start here: [`PLAN.md`](PLAN.md)** — the on-device phased plan (built for handing to Claude Code).
TL;DR: **runs on the Orin NX**; GMR + SONIC are Jetson-ready (SONIC is NVIDIA's official G1-Orin deploy);
**GVHMR is swapped for ROMP** (the only stage that can't run on ARM) — costs some spin/rotation crispness.
29-DOF sim-first; real 23-DOF robot deferred.

## What runs where (all on the Orin NX)
- **Record + dashboard + view:** `app/` (FastAPI routes + Motion tab). No ML.
- **Pipeline (behind a "pause live perception to free RAM" step):** `pipeline/` — ROMP → GMR → SONIC.
- No network, no second machine — everything is local files (`data/`) + `/dev/shm`.

## Layout
```
config/        pipeline.yaml — rates, robot/dof, pose estimator, which containers to pause, model paths
app/           recorder, job store, replay providers (stub → sonic), FastAPI /motion/* routes, web/ tab assets
pipeline/      stage_pose (ROMP) / stage_gmr / stage_sonic + glue/ (fps, frames, pose→smpl, joint maps, gmr→sonic CSV)
third_party/   submodules: ROMP, GMR, GR00T-WholeBodyControl (weights gitignored)
models/        weights (gitignored): smpl/ (MPI-gated), sonic/
sim/           MuJoCo sim2sim harness + framebuffer→dashboard bridge (renders on-device)
data/          per-job artifacts (gitignored): clips/, jobs/ (smpl, gmr.pkl, csv, replay.mp4, status.json)
tests/         recorder fps, joint-map round-trip, gmr→sonic CSV parity
```
(`transfer/` exists only for the optional two-machine fallback — unused on-device.)

## Build order
Phase 1 (record/tab/stub — buildable now) → JetPack6/TensorRT-10.7 base → spike SONIC's own examples → Phase 2 ROMP → 3 GMR → 4 SONIC → 5 wire it up. Each phase is a standalone checkpoint. See `PLAN.md`.
