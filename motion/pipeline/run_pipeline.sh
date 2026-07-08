#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — ON-DEVICE per-stage dispatcher for the motion Recreate pipeline.
#
# Called ONE STAGE AT A TIME by motion.app.replay.SonicProvider (which owns the
# sequence, the stepper callbacks, the perception pause/restart and the fallback).
# Keeping the sequencing in Python + the heavy tool calls here means the orchestration
# is unit-tested off-robot while every heavy dependency (ROMP / GMR / SONIC / MuJoCo)
# stays on the Orin.
#
#   run_pipeline.sh --stage POSE|GMR|SONIC|SONIC_FALLBACK \
#                   --job <job_dir> --clip <clip.mp4> --motion-root <motion/>
#
# Each stage EXITS NON-ZERO on failure so SonicProvider flips the job to ERROR (or, for
# SONIC, falls back to a kinematic GMR render). ON-DEVICE ONLY: ROMP/GMR/SONIC/MuJoCo are
# not present on a workstation, so the sub-scripts no-op-exit there.
# =============================================================================
set -euo pipefail

STAGE="" JOB="" CLIP="" MROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)       STAGE="$2"; shift 2;;
    --job)         JOB="$2";   shift 2;;
    --clip)        CLIP="$2";  shift 2;;
    --motion-root) MROOT="$2"; shift 2;;
    *) echo "run_pipeline.sh: unknown arg $1" >&2; exit 2;;
  esac
done
[[ -n "$STAGE" && -n "$JOB" && -n "$MROOT" ]] || {
  echo "usage: $0 --stage <POSE|GMR|SONIC|SONIC_FALLBACK> --job <dir> --clip <mp4> --motion-root <dir>" >&2
  exit 2
}

PY="${PYTHON:-python3}"
GMR_PKL="$JOB/gmr.pkl"
CSV_DIR="$JOB/sonic_csv"
SONIC_OUT="$JOB/sonic_out"
REPLAY="$JOB/replay.mp4"

banner() { echo "[run_pipeline:$STAGE] $*"; }

case "$STAGE" in
  POSE)
    # ROMP on clip.mp4 -> per-frame SMPL, then pose_to_smpl -> smpl.npz (run_pose.sh does both).
    banner "ROMP + pose_to_smpl (on-device)"
    bash "$MROOT/pipeline/stage_pose/run_pose.sh" "$JOB"
    ;;
  GMR)
    # SMPL sequence -> gmr.pkl (29-DOF G1 joints). tgt fps read from clip.json inside the script.
    banner "GMR retarget (on-device)"
    bash "$MROOT/pipeline/stage_gmr/run_gmr.sh" "$JOB"
    "$PY" -m motion.pipeline.glue.gmr_pkl --pkl "$GMR_PKL" >/dev/null   # validate loud
    ;;
  SONIC)
    # gmr.pkl -> 6-CSV bundle (pure Python, always available) -> SONIC track -> MuJoCo render.
    banner "gmr_to_sonic_csv"
    "$PY" -m motion.pipeline.glue.gmr_to_sonic_csv --gmr "$GMR_PKL" --out "$CSV_DIR"
    banner "SONIC track (on-device)"
    bash "$MROOT/pipeline/stage_sonic/run_sonic.sh" "$CSV_DIR"     # writes $SONIC_OUT (VERIFY path on-robot)
    banner "render -> replay.mp4 (on-device)"
    "$PY" "$MROOT/sim/sim_to_feed.py" --sonic "$SONIC_OUT" --out "$REPLAY"
    ;;
  SONIC_FALLBACK)
    # SONIC couldn't track -> render GMR's kinematic playback so Recreate still ships a clip.
    banner "kinematic GMR render -> replay.mp4 (on-device fallback)"
    "$PY" "$MROOT/sim/sim_to_feed.py" --kinematic "$GMR_PKL" --out "$REPLAY"
    ;;
  *)
    echo "run_pipeline.sh: unknown stage '$STAGE'" >&2; exit 2;;
esac

banner "ok"
