#!/usr/bin/env bash
# G1 stairs pipeline: smoke test -> convergence test -> real training.
# Each stage gates the next — a failure stops the script before wasting GPU
# time on the next, more expensive stage.
#
# Run from the training box (needs isaaclab + isaaclab_rl + rsl_rl installed,
# repo root on PYTHONPATH so `g1_stairs_rl_policy.*` imports resolve):
#   ./g1_stairs_rl_policy/scripts/run_pipeline.sh
#
# Flags (all optional):
#   --skip-smoke          skip stage 1
#   --skip-convergence    skip stage 2
#   --skip-train          stop after stage 2, don't launch real training
#   --isaaclab PATH        path to isaaclab.sh (default: isaaclab.sh on PATH)
#
# Unrecognized args are passed straight through to the TRAIN stage only
# (e.g. --num_envs 4096 --max_iterations 8000), so you can tune the real run
# without editing this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ISAACLAB="${ISAACLAB_SH:-isaaclab.sh}"
SKIP_SMOKE=0
SKIP_CONVERGENCE=0
SKIP_TRAIN=0
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --skip-convergence) SKIP_CONVERGENCE=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --isaaclab) ISAACLAB="$2"; shift 2 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

log() { echo -e "\n=== [$(date +%H:%M:%S)] $* ===\n"; }

# ---------------------------------------------------------------------------
if [[ $SKIP_SMOKE -eq 0 ]]; then
    log "STAGE 1/3: smoke test (seconds — does the env even come up)"
    "$ISAACLAB" -p "$SCRIPT_DIR/smoke_test.py" --headless --num_envs 4 --steps 50
    log "smoke test PASSED"
else
    log "STAGE 1/3: smoke test SKIPPED (--skip-smoke)"
fi

# ---------------------------------------------------------------------------
if [[ $SKIP_CONVERGENCE -eq 0 ]]; then
    log "STAGE 2/3: convergence test (minutes — easiest 2 rows only, does reward improve at all)"
    "$ISAACLAB" -p "$SCRIPT_DIR/convergence_test.py" --headless \
        --num_envs 64 --iterations 40 --terrain_rows 2 --terrain_cols 4
    log "convergence test PASSED"
else
    log "STAGE 2/3: convergence test SKIPPED (--skip-convergence)"
fi

# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 1 ]]; then
    log "STAGE 3/3: real training SKIPPED (--skip-train). Pipeline stops here."
    exit 0
fi

log "STAGE 3/3: real training (hours — full 10x20 curriculum, ppo_cfg.py's max_iterations by default)"
# Isaac Lab's own generic rsl_rl train script — not something we wrote, reused
# as-is (same path referenced in ppo_cfg.py's own launch comment).
"$ISAACLAB" -p "$REPO_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Unitree-G1-29dof-Stairs-v0 --headless "${TRAIN_ARGS[@]}"

log "training run finished (or was stopped) — checkpoints under logs/rsl_rl/g1_stairs/"
