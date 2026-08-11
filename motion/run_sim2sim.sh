#!/usr/bin/env bash
# Easy button: run a motion_library clip through HoloMotion sim2sim eval on
# the Jetson deploy container. Starts the (already-provisioned) container if
# it isn't running, then execs holomotion/scripts/sim2sim.py inside it.
#
# Usage:
#   ./motion/run_sim2sim.sh wave_v2
#   ./motion/run_sim2sim.sh cartwheel --gui
#   ./motion/run_sim2sim.sh --list
#
# Output (video + robot-trajectory npz) lands in
#   motion/holomotion_ckpt/exported/mujoco_output_model_14000/
# since that dir is bind-mounted straight into the container -- no copy-out
# step needed.
#
# One-time setup this assumes (already done as of 2026-08-11): a container
# named holomotion_sim2sim exists, built from
# horizonrobotics/holomotion:v1.4.0-orin-jp5.1-arm64 with --runtime nvidia
# and --entrypoint sleep, and bind-mounts:
#   motion/holomotion       -> /workspace/holomotion
#   motion/holomotion_ckpt  -> /workspace/ckpt
#   motion/motion_library   -> /workspace/motions
# If that container doesn't exist (fresh machine), recreate it -- see the
# `docker run` invocation in git history (commit that added this script) or
# ask Claude to redo it.

set -euo pipefail

CONTAINER=holomotion_sim2sim

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container '$CONTAINER' doesn't exist. See the setup notes at the" >&2
    echo "top of this script (or motion/holomotion/HOW_IT_WORKS.md) to recreate it." >&2
    exit 1
fi

status="$(docker inspect "$CONTAINER" --format '{{.State.Status}}')"
if [[ "$status" != "running" ]]; then
    echo "Starting $CONTAINER (was: $status)..."
    docker start "$CONTAINER" >/dev/null
fi

exec docker exec -i "$CONTAINER" \
    /root/miniconda3/envs/holomotion_deploy/bin/python \
    /workspace/holomotion/scripts/sim2sim.py "$@"
