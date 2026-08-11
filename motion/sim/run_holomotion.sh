#!/usr/bin/env bash
# Easy button: run a motion_library clip through HoloMotion sim2sim eval on
# the Jetson deploy container.
#
# Usage:
#   ./motion/sim/run_holomotion.sh wave_v2
#   ./motion/sim/run_holomotion.sh cartwheel --gui   # real interactive MuJoCo window,
#                                                     # X11-forwarded to wherever you're
#                                                     # ssh'd in from -- see README.md.
#                                                     # Run this from a plain `ssh -X`
#                                                     # terminal, not a VS Code Remote-SSH
#                                                     # one (that tunnel doesn't carry X11).
#   ./motion/sim/run_holomotion.sh --list
#
# Output (video + robot-trajectory npz) lands in
#   motion/holomotion_ckpt/exported/mujoco_output_model_14000/
# since that dir is bind-mounted straight into the container -- no copy-out
# step needed.
#
# Fully self-contained: if the holomotion_sim2sim container doesn't exist
# (fresh machine, or it got removed), this bootstraps it via
# setup_container.sh before running -- no manual docker setup required.
#
# Heads up: this Jetson has 16GB shared RAM/GPU across everything. If the
# robot's perception (g1-detect/g1-pose/g1-hands) is running, pause it first
# or this can OOM mid-run:
#   docker stop g1-detect g1-pose g1-hands
#   ./motion/sim/run_holomotion.sh <name>
#   docker start g1-detect g1-pose g1-hands

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=holomotion_sim2sim

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container '$CONTAINER' doesn't exist yet -- setting it up..."
    "$HERE/setup_container.sh"
fi

status="$(docker inspect "$CONTAINER" --format '{{.State.Status}}')"
if [[ "$status" != "running" ]]; then
    echo "Starting $CONTAINER (was: $status)..."
    docker start "$CONTAINER" >/dev/null
fi

exec docker exec -i -e DISPLAY="${DISPLAY:-}" "$CONTAINER" \
    /root/miniconda3/envs/holomotion_deploy/bin/python \
    /workspace/sim/sim2sim.py "$@"
