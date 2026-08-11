#!/usr/bin/env bash
# (Re)create the holomotion_sim2sim container from scratch. run_holomotion.sh
# calls this automatically if the container is missing -- you normally don't
# need to run it by hand.
#
# Two-tier image strategy:
#   1. If a local `holomotion-sim2sim-provisioned` image exists (a snapshot
#      of a container that already had the extra pip deps installed), reuse
#      it -- instant, no reinstall.
#   2. Otherwise, start fresh from the upstream deploy image, install the
#      deps eval_mujoco_sim2sim.py needs that the deploy image doesn't ship
#      (it's built for real-time ROS control, not offline MuJoCo eval), and
#      commit the result as `holomotion-sim2sim-provisioned` so this never
#      has to happen again on this machine.
#
# Either way, the final container is recreated with all bind mounts fresh,
# so a stale mount (e.g. after a motion_library reorg) always self-heals too.
#
# Also: --network host, plus mounts of /tmp/.X11-unix and ~/.Xauthority (as
# /root/.ssh-xauth + XAUTHORITY env -- the base image ships /root/.Xauthority
# as a directory already, which blocks a plain file bind mount there), for
# --gui (the real interactive MuJoCo/GLFW window, X11-forwarded to your
# machine over `ssh -X`) -- see run_holomotion.sh's --gui path and
# motion/sim/README.md's X11 forwarding section.
#
# --network host specifically: this sshd's X11 forwarding only creates a TCP
# listener on 127.0.0.1:<6000+display> (X11UseLocalhost yes, the default) --
# no /tmp/.X11-unix socket ever gets created despite that mount. On the
# container's own (bridge) network, "localhost" in DISPLAY=localhost:10.0
# means the CONTAINER's loopback, not the host's, so the connection goes
# nowhere. --network host makes them the same loopback. Confirmed needed
# 2026-08-11: GLFW errored "X11: Failed to open display localhost:10.0"
# without this even with DISPLAY/XAUTHORITY correctly forwarded in.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOTION_ROOT="$REPO_ROOT/motion"
CONTAINER=holomotion_sim2sim
BASE_IMAGE="horizonrobotics/holomotion:v1.4.0-orin-jp5.1-arm64"
PROVISIONED_IMAGE="holomotion-sim2sim-provisioned:latest"
DEPLOY_PIP=/root/miniconda3/envs/holomotion_deploy/bin/pip
# Extras eval_mujoco_sim2sim.py needs that holomotion_deploy doesn't ship
# out of the box. torch/onnxruntime-gpu are already baked in -- don't touch.
# hydra-core installs --no-deps (its dep resolution wants to drag in a newer
# omegaconf than the one already pinned/working in this env).
HYDRA_DEP="hydra-core==1.3.2"
EXTRA_DEPS="mujoco ray pandas tabulate tqdm"

if docker image inspect "$PROVISIONED_IMAGE" >/dev/null 2>&1; then
    RUN_IMAGE="$PROVISIONED_IMAGE"
    echo "Using existing provisioned image: $PROVISIONED_IMAGE"
else
    echo "No provisioned image found. Building one from $BASE_IMAGE (first-time setup, a few minutes)..."
    docker rm -f holomotion_sim2sim_provision_tmp >/dev/null 2>&1 || true
    docker run -d --name holomotion_sim2sim_provision_tmp --entrypoint sleep "$BASE_IMAGE" infinity >/dev/null
    docker exec holomotion_sim2sim_provision_tmp "$DEPLOY_PIP" install --no-deps "$HYDRA_DEP"
    docker exec holomotion_sim2sim_provision_tmp "$DEPLOY_PIP" install $EXTRA_DEPS
    docker commit holomotion_sim2sim_provision_tmp "$PROVISIONED_IMAGE" >/dev/null
    docker rm -f holomotion_sim2sim_provision_tmp >/dev/null
    RUN_IMAGE="$PROVISIONED_IMAGE"
    echo "Built and saved $PROVISIONED_IMAGE for next time."
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" \
    --runtime nvidia \
    --restart unless-stopped \
    --network host \
    --entrypoint sleep \
    -e ACCEPT_EULA=Y \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e Deploy_CONDA_PREFIX=/root/miniconda3/envs/holomotion_deploy \
    -e CUDA_HOME=/usr/local/cuda-12.2 \
    -v "$MOTION_ROOT/holomotion:/workspace/holomotion" \
    -v "$MOTION_ROOT/holomotion_ckpt:/workspace/ckpt" \
    -v "$MOTION_ROOT/motion_library:/workspace/motions" \
    -v "$MOTION_ROOT/sim:/workspace/sim" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/root/.ssh-xauth:ro" \
    -e XAUTHORITY=/root/.ssh-xauth \
    "$RUN_IMAGE" \
    infinity >/dev/null

echo "Container '$CONTAINER' ready (image: $RUN_IMAGE)."
