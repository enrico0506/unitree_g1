#!/usr/bin/env bash
# Easy button: run a motion_library clip through HoloMotion sim2sim eval on
# the Jetson deploy container.
#
# Usage:
#   ./motion/sim/run_holomotion.sh wave_v2
#   ./motion/sim/run_holomotion.sh --list
#
# Watch it live in a browser while it runs (recommended -- see README.md's
# "Watching it live" section for why): every headless run auto-starts
# live_view_server.py if it isn't already running and prints the URL below.
# Open that in any browser on your laptop, no X11/ssh -X/MobaXterm needed.
#
# --gui also exists (the real interactive MuJoCo/GLFW window, X11-forwarded)
# but hits an OpenGL-version ceiling that indirect GLX can't clear on this
# hardware -- kept around in case you set up VirtualGL later, but the
# browser view above is the reliable path today.
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
# This Jetson has 16GB shared RAM/GPU across everything, and running this
# alongside live perception reliably OOMs (SIGKILL, exit code 247). So for
# any actual run (not --list), this pauses g1-detect/g1-pose/g1-hands first
# and restarts them after -- only the ones that were actually running, and
# even if the run fails or you Ctrl+C it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=holomotion_sim2sim
LIVE_VIEW_PORT=8098
PERCEPTION_CONTAINERS=(g1-detect g1-pose g1-hands)

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container '$CONTAINER' doesn't exist yet -- setting it up..."
    "$HERE/setup_container.sh"
fi

status="$(docker inspect "$CONTAINER" --format '{{.State.Status}}')"
if [[ "$status" != "running" ]]; then
    echo "Starting $CONTAINER (was: $status)..."
    docker start "$CONTAINER" >/dev/null
fi

is_headless_run=true
is_actual_run=false
for arg in "$@"; do
    [[ "$arg" == "--gui" ]] && is_headless_run=false
    [[ "$arg" != "--list" ]] && is_actual_run=true
done
[[ $# -eq 0 ]] && is_actual_run=false

paused_containers=()
perception_restored=false
restore_perception() {
    [[ "$perception_restored" == true ]] && return
    perception_restored=true
    if [[ ${#paused_containers[@]} -gt 0 ]]; then
        echo "Restoring perception (${paused_containers[*]})..."
        docker start "${paused_containers[@]}" >/dev/null 2>&1 || true
    fi
}

if [[ "$is_actual_run" == true ]]; then
    for c in "${PERCEPTION_CONTAINERS[@]}"; do
        c_status="$(docker inspect "$c" --format '{{.State.Status}}' 2>/dev/null || echo absent)"
        [[ "$c_status" == "running" ]] && paused_containers+=("$c")
    done
    if [[ ${#paused_containers[@]} -gt 0 ]]; then
        echo "Pausing perception (${paused_containers[*]}) to free RAM for this run..."
        docker stop "${paused_containers[@]}" >/dev/null
        trap restore_perception EXIT
    fi
fi

if [[ "$is_headless_run" == true && "$is_actual_run" == true ]]; then
    port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- 3>&-; }
    if ! port_open "$LIVE_VIEW_PORT"; then
        nohup python3 "$HERE/live_view_server.py" --port "$LIVE_VIEW_PORT" \
            >/tmp/motion_sim_live_view.log 2>&1 &
        disown
        sleep 0.3
    fi
    # Prefer the address you actually connected to sshd on (3rd field of
    # SSH_CONNECTION: "client_ip client_port server_ip server_port") --
    # this box has multiple interfaces (eth0 to the robot, wlan0 to your
    # laptop) and `hostname -I` has no way to know which one you can reach.
    # Falls back to hostname -I's first address when not run over SSH.
    ip="$(awk '{print $3}' <<< "${SSH_CONNECTION:-}")"
    if [[ -z "$ip" ]]; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    echo "Live view: http://${ip:-<jetson-ip>}:${LIVE_VIEW_PORT}/  (open this in your browser now)"
fi

if docker exec -i -e DISPLAY="${DISPLAY:-}" "$CONTAINER" \
    /root/miniconda3/envs/holomotion_deploy/bin/python \
    /workspace/sim/sim2sim.py "$@"; then
    rc=0
else
    rc=$?
fi

restore_perception
exit $rc
