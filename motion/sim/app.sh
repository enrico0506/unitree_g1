#!/usr/bin/env bash
# The "app": pick a motion from a menu, watch it live in MuJoCo, land back on
# the menu for the next one. Stays open until you quit -- run this once and
# leave it running instead of retyping run_holomotion.sh each time.
#
# Usage: ./motion/sim/app.sh
# Needs the same X11 forwarding setup as --gui -- see README.md's
# "Watching it live" section (X server on your end + ssh -X/-Y, or MobaXterm).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Loading motion list..."
mapfile -t motions < <("$HERE/run_holomotion.sh" --list | tail -n +2 | awk '{print $1}')
if [[ ${#motions[@]} -eq 0 ]]; then
    echo "No motions found in motion_library." >&2
    exit 1
fi
motions+=("Quit")

while true; do
    echo
    echo "=== HoloMotion sim2sim -- pick a motion to watch live ==="
    PS3="> "
    select choice in "${motions[@]}"; do
        if [[ "$choice" == "Quit" ]]; then
            echo "Bye."
            exit 0
        elif [[ -n "$choice" ]]; then
            "$HERE/run_holomotion.sh" "$choice" --gui || echo "(that run failed -- see errors above)"
            break
        else
            echo "Invalid selection ($REPLY), try again."
        fi
    done
done
