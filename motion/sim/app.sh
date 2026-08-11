#!/usr/bin/env bash
# The "app": pick a motion from a menu, watch it live (in your browser --
# see README.md), land back on the menu for the next one. Stays open until
# you quit -- run this once and leave it running instead of retyping
# run_holomotion.sh each time.
#
# Usage: ./motion/sim/app.sh
# First run of the session prints a live-view URL -- open it in your
# browser and leave the tab open, it'll show whatever's currently playing.

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
            "$HERE/run_holomotion.sh" "$choice"
            rc=$?
            if [[ $rc -ne 0 ]]; then
                echo "(that run failed -- exit code $rc, see errors above)"
            fi
            break
        else
            echo "Invalid selection ($REPLY), try again."
        fi
    done
done
