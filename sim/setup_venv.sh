#!/usr/bin/env bash
# Creates an ISOLATED venv for the mujoco G1 sim (Track B / B0).
#
# Why isolated: mujoco drags in a modern numpy as a transitive dependency, and
# this Jetson's system python3 (3.8) has an ancient numpy==1.17.4 that other
# scripts (e.g. scripts/fused_odometry.py) depend on. `--system-site-packages`
# is deliberately NOT used, so nothing here can shadow or clobber the system
# numpy. Run this script once from the repo root (or anywhere -- paths are
# resolved relative to this file):
#
#   bash sim/setup_venv.sh
#
# Afterwards, run everything sim-related via the venv's own interpreter, e.g.:
#   sim/.venv/bin/python3 scripts/sim_runner.py --selftest
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"

echo "[setup_venv] creating venv at $VENV_DIR (no system-site-packages)..."
# NOTE: plain `python3 -m venv "$VENV_DIR"` FAILS on this Jetson image with
# "ensurepip is not available ... apt install python3.8-venv" -- that package
# isn't installed and there's no passwordless sudo here to add it. Workaround
# (verified 2026-07-23): build the venv WITHOUT pip's bundled ensurepip step,
# then bootstrap pip into it directly via get-pip.py. Functionally identical
# venv once done (isolated site-packages, no system-site-packages leak).
python3 -m venv --without-pip "$VENV_DIR"

if [ ! -x "$VENV_DIR/bin/pip" ]; then
    echo "[setup_venv] bootstrapping pip via get-pip.py (ensurepip unavailable)..."
    GET_PIP="$(mktemp /tmp/get-pip.XXXXXX.py)"
    curl -sS -o "$GET_PIP" https://bootstrap.pypa.io/pip/3.8/get-pip.py
    "$VENV_DIR/bin/python3" "$GET_PIP"
    rm -f "$GET_PIP"
fi

echo "[setup_venv] installing pinned requirements..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$HERE/requirements.txt"

echo "[setup_venv] done. venv numpy:"
"$VENV_DIR/bin/python3" -c "import numpy; print('  venv numpy  :', numpy.__version__)"
echo "  system numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '(system numpy not importable from this shell)')"

# -----------------------------------------------------------------------------
# G1 MJCF model (B1): NOT fetched here by default -- if sim/models/g1/ already
# exists in this repo (vendored), nothing to do. If you deleted it and need to
# re-fetch, the source is:
#
#   git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie /tmp/mujoco_menagerie_clone
#   cp -r /tmp/mujoco_menagerie_clone/unitree_g1 sim/models/g1
#
# See sim/models/README.md for the license note and exact provenance.
