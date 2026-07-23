#!/usr/bin/env bash
# Fetches the Unitree G1 MJCF model (+ meshes) from google-deepmind/mujoco_menagerie
# into sim/models/g1/. Track B / B1.
#
# WHY THIS IS A SEPARATE FETCH-AT-SETUP-TIME STEP AND NOT VENDORED INTO GIT:
# The model+meshes are ~38 MB of binary mesh assets. This repo already treats
# large downloadable model assets as "not source" (see .gitignore's *.pt/*.onnx/
# perception/*/models/ entries) -- re-fetch rather than bloat git history. The
# license (unitree_g1/LICENSE inside the upstream repo) is BSD-3-Clause, which
# explicitly permits redistribution; there's no license doubt here, this is a
# repo-hygiene choice, not a legal one.
#
# Usage:
#   bash sim/fetch_model.sh
#
# Result: sim/models/g1/{g1.xml, g1_with_hands.xml, scene*.xml, assets/, LICENSE, ...}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/models/g1"
CLONE_DIR="$(mktemp -d /tmp/mujoco_menagerie_clone.XXXXXX)"

echo "[fetch_model] cloning google-deepmind/mujoco_menagerie (shallow)..."
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$CLONE_DIR"

echo "[fetch_model] copying unitree_g1/ -> $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$CLONE_DIR/unitree_g1/." "$DEST/"

rm -rf "$CLONE_DIR"
echo "[fetch_model] done. Model root: $DEST/g1.xml"
echo "[fetch_model] license: $DEST/LICENSE (BSD-3-Clause, from Unitree Robotics)"
