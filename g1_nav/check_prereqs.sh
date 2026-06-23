#!/usr/bin/env bash
# =============================================================================
# check_prereqs.sh -- Phase-0 prerequisite check for the G1 Nav2 migration.
#
# Reports what's present vs missing for the Nav2 stack on this robot (Foxy).
# Does NOT modify anything. Source-built deps (STVL, Patchwork++) only WARN --
# they are expected to be missing until Enrico builds them. Hard requirements
# (nav2 core, DWB, live topics) drive the final PASS/MISSING verdict.
#
# Run:  bash /home/unitree/projects/g1/g1_nav/check_prereqs.sh
# =============================================================================

# Source the isolated ROS 2 Foxy + mapping/Nav2 environment (domain 99).
source /home/unitree/g1_mapping_ws/env.sh 2>/dev/null || {
  echo "FATAL: could not source /home/unitree/g1_mapping_ws/env.sh"; exit 1; }

FAIL=0   # incremented by any hard-requirement miss

echo "============================================================"
echo " G1 Nav2 -- Phase-0 prerequisite check"
echo "============================================================"
echo "ROS_DISTRO      : ${ROS_DISTRO:-<unset>}"
echo "ROS_DOMAIN_ID   : ${ROS_DOMAIN_ID:-<unset>} (expect 99, isolated)"
echo "RMW             : ${RMW_IMPLEMENTATION:-<unset>}"
echo

# --- snapshot package + topic lists once (these calls are slowish) ----------
PKGS="$(ros2 pkg list 2>/dev/null)"
TOPICS="$(ros2 topic list -t 2>/dev/null)"

# ---------------------------------------------------------------------------
# 1. Nav2 core packages (HARD requirements)
# ---------------------------------------------------------------------------
echo "--- Nav2 core packages (required) ---"
for p in nav2_bringup nav2_costmap_2d nav2_controller nav2_planner \
         nav2_lifecycle_manager nav2_dwb_controller; do
  if grep -qx "$p" <<<"$PKGS"; then
    echo "  [ OK ]     $p"
  else
    echo "  [MISSING]  $p   <-- required"
    FAIL=$((FAIL + 1))
  fi
done
echo

# ---------------------------------------------------------------------------
# 2. Source-built layers (WARN only -- expected missing until built)
# ---------------------------------------------------------------------------
echo "--- Source-built deps (warn only) ---"
if grep -q "spatio_temporal_voxel_layer" <<<"$PKGS"; then
  echo "  [ OK ]     spatio_temporal_voxel_layer (STVL)"
else
  echo "  [ WARN ]   spatio_temporal_voxel_layer (STVL) not found"
  echo "             -> source-build the Foxy branch & overlay it."
  echo "             (only nav2_voxel_grid is present on this robot)"
fi
if grep -qi "patchwork" <<<"$PKGS"; then
  echo "  [ OK ]     patchwork (Patchwork++ ground seg)"
else
  echo "  [ WARN ]   patchwork (Patchwork++) not found"
  echo "             -> source-build patchwork-plusplus-ros & overlay it."
fi
echo

# ---------------------------------------------------------------------------
# 3. Live topics (HARD requirements -- the sensor + odom feeds)
# ---------------------------------------------------------------------------
echo "--- Live topics (required -- start the Livox driver + FAST-LIO first) ---"
for t in /livox/lidar /Odometry; do
  if grep -q "^${t} " <<<"$TOPICS"; then
    echo "  [ OK ]     $t   $(grep "^${t} " <<<"$TOPICS" | sed "s#^${t} ##")"
  else
    echo "  [MISSING]  $t   <-- required (is the Livox driver / FAST-LIO up?)"
    FAIL=$((FAIL + 1))
  fi
done
echo

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
  echo " RESULT: PASS -- hard requirements met."
  echo "         (Check WARN items above before enabling STVL/Patchwork.)"
  exit 0
else
  echo " RESULT: MISSING -- $FAIL hard requirement(s) not satisfied."
  echo "         Fix the [MISSING] items above before running run_nav.sh."
  exit 1
fi
