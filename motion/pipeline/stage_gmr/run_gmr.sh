#!/usr/bin/env bash
# =============================================================================
# pipeline/stage_gmr/run_gmr.sh  —  Phase 3 GMR (retarget) stage runner
#
#   smpl.npz  ->  GMR smplx_to_robot.py --robot unitree_g1  ->  gmr.pkl
#              ->  [validate: pipeline.glue.gmr_pkl]  ->  Phase-4 seam
#
# Retargets the 6-key AMASS-style SMPL sequence pose_to_smpl produced (Phase 2)
# onto the 29-DOF Unitree G1, then VALIDATES the pkl GMR wrote before it reaches
# the SONIC seam. GMR retargeting runs ON-DEVICE (CPU, aarch64); the validation
# step is the pure-python gmr_pkl checkpoint (numpy+scipy) and runs anywhere.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHERE THIS RUNS
#   >>> The GMR retarget half is ON-DEVICE (Jetson Orin NX 16GB). <<<
#   GMR is CPU Python with native aarch64 wheels + the unitree_g1 MJCF asset —
#   NEITHER the GMR checkout nor its asset live on the workstation. This script
#   is therefore NO-OP-SAFE off-device: if it can't find GMR's smplx_to_robot.py
#   it prints setup guidance and exits 0 WITHOUT touching anything.
#
# WHAT IS VERIFY-on-robot (untested — do NOT trust blindly)
#   The exact smplx_to_robot.py flag names (--smplx_file / --robot / --tgt_fps /
#   --save_path) and its on-disk pkl layout are BEST-GUESS from PLAN.md. Every
#   such spot is tagged `# VERIFY-on-robot`. On the first Orin run, confirm the
#   real flags (GMR prints robot.robot_dof_names — that MUST equal G1_MJCF_ORDER)
#   and eyeball GMR's robot_motion_viewer: recognizable motion, UPRIGHT, right
#   speed. The gmr_pkl validator here is a programmatic tripwire, not a substitute
#   for that visual check.
#
# USAGE
#   pipeline/stage_gmr/run_gmr.sh <job_dir> [tgt_fps]
#     <job_dir>  a job that finished Phase 2: must contain pose/smpl.npz and
#                clip.json. tgt_fps defaults to the clip's TRUE (measured) fps —
#                pass it explicitly to override.
#   Writes: <job_dir>/gmr/gmr.pkl   (root_pos/root_rot XYZW/dof_pos MJCF/fps)
#
# fps NOTE (PLAN "fps" gotcha): GMR hardcodes tgt_fps=30 by default; we pass the
# clip's measured fps so the retarget rate matches the real capture rate. That
# fps is carried in the pkl and drives the Phase-4 exact-50 Hz resample.
# =============================================================================
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # …/unitree_g1/motion
CONFIG_YAML="${MOTION_DIR}/config/pipeline.yaml"

# GMR checkout (override with GMR_DIR=… ). third_party/GMR is the PLAN location.
GMR_DIR="${GMR_DIR:-${MOTION_DIR}/third_party/GMR}"
SMPLX_SCRIPT="${GMR_DIR}/scripts/smplx_to_robot.py"   # VERIFY-on-robot: real path

# ── args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <job_dir> [tgt_fps]   (job dir with pose/smpl.npz + clip.json)" >&2
  exit 2
fi
JOB_DIR="$(cd "$1" && pwd)"
TGT_FPS_ARG="${2:-}"
SMPL_NPZ="${JOB_DIR}/pose/smpl.npz"
CLIP_JSON="${JOB_DIR}/clip.json"
GMR_OUT_DIR="${JOB_DIR}/gmr"
GMR_PKL="${GMR_OUT_DIR}/gmr.pkl"

log() { printf '[run_gmr] %s\n' "$*"; }
die() { printf '[run_gmr] ERROR: %s\n' "$*" >&2; exit 1; }

# ── read robot name from config (single source of truth; default unitree_g1) ──
ROBOT="unitree_g1"                                  # robot.name
if [[ -f "${CONFIG_YAML}" ]] && python3 -c 'import yaml' 2>/dev/null; then
  _r="$(python3 - "${CONFIG_YAML}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("robot", {}) or {}).get("name", "unitree_g1"))
PY
)"
  [[ -n "${_r}" ]] && ROBOT="${_r}"
fi

# ── pre-flight ───────────────────────────────────────────────────────────────
[[ -f "${SMPL_NPZ}"  ]] || die "no pose/smpl.npz in job dir (run Phase 2 first): ${SMPL_NPZ}"
[[ -f "${CLIP_JSON}" ]] || die "no clip.json in job dir: ${CLIP_JSON}"

# ── resolve tgt_fps: CLI arg wins, else clip.json measured_fps -> true_fps -> fps ─
if [[ -n "${TGT_FPS_ARG}" ]]; then
  TGT_FPS="${TGT_FPS_ARG}"
else
  TGT_FPS="$(python3 - "${CLIP_JSON}" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
for k in ("measured_fps", "true_fps", "fps"):
    v = d.get(k)
    if v is not None:
        print(float(v)); break
else:
    sys.exit("clip.json has no measured_fps/true_fps/fps")
PY
)" || die "could not read tgt_fps from ${CLIP_JSON}"
fi
log "config: robot=${ROBOT}  tgt_fps=${TGT_FPS}  (from ${TGT_FPS_ARG:+CLI arg}${TGT_FPS_ARG:-clip.json})"

# ── NO-OP-SAFE guard: is GMR even present? ───────────────────────────────────
# If not, we are almost certainly off-device (workstation). Print guidance and
# exit 0 WITHOUT writing anything — the pure-python validator is still testable.
if [[ ! -f "${SMPLX_SCRIPT}" ]]; then
  cat >&2 <<EOF
[run_gmr] GMR not found — this retarget stage is ON-DEVICE (Jetson Orin) ONLY.
[run_gmr] expected GMR script at: ${SMPLX_SCRIPT}
[run_gmr] Nothing was changed. To enable on the Orin:
[run_gmr]   1) git submodule add the GMR repo under: ${GMR_DIR}
[run_gmr]   2) pip install -e GMR  (CPU, native aarch64 wheels),
[run_gmr]   3) place the unitree_g1 MJCF/asset GMR expects.
[run_gmr] The pure-python validator is still runnable/testable here:
[run_gmr]   pytest motion/tests/test_gmr_pkl.py
[run_gmr]   python -m pipeline.glue.gmr_pkl <some_gmr.pkl>
EOF
  exit 0
fi

# ── (A) ON-DEVICE: retarget SMPL -> 29-DOF G1 pkl ────────────────────────────
# VERIFY-on-robot: flag names are BEST-GUESS from PLAN.md. GMR applies its own
# Y-up -> Z-up fix internally, so the glue downstream does NOT rotate again.
mkdir -p "${GMR_OUT_DIR}"
log "retargeting ${SMPL_NPZ} -> ${GMR_PKL}  (robot=${ROBOT}, tgt_fps=${TGT_FPS})"
( cd "${GMR_DIR}" && python3 "${SMPLX_SCRIPT}" \
    --smplx_file "${SMPL_NPZ}" \
    --robot "${ROBOT}" \
    --tgt_fps "${TGT_FPS}" \
    --save_path "${GMR_PKL}" ) \
  || die "GMR smplx_to_robot.py failed (check flags + unitree_g1 asset — VERIFY-on-robot)"

[[ -f "${GMR_PKL}" ]] || die "GMR reported success but ${GMR_PKL} is missing"

# ── (B) VALIDATE the pkl (Phase-3 checkpoint) ────────────────────────────────
# Runs as a module so `pipeline.glue.gmr_pkl` resolves from MOTION_DIR. Hard-
# checks shapes (root_pos T×3 / root_rot T×4 XYZW / dof_pos T×29), fps, finite-
# ness, and asserts the pelvis is upright (not sideways / inverted).
log "validating gmr.pkl via pipeline.glue.gmr_pkl"
( cd "${MOTION_DIR}" && python3 -m pipeline.glue.gmr_pkl "${GMR_PKL}" ) \
  || die "gmr.pkl failed validation (see message above) — STOP before SONIC"

# ── Verify banner ────────────────────────────────────────────────────────────
log "done -> ${GMR_PKL}"
log "VERIFY-on-robot (manual): load ${GMR_PKL} in GMR robot_motion_viewer on the"
log "                          G1 — motion recognizable, UPRIGHT, right speed —"
log "                          before trusting the Phase-4 SONIC-CSV seam."
