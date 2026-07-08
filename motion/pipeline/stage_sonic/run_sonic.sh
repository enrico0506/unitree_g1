#!/usr/bin/env bash
# =============================================================================
# pipeline/stage_sonic/run_sonic.sh  —  Phase 4 SONIC (whole-body track) runner
#
#   <csv_dir> (SONIC 6-CSV bundle)  ->  gear_sonic_deploy deploy.sh --motion-data
#                                       <csv_dir> sim  ->  G1 tracks it in MuJoCo
#
# Hands the 6-CSV bundle the Phase-4 glue (gmr_to_sonic_csv) produced to SONIC's
# on-Orin sim2sim loop. SONIC is the HARDEST stage: it is literally NVIDIA's G1
# on-device deploy target (JetPack 6, TensorRT 10.7, a ~91 MB ONNX built into a
# TensorRT engine ON-DEVICE). Nothing here runs on the workstation.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHERE THIS RUNS
#   >>> ON-DEVICE (Jetson Orin NX 16GB) ONLY. <<<
#   Needs the gear_sonic_deploy build + TensorRT 10.7 (stock JetPack 6.2 ships
#   10.3 — a version MISMATCH produces WRONG motion, dangerous on real hardware;
#   sim-first protects you). This script is NO-OP-SAFE off-device: if it can't
#   find SONIC's deploy.sh it prints setup guidance and exits 0 WITHOUT running.
#
# WHAT IS VERIFY-on-robot (untested — do NOT trust blindly)
#   The deploy.sh path + its exact CLI (`--motion-data <dir> sim`) are BEST-GUESS
#   from PLAN.md. DERISK ORDER (PLAN): FIRST spike SONIC's OWN example motions
#   through this same loop (`deploy.sh --motion-data <example_dir> sim`) — if that
#   doesn't run on the Orin, fix TensorRT/CUDA before anything downstream matters.
#   Only then feed a glue-produced bundle. Also confirm the observation_config
#   expects WXYZ body_quat + IsaacLab joint order (the glue's frozen assumptions).
#
# USAGE
#   pipeline/stage_sonic/run_sonic.sh <csv_dir>
#     <csv_dir>  the 6-CSV bundle dir from Phase 4: joint_pos.csv, joint_vel.csv,
#                body_pos.csv, body_quat.csv (+ body_lin_vel/body_ang_vel), and
#                metadata.txt — all 50 Hz, header row each, EQUAL row counts.
# =============================================================================
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # …/unitree_g1/motion
CONFIG_YAML="${MOTION_DIR}/config/pipeline.yaml"

# SONIC deploy checkout (override with SONIC_DIR=… / DEPLOY_SH=… ).
SONIC_DIR="${SONIC_DIR:-${MOTION_DIR}/third_party/GR00T-WholeBodyControl}"
DEPLOY_SH="${DEPLOY_SH:-${SONIC_DIR}/gear_sonic_deploy/deploy.sh}"   # VERIFY-on-robot

# ── args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <csv_dir>   (Phase-4 6-CSV bundle dir)" >&2
  exit 2
fi
CSV_DIR="$(cd "$1" && pwd)"

log() { printf '[run_sonic] %s\n' "$*"; }
die() { printf '[run_sonic] ERROR: %s\n' "$*" >&2; exit 1; }

# ── read TensorRT pin from config (banner only; default 10.7) ─────────────────
TRT_PIN="10.7"                                       # deploy.tensorrt
if [[ -f "${CONFIG_YAML}" ]] && python3 -c 'import yaml' 2>/dev/null; then
  _t="$(python3 - "${CONFIG_YAML}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("deploy", {}) or {}).get("tensorrt", "10.7"))
PY
)"
  [[ -n "${_t}" ]] && TRT_PIN="${_t}"
fi

# ── pre-flight: the 4 required CSVs exist + equal data-row counts ─────────────
# Light, pure-python (numpy-free) gate so a truncated/ragged bundle is caught
# HERE, not by SONIC three layers down. body_lin_vel/body_ang_vel are optional
# (glue write_body_vel flag); when present they must share the row count.
[[ -d "${CSV_DIR}" ]] || die "csv_dir does not exist: ${CSV_DIR}"
log "checking 6-CSV bundle in ${CSV_DIR}"
python3 - "${CSV_DIR}" <<'PY' || die "CSV bundle check failed — fix Phase-4 glue output before SONIC"
import sys, os
d = sys.argv[1]
required = ["joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv"]
optional = ["body_lin_vel.csv", "body_ang_vel.csv"]
missing = [f for f in required + ["metadata.txt"] if not os.path.isfile(os.path.join(d, f))]
if missing:
    sys.exit("missing required file(s): %s" % missing)

def datarows(path):  # total lines minus the 1 header line
    with open(path) as fh:
        return sum(1 for _ in fh) - 1

counts = {f: datarows(os.path.join(d, f)) for f in required
          if os.path.isfile(os.path.join(d, f))}
for f in optional:
    p = os.path.join(d, f)
    if os.path.isfile(p):
        counts[f] = datarows(p)
uniq = set(counts.values())
if len(uniq) != 1:
    sys.exit("unequal data-row counts across CSVs: %s" % counts)
n = uniq.pop()
if n < 1:
    sys.exit("CSV bundle has 0 data rows")
print("[run_sonic] bundle OK: %d files, %d data rows each (50 Hz)" % (len(counts), n))
PY

# ── NO-OP-SAFE guard: is SONIC's deploy.sh present? ──────────────────────────
# If not, we are almost certainly off-device (workstation). Print guidance and
# exit 0 WITHOUT running — the bundle check above already validated the input.
if [[ ! -x "${DEPLOY_SH}" && ! -f "${DEPLOY_SH}" ]]; then
  cat >&2 <<EOF
[run_sonic] SONIC deploy.sh not found — this stage is ON-DEVICE (Jetson Orin) ONLY.
[run_sonic] expected deploy script at: ${DEPLOY_SH}
[run_sonic] Nothing was run. To enable on the Orin:
[run_sonic]   1) git submodule add NVlabs/GR00T-WholeBodyControl under: ${SONIC_DIR}
[run_sonic]   2) install TensorRT ${TRT_PIN} (stock JetPack 6.2 ships 10.3 —
[run_sonic]      a mismatch produces WRONG motion), then build gear_sonic_deploy,
[run_sonic]   3) fetch model_encoder.onnx / model_decoder.onnx / observation_config.yaml.
[run_sonic] DERISK FIRST: run SONIC's OWN example motions through this loop before
[run_sonic]              any glue bundle: ${DEPLOY_SH} --motion-data <example_dir> sim
EOF
  exit 0
fi

# ── ON-DEVICE: hand the bundle to SONIC's sim2sim loop ───────────────────────
# VERIFY-on-robot: CLI is BEST-GUESS. `sim` selects the MuJoCo sim2sim target
# (real-hardware target is a separate, deferred, e-stop-gated run).
log "launching SONIC sim2sim: ${DEPLOY_SH} --motion-data ${CSV_DIR} sim  (TensorRT ${TRT_PIN})"
bash "${DEPLOY_SH}" --motion-data "${CSV_DIR}" sim \
  || die "SONIC deploy.sh failed (check TensorRT ${TRT_PIN} + ONNX + obs cfg — VERIFY-on-robot)"

log "done — SONIC tracked the bundle in sim."
log "VERIFY-on-robot (manual): the G1 reproduces the recorded motion, stays"
log "                          upright, and tracks at the right speed in MuJoCo."
