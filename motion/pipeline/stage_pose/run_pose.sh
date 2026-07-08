#!/usr/bin/env bash
# =============================================================================
# pipeline/stage_pose/run_pose.sh  —  Phase 2 POSE stage runner
#
#   clip.mp4  ->  ROMP (body SMPL, per-frame)  ->  [glue: pose_to_smpl]  ->  smpl.npz
#
# Turns one recorded clip into the exact 6-key AMASS-style SMPL npz that GMR's
# scripts/smplx_to_robot.py consumes in Phase 3. Two halves:
#   (A) ON-DEVICE  : run simple_romp on the Orin (needs L4T torch + the SMPL body
#                    model) -> per-frame results, collapsed to ONE subject/frame.
#   (B) PURE-PYTHON: pack + SLERP-smooth via `python -m pipeline.glue.pose_to_smpl`
#                    (numpy+scipy only) -> <job>/pose/smpl.npz.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHERE THIS RUNS
#   >>> ON-DEVICE (Jetson Orin NX 16GB) ONLY. <<<
#   ROMP (simple_romp) needs the L4T PyTorch wheel and the MPI-registration-gated
#   SMPL body model — NEITHER exists on the workstation, so ROMP CANNOT run here.
#   This script is therefore NO-OP-SAFE off-device: if it can't find ROMP it
#   prints setup guidance and exits 0 WITHOUT stopping any containers (so it can
#   never leave perception down on a box that can't finish the job).
#
# WHAT IS VERIFY-on-robot (untested — do NOT trust blindly)
#   The exact simple_romp CLI flags, its on-disk output layout, and its per-frame
#   dict keys are BEST-GUESS. Every such spot is tagged `# VERIFY-on-robot`.
#   First real run: confirm the ROMP output keys match what the glue's version
#   guards expect (smpl_thetas / smpl_betas / cam|cam_trans / center_confs).
#
# USAGE
#   pipeline/stage_pose/run_pose.sh <job_dir>
#     <job_dir>  a recorded job: must contain clip.mp4 + clip.json (Phase 1).
#   Writes: <job_dir>/pose/romp/frame_XXXXXX.npz  (per-frame, 1 subject)
#           <job_dir>/pose/smpl.npz               (GMR-ready, 6 keys)
#
# RAM (critical, 16GB unified): stops the live g1-pose/g1-hands/g1-detect
# containers for the duration and ALWAYS restarts them via an EXIT trap — even on
# failure or Ctrl-C.
# =============================================================================
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # …/unitree_g1/motion
CONFIG_YAML="${MOTION_DIR}/config/pipeline.yaml"

# ── args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <job_dir>   (job dir with clip.mp4 + clip.json)" >&2
  exit 2
fi
JOB_DIR="$(cd "$1" && pwd)"
CLIP_MP4="${JOB_DIR}/clip.mp4"
CLIP_JSON="${JOB_DIR}/clip.json"
ROMP_DIR="${JOB_DIR}/pose/romp"          # per-frame frame_XXXXXX.npz (1 subject)
ROMP_RAW="${JOB_DIR}/pose/romp_raw"      # simple_romp's own output (pre-collapse)
SMPL_NPZ="${JOB_DIR}/pose/smpl.npz"      # <- the glue's GMR-ready product

log() { printf '[run_pose] %s\n' "$*"; }
die() { printf '[run_pose] ERROR: %s\n' "$*" >&2; exit 1; }

# ── read config (pause_containers, smpl_dir, static_cam) ─────────────────────
# Defaults mirror config/pipeline.yaml so the script still works if PyYAML is
# absent. When PyYAML is present we read the live values (single source of truth).
PAUSE_CONTAINERS=(g1-pose g1-hands g1-detect)   # recreate.pause_containers
SMPL_DIR_REL="models/smpl"                        # models.smpl_dir
STATIC_CAM="true"                                 # pose.static_cam
if [[ -f "${CONFIG_YAML}" ]] && python3 -c 'import yaml' 2>/dev/null; then
  # shellcheck disable=SC2016
  _cfg="$(python3 - "${CONFIG_YAML}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
pc = (c.get("recreate", {}) or {}).get("pause_containers", []) or []
sd = (c.get("models", {}) or {}).get("smpl_dir", "models/smpl")
sc = (c.get("pose", {}) or {}).get("static_cam", True)
print(" ".join(pc))
print(sd)
print("true" if sc else "false")
PY
)"
  mapfile -t _cfg_lines <<<"${_cfg}"
  [[ -n "${_cfg_lines[0]:-}" ]] && read -r -a PAUSE_CONTAINERS <<<"${_cfg_lines[0]}"
  [[ -n "${_cfg_lines[1]:-}" ]] && SMPL_DIR_REL="${_cfg_lines[1]}"
  [[ -n "${_cfg_lines[2]:-}" ]] && STATIC_CAM="${_cfg_lines[2]}"
fi
SMPL_DIR="${MOTION_DIR}/${SMPL_DIR_REL}"
log "config: pause_containers=[${PAUSE_CONTAINERS[*]}] smpl_dir=${SMPL_DIR_REL} static_cam=${STATIC_CAM}"

# ── pre-flight ───────────────────────────────────────────────────────────────
[[ -f "${CLIP_MP4}"  ]] || die "no clip.mp4 in job dir: ${CLIP_MP4}"
[[ -f "${CLIP_JSON}" ]] || die "no clip.json in job dir: ${CLIP_JSON}"

# ── NO-OP-SAFE guard: is ROMP even installed? ────────────────────────────────
# If not, we are almost certainly off-device (workstation). Print guidance and
# exit 0 WITHOUT touching any container — never leave perception down here.
ROMP_OK=0
if command -v romp >/dev/null 2>&1; then
  ROMP_OK=1
elif python3 -c 'import romp' 2>/dev/null || python3 -c 'import simple_romp' 2>/dev/null; then
  ROMP_OK=1
fi
if [[ "${ROMP_OK}" -eq 0 ]]; then
  cat >&2 <<EOF
[run_pose] simple_romp not found — this stage is ON-DEVICE (Jetson Orin) ONLY.
[run_pose] ROMP needs the L4T PyTorch wheel + the MPI-gated SMPL body model.
[run_pose] Nothing was changed (no containers stopped). To enable on the Orin:
[run_pose]   1) install the NVIDIA L4T PyTorch wheel (JetPack 6),
[run_pose]   2) pip install simple_romp,
[run_pose]   3) place the SMPL body model under: ${SMPL_DIR}
[run_pose] The pure-python glue is still testable here:
[run_pose]   pytest motion/tests/test_pose_to_smpl.py
EOF
  exit 0
fi
[[ -d "${SMPL_DIR}" ]] || log "WARN: SMPL body model dir missing: ${SMPL_DIR} (ROMP will fail without it)"

# ── FREE RAM: stop live perception, ALWAYS restart on EXIT ───────────────────
# The trap fires on success, failure, or Ctrl-C so we never strand perception.
restart_containers() {
  local rc=$?
  log "restarting paused containers: ${PAUSE_CONTAINERS[*]}"
  for c in "${PAUSE_CONTAINERS[@]}"; do
    sudo docker start "${c}" >/dev/null 2>&1 || log "WARN: could not restart ${c}"
  done
  exit "${rc}"
}
trap restart_containers EXIT

log "stopping perception containers to free RAM: ${PAUSE_CONTAINERS[*]}"
for c in "${PAUSE_CONTAINERS[@]}"; do
  sudo docker stop "${c}" >/dev/null 2>&1 || log "WARN: ${c} not running (skip)"
done

# ── (A) ON-DEVICE: run simple_romp on clip.mp4 (video / tracking mode) ───────
# VERIFY-on-robot: flags + output layout are BEST-GUESS. simple_romp's video
# mode tracks a person across frames and can dump per-frame results. Confirm the
# real flag names and where it writes on the first Orin run.
mkdir -p "${ROMP_RAW}" "${ROMP_DIR}"
STATIC_FLAG=()
[[ "${STATIC_CAM}" == "true" ]] && STATIC_FLAG+=(--calib_frames)   # VERIFY-on-robot: static-cam hint
log "running simple_romp on ${CLIP_MP4} (video mode) -> ${ROMP_RAW}"
# VERIFY-on-robot: the exact CLI. Kept as a single documented invocation.
romp \
  --mode video \
  --input "${CLIP_MP4}" \
  --save_path "${ROMP_RAW}" \
  --smpl_path "${SMPL_DIR}" \
  --show_largest \
  "${STATIC_FLAG[@]}" \
  || die "simple_romp failed (check L4T torch + SMPL model + flags — all VERIFY-on-robot)"

# ── collapse N->1 subject/frame AT THE RUNNER (not the glue's math) ──────────
# Re-save each frame as np.savez(results=<one-subject dict>) into frame_XXXXXX.npz.
# VERIFY-on-robot: this reads ROMP's raw dump — adapt the loader to ROMP's actual
# output structure. Subject pick = argmax(center_confs), else min(track_ids).
log "collapsing ROMP output to 1 subject/frame -> ${ROMP_DIR}"
python3 - "${ROMP_RAW}" "${ROMP_DIR}" <<'PY'
import sys, glob, os
import numpy as np
raw_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

def pick(frame):
    confs = frame.get("center_confs")
    if confs is not None and np.asarray(confs).size:
        return int(np.argmax(np.asarray(confs).reshape(-1)))
    tids = frame.get("track_ids")
    if tids is not None and np.asarray(tids).size:
        return int(np.argmin(np.asarray(tids).reshape(-1)))
    return 0

def one_subject(frame, i):
    out = {}
    for k, v in frame.items():
        a = np.asarray(v)
        out[k] = a[i:i+1] if a.ndim >= 1 and a.shape[0] > i else a
    return out

# VERIFY-on-robot: ROMP video mode may write per-frame *.npz OR one results.npz
# holding a sequence. Handle both; adjust to the real layout as needed.
per_frame = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
frames = []
if len(per_frame) > 1:
    for f in per_frame:
        with np.load(f, allow_pickle=True) as d:
            frames.append(d["results"][()] if "results" in d
                          else {k: d[k] for k in d.files})
elif per_frame:
    with np.load(per_frame[0], allow_pickle=True) as d:
        res = d["results"][()] if "results" in d else {k: d[k] for k in d.files}
    frames = list(res) if isinstance(res, (list, tuple, np.ndarray)) else [res]
if not frames:
    sys.exit("no ROMP frames found in %r (VERIFY-on-robot: output layout)" % raw_dir)

for t, fr in enumerate(frames):
    np.savez(os.path.join(out_dir, "frame_%06d.npz" % t),
             results=one_subject(fr, pick(fr)))
print("[run_pose] wrote %d single-subject frames" % len(frames))
PY

# ── (B) PURE-PYTHON glue: pack + SLERP-smooth -> smpl.npz ────────────────────
# Runs as a module so `pipeline.glue.pose_to_smpl` resolves from MOTION_DIR.
log "packing SMPL sequence via pose_to_smpl glue -> ${SMPL_NPZ}"
( cd "${MOTION_DIR}" && python3 -m pipeline.glue.pose_to_smpl \
    --romp "${ROMP_DIR}" \
    --clip-json "${CLIP_JSON}" \
    --out "${SMPL_NPZ}" ) \
  || die "pose_to_smpl glue failed"

# ── Verify (no GMR yet) ──────────────────────────────────────────────────────
# 1) T == clip.json.frame_count  2) root turns smoothly (glue already SLERP-
# smoothed + validated shapes/finiteness). Upright check is VERIFY-on-robot and
# needs ROMP `joints` (head-above-pelvis) which the glue doesn't carry — do it
# here once the joints are available in the raw dump.
python3 - "${SMPL_NPZ}" "${CLIP_JSON}" <<'PY'
import sys, json
import numpy as np
smpl, clip = sys.argv[1], sys.argv[2]
d = np.load(smpl, allow_pickle=True)
T = d["root_orient"].shape[0]
fc = json.load(open(clip)).get("frame_count")
assert fc is None or T == fc, "T=%d != clip.json frame_count=%s" % (T, fc)
assert set(d.files) == {"root_orient","pose_body","trans","betas","gender","mocap_frame_rate"}
assert np.isfinite(d["root_orient"]).all() and np.isfinite(d["pose_body"]).all()
print("[run_pose] VERIFY ok: T=%d frames, fps=%.3f, 6 keys, finite." %
      (T, float(d["mocap_frame_rate"])))
print("[run_pose] VERIFY-on-robot (manual): eyeball the root turns smoothly "
      "(no per-frame flips) and the person is upright, not lying sideways.")
PY

log "done -> ${SMPL_NPZ}"
# trap restarts the paused containers on EXIT.
