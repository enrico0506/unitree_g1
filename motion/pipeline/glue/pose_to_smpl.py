"""pose_to_smpl — Phase 2 glue: per-frame ROMP output -> GMR-ready SMPL sequence.

Pipeline seat:  ``clip.mp4 -> ROMP -> [THIS GLUE] -> GMR -> SONIC``.

This module is a PURE-ARRAY reshaper. It imports with **numpy + scipy only**
(no torch / smplx / ROMP / GMR), so it runs and is unit-tested on the
workstation even though ROMP itself only runs on the Orin. It never touches a
body model; it just packs per-frame SMPL parameters that ROMP already regressed
into the exact AMASS-style npz that GMR's ``scripts/smplx_to_robot.py`` reads.

Verified env: numpy 2.4.4, scipy 1.15.2 (``Rotation`` + ``Slerp`` available).

What it does, per the FROZEN CONTRACT (motion/PLAN.md Phase 2):
  READ  : select a subject, split ROMP's 72-dim (24-joint) axis-angle theta into
          global_orient(3) + body_pose(63) [drop the 6 zero hand dims], grab
          betas(10) and a weak monocular translation(3).
  FIX   : rotate root orientation + translation from ROMP camera frame into
          GMR's Z-up world frame (GMR's smplx path applies NO up-axis fix, so we
          must). Behind a VERIFY-on-robot guard.
  SMOOTH: SLERP low-pass on the root orientation ONLY (body_pose/trans untouched)
          to kill the per-frame "flips" ROMP emits on spins (quaternion double
          cover + HF yaw jitter). See ``_slerp_lowpass`` for the frozen algo.
  WRITE : the exact 6-key npz GMR consumes: root_orient, pose_body, trans,
          betas, gender, mocap_frame_rate.

Honest note (from PLAN): ROMP's monocular root TRANSLATION is weak/approximate
and global spins are the roughest part. ``zero_transl=True`` is the escape hatch
if the monocular trans destabilizes GMR.

Every version-uncertain assumption sits behind a ``# GUARD / VERIFY-on-robot``
comment — none of the real-ROMP-format guesses have been checked against a real
simple_romp run yet (ROMP cannot run on this workstation).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

__all__ = [
    "RompFormatError",
    "ROMP_CAM_TO_ZUP",
    "pose_to_smpl",
    "save_smpl_sequence",
]

# Guard against a divide-by-tiny scale when deriving trans from ROMP's [s,tx,ty].
_EPS = 1e-6

# The 6 keys GMR reads. Anything else in the npz is silently ignored by GMR, so
# we write EXACTLY these — no more, no less.
_OUT_KEYS = ("root_orient", "pose_body", "trans", "betas", "gender", "mocap_frame_rate")


class RompFormatError(ValueError):
    """Raised LOUDLY whenever a ROMP frame / packed sequence fails a shape,
    dtype, or key contract. Never fail silently: a bad pose that slips through
    here becomes a robot standing sideways three stages downstream."""


# ---------------------------------------------------------------------------
# Up-axis fix.  ROMP is camera-relative; GMR's smplx path applies NO up-axis
# correction, so the glue must hand GMR a Z-up world sequence.
# ---------------------------------------------------------------------------
# R_fix = R_x(-90 deg). Maps ROMP camera axes (X-right, Y-down, Z-forward) into
# GMR world Z-up: down -> -Z, forward -> +Y, head -> +Z. This is the AMASS
# Y-up->Z-up matrix [[1,0,0],[0,0,-1],[0,1,0]] composed with aitviewer's ROMP
# R_x(180 deg), which collapses to R_x(-90 deg).
#
# GUARD / VERIFY-on-robot: this matrix is a FROZEN CANDIDATE, untested against a
# real GMR run. Tune it by loading one GMR-shipped AMASS ``*_stageii.npz``,
# matching its up-axis, and confirming the retargeted G1 stands UPRIGHT in
# robot_motion_viewer. Do NOT trust this matrix untested.
ROMP_CAM_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

# Named up-axis options, so ``up_axis_fix`` stays a readable constant not a magic
# 3x3. "identity" is here only as an explicit VERIFY escape (no rotation).
_UP_AXIS_MATRICES = {
    "romp_cam_to_zup": ROMP_CAM_TO_ZUP,
    "identity": np.eye(3, dtype=np.float64),
}


def _up_axis_matrix(name: str) -> np.ndarray:
    try:
        return _UP_AXIS_MATRICES[name]
    except KeyError:
        raise RompFormatError(
            f"unknown up_axis_fix {name!r}; expected one of "
            f"{sorted(_UP_AXIS_MATRICES)}"
        )


# ---------------------------------------------------------------------------
# Subject selection + per-frame extraction. ALL version guards live in here.
# ---------------------------------------------------------------------------
def _pick(arr, idx: int) -> np.ndarray:
    """Return one subject's flat vector from a frame value that may still carry
    the leading person axis N (shape (N, D)) OR be already collapsed to (D,).

    ndim >= 2  -> index the leading person axis with ``idx`` then flatten.
    ndim == 1  -> already one subject; flatten (a no-op) and return.
    """
    a = np.asarray(arr)
    if a.ndim >= 2:
        a = a[idx]
    return np.asarray(a, dtype=np.float32).reshape(-1)


def _infer_n(frame: dict) -> int:
    """Best-effort person-count from the first 2-D pose/betas/cam array. A 1-D
    array means the frame is already collapsed to a single subject -> N == 1."""
    for key in ("smpl_thetas", "poses", "smpl_betas", "betas", "cam", "cam_trans"):
        if key in frame and frame[key] is not None:
            a = np.asarray(frame[key])
            if a.ndim >= 2:
                return int(a.shape[0])
            if a.ndim == 1:
                return 1
    return 0


def _select_subject(frame: dict) -> int:
    """Pick which detected person to keep. N>1 -> most-confident (center_confs)
    or earliest track (min track_id); N==1 -> 0; N==0 -> raise.

    NOTE the runner already collapses N->1 (largest/most-confident) before the
    glue sees a frame, so this is a defensive fallback, not the main path.
    """
    confs = frame.get("center_confs")
    if confs is not None:
        confs = np.asarray(confs).reshape(-1)
        if confs.size == 0:
            raise RompFormatError("frame has N==0 people (empty center_confs)")
        return int(np.argmax(confs))

    tids = frame.get("track_ids")
    if tids is not None:
        tids = np.asarray(tids).reshape(-1)
        if tids.size == 0:
            raise RompFormatError("frame has N==0 people (empty track_ids)")
        # earliest / most-stable track wins when no confidences are available.
        return int(np.argmin(tids))

    n = _infer_n(frame)
    if n == 0:
        raise RompFormatError("frame has N==0 people (no detections in frame)")
    return 0


def _get_theta(frame: dict, idx: int) -> np.ndarray:
    """Return the canonical (72,) 24-joint SMPL axis-angle for subject ``idx``.

    GUARD (a) — thetas key order / VERIFY-on-robot:
        smpl_thetas (the key simple_romp actually writes)  ->
        poses (some ROMP forks) ->
        assemble concat(global_orient(3), body_pose).
    GUARD (b) — body_pose 63-vs-69:
        canonical SMPL body_pose is 63 (21 joints); some stacks emit 69
        (23 joints incl. the two always-zero hand joints) -> slice to 63.
        Whatever path we take, the FINAL theta must be length 72 or we raise.
    """
    if "smpl_thetas" in frame:                       # GUARD (a): primary key
        theta = _pick(frame["smpl_thetas"], idx)
    elif "poses" in frame:                           # GUARD (a): alt fork key
        theta = _pick(frame["poses"], idx)
    elif "global_orient" in frame and "body_pose" in frame:
        go = _pick(frame["global_orient"], idx)      # (3,)
        bp = _pick(frame["body_pose"], idx)          # (63,) or (69,)
        if bp.shape[0] == 69:                        # GUARD (b): 69 -> 63
            bp = bp[:63]
        theta = np.concatenate([go, bp])             # 3 + 63 = 66
    else:
        raise RompFormatError(
            "frame exposes no pose: need one of smpl_thetas / poses / "
            "(global_orient + body_pose)"
        )

    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    if theta.shape[0] == 66:
        # Assembled from a 63-dim body_pose: restore the 6 hand slots ROMP
        # always zeroes, so the 72-dim split below is uniform across all paths.
        theta = np.concatenate([theta, np.zeros(6, dtype=np.float32)])

    if theta.shape[0] != 72:                         # GUARD (b): require 72
        raise RompFormatError(
            f"theta length {theta.shape[0]} != 72 "
            f"(expected 24-joint SMPL axis-angle)"
        )
    return theta


def _get_betas(frame: dict, idx: int) -> np.ndarray:
    """Return subject ``idx`` betas as (10,).

    GUARD (c) — betas key + dim: smpl_betas -> betas; ROMP=10, BEV=11. An
    11-dim betas is the tell-tale of BEV output, not ROMP -> raise.
    """
    if "smpl_betas" in frame:
        betas = _pick(frame["smpl_betas"], idx)
    elif "betas" in frame:
        betas = _pick(frame["betas"], idx)
    else:
        raise RompFormatError("frame exposes no betas (need smpl_betas or betas)")

    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] == 11:
        raise RompFormatError(
            "betas has 11 dims -> looks like BEV output, not ROMP (expected 10)"
        )
    if betas.shape[0] != 10:
        raise RompFormatError(f"betas dim {betas.shape[0]} != 10")
    return betas


def _get_transl(frame: dict, idx: int) -> np.ndarray:
    """Return subject ``idx`` root translation as (3,). Monocular and WEAK.

    GUARD (e) — two code paths + monocular weakness:
        prefer cam_trans (a direct 3D root trans) ->
        else derive from cam=[s, tx, ty]:  [tx/s, ty/s, 1/s] * 2.0
            (guard abs(s) > eps, else zeros) ->
        else zeros.
    The caller's ``zero_transl`` escape hatch overrides all of this with zeros.
    """
    if "cam_trans" in frame:
        transl = _pick(frame["cam_trans"], idx)
        if transl.shape[0] != 3:
            raise RompFormatError(f"cam_trans dim {transl.shape[0]} != 3")
        return transl.astype(np.float32)

    if "cam" in frame:
        cam = _pick(frame["cam"], idx)               # [s, tx, ty]
        if cam.shape[0] != 3:
            raise RompFormatError(f"cam dim {cam.shape[0]} != 3 (expected [s,tx,ty])")
        s = float(cam[0])
        if abs(s) > _EPS:
            return (np.array([cam[1] / s, cam[2] / s, 1.0 / s],
                             dtype=np.float32) * 2.0)
        # Degenerate scale -> a divide would blow up; fall back to zeros.
        return np.zeros(3, dtype=np.float32)

    return np.zeros(3, dtype=np.float32)


def _extract_frame(frame: dict):
    """ONE subject -> (theta72:(72,), betas10:(10,), transl3:(3,)). All the
    version guards funnel through here so the sequence loop stays clean."""
    idx = _select_subject(frame)
    theta = _get_theta(frame, idx)
    betas = _get_betas(frame, idx)
    transl = _get_transl(frame, idx)

    # GUARD (d) — hand-zero: WARN (not fatal). SMPL joints 22,23 (theta[66:72])
    # are hands and ROMP always regresses them to 0; a non-zero here means the
    # key/layout assumption is off. VERIFY-on-robot.
    if not np.allclose(theta[66:72], 0.0, atol=1e-4):
        warnings.warn(
            "theta[66:72] (SMPL hand joints) is not ~0 — ROMP normally zeroes "
            "these. Check the smpl_thetas layout on-robot.",
            RuntimeWarning,
            stacklevel=2,
        )
    return theta, betas, transl


# ---------------------------------------------------------------------------
# Root-orientation SLERP low-pass (kills per-frame flips). Root ONLY.
# ---------------------------------------------------------------------------
def _hemisphere_align(quat_xyzw: np.ndarray) -> np.ndarray:
    """Fix the quaternion double cover: q and -q are the SAME rotation, but ROMP
    regresses each frame's root independently and can emit either sign. Force
    every frame onto the same hemisphere as its predecessor so the trajectory is
    one continuous short-arc curve on S^3 (no naive-average long-way swings).

    Input/return: (T, 4) XYZW. Pure numpy — independent of any scipy quaternion
    canonicalization, so the caller controls sign continuity explicitly.
    """
    q = np.array(quat_xyzw, dtype=np.float64, copy=True)
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0.0:
            q[t] = -q[t]
    return q


def _slerp_lowpass(root_aa: np.ndarray, window: int) -> np.ndarray:
    """(T,3) axis-angle -> (T,3) axis-angle, SLERP moving-average low-pass.

    FROZEN algorithm (param = window, odd, default 7; <=1 disables):
      1) axis-angle -> quaternion  q = R.from_rotvec(root_aa).as_quat()  (T,4 XYZW)
      2) HEMISPHERE-ALIGN q (double-cover fix, ``_hemisphere_align``).
      3) WINDOWED GEODESIC (Karcher) MEAN — a true, order-independent moving average.
         For output t, window = clip([t-w, t+w], 0, T-1), w=(window-1)//2; the smoothed
         rotation is the equal-weight geodesic mean of the window's quaternions via
         scipy ``Rotation.mean()`` (minimizes the chordal L2 on S^3). Because the window
         is hemisphere-aligned (step 2), the mean can't split across the double cover, so
         it stays inside the window arc and cannot introduce NEW flips.
      4) quaternion -> axis-angle.
    """
    root_aa = np.asarray(root_aa, dtype=np.float64)
    T = root_aa.shape[0]
    if window is None or window <= 1 or T <= 2:
        return root_aa.astype(np.float32)

    w = (window - 1) // 2
    q = R.from_rotvec(root_aa).as_quat()             # (T,4) XYZW  [step 1]
    q = _hemisphere_align(q)                          # [step 2]

    q_smooth = np.empty_like(q)
    for t in range(T):                                # [step 3]
        lo = max(0, t - w)
        hi = min(T - 1, t + w)
        # Equal-weight Karcher mean over the window: order-independent + unbiased
        # (unlike an incremental Slerp fold). Window is hemisphere-aligned above.
        q_smooth[t] = R.from_quat(q[lo:hi + 1]).mean().as_quat()

    return R.from_quat(q_smooth).as_rotvec().astype(np.float32)  # [step 4]


# ---------------------------------------------------------------------------
# betas reduction over the sequence
# ---------------------------------------------------------------------------
def _reduce_betas(betas_all: np.ndarray, policy: str) -> np.ndarray:
    """(T,10) per-frame betas -> a single robust (10,) vector."""
    if policy == "median":
        return np.median(betas_all, axis=0).astype(np.float32)
    if policy == "mean":
        return np.mean(betas_all, axis=0).astype(np.float32)
    if policy == "first":
        return betas_all[0].astype(np.float32)
    raise RompFormatError(
        f"unknown betas_policy {policy!r}; expected median|mean|first"
    )


# ---------------------------------------------------------------------------
# fps from clip.json
# ---------------------------------------------------------------------------
def _fps_from_clip_json(clip_json_path) -> float:
    """measured_fps -> true_fps -> fps.

    GUARD (h): the recorder writes ``measured_fps``; PLAN Phase-2 text calls it
    ``true_fps`` (which does NOT exist in the recorder output). Read measured_fps
    first, fall back to PLAN's true_fps, then a bare fps.
    """
    with open(clip_json_path) as f:
        doc = json.load(f)
    for key in ("measured_fps", "true_fps", "fps"):
        val = doc.get(key)
        if val is not None:
            return float(val)
    raise RompFormatError(
        f"{clip_json_path}: no measured_fps / true_fps / fps field"
    )


# ---------------------------------------------------------------------------
# Runtime shape validation — fail LOUDLY, never silently.
# ---------------------------------------------------------------------------
def _validate_sequence(seq: dict, expected_T: int) -> None:
    ro = seq["root_orient"]
    pb = seq["pose_body"]
    tr = seq["trans"]
    be = seq["betas"]

    if ro.shape != (expected_T, 3):
        raise RompFormatError(f"root_orient shape {ro.shape} != {(expected_T, 3)}")
    if pb.shape != (expected_T, 63):
        raise RompFormatError(f"pose_body shape {pb.shape} != {(expected_T, 63)}")
    if tr.shape != (expected_T, 3):
        raise RompFormatError(f"trans shape {tr.shape} != {(expected_T, 3)}")
    if be.shape != (16,):
        raise RompFormatError(f"betas shape {be.shape} != (16,)")

    for name, arr in (("root_orient", ro), ("pose_body", pb),
                      ("trans", tr), ("betas", be)):
        if arr.dtype != np.float32:
            raise RompFormatError(f"{name} dtype {arr.dtype} != float32")
        # NaN/Inf (e.g. from a bad monocular trans divide) must never pass.
        if not np.isfinite(arr).all():
            raise RompFormatError(f"{name} contains NaN/Inf")

    if np.ndim(seq["gender"]) != 0:
        raise RompFormatError("gender must be a 0-d array")
    if np.ndim(seq["mocap_frame_rate"]) != 0:
        raise RompFormatError("mocap_frame_rate must be a 0-d array")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pose_to_smpl(
    romp_frames: list,
    fps: float,
    *,
    betas_policy: str = "median",
    smooth_window: int = 7,
    up_axis_fix: str = "romp_cam_to_zup",
    zero_transl: bool = False,
    gender: str = "neutral",
) -> dict:
    """Pack per-frame ROMP dicts (in clip order) into an in-memory GMR-ready
    SMPL sequence.

    Parameters
    ----------
    romp_frames : list[dict]
        One ``np.savez(...)['results'][()]`` dict PER FRAME, in clip order. Each
        value may still carry the leading person axis N, or already be collapsed
        to one subject.
    fps : float
        Source fps == ``clip.json["measured_fps"]`` (PLAN calls it true_fps).
    betas_policy : {"median","mean","first"}
        Sequence reduction of per-frame betas (median = robust default).
    smooth_window : int
        Odd SLERP low-pass window in frames; <=1 disables root smoothing.
    up_axis_fix : {"romp_cam_to_zup","identity"}
        Named ROMP-cam -> GMR-world rotation (VERIFY-on-robot).
    zero_transl : bool
        Force trans=0 (honest-note escape hatch when monocular trans is unstable).
    gender : str
        ROMP is gender-neutral; pass through as-is.

    Returns
    -------
    dict with keys root_orient(T,3), pose_body(T,63), trans(T,3), betas(16,),
    gender(0-d), mocap_frame_rate(0-d) — the exact 6 keys GMR reads.
    """
    if len(romp_frames) == 0:
        raise RompFormatError("romp_frames is empty (no frames to pack)")

    T = len(romp_frames)
    thetas = np.empty((T, 72), dtype=np.float32)
    betas_all = np.empty((T, 10), dtype=np.float32)
    transl_all = np.empty((T, 3), dtype=np.float32)

    # --- READ: per-frame extraction (all version guards inside _extract_frame)
    for t, frame in enumerate(romp_frames):
        theta, betas, transl = _extract_frame(frame)
        thetas[t] = theta
        betas_all[t] = betas
        transl_all[t] = transl

    # The 72 -> (3 + 63) split; DROP theta[66:72] (SMPL hand joints, ROMP == 0).
    root_aa_cam = thetas[:, 0:3]          # (T,3) pelvis/root, camera frame
    pose_body = thetas[:, 3:66].copy()    # (T,63) SMPL joints 1..21 (untouched)

    # --- FIX: rotate root + trans from ROMP camera frame into GMR world (Z-up),
    # per frame, BEFORE smoothing. GMR's smplx path applies NO up-axis fix.
    R_fix = _up_axis_matrix(up_axis_fix)                  # (3,3)
    R_fix_rot = R.from_matrix(R_fix)
    # root_R_world = R_fix @ from_rotvec(root_aa)  (single * stack broadcasts)
    root_aa_world = (R_fix_rot * R.from_rotvec(root_aa_cam)).as_rotvec()
    root_aa_world = root_aa_world.astype(np.float32)      # (T,3)

    if zero_transl:
        trans_world = np.zeros((T, 3), dtype=np.float32)
    else:
        # trans_world = R_fix @ transl  (apply to every frame)
        trans_world = (R_fix @ transl_all.T).T.astype(np.float32)

    # --- SMOOTH: SLERP low-pass, root ONLY, on the world-frame axis-angle.
    root_smooth = _slerp_lowpass(root_aa_world, smooth_window)  # (T,3)

    # --- betas: reduce over sequence, then pad 10 -> 16 (GUARD g: GMR/AMASS
    # betas are 16; ROMP gives 10 -> zero-pad the 6 extra shape dims).
    betas10 = _reduce_betas(betas_all, betas_policy)
    betas16 = np.concatenate([betas10, np.zeros(6, dtype=np.float32)]).astype(np.float32)

    seq = {
        "root_orient": root_smooth.astype(np.float32),          # (T,3)
        "pose_body": pose_body.astype(np.float32),              # (T,63)
        "trans": trans_world.astype(np.float32),                # (T,3)
        "betas": betas16,                                       # (16,)
        "gender": np.array(gender),                             # 0-d
        # GMR gotcha carried through: it hardcodes tgt_fps=30 and only
        # SLERP-downsamples when mocap_frame_rate > 30; <=30 keeps all frames
        # but stamps the output 30fps. Key MUST be mocap_frame_rate (underscore).
        "mocap_frame_rate": np.array(float(fps), dtype=np.float32),  # 0-d
    }
    _validate_sequence(seq, T)
    return seq


def save_smpl_sequence(seq: dict, out_path: str | os.PathLike) -> Path:
    """Write ONLY the 6 keys GMR reads to ``out_path`` as an npz. Returns the
    resolved Path actually written (np.savez appends .npz if absent)."""
    missing = [k for k in _OUT_KEYS if k not in seq]
    if missing:
        raise RompFormatError(f"sequence missing keys {missing}; cannot save")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        root_orient=seq["root_orient"],
        pose_body=seq["pose_body"],
        trans=seq["trans"],
        betas=seq["betas"],
        gender=seq["gender"],
        mocap_frame_rate=seq["mocap_frame_rate"],
    )
    # np.savez tacks on .npz iff the name lacks it.
    written = out_path if out_path.suffix == ".npz" else Path(str(out_path) + ".npz")
    return written.resolve()


# ---------------------------------------------------------------------------
# Frame loading for the CLI (runner already collapsed N->1, but we don't assume)
# ---------------------------------------------------------------------------
def _load_frame_npz(path) -> dict:
    """One ``frame_XXXXXX.npz`` (saved as ``np.savez(results=outputs)``) -> dict."""
    with np.load(path, allow_pickle=True) as data:
        if "results" in data:
            return data["results"][()]
        # Fallback: some dumps store the dict fields at top level.
        return {k: data[k] for k in data.files}


def _load_frames(romp_path) -> list:
    """Load per-frame ROMP dicts in clip order.

    ``--romp`` is either a directory of ``frame_*.npz`` (one dict each) or a
    single ``video_results.npz`` whose ``results`` holds a per-frame sequence.
    """
    romp_path = Path(romp_path)
    if romp_path.is_dir():
        files = sorted(glob.glob(str(romp_path / "frame_*.npz")))
        if not files:
            raise RompFormatError(f"no frame_*.npz found under {romp_path}")
        return [_load_frame_npz(f) for f in files]

    if romp_path.is_file():
        with np.load(romp_path, allow_pickle=True) as data:
            if "results" in data:
                res = data["results"][()]
                # results may be a list/array of per-frame dicts, or a single dict.
                if isinstance(res, (list, tuple, np.ndarray)):
                    return list(res)
                if isinstance(res, dict):
                    return [res]
        raise RompFormatError(
            f"{romp_path}: single-file ROMP npz must carry a 'results' "
            f"per-frame sequence"
        )

    raise RompFormatError(f"--romp path does not exist: {romp_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.glue.pose_to_smpl",
        description="Pack per-frame ROMP output into a GMR-ready SMPL npz.",
    )
    ap.add_argument("--romp", required=True,
                    help="dir of frame_*.npz OR a single video_results.npz")
    ap.add_argument("--clip-json", required=True,
                    help="clip.json; fps read from measured_fps (-> true_fps -> fps)")
    ap.add_argument("--out", required=True, help="output smpl.npz path")
    ap.add_argument("--betas-policy", default="median",
                    choices=["median", "mean", "first"])
    ap.add_argument("--smooth-window", type=int, default=7,
                    help="odd SLERP low-pass window; <=1 disables")
    ap.add_argument("--up-axis-fix", default="romp_cam_to_zup",
                    choices=sorted(_UP_AXIS_MATRICES))
    ap.add_argument("--zero-transl", action="store_true",
                    help="force trans=0 (monocular trans escape hatch)")
    ap.add_argument("--gender", default="neutral")
    args = ap.parse_args(argv)

    frames = _load_frames(args.romp)
    fps = _fps_from_clip_json(args.clip_json)
    seq = pose_to_smpl(
        frames,
        fps,
        betas_policy=args.betas_policy,
        smooth_window=args.smooth_window,
        up_axis_fix=args.up_axis_fix,
        zero_transl=args.zero_transl,
        gender=args.gender,
    )
    written = save_smpl_sequence(seq, args.out)

    # Runner-level cross-check (glue also asserts T == len(frames) internally).
    T = seq["root_orient"].shape[0]
    print(f"[pose_to_smpl] frames={T} fps={fps} -> {written}")
    print(f"[pose_to_smpl] keys={list(_OUT_KEYS)} "
          f"root_orient={seq['root_orient'].shape} "
          f"pose_body={seq['pose_body'].shape} "
          f"trans={seq['trans'].shape} betas={seq['betas'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
