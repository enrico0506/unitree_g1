"""gmr_pkl — Phase 3 checkpoint: load + VALIDATE GMR's ``gmr.pkl``.

Pipeline seat::

    clip -> ROMP -> pose_to_smpl -> GMR -> [THIS VALIDATOR] -> gmr_to_sonic_csv -> SONIC

GMR's ``scripts/smplx_to_robot.py --robot unitree_g1`` retargets the SMPL
sequence ``pose_to_smpl`` produced onto the 29-DOF G1 and pickles a *flat* dict
(``gmr.pkl``). Before Phase 4 turns that dict into SONIC's 6-CSV bundle, this
module LOADS the pkl, checks it HARD (shapes, dtypes, finiteness, fps, and an
upright sanity assert on the pelvis quaternion), renormalizes the quaternion,
and returns a clean typed dict. A bad pkl caught here is a robot that does NOT
move wrong three stages downstream — sim-first protects, this is the gate.

Import posture (FROZEN, same as the sibling glue): numpy + scipy ONLY
(``scipy.spatial.transform.Rotation``) + stdlib ``pickle`` — NO
torch / mujoco / isaaclab / GMR — so it runs unit-tested on the workstation
even though GMR itself only runs on the Orin. Verified env: numpy 2.4.4,
scipy 1.15.2.

Input contract (what GMR writes / this reads) — the 4 load-bearing keys:

    fps       float scalar   pkl frame rate (~30, may be non-integer)
    root_pos  (T,3) float    pelvis world position, METERS, Z-up
    root_rot  (T,4) float    pelvis quaternion, XYZW (scalar-LAST, scipy order)
    dof_pos   (T,29) float   joint angles, RADIANS, MJCF/mujoco order

``local_body_pos`` / ``link_body_list`` are ``None`` on GMR's smplx path and are
intentionally dropped here — that ``None`` is exactly WHY the Phase-4 body
arrays start root-only. The clean dict this returns is drop-in for the Phase-4
glue, which reads only ``fps / root_pos / root_rot / dof_pos``.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "GmrValidationError",
    "GmrData",
    "validate_gmr",
    "load_gmr_pkl",
    "upright_tilt_deg",
]

# --- pinned dimensions (SOURCE OF TRUTH: gmr.pkl layout) ----------------------
_NUM_DOF = 29     # dof_pos columns — G1_MJCF_ORDER length (Phase-4 joint_maps).
_POS_DIM = 3      # root_pos columns (x, y, z)
_QUAT_DIM = 4     # root_rot columns (XYZW)

# A quaternion whose norm drops below this is degenerate (cannot be normalized
# into a rotation) — raise rather than divide by ~0.
_QUAT_NORM_EPS = 1e-6

# --- upright sanity thresholds (median pelvis up-axis tilt, degrees) ----------
# VERIFY-on-robot: "up" is taken as the pelvis body-frame +Z. GMR applies a
# Y-up -> Z-up fix so its world is Z-up and an upright G1 at home has root_rot ~
# identity (pelvis +Z == world +Z, tilt 0). This is authoritative-on-PAPER only;
# confirm in GMR's robot_motion_viewer that the retargeted G1 stands upright
# (not sideways) before trusting the number. See PLAN "Up-axis" gotcha.
_UPRIGHT_WARN_DEG = 45.0     # >= this: warn (person leaning / possibly sideways)
_UPRIGHT_ERROR_DEG = 120.0   # >= this: raise (pelvis inverted — almost certainly
                             #   an up-axis or XYZW/WXYZ quaternion-order bug)

# world up-axis (Z-up, GMR convention after its Y-up->Z-up fix)
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class GmrValidationError(ValueError):
    """Raised LOUDLY on any gmr.pkl contract violation — a wrong shape, a
    non-finite value, a missing/absurd fps, a degenerate quaternion, or an
    inverted pelvis. Never fail silently: a corrupt pkl that slips through
    becomes a robot moving the wrong way three stages downstream."""


class GmrData(TypedDict):
    """Clean, validated gmr.pkl payload. All arrays are C-contiguous float64;
    ``root_rot`` is renormalized to unit XYZW. Drop-in for the Phase-4 glue
    (which reads ``fps / root_pos / root_rot / dof_pos``)."""

    fps: float           # > 0, finite
    root_pos: np.ndarray  # (T, 3) float64, meters, Z-up
    root_rot: np.ndarray  # (T, 4) float64, XYZW, unit norm
    dof_pos: np.ndarray   # (T, 29) float64, radians, MJCF order
    T: int                # frame count (== rows of every array)


# ---------------------------------------------------------------------------
# scalar + array field validators (each raises GmrValidationError, never silent)
# ---------------------------------------------------------------------------
def _validate_fps(value, source: str) -> float:
    """fps must be a finite positive scalar (GMR writes a python/np float)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        fps = float(arr)
    elif arr.size == 1:
        fps = float(arr.reshape(-1)[0])   # tolerate a 1-element array wrapper
    else:
        raise GmrValidationError(
            f"{source}: 'fps' must be a scalar, got shape {arr.shape}"
        )
    if not np.isfinite(fps):
        raise GmrValidationError(f"{source}: 'fps' is not finite ({fps!r})")
    if fps <= 0.0:
        raise GmrValidationError(f"{source}: 'fps' must be > 0, got {fps!r}")
    return fps


def _to_float_2d(value, name: str, ncols: int, source: str) -> np.ndarray:
    """Coerce to (T, ncols) float64, checking rank, column count, and
    finiteness with a message that says exactly which invariant failed."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise GmrValidationError(
            f"{source}: '{name}' must be 2-D (T, {ncols}), got ndim={arr.ndim} "
            f"shape={arr.shape}"
        )
    if arr.shape[1] != ncols:
        raise GmrValidationError(
            f"{source}: '{name}' has {arr.shape[1]} columns, expected {ncols} "
            f"({'XYZW quaternion' if ncols == _QUAT_DIM else 'columns'})"
        )
    if not np.isfinite(arr).all():
        raise GmrValidationError(f"{source}: '{name}' contains NaN/Inf")
    # C-contiguous copy so downstream slicing/reindexing is predictable.
    return np.ascontiguousarray(arr)


def _renormalize_quats(root_rot: np.ndarray, source: str) -> np.ndarray:
    """Renormalize each XYZW row to unit norm (GMR's own quats are ~unit but
    float drift accumulates; the Phase-4 Slerp needs unit quats). A near-zero
    norm is a genuinely degenerate rotation -> raise, don't divide by ~0."""
    norms = np.linalg.norm(root_rot, axis=1)
    if np.any(norms < _QUAT_NORM_EPS):
        bad = int(np.argmin(norms))
        raise GmrValidationError(
            f"{source}: 'root_rot' row {bad} has ~zero norm "
            f"({norms[bad]:.3e}); degenerate quaternion, cannot normalize"
        )
    return root_rot / norms[:, None]


# ---------------------------------------------------------------------------
# upright sanity check on the pelvis quaternion (yaw-invariant tilt)
# ---------------------------------------------------------------------------
def upright_tilt_deg(root_rot_xyzw: np.ndarray) -> np.ndarray:
    """Per-frame angle (deg) between the pelvis body +Z axis (rotated into the
    world by ``root_rot``) and world +Z. Yaw about world Z leaves this at ~0, so
    the metric is turn-invariant: 0 == upright, ~90 == lying sideways, ~180 ==
    upside-down. Input is (T,4) XYZW (scipy scalar-last)."""
    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    up_world = R.from_quat(q).apply(_WORLD_UP)     # pelvis +Z in world
    cos = np.clip(up_world @ _WORLD_UP, -1.0, 1.0)  # == up_world[:, 2]
    return np.degrees(np.arccos(cos))


def _check_upright(root_rot: np.ndarray, strict: bool, source: str) -> None:
    """Median-tilt verdict. Median (not mean/max) so a few noisy ROMP frames
    can't trip it, but a genuinely inverted sequence always does."""
    med = float(np.median(upright_tilt_deg(root_rot)))
    if med >= _UPRIGHT_ERROR_DEG:
        raise GmrValidationError(
            f"{source}: pelvis appears INVERTED (median up-axis tilt "
            f"{med:.1f} deg >= {_UPRIGHT_ERROR_DEG:.0f}). Almost certainly an "
            f"up-axis or XYZW/WXYZ quaternion-order bug upstream. "
            f"VERIFY-on-robot: confirm in GMR robot_motion_viewer."
        )
    if med >= _UPRIGHT_WARN_DEG:
        msg = (
            f"{source}: pelvis tilted (median up-axis tilt {med:.1f} deg >= "
            f"{_UPRIGHT_WARN_DEG:.0f}) — person may be leaning or sideways. "
            f"VERIFY-on-robot: eyeball robot_motion_viewer (upright, not lying)."
        )
        if strict:
            raise GmrValidationError(msg + " [strict_upright]")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def validate_gmr(
    gmr: dict,
    *,
    check_upright: bool = True,
    strict_upright: bool = False,
    source: str = "<dict>",
) -> GmrData:
    """Validate an in-memory gmr.pkl dict and return a clean typed dict.

    Parameters
    ----------
    gmr : dict
        The flat dict GMR pickled. Only ``fps / root_pos / root_rot / dof_pos``
        are read; other keys (``local_body_pos``, ``link_body_list``) are
        ignored — they are ``None`` on GMR's smplx path, which is exactly why
        Phase-4 body arrays start root-only.
    check_upright : bool
        Run the pelvis upright sanity assert (default on).
    strict_upright : bool
        Escalate the "leaning / sideways" WARNING (>= 45 deg median tilt) to a
        raised ``GmrValidationError``. The inverted case (>= 120 deg) always
        raises regardless.
    source : str
        Label used in error messages (a path, or "<dict>").

    Returns
    -------
    GmrData
        ``fps`` (float>0), ``root_pos`` (T,3) f64, ``root_rot`` (T,4) f64 unit
        XYZW, ``dof_pos`` (T,29) f64, ``T`` (int). Drop-in for the Phase-4 glue.

    Raises
    ------
    GmrValidationError
        On any missing key, wrong shape/dtype, non-finite value, absurd fps,
        degenerate quaternion, frame-count mismatch, or inverted pelvis.
    """
    if not isinstance(gmr, dict):
        raise GmrValidationError(
            f"{source}: gmr.pkl payload must be a dict, got {type(gmr).__name__}"
        )

    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [k for k in required if k not in gmr]
    if missing:
        raise GmrValidationError(
            f"{source}: gmr.pkl missing key(s) {missing} "
            f"(need {list(required)}; got {sorted(gmr.keys())})"
        )

    fps = _validate_fps(gmr["fps"], source)
    root_pos = _to_float_2d(gmr["root_pos"], "root_pos", _POS_DIM, source)
    root_rot = _to_float_2d(gmr["root_rot"], "root_rot", _QUAT_DIM, source)
    dof_pos = _to_float_2d(gmr["dof_pos"], "dof_pos", _NUM_DOF, source)

    # frame counts: T >= 1 and identical across the three time arrays.
    T = root_pos.shape[0]
    if T < 1:
        raise GmrValidationError(f"{source}: T == 0 (no frames in gmr.pkl)")
    if root_rot.shape[0] != T or dof_pos.shape[0] != T:
        raise GmrValidationError(
            f"{source}: frame-count mismatch — root_pos={root_pos.shape[0]}, "
            f"root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]} "
            f"(must all be equal)"
        )

    root_rot = _renormalize_quats(root_rot, source)

    if check_upright:
        _check_upright(root_rot, strict_upright, source)

    return GmrData(
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        T=T,
    )


def load_gmr_pkl(
    path,
    *,
    check_upright: bool = True,
    strict_upright: bool = False,
) -> GmrData:
    """Read a pickled gmr.pkl from ``path`` and validate it (see ``validate_gmr``).

    ``pickle.load`` on a GMR-written dict of numpy arrays needs no special
    globals, so this stays torch/mujoco-free."""
    path = Path(path)
    if not path.is_file():
        raise GmrValidationError(f"gmr.pkl not found: {path}")
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise GmrValidationError(f"{path}: not a readable pickle ({exc})") from exc
    return validate_gmr(
        raw,
        check_upright=check_upright,
        strict_upright=strict_upright,
        source=str(path),
    )


# ---------------------------------------------------------------------------
# CLI — the Phase-3 checkpoint you run right after GMR, before the Phase-4 seam.
# ---------------------------------------------------------------------------
def _summarize(data: GmrData, path: str) -> None:
    T, fps = data["T"], data["fps"]
    dur = T / fps
    pos = data["root_pos"]
    tilt = upright_tilt_deg(data["root_rot"])
    print(f"[gmr_pkl] OK: {path}")
    print(f"[gmr_pkl]   T={T} frames  fps={fps:.4f}  duration={dur:.3f}s")
    print(
        f"[gmr_pkl]   root_pos xyz range "
        f"x[{pos[:, 0].min():.3f},{pos[:, 0].max():.3f}] "
        f"y[{pos[:, 1].min():.3f},{pos[:, 1].max():.3f}] "
        f"z[{pos[:, 2].min():.3f},{pos[:, 2].max():.3f}] (m)"
    )
    print(
        f"[gmr_pkl]   dof_pos range [{data['dof_pos'].min():.3f},"
        f"{data['dof_pos'].max():.3f}] rad over {_NUM_DOF} joints"
    )
    print(
        f"[gmr_pkl]   pelvis up-axis tilt: median={np.median(tilt):.1f} "
        f"max={tilt.max():.1f} deg (0=upright, ~90=sideways, ~180=inverted)"
    )
    print(
        "[gmr_pkl] VERIFY-on-robot: also eyeball GMR robot_motion_viewer — "
        "recognizable motion, upright, right speed — before the SONIC seam."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.glue.gmr_pkl",
        description="Load + validate GMR's gmr.pkl (Phase-3 checkpoint).",
    )
    ap.add_argument("path", help="path to gmr.pkl")
    ap.add_argument(
        "--no-upright",
        action="store_true",
        help="skip the pelvis upright sanity assert",
    )
    ap.add_argument(
        "--strict-upright",
        action="store_true",
        help="raise (not warn) when the pelvis is >= 45 deg off upright",
    )
    args = ap.parse_args(argv)
    try:
        data = load_gmr_pkl(
            args.path,
            check_upright=not args.no_upright,
            strict_upright=args.strict_upright,
        )
    except GmrValidationError as exc:
        print(f"[gmr_pkl] INVALID: {exc}", flush=True)
        return 1
    _summarize(data, args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
