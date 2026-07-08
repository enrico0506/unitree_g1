"""gmr_to_sonic_csv — Phase 4 glue: GMR ``gmr.pkl`` -> SONIC's 6-CSV bundle.

Pipeline seat: ``clip -> ROMP -> pose_to_smpl -> GMR -> [THIS GLUE] -> SONIC
-> MuJoCo G1``.  This is the make-or-break seam: it reindexes the 29 joints
from GMR's MJCF/mujoco order into SONIC's IsaacLab order (BY NAME — issue #78
guard, see ``joint_maps``), converts the root quaternion XYZW->WXYZ, resamples
everything to EXACTLY the control rate (50 Hz), finite-diffs velocities at
dt = 1/control_hz (0.02 s), and writes the SONIC CSV bundle with headers and
equal row counts.

PURE-ARRAY reshaper.  Imports **numpy + scipy ONLY** (``Rotation``, ``Slerp``)
plus stdlib (``csv`` is not used — we format ``%.6f`` by hand; ``pickle`` only
for the CLI).  NO torch / mujoco / isaaclab / GMR, so it runs unit-tested on the
workstation though GMR + SONIC themselves are on-device.  Verified env:
numpy 2.4.4, scipy 1.15.2.

FROZEN CONTRACT (motion/PLAN.md Phase 4):
  READ    : gmr['fps','root_pos','root_rot','dof_pos'] only (4 keys), each
            validated LOUD (GmrFormatError).  root_pos m Z-up, root_rot XYZW,
            dof_pos rad in MJCF order, fps ~30.
  REINDEX : dof MJCF->IsaacLab, BY NAME (dof[:, MJCF_TO_ISAAC]).
  QUAT    : root XYZW -> WXYZ ([:, [3,0,1,2]]).
  RESAMPLE: to EXACTLY control_hz.  N = floor((T-1)/fps * control_hz) + 1 (a
            uniform control_hz grid over the true sample span, so the last point
            never overshoots -> no end-velocity clamp).  Slerp for
            rotation, linear (np.interp) for joints/positions.  Endpoint clamp
            prevents extrapolation.  T==1 -> tile single frame to N rows.
  VEL     : finite-diff at dt = 1/control_hz (np.gradient: central interior,
            one-sided ends).  N==1 -> zeros.
  BODY    : ROOT-ONLY (B=1) — body_pos=root_pos, body_quat=root WXYZ,
            body_lin_vel/body_ang_vel from the resampled root.  Full multi-body
            FK needs the mujoco model (on-device) -> DEFERRED, flagged below.
  WRITE   : 6 CSVs + metadata.txt, headers on line 1 (reader SKIPS), %.6f cells,
            equal data-row count N in every file.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

try:  # normal case: imported as a package module (motion.pipeline.glue.*)
    from .joint_maps import (
        MJCF_TO_ISAAC,
        assert_runtime_joint_order,  # noqa: F401  (re-exported VERIFY hook)
        xyzw_to_wxyz,
    )
except ImportError:  # pragma: no cover - direct-script fallback (glue/ on sys.path)
    from joint_maps import (  # type: ignore
        MJCF_TO_ISAAC,
        assert_runtime_joint_order,  # noqa: F401
        xyzw_to_wxyz,
    )

__all__ = [
    "GmrFormatError",
    "gmr_to_sonic_csv",
    "DEFAULT_CONTROL_HZ",
    "DT",
]

DEFAULT_CONTROL_HZ = 50.0
DT = 1.0 / DEFAULT_CONTROL_HZ  # = 0.02 s at 50 Hz (the default; per-call dt = 1/control_hz)

_NDOF = 29
_EPS = 1e-12  # zero-norm quaternion guard


class GmrFormatError(ValueError):
    """Raised LOUDLY whenever the input ``gmr`` dict fails the shape / dtype /
    finiteness contract, or a bundle invariant is violated.  Never fail
    silently: a silently-truncated array or a wrong T becomes a robot moving
    the wrong way three stages downstream."""


# ---------------------------------------------------------------------------
# Input validation — read exactly the 4 keys the glue needs, fail loud.
# ---------------------------------------------------------------------------
def _read_and_validate(gmr: dict):
    """Return (fps, root_pos, root_rot, dof_pos) as validated float64 arrays.

    Reads ONLY 'fps','root_pos','root_rot','dof_pos'.  Ignores everything else
    (GMR's smplx path leaves 'local_body_pos'/'link_body_list' == None — this is
    WHY the body arrays start root-only; the glue never dereferences them).
    """
    if not isinstance(gmr, dict):
        raise GmrFormatError(f"gmr must be a dict, got {type(gmr).__name__}")
    for k in ("fps", "root_pos", "root_rot", "dof_pos"):
        if k not in gmr:
            raise GmrFormatError(
                f"gmr missing required key {k!r} "
                f"(need fps, root_pos, root_rot, dof_pos)"
            )

    # fps: a plain scalar float
    try:
        fps = float(np.asarray(gmr["fps"]).reshape(()))
    except Exception as exc:  # noqa: BLE001 - re-raised as our loud error
        raise GmrFormatError(f"fps is not a scalar float: {exc}") from exc
    if not np.isfinite(fps) or fps <= 0.0:
        raise GmrFormatError(f"fps must be finite and > 0, got {fps!r}")

    root_pos = np.asarray(gmr["root_pos"], dtype=np.float64)
    root_rot = np.asarray(gmr["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(gmr["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise GmrFormatError(f"root_pos shape {root_pos.shape} != (T,3)")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise GmrFormatError(f"root_rot shape {root_rot.shape} != (T,4) XYZW")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != _NDOF:
        raise GmrFormatError(
            f"dof_pos shape {dof_pos.shape} != (T,{_NDOF}) MJCF order"
        )

    T = dof_pos.shape[0]
    if T < 1:
        raise GmrFormatError("empty sequence (T == 0)")
    if root_pos.shape[0] != T or root_rot.shape[0] != T:
        raise GmrFormatError(
            f"T mismatch across time arrays: dof_pos T={T}, "
            f"root_pos T={root_pos.shape[0]}, root_rot T={root_rot.shape[0]}"
        )

    for name, arr in (("root_pos", root_pos), ("root_rot", root_rot), ("dof_pos", dof_pos)):
        if not np.isfinite(arr).all():
            raise GmrFormatError(f"{name} contains NaN/Inf")

    # Renormalize quaternions before Slerp (guard zero-norm rows).
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if not np.all(norms > _EPS):
        raise GmrFormatError("root_rot contains ~zero-norm quaternion(s)")
    root_rot = root_rot / norms

    return fps, root_pos, root_rot, dof_pos


# ---------------------------------------------------------------------------
# Frozen internal helpers
# ---------------------------------------------------------------------------
def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Frozen-internal alias for the quaternion channel reindex.  Delegates to
    ``joint_maps.xyzw_to_wxyz`` so there is exactly ONE implementation."""
    return xyzw_to_wxyz(q)


def _resample(fps: float, lin_arrays: list[np.ndarray], quat_xyzw: np.ndarray,
              control_hz: float):
    """Resample a bundle of linear arrays + one quaternion track to EXACTLY
    ``control_hz``.

    Returns ``(N, lin_out, quat_wxyz)`` where ``N`` is the output row count,
    ``lin_out`` mirrors ``lin_arrays`` (each (N, D), np.interp per column) and
    ``quat_wxyz`` is (N, 4) scalar-first (Slerp on the XYZW input, then WXYZ).

    Frozen formula (span = the TRUE sample span, not T/fps):
      span = (T - 1) / fps                      # T samples span (T-1) intervals
      N = max(1, floor(span * control_hz) + 1)  # fit a uniform control_hz grid, no overshoot
      t_src = arange(T)/fps
      t_new = arange(N)/control_hz              # last point <= t_src[-1], so NO clamp
    Using the true span (not T/fps) keeps t_new inside the source range, so no
    trailing samples are clamped to t_src[-1] -- a clamp would collapse the final
    finite-diff velocities (they'd read ~0). Up-/down-sampling handled identically.
    ``T == 1`` cannot be interpolated, so the single frame is tiled (velocities -> 0).
    """
    T = quat_xyzw.shape[0]

    # Degenerate single frame: no span to interpolate over -> tile to a nominal
    # row count (velocities -> 0). No clamp issue here (no interpolation).
    if T == 1:
        N = max(1, int(round(T / fps * control_hz)))
        lin_out = [np.repeat(a, N, axis=0) for a in lin_arrays]
        quat_wxyz = np.repeat(_xyzw_to_wxyz(quat_xyzw), N, axis=0)
        return N, lin_out, quat_wxyz

    span = (T - 1) / fps
    N = max(1, int(np.floor(span * control_hz)) + 1)
    t_src = np.arange(T) / fps
    # Clamp target times into the source span so neither np.interp (already
    # clamps) nor Slerp (raises on out-of-range) extrapolates on the <=1
    # trailing sample that arange(N)/control_hz can push just past t_src[-1].
    t_new = np.clip(np.arange(N) / control_hz, t_src[0], t_src[-1])

    lin_out = []
    for a in lin_arrays:
        cols = [np.interp(t_new, t_src, a[:, c]) for c in range(a.shape[1])]
        lin_out.append(np.stack(cols, axis=1))

    slerp = Slerp(t_src, R.from_quat(quat_xyzw))
    quat_new_xyzw = slerp(t_new).as_quat()  # (N,4) XYZW
    quat_wxyz = _xyzw_to_wxyz(quat_new_xyzw)  # (N,4) WXYZ

    return N, lin_out, quat_wxyz


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """d/dt of a time series (axis 0) by ``np.gradient`` — central difference in
    the interior, one-sided (correct) at the endpoints.  N < 2 -> zeros."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def _body_ang_vel(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (rad/s), (N,3), from a resampled orientation
    track (input (N,4) XYZW).

    Central interior:  rotvec(R[i+1] * R[i-1]^-1) / (2 dt)
    One-sided ends:    rotvec(R[1] * R[0]^-1) / dt , rotvec(R[-1]*R[-2]^-1)/dt
    Mirrors ``np.gradient``'s stencil, but composed on SO(3) so double-cover /
    wrap issues can't leak in.  N < 2 -> zeros.
    """
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    N = quat_xyzw.shape[0]
    if N < 2:
        return np.zeros((N, 3), dtype=np.float64)

    rot = R.from_quat(quat_xyzw)
    w = np.zeros((N, 3), dtype=np.float64)
    if N >= 3:
        rel = rot[2:] * rot[:-2].inv()          # (N-2,) relative rotations
        w[1:-1] = rel.as_rotvec() / (2.0 * dt)
    # one-sided endpoints
    w[0] = (rot[1] * rot[0].inv()).as_rotvec() / dt
    w[-1] = (rot[-1] * rot[-2].inv()).as_rotvec() / dt
    return w


def _write_csv(path, header: list[str], rows: np.ndarray) -> None:
    """Write one SONIC CSV: line 1 = comma-joined ``header`` (reader SKIPS it),
    then each data row as ``%.6f`` cells, comma-separated, ``\\n``-terminated."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise GmrFormatError(f"{path}: rows must be 2-D, got shape {rows.shape}")
    if rows.shape[1] != len(header):
        raise GmrFormatError(
            f"{path}: {rows.shape[1]} data columns != {len(header)} header columns"
        )
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("%.6f" % v for v in r) + "\n")


def _write_metadata(path, *, motion_name: str, N: int,
                    body_indexes: np.ndarray, arrays_summary: list) -> None:
    """Write ``metadata.txt`` in SONIC ``save_metadata()`` format.  The reader's
    ONLY hard requirement is the bracketed 'Body part indexes:' line (root-only
    -> ``[0]``); the rest is human-readable summary."""
    lines = [
        f"Metadata for: {motion_name}",
        "=" * 30,
        "",
        "Body part indexes:",
        str(np.asarray(body_indexes)),   # numpy str: root-only [0]; e.g. [0 4 10 ...]
        "",
        f"Total timesteps: {N}",
        "",
        "Data arrays summary:",
    ]
    for name, shape in arrays_summary:
        lines.append(f"  {name}: {tuple(shape)} (float32)")
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Header builders (exact strings from the FROZEN csv_bundle spec).
# ---------------------------------------------------------------------------
def _joint_pos_header() -> list[str]:
    return [f"joint_{i}" for i in range(_NDOF)]


def _joint_vel_header() -> list[str]:
    return [f"joint_vel_{i}" for i in range(_NDOF)]


def _body_pos_header(B: int) -> list[str]:
    return [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(3 * B)]


def _body_quat_header(B: int) -> list[str]:
    return [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(4 * B)]


def _body_lin_vel_header(B: int) -> list[str]:
    return [f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(3 * B)]


def _body_ang_vel_header(B: int) -> list[str]:
    return [f"body_{i // 3}_angvel_{'xyz'[i % 3]}" for i in range(3 * B)]


def _check_bundle(N: int, B: int, files: list) -> None:
    """Enforce the writer invariants BEFORE anything hits disk (raise on
    violation): every file has exactly N data rows; joint files have 29 cols;
    body_pos/lin/ang have 3*B cols and body_quat 4*B; the body-velocity files
    (if present) share B with body_pos."""
    for name, header, rows in files:
        rows = np.asarray(rows)
        if rows.shape[0] != N:
            raise GmrFormatError(
                f"{name}: row count {rows.shape[0]} != N={N} (equal-row-count invariant)"
            )
        if rows.shape[1] != len(header):
            raise GmrFormatError(
                f"{name}: {rows.shape[1]} cols != header {len(header)} cols"
            )
    cols = {name: np.asarray(rows).shape[1] for name, _, rows in files}
    if cols["joint_pos.csv"] != _NDOF or cols["joint_vel.csv"] != _NDOF:
        raise GmrFormatError("joint_pos/joint_vel must each have exactly 29 columns")
    if cols["body_pos.csv"] != 3 * B or cols["body_quat.csv"] != 4 * B:
        raise GmrFormatError("body_pos must be 3*B cols and body_quat 4*B cols")
    for opt in ("body_lin_vel.csv", "body_ang_vel.csv"):
        if opt in cols and cols[opt] != 3 * B:
            raise GmrFormatError(f"{opt}: {cols[opt]} cols != 3*B={3 * B} (B mismatch vs body_pos)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def gmr_to_sonic_csv(gmr: dict, out_dir: str | os.PathLike, *,
                     control_hz: float = 50.0,
                     motion_name: str = "motion",
                     root_only: bool = True,
                     write_body_vel: bool = True) -> pathlib.Path:
    """Convert a GMR ``gmr.pkl`` dict into SONIC's CSV bundle in ``out_dir``.

    Reads ``gmr['fps','root_pos','root_rot','dof_pos']``; writes 6 CSVs
    (``joint_pos``, ``joint_vel``, ``body_pos``, ``body_quat``, and — when
    ``write_body_vel`` — ``body_lin_vel``, ``body_ang_vel``) plus
    ``metadata.txt``, all at ``control_hz`` (50 Hz) with headers and identical
    data-row count N.  Returns ``out_dir`` as a ``pathlib.Path``.

    Parameters
    ----------
    gmr : dict
        The flat dict pickled by GMR ``smplx_to_robot.py``.
    out_dir : path
        Destination directory (created if absent).
    control_hz : float
        Output/control rate; SONIC wants 50 Hz -> dt = 0.02 s.
    motion_name : str
        Label written into ``metadata.txt``.
    root_only : bool
        Body arrays use the pelvis/root only (B=1).  Full-body FK needs the
        mujoco model (on-device) and is DEFERRED — see the VERIFY-on-robot flag.
    write_body_vel : bool
        Emit the two body-velocity CSVs (default).  False -> the 4-CSV minimum
        set SONIC also accepts.
    """
    if not root_only:
        # VERIFY-on-robot: full multi-body FK needs the mujoco model (on-device).
        # SONIC's observation_config.yaml may enable VR-3point/5point BODY
        # observations (isaaclab body indexes {28,29,9} / {28,29,0,18,19}); if
        # active, B=1 is INSUFFICIENT and full FK (e.g. 14-body
        # _body_indexes=[0 4 10 18 5 11 19 9 16 22 28 17 23 29]) is required.
        raise GmrFormatError(
            "root_only=False (full-body FK) is DEFERRED: needs the mujoco model "
            "on-device. Check observation_config.yaml; keep root_only=True until then."
        )

    if not (np.isfinite(control_hz) and control_hz > 0):
        raise GmrFormatError(f"control_hz must be finite and > 0, got {control_hz!r}")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / control_hz

    fps, root_pos, root_rot, dof_pos = _read_and_validate(gmr)

    # (1) Reindex dof MJCF -> IsaacLab, BY NAME.  The single most dangerous step.
    # VERIFY-on-robot: MJCF_TO_ISAAC is built by name in joint_maps (issue #78
    # guard). Before any real run, call assert_runtime_joint_order(
    # robot.data.joint_names) and/or replay the CSVs back through GMR/MuJoCo; if
    # names disagree, STOP.
    dof_isaac = dof_pos[:, MJCF_TO_ISAAC]  # (T,29) IsaacLab order

    # (2)+(3) Resample to exactly control_hz. Linear on [dof_isaac, root_pos];
    # Slerp on root_rot (XYZW) -> WXYZ inside _resample.
    N, (joint_pos, body_pos), body_quat = _resample(
        fps, [dof_isaac, root_pos], root_rot, control_hz
    )
    # joint_pos (N,29) rad IsaacLab order; body_pos (N,3) m world; body_quat (N,4) WXYZ.

    # (4) Joint velocities: finite-diff of the resampled joint positions at dt.
    joint_vel = _finite_diff(joint_pos, dt)  # (N,29) rad/s

    # (5) Body arrays — ROOT-ONLY (B=1).
    # VERIFY-on-robot: root-only sufficiency is UNPROVEN (VR-point observations
    # may require full-body FK). body_pos/body_quat are the pelvis; column-group
    # 0 is ALWAYS root.
    B = 1
    body_lin_vel = _finite_diff(body_pos, dt)  # (N,3) m/s world
    # angular velocity from the resampled orientation; _body_ang_vel wants XYZW,
    # so convert the WXYZ track back: wxyz[:, [1,2,3,0]] == xyzw.
    body_ang_vel = _body_ang_vel(body_quat[:, [1, 2, 3, 0]], dt)  # (N,3) rad/s world

    # (6) Assemble the bundle (order matters only for the metadata summary).
    files = [
        ("joint_pos.csv", _joint_pos_header(), joint_pos),
        ("joint_vel.csv", _joint_vel_header(), joint_vel),
        ("body_pos.csv", _body_pos_header(B), body_pos),
        # VERIFY-on-robot: WXYZ order is 'critical' per SONIC docs — confirm the
        # shipped ONNX/observation_config expects scalar-first before trusting.
        ("body_quat.csv", _body_quat_header(B), body_quat),
    ]
    if write_body_vel:
        files.append(("body_lin_vel.csv", _body_lin_vel_header(B), body_lin_vel))
        files.append(("body_ang_vel.csv", _body_ang_vel_header(B), body_ang_vel))

    # Enforce invariants before writing a single byte.
    _check_bundle(N, B, files)

    for name, header, rows in files:
        _write_csv(out_dir / name, header, rows)

    # metadata.txt — root-only body index is [0]. Summary lists ONLY the arrays
    # actually written.
    arrays_summary = [
        ("joint_pos", (N, _NDOF)),
        ("joint_vel", (N, _NDOF)),
        ("body_pos", (N, B, 3)),
        ("body_quat", (N, B, 4)),
    ]
    if write_body_vel:
        arrays_summary.append(("body_lin_vel", (N, B, 3)))
        arrays_summary.append(("body_ang_vel", (N, B, 3)))
    _write_metadata(
        out_dir / "metadata.txt",
        motion_name=motion_name,
        N=N,
        body_indexes=np.arange(B),  # root-only -> [0]
        arrays_summary=arrays_summary,
    )

    return out_dir


# ---------------------------------------------------------------------------
# CLI — load a gmr.pkl, write the bundle.  (pickle import lives here only; the
# library path above never unpickles.)
# ---------------------------------------------------------------------------
def _load_gmr_pkl(path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise GmrFormatError(f"{path}: pickle payload is {type(obj).__name__}, not a dict")
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m motion.pipeline.glue.gmr_to_sonic_csv",
        description="Convert a GMR gmr.pkl into SONIC's 6-CSV bundle (50 Hz).",
    )
    ap.add_argument("--gmr", required=True, help="path to gmr.pkl (from GMR smplx_to_robot.py)")
    ap.add_argument("--out", required=True, help="output directory for the CSV bundle")
    ap.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ,
                    help="output/control rate in Hz (default 50)")
    ap.add_argument("--motion-name", default="motion", help="label for metadata.txt")
    ap.add_argument("--no-body-vel", action="store_true",
                    help="write only the 4-CSV minimum set (drop body_lin_vel/body_ang_vel)")
    args = ap.parse_args(argv)

    gmr = _load_gmr_pkl(args.gmr)
    out = gmr_to_sonic_csv(
        gmr,
        args.out,
        control_hz=args.control_hz,
        motion_name=args.motion_name,
        write_body_vel=not args.no_body_vel,
    )
    print(f"wrote SONIC bundle -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
