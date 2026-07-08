"""pytest for the Phase-3 gmr.pkl validator (``pipeline/glue/gmr_pkl.py``).

GMR itself CANNOT run on this workstation (aarch64 wheels + the unitree_g1 asset
are Orin-only), so every gmr.pkl here is SYNTHETIC. ``make_synthetic_gmr`` builds
a dict shaped exactly like GMR's ``smplx_to_robot.py`` output (the FROZEN-CONTRACT
6-key layout) with every value chosen so the validator's checks are provable by
closed form: root_pos is a linear +x ramp, root_rot is a slow pure yaw about
world Z (upright at every frame), dof_pos encodes each MJCF column index.

The validator imports numpy + scipy + stdlib pickle ONLY (no torch / mujoco /
GMR), so this suite runs under the workstation base python.

Run:  ``pytest motion/tests/test_gmr_pkl.py``  from the repo root.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Repo root (parent of motion/) on sys.path so the dotted import resolves no
# matter where pytest is invoked from — same posture as test_pose_to_smpl.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motion.pipeline.glue.gmr_pkl import (  # noqa: E402  (after sys.path insert)
    GmrValidationError,
    load_gmr_pkl,
    upright_tilt_deg,
    validate_gmr,
)

_CLEAN_KEYS = {"fps", "root_pos", "root_rot", "dof_pos", "T"}
_NUM_DOF = 29


# ---------------------------------------------------------------------------
# Synthetic gmr.pkl (mirrors the FROZEN-CONTRACT builder)
# ---------------------------------------------------------------------------
def make_synthetic_gmr(T=60, fps=30.0, seed=0):
    """Flat dict identical in structure to GMR smplx_to_robot.py output."""
    t = np.arange(T) / fps
    # root_pos: linear +x ramp @ 0.5 m/s, constant 0.8 m height (Z-up).
    root_pos = np.stack(
        [0.5 * t, np.zeros(T), 0.8 * np.ones(T)], axis=1
    ).astype(np.float64)
    # root_rot: slow pure yaw about world Z, stored XYZW (scalar-LAST) — upright
    # at every frame, so upright_tilt_deg == 0.
    root_rot = R.from_euler("z", 0.3 * t).as_quat().astype(np.float64)
    # dof_pos: column j == its MJCF index + a per-column linear ramp.
    j = np.arange(_NUM_DOF)[None, :]
    dof_pos = (j + 0.1 * j * t[:, None]).astype(np.float64)
    return {
        "fps": float(fps),
        "root_pos": root_pos,       # (T,3) f64 m, Z-up
        "root_rot": root_rot,       # (T,4) f64 XYZW
        "dof_pos": dof_pos,         # (T,29) f64 rad, MJCF order
        "local_body_pos": None,     # GMR smplx path leaves these None
        "link_body_list": None,
    }


# ---------------------------------------------------------------------------
# (1) a valid gmr.pkl loads: clean typed dict, exact shapes/dtypes, fps carried
# ---------------------------------------------------------------------------
def test_valid_returns_clean_typed_dict():
    T = 60
    gmr = make_synthetic_gmr(T=T, fps=30.0)
    data = validate_gmr(gmr)

    # exactly the clean keys — local_body_pos/link_body_list dropped.
    assert set(data.keys()) == _CLEAN_KEYS

    # shapes / dtypes / frame count
    assert data["T"] == T
    assert data["root_pos"].shape == (T, 3)
    assert data["root_rot"].shape == (T, 4)
    assert data["dof_pos"].shape == (T, _NUM_DOF)
    for k in ("root_pos", "root_rot", "dof_pos"):
        assert data[k].dtype == np.float64, k

    # fps carried through as a plain positive float
    assert isinstance(data["fps"], float)
    assert data["fps"] == pytest.approx(30.0)

    # everything finite
    for k in ("root_pos", "root_rot", "dof_pos"):
        assert np.isfinite(data[k]).all(), k


def test_valid_upright_passes_without_warning():
    # pure-yaw synthetic is upright at every frame -> tilt ~0, no warning.
    gmr = make_synthetic_gmr(T=40, fps=30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        data = validate_gmr(gmr)
    assert float(np.median(upright_tilt_deg(data["root_rot"]))) < 1.0


def test_root_rot_renormalized_to_unit():
    # non-unit quats (scaled x3) must come back unit-norm.
    gmr = make_synthetic_gmr(T=20)
    gmr["root_rot"] = gmr["root_rot"] * 3.0
    data = validate_gmr(gmr)
    norms = np.linalg.norm(data["root_rot"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_clean_dict_is_phase4_ready():
    # the 4 load-bearing keys the Phase-4 glue reads are all present + shaped.
    data = validate_gmr(make_synthetic_gmr(T=15))
    for k in ("fps", "root_pos", "root_rot", "dof_pos"):
        assert k in data
    assert data["root_pos"].shape[0] == data["dof_pos"].shape[0] == data["T"]


# ---------------------------------------------------------------------------
# (2) pickle-file round-trip through load_gmr_pkl
# ---------------------------------------------------------------------------
def test_load_gmr_pkl_roundtrip(tmp_path):
    gmr = make_synthetic_gmr(T=25, fps=29.97)
    p = tmp_path / "gmr.pkl"
    with open(p, "wb") as f:
        pickle.dump(gmr, f)

    data = load_gmr_pkl(p)
    assert data["T"] == 25
    assert data["fps"] == pytest.approx(29.97)
    assert np.allclose(data["root_pos"], gmr["root_pos"])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(GmrValidationError, match="not found"):
        load_gmr_pkl(tmp_path / "nope.pkl")


def test_load_garbage_pickle_raises(tmp_path):
    p = tmp_path / "junk.pkl"
    p.write_bytes(b"not a pickle at all \x00\x01")
    with pytest.raises(GmrValidationError):
        load_gmr_pkl(p)


# ---------------------------------------------------------------------------
# (3) malformed gmr.pkl raises LOUDLY — the negative-path fixtures
# ---------------------------------------------------------------------------
def test_dof_28_columns_raises():
    gmr = make_synthetic_gmr(T=10)
    gmr["dof_pos"] = gmr["dof_pos"][:, :28]
    with pytest.raises(GmrValidationError, match="dof_pos.*28.*expected 29"):
        validate_gmr(gmr)


def test_dof_30_columns_raises():
    gmr = make_synthetic_gmr(T=10)
    gmr["dof_pos"] = np.concatenate(
        [gmr["dof_pos"], gmr["dof_pos"][:, :1]], axis=1
    )
    with pytest.raises(GmrValidationError, match="dof_pos.*30.*expected 29"):
        validate_gmr(gmr)


def test_root_rot_3_columns_raises():
    gmr = make_synthetic_gmr(T=10)
    gmr["root_rot"] = gmr["root_rot"][:, :3]      # (T,3) — not a quaternion
    with pytest.raises(GmrValidationError, match="root_rot.*3.*expected 4"):
        validate_gmr(gmr)


def test_frame_count_mismatch_raises():
    gmr = make_synthetic_gmr(T=20)
    gmr["root_pos"] = gmr["root_pos"][:-1]        # 19 vs 20
    with pytest.raises(GmrValidationError, match="frame-count mismatch"):
        validate_gmr(gmr)


def test_missing_key_raises():
    gmr = make_synthetic_gmr(T=8)
    del gmr["fps"]
    with pytest.raises(GmrValidationError, match="missing key"):
        validate_gmr(gmr)


def test_fps_non_positive_raises():
    gmr = make_synthetic_gmr(T=8)
    gmr["fps"] = 0.0
    with pytest.raises(GmrValidationError, match="fps.*> 0"):
        validate_gmr(gmr)


def test_fps_non_finite_raises():
    gmr = make_synthetic_gmr(T=8)
    gmr["fps"] = float("nan")
    with pytest.raises(GmrValidationError, match="fps.*finite"):
        validate_gmr(gmr)


def test_nan_in_array_raises():
    gmr = make_synthetic_gmr(T=8)
    gmr["root_pos"][0, 0] = np.nan
    with pytest.raises(GmrValidationError, match="root_pos.*NaN/Inf"):
        validate_gmr(gmr)


def test_empty_sequence_raises():
    gmr = make_synthetic_gmr(T=1)
    for k in ("root_pos", "root_rot", "dof_pos"):
        gmr[k] = gmr[k][:0]                        # T == 0
    with pytest.raises(GmrValidationError, match="T == 0"):
        validate_gmr(gmr)


def test_degenerate_zero_quat_raises():
    gmr = make_synthetic_gmr(T=8)
    gmr["root_rot"][2] = 0.0                       # zero-norm quaternion
    with pytest.raises(GmrValidationError, match="~zero norm|degenerate"):
        validate_gmr(gmr)


def test_not_a_dict_raises():
    with pytest.raises(GmrValidationError, match="must be a dict"):
        validate_gmr(["not", "a", "dict"])         # type: ignore[arg-type]


def test_root_rot_wrong_rank_raises():
    gmr = make_synthetic_gmr(T=8)
    gmr["root_rot"] = gmr["root_rot"].ravel()      # 1-D, not (T,4)
    with pytest.raises(GmrValidationError, match="root_rot.*2-D"):
        validate_gmr(gmr)


# ---------------------------------------------------------------------------
# (4) upright sanity assert — inverted raises, sideways warns / strict-raises
# ---------------------------------------------------------------------------
def _fixed_rot_gmr(T, rot: R):
    """gmr.pkl whose pelvis holds a single fixed orientation for all frames."""
    gmr = make_synthetic_gmr(T=T)
    gmr["root_rot"] = np.tile(rot.as_quat().astype(np.float64), (T, 1))
    return gmr


def test_inverted_pelvis_raises():
    # roll 180 deg -> pelvis +Z points to world -Z -> median tilt 180 deg.
    gmr = _fixed_rot_gmr(30, R.from_euler("x", np.pi))
    with pytest.raises(GmrValidationError, match="INVERTED"):
        validate_gmr(gmr)


def test_sideways_pelvis_warns_by_default_and_strict_raises():
    # roll 90 deg -> lying on side -> median tilt 90 deg (warn zone).
    gmr = _fixed_rot_gmr(30, R.from_euler("x", np.pi / 2))

    with pytest.warns(RuntimeWarning, match="tilted"):
        validate_gmr(gmr)                          # default: warns, still returns

    with pytest.raises(GmrValidationError, match="strict_upright"):
        validate_gmr(gmr, strict_upright=True)     # strict: raises


def test_upright_check_can_be_disabled():
    # inverted pelvis is accepted when the check is off (no raise).
    gmr = _fixed_rot_gmr(20, R.from_euler("x", np.pi))
    data = validate_gmr(gmr, check_upright=False)
    assert data["T"] == 20


def test_upright_tilt_deg_closed_form():
    # yaw about Z is upright (0 deg); pitch 30 deg tilts the up-axis by 30 deg.
    yaw = R.from_euler("z", np.linspace(0, np.pi, 10)).as_quat()
    assert np.allclose(upright_tilt_deg(yaw), 0.0, atol=1e-6)
    pitch = R.from_euler("y", np.full(5, np.radians(30.0))).as_quat()
    assert np.allclose(upright_tilt_deg(pitch), 30.0, atol=1e-6)
