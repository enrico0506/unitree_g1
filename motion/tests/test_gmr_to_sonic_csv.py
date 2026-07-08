"""pytest for the Phase-4 SONIC glue (``pipeline/glue/gmr_to_sonic_csv.py`` +
``pipeline/glue/joint_maps.py``).

GMR and SONIC CANNOT run on this workstation (GMR needs the retargeter + a
mujoco G1 asset; SONIC needs TensorRT 10.7 on a Jetson), so every input here is
a SYNTHETIC ``gmr.pkl``-shaped dict built by ``make_synthetic_gmr`` — identical
in structure to GMR ``smplx_to_robot.py`` output.  Every value is designed so
the glue's transforms are checkable in closed form:

  * root_pos is a linear +x ramp  -> body_lin_vel is a known constant [0.5,0,0]
  * root_rot is a slow pure yaw    -> tests XYZW->WXYZ and Slerp monotonicity
  * dof[t,j] = j + 0.1*j*t          -> row0 == source MJCF index (reindex probe),
                                        slope 0.1*j (velocity probe); linear in t
                                        so linear resampling preserves it exactly

The glue imports with numpy + scipy ONLY (no torch / mujoco / isaaclab / GMR),
so this suite runs under the workstation base python.  NO module-level
``sys.exit`` — proper ``def test_*`` functions.

Run:  ``pytest motion/tests/test_gmr_to_sonic_csv.py``  from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Put the repo root (parent of motion/) on sys.path so the package resolves
# regardless of where pytest is invoked from — same posture as test_pose_to_smpl.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motion.pipeline.glue.joint_maps import (  # noqa: E402  (after sys.path insert)
    G1_ISAACLAB_ORDER,
    G1_MJCF_ORDER,
    ISAAC_TO_MJCF,
    MJCF_TO_ISAAC,
    JointOrderError,
    assert_runtime_joint_order,
    build_permutation,
    xyzw_to_wxyz,
)
from motion.pipeline.glue.gmr_to_sonic_csv import (  # noqa: E402
    GmrFormatError,
    gmr_to_sonic_csv,
)

_NDOF = 29
_ALL_CSVS = (
    "joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv",
    "body_lin_vel.csv", "body_ang_vel.csv",
)
_FROZEN_MJCF_TO_ISAAC = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]


# ---------------------------------------------------------------------------
# Synthetic gmr.pkl (mirrors GMR smplx_to_robot.py output — the FROZEN builder)
# ---------------------------------------------------------------------------
def make_synthetic_gmr(T=60, fps=30.0, seed=0):
    """Flat dict identical in structure to GMR smplx_to_robot.py output."""
    t = np.arange(T) / fps  # source sample times (s)
    # root_pos: linear +x ramp so linear interp is EXACT and body_lin_vel is a
    # known constant (0.5 m/s in x); z held at 0.8 m (Z-up).
    root_pos = np.stack([0.5 * t, np.zeros(T), 0.8 * np.ones(T)], axis=1).astype(np.float64)
    # root_rot: slow pure yaw about world Z, stored XYZW (scalar-LAST). yaw(0)=0.
    root_rot = R.from_euler("z", 0.3 * t).as_quat().astype(np.float64)
    # dof_pos: column j = j + 0.1*j*t -> value at t=0 is exactly the MJCF index.
    j = np.arange(_NDOF)[None, :]
    dof_pos = (j + 0.1 * j * t[:, None]).astype(np.float64)
    return {
        "fps": float(fps),
        "root_pos": root_pos,        # (T,3) m, Z-up
        "root_rot": root_rot,        # (T,4) XYZW
        "dof_pos": dof_pos,          # (T,29) rad, MJCF order
        "local_body_pos": None,      # GMR smplx path leaves these None
        "link_body_list": None,
    }


def _read_csv(path):
    """Return (header:list[str], data:np.ndarray (N,C)). Reader SKIPS line 1."""
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0].split(",")
    if len(lines) == 1:
        return header, np.zeros((0, len(header)))
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return header, data


# ---------------------------------------------------------------------------
# (A) joint_maps: the reindex table is a true permutation and matches the freeze
# ---------------------------------------------------------------------------
def test_reindex_matches_frozen_literal_and_is_a_true_permutation():
    assert MJCF_TO_ISAAC.tolist() == _FROZEN_MJCF_TO_ISAAC
    # a permutation: sorted == arange, and it maps MJCF names onto IsaacLab names
    assert sorted(MJCF_TO_ISAAC.tolist()) == list(range(_NDOF))
    assert [G1_MJCF_ORDER[i] for i in MJCF_TO_ISAAC] == G1_ISAACLAB_ORDER


def test_forward_and_inverse_round_trip():
    # ISAAC_TO_MJCF is the inverse gather: applying both returns identity.
    assert np.array_equal(ISAAC_TO_MJCF[MJCF_TO_ISAAC], np.arange(_NDOF))
    assert np.array_equal(MJCF_TO_ISAAC[ISAAC_TO_MJCF], np.arange(_NDOF))
    # a data round-trip: reindex there and back is a no-op.
    x = np.random.default_rng(0).standard_normal((7, _NDOF))
    there = x[:, MJCF_TO_ISAAC]
    back = there[:, ISAAC_TO_MJCF]
    assert np.allclose(back, x)


def test_inverse_is_not_the_forward_gather_issue_78_trap():
    # The whole point of issue #78: the inverse != the forward gather. Using the
    # inverse as the forward gather would put the wrong joint's name at index 0+.
    assert ISAAC_TO_MJCF.tolist() != MJCF_TO_ISAAC.tolist()
    wrong = [G1_MJCF_ORDER[i] for i in ISAAC_TO_MJCF]
    assert wrong != G1_ISAACLAB_ORDER  # garbage, as expected


def test_build_permutation_raises_on_unmatched_name():
    corrupted = list(G1_MJCF_ORDER)
    corrupted[3] = "left_knee_TYPO_joint"     # rename one joint
    with pytest.raises(JointOrderError, match="not found in src"):
        build_permutation(corrupted, G1_ISAACLAB_ORDER)


def test_build_permutation_raises_on_length_and_duplicate():
    with pytest.raises(JointOrderError, match="length mismatch"):
        build_permutation(G1_MJCF_ORDER[:-1], G1_ISAACLAB_ORDER)
    dup = list(G1_MJCF_ORDER)
    dup[1] = dup[0]                            # duplicate -> ambiguous .index
    with pytest.raises(JointOrderError, match="duplicate"):
        build_permutation(dup, G1_ISAACLAB_ORDER)


def test_assert_runtime_joint_order_hook():
    assert_runtime_joint_order(G1_ISAACLAB_ORDER)  # matching -> no raise
    bad = list(G1_ISAACLAB_ORDER)
    bad[0], bad[1] = bad[1], bad[0]
    with pytest.raises(JointOrderError):
        assert_runtime_joint_order(bad)


def test_xyzw_to_wxyz_helper():
    q = np.array([[0.1, 0.2, 0.3, 0.9]])       # XYZW
    assert np.allclose(xyzw_to_wxyz(q), [[0.9, 0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        xyzw_to_wxyz(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# (B) bundle shape: all 6 files written, equal row counts, headers present
# ---------------------------------------------------------------------------
def test_six_files_written_with_headers_and_equal_row_counts(tmp_path):
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    out = gmr_to_sonic_csv(gmr, tmp_path)
    assert out == tmp_path

    # all 6 CSVs + metadata.txt exist
    for name in _ALL_CSVS:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "metadata.txt").exists()

    # equal data-row count across every CSV
    counts = {}
    for name in _ALL_CSVS:
        header, data = _read_csv(tmp_path / name)
        assert len(header) == data.shape[1]       # header present, width matches
        counts[name] = data.shape[0]
    assert len(set(counts.values())) == 1, counts
    N = next(iter(counts.values()))
    assert N == 99                                 # 60 @ 30 -> span 59/30 s -> floor(*50)+1 = 99


def test_exact_headers_match_frozen_spec(tmp_path):
    gmr = make_synthetic_gmr(T=30, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    hdr = lambda n: _read_csv(tmp_path / n)[0]  # noqa: E731
    assert hdr("joint_pos.csv") == [f"joint_{i}" for i in range(29)]
    assert hdr("joint_vel.csv") == [f"joint_vel_{i}" for i in range(29)]
    assert hdr("body_pos.csv") == ["body_0_x", "body_0_y", "body_0_z"]
    assert hdr("body_quat.csv") == ["body_0_w", "body_0_x", "body_0_y", "body_0_z"]
    assert hdr("body_lin_vel.csv") == ["body_0_vel_x", "body_0_vel_y", "body_0_vel_z"]
    assert hdr("body_ang_vel.csv") == ["body_0_angvel_x", "body_0_angvel_y", "body_0_angvel_z"]


def test_column_widths(tmp_path):
    gmr = make_synthetic_gmr(T=20, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, jp = _read_csv(tmp_path / "joint_pos.csv")
    _, jv = _read_csv(tmp_path / "joint_vel.csv")
    _, bp = _read_csv(tmp_path / "body_pos.csv")
    _, bq = _read_csv(tmp_path / "body_quat.csv")
    assert jp.shape[1] == 29 and jv.shape[1] == 29
    assert bp.shape[1] == 3 and bq.shape[1] == 4


# ---------------------------------------------------------------------------
# (C) exact 50 Hz resample — row count AND spacing, up- and down-sample
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T,fps,expected_N", [
    (60, 30.0, 99),      # baseline 30 -> 50 (upsample)
    (120, 60.0, 100),    # downsample 60 -> 50
    (50, 25.0, 99),      # upsample 25 -> 50
    (60, 29.97, 99),     # non-integer fps
    (90, 30.0, 149),     # ~3 s
])
def test_row_count_is_exactly_span_times_50_plus1(tmp_path, T, fps, expected_N):
    gmr = make_synthetic_gmr(T=T, fps=fps)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, data = _read_csv(tmp_path / "joint_pos.csv")
    assert data.shape[0] == expected_N
    # frozen formula: fit a uniform 50 Hz grid over the TRUE sample span (T-1)/fps,
    # so t_new never overshoots t_src[-1] (no end-velocity clamp).
    import math
    assert expected_N == math.floor((T - 1) / fps * 50.0) + 1


def test_body_pos_row0_and_lin_vel_constant_downsample(tmp_path):
    # Downsample 60 -> 50 Hz. Linear ramp is preserved exactly.
    gmr = make_synthetic_gmr(T=120, fps=60.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, bp = _read_csv(tmp_path / "body_pos.csv")
    _, blv = _read_csv(tmp_path / "body_lin_vel.csv")
    # row 0 == [0,0,0.8]
    assert np.allclose(bp[0], [0.0, 0.0, 0.8], atol=1e-6)
    # interior body_lin_vel == [0.5, 0, 0] exactly (finite-diff of linear pos)
    assert np.allclose(blv[1:-1], [0.5, 0.0, 0.0], atol=1e-6)


def test_resample_step_is_uniform_002s(tmp_path):
    # The x-position ramp is 0.5*t; after resampling, consecutive interior rows
    # must differ by 0.5 * 0.02 = 0.01 m in x, proving a uniform 50 Hz grid.
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, bp = _read_csv(tmp_path / "body_pos.csv")
    dx = np.diff(bp[:, 0])
    # drop the last step (target-time clamp can shorten it); interior is uniform.
    assert np.allclose(dx[:-1], 0.01, atol=1e-6)


# ---------------------------------------------------------------------------
# (D) THE reindex probe — row0 == source MJCF index, vel == 0.1*index
# ---------------------------------------------------------------------------
def test_reindex_probe_row0_equals_source_index(tmp_path):
    # dof[0,j] = j, so after gathering with MJCF_TO_ISAAC each output column
    # shows the MJCF source index it pulled. A wrong permutation fails LOUD.
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, jp = _read_csv(tmp_path / "joint_pos.csv")
    assert np.allclose(jp[0], MJCF_TO_ISAAC.astype(float), atol=1e-6)


def test_reindex_probe_velocity_equals_scaled_index(tmp_path):
    # slope of column j is 0.1*j; after permutation, interior joint_vel row k
    # must equal 0.1 * MJCF_TO_ISAAC[k].
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, jv = _read_csv(tmp_path / "joint_vel.csv")
    expected = 0.1 * MJCF_TO_ISAAC.astype(float)
    assert np.allclose(jv[10], expected, atol=1e-6)   # a well-interior row


# ---------------------------------------------------------------------------
# (E) velocities == finite-diff of positions at dt=0.02
# ---------------------------------------------------------------------------
def test_joint_vel_is_finite_diff_of_joint_pos(tmp_path):
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, jp = _read_csv(tmp_path / "joint_pos.csv")
    _, jv = _read_csv(tmp_path / "joint_vel.csv")
    grad = np.gradient(jp, 0.02, axis=0)
    # Interior positions are exact at %.6f, so re-differentiating the CSV
    # reproduces the glue's velocity to full precision -> exact relationship.
    assert np.allclose(jv[1:-2], grad[1:-2], atol=1e-6)
    # The whole array agrees within %.6f re-differentiation error: only the last
    # (time-clamped) resampled sample isn't an exact 6-decimal value, and
    # dividing that rounding by dt=0.02 magnifies it to ~1e-5 at the tail.
    assert np.allclose(jv, grad, atol=1e-4)


def test_body_lin_vel_is_finite_diff_of_body_pos(tmp_path):
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, bp = _read_csv(tmp_path / "body_pos.csv")
    _, blv = _read_csv(tmp_path / "body_lin_vel.csv")
    grad = np.gradient(bp, 0.02, axis=0)
    assert np.allclose(blv[1:-2], grad[1:-2], atol=1e-6)   # interior exact
    assert np.allclose(blv, grad, atol=1e-4)               # tail within %.6f error


def test_body_ang_vel_matches_pure_yaw_rate(tmp_path):
    # root_rot is yaw = 0.3*t rad -> yaw rate 0.3 rad/s about +Z. Well-interior
    # rows must be ~[0,0,0.3]. The last two rows sit at the time-clamped tail
    # (only the final resampled sample is clamped, so the last central-diff and
    # the one-sided endpoint dip below 0.3) — a documented resample artifact.
    gmr = make_synthetic_gmr(T=90, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, bav = _read_csv(tmp_path / "body_ang_vel.csv")
    assert np.allclose(bav[1:-2], [0.0, 0.0, 0.3], atol=1e-4)


# ---------------------------------------------------------------------------
# (F) quaternion is WXYZ, unit norm, and yaw is recovered correctly
# ---------------------------------------------------------------------------
def test_body_quat_is_wxyz_unit_norm_and_yaw_correct(tmp_path):
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, bq = _read_csv(tmp_path / "body_quat.csv")
    # row 0 (yaw 0) == (w,x,y,z) = (1,0,0,0)
    assert np.allclose(bq[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    # unit norm everywhere
    assert np.allclose(np.linalg.norm(bq, axis=1), 1.0, atol=1e-6)
    # pure yaw: x==y==0, and (w,z) == (cos(th/2), sin(th/2)) with th = 0.3*t_new.
    assert np.allclose(bq[:, 1], 0.0, atol=1e-6)      # x
    assert np.allclose(bq[:, 2], 0.0, atol=1e-6)      # y
    t_new = np.arange(bq.shape[0]) / 50.0
    th = 0.3 * np.clip(t_new, 0.0, (60 - 1) / 30.0)   # clamp mirrors the glue
    assert np.allclose(bq[:, 0], np.cos(th / 2), atol=1e-5)   # w
    assert np.allclose(bq[:, 3], np.sin(th / 2), atol=1e-5)   # z
    # z increases monotonically -> Slerp yaw is monotone
    assert np.all(np.diff(bq[:, 3]) >= -1e-9)


# ---------------------------------------------------------------------------
# (G) metadata.txt — body index [0], timestep count, format
# ---------------------------------------------------------------------------
def test_metadata_body_index_and_timesteps(tmp_path):
    gmr = make_synthetic_gmr(T=60, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path, motion_name="unit_test_motion")
    text = (tmp_path / "metadata.txt").read_text()
    assert "Metadata for: unit_test_motion" in text
    assert "=" * 30 in text
    assert "Body part indexes:" in text
    assert "[0]" in text                    # root-only body index
    assert "Total timesteps: 99" in text
    assert "joint_pos: (99, 29) (float32)" in text
    assert "body_pos: (99, 1, 3) (float32)" in text
    assert "body_quat: (99, 1, 4) (float32)" in text


# ---------------------------------------------------------------------------
# (H) write_body_vel=False -> the 4-CSV minimum set
# ---------------------------------------------------------------------------
def test_write_body_vel_false_drops_two_csvs(tmp_path):
    gmr = make_synthetic_gmr(T=30, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path, write_body_vel=False)
    for name in ("joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "body_lin_vel.csv").exists()
    assert not (tmp_path / "body_ang_vel.csv").exists()
    # metadata summary must not mention the dropped arrays
    text = (tmp_path / "metadata.txt").read_text()
    assert "body_lin_vel" not in text
    assert "body_ang_vel" not in text


# ---------------------------------------------------------------------------
# (I) degenerate T == 1 -> single frame tiled to N rows, velocities zero
# ---------------------------------------------------------------------------
def test_single_frame_tiled_and_zero_velocity(tmp_path):
    gmr = make_synthetic_gmr(T=1, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    _, jp = _read_csv(tmp_path / "joint_pos.csv")
    _, jv = _read_csv(tmp_path / "joint_vel.csv")
    _, bav = _read_csv(tmp_path / "body_ang_vel.csv")
    assert jp.shape[0] == 2                  # round(1/30 * 50) = 2
    assert np.allclose(jp[0], jp[1])         # tiled: identical rows
    assert np.allclose(jv, 0.0)              # constant -> zero velocity
    assert np.allclose(bav, 0.0)


# ---------------------------------------------------------------------------
# (J) root-only: glue never dereferences None link arrays, B == 1
# ---------------------------------------------------------------------------
def test_root_only_ignores_none_link_arrays(tmp_path):
    gmr = make_synthetic_gmr(T=20, fps=30.0)
    assert gmr["local_body_pos"] is None and gmr["link_body_list"] is None
    gmr_to_sonic_csv(gmr, tmp_path)          # must not crash dereferencing None
    _, bp = _read_csv(tmp_path / "body_pos.csv")
    assert bp.shape[1] == 3                   # B == 1


def test_full_body_fk_is_deferred(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    with pytest.raises(GmrFormatError, match="DEFERRED"):
        gmr_to_sonic_csv(gmr, tmp_path, root_only=False)


# ---------------------------------------------------------------------------
# (K) negative-path fixtures — each must raise GmrFormatError LOUDLY
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ncol", [28, 30])
def test_wrong_dof_column_count_raises(tmp_path, ncol):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr["dof_pos"] = np.zeros((10, ncol))
    with pytest.raises(GmrFormatError, match="dof_pos"):
        gmr_to_sonic_csv(gmr, tmp_path)


def test_root_rot_wrong_width_raises(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr["root_rot"] = np.zeros((10, 3))       # (T,3) not (T,4)
    with pytest.raises(GmrFormatError, match="root_rot"):
        gmr_to_sonic_csv(gmr, tmp_path)


def test_T_mismatch_raises(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr["root_pos"] = gmr["root_pos"][:8]     # 8 != 10
    with pytest.raises(GmrFormatError, match="mismatch"):
        gmr_to_sonic_csv(gmr, tmp_path)


def test_missing_key_raises(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    del gmr["fps"]
    with pytest.raises(GmrFormatError, match="fps"):
        gmr_to_sonic_csv(gmr, tmp_path)


def test_non_finite_raises(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr["dof_pos"][3, 5] = np.nan
    with pytest.raises(GmrFormatError, match="NaN/Inf"):
        gmr_to_sonic_csv(gmr, tmp_path)


def test_bad_fps_raises(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr["fps"] = 0.0
    with pytest.raises(GmrFormatError, match="fps"):
        gmr_to_sonic_csv(gmr, tmp_path)


# ---------------------------------------------------------------------------
# (L) %.6f cell formatting is honored
# ---------------------------------------------------------------------------
def test_cells_are_six_decimal_places(tmp_path):
    gmr = make_synthetic_gmr(T=10, fps=30.0)
    gmr_to_sonic_csv(gmr, tmp_path)
    line = (tmp_path / "body_pos.csv").read_text().splitlines()[1]
    for cell in line.split(","):
        # every numeric cell has exactly 6 digits after the decimal point
        assert "." in cell and len(cell.split(".")[1]) == 6, cell
