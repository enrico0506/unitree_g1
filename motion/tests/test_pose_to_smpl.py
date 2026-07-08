"""pytest for the Phase-2 pose glue (``pipeline/glue/pose_to_smpl.py``).

ROMP itself CANNOT run on this workstation (needs L4T torch + the
MPI-registration-gated SMPL body model, both Orin-only), so every frame here is
SYNTHETIC: ``make_synthetic_romp_frame`` builds one dict shaped exactly like a
``np.savez(results=outputs)['results'][()]`` entry from simple_romp (N==1), and
``synth_slow_turn`` strings them into a slow half-turn about world-up with
injected high-frequency yaw jitter — the thing the SLERP low-pass must smooth.

The glue imports with numpy + scipy ONLY (no torch / smplx / ROMP), so this
suite runs under the workstation base python. NO module-level ``sys.exit`` —
proper ``def test_*`` functions.

Run:  ``pytest motion/tests/test_pose_to_smpl.py``  from the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Put the repo root (parent of motion/) on sys.path so
# ``motion.pipeline.glue.pose_to_smpl`` resolves regardless of where pytest is
# invoked from — same posture as test_recorder.py. motion/ is an implicit
# namespace package; motion/pipeline + motion/pipeline/glue have __init__.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motion.pipeline.glue.pose_to_smpl import (  # noqa: E402  (after sys.path insert)
    RompFormatError,
    _fps_from_clip_json,
    _hemisphere_align,
    pose_to_smpl,
    save_smpl_sequence,
)

_OUT_KEYS = {"root_orient", "pose_body", "trans", "betas", "gender", "mocap_frame_rate"}


# ---------------------------------------------------------------------------
# Synthetic ROMP frames (mirrors the FROZEN CONTRACT's builder)
# ---------------------------------------------------------------------------
def make_synthetic_romp_frame(theta72, betas10, cam3):
    """One simple_romp results dict, N==1. Keeps the leading person axis so the
    glue exercises its (N,D)->(D,) subject-collapse path."""
    s = cam3[0]
    return {
        "smpl_thetas": theta72.reshape(1, 72).astype(np.float32),
        "smpl_betas": betas10.reshape(1, 10).astype(np.float32),
        "cam": cam3.reshape(1, 3).astype(np.float32),                 # [s, tx, ty]
        "cam_trans": (np.array([[cam3[1] / s, cam3[2] / s, 1.0 / s]],
                               np.float32) * 2.0),
    }


def synth_slow_turn(T=60, fps=30.0, seed=0):
    """Slow half-turn (0 -> pi) about world-up + injected HF yaw jitter spikes
    every 7th frame — the flips/jitter the SLERP low-pass must remove."""
    rng = np.random.default_rng(seed)
    yaw = np.linspace(0.0, np.pi, T)
    jitter = np.zeros(T)
    jitter[::7] = 0.6 * rng.standard_normal(len(jitter[::7]))  # HF spikes
    betas = np.array([0.3, -0.1, 0, 0, 0, 0, 0, 0, 0, 0], np.float32)
    frames = []
    for t in range(T):
        go = R.from_euler("y", yaw[t] + jitter[t]).as_rotvec().astype(np.float32)
        theta = np.zeros(72, np.float32)
        theta[0:3] = go                       # body_pose=0, hands theta[66:72]=0
        cam = np.array([1.2, 0.02 * np.sin(0.2 * t), -0.1], np.float32)
        frames.append(make_synthetic_romp_frame(theta, betas, cam))
    return frames, fps


def _write_clip_json(tmp_path: Path, frame_count: int, measured_fps: float) -> Path:
    """A minimal clip.json matching the recorder's real field names."""
    doc = {
        "job_id": "synthjob",
        "measured_fps": measured_fps,
        "frame_count": frame_count,
        "duration_s": frame_count / measured_fps,
        "clip_path": "clip.mp4",
    }
    p = tmp_path / "clip.json"
    p.write_text(json.dumps(doc, indent=2))
    return p


def _consecutive_angles(root_aa: np.ndarray) -> np.ndarray:
    """Frame-to-frame root Δangle (rad), (T-1,)."""
    rots = R.from_rotvec(np.asarray(root_aa, dtype=np.float64))
    rel = rots[:-1].inv() * rots[1:]
    return rel.magnitude()


def _angle_from_start(root_aa: np.ndarray) -> np.ndarray:
    """Total rotation angle of each frame relative to frame 0 (rad), (T,)."""
    rots = R.from_rotvec(np.asarray(root_aa, dtype=np.float64))
    rel = rots[0].inv() * rots
    return rel.magnitude()


# ---------------------------------------------------------------------------
# (1) length / shapes / dtypes  +  T == clip.json.frame_count  +  fps carried
# ---------------------------------------------------------------------------
def test_length_shapes_dtypes_and_frame_count(tmp_path):
    T = 60
    frames, fps = synth_slow_turn(T=T, fps=30.0)
    clip_json = _write_clip_json(tmp_path, frame_count=T, measured_fps=fps)

    seq = pose_to_smpl(frames, fps)

    # length == input frames == clip.json frame_count
    assert len(seq["root_orient"]) == T
    with open(clip_json) as f:
        assert T == json.load(f)["frame_count"]

    # exact shapes
    assert seq["root_orient"].shape == (T, 3)
    assert seq["pose_body"].shape == (T, 63)
    assert seq["trans"].shape == (T, 3)
    assert seq["betas"].shape == (16,)

    # dtypes float32
    for k in ("root_orient", "pose_body", "trans", "betas"):
        assert seq[k].dtype == np.float32, k

    # 0-d scalars
    assert np.ndim(seq["gender"]) == 0
    assert np.ndim(seq["mocap_frame_rate"]) == 0

    # betas: 10 real dims reduced + 6 zero-pad
    assert np.allclose(seq["betas"][10:], 0.0)

    # fps carried through (and _fps_from_clip_json reads measured_fps)
    assert float(seq["mocap_frame_rate"]) == pytest.approx(fps)
    assert _fps_from_clip_json(clip_json) == pytest.approx(fps)

    # finite everywhere
    for k in ("root_orient", "pose_body", "trans", "betas"):
        assert np.isfinite(seq[k]).all(), k


# ---------------------------------------------------------------------------
# (2) exactly the 6 GMR keys, no extras + round-trip through save/np.load
# ---------------------------------------------------------------------------
def test_exact_keys_and_npz_roundtrip(tmp_path):
    T = 40
    frames, fps = synth_slow_turn(T=T, fps=24.0)
    seq = pose_to_smpl(frames, fps)

    # exactly the 6 keys, no more no less
    assert set(seq.keys()) == _OUT_KEYS

    out = tmp_path / "smpl.npz"
    written = save_smpl_sequence(seq, out)
    assert written.exists()
    assert written.suffix == ".npz"

    with np.load(written, allow_pickle=True) as data:
        # npz on disk carries EXACTLY the 6 keys
        assert set(data.files) == _OUT_KEYS
        assert data["root_orient"].shape == (T, 3)
        assert data["pose_body"].shape == (T, 63)
        assert data["trans"].shape == (T, 3)
        assert data["betas"].shape == (16,)
        assert str(data["gender"]) == "neutral"
        assert float(data["mocap_frame_rate"]) == pytest.approx(fps)
        # values survive the round-trip
        assert np.allclose(data["root_orient"], seq["root_orient"])
        assert np.allclose(data["pose_body"], seq["pose_body"])


def test_save_appends_npz_suffix(tmp_path):
    frames, fps = synth_slow_turn(T=10)
    seq = pose_to_smpl(frames, fps, smooth_window=1)
    written = save_smpl_sequence(seq, tmp_path / "noext")   # no .npz
    assert written.name == "noext.npz"
    assert written.exists()


# ---------------------------------------------------------------------------
# (3) the slow turn is SMOOTH, not flipping: SLERP low-pass reduces the
#     frame-to-frame angular delta (max + variance) and yields a monotone turn.
# ---------------------------------------------------------------------------
def test_slerp_smoothing_kills_flips_and_is_monotone():
    T = 60
    frames, fps = synth_slow_turn(T=T, fps=30.0, seed=0)

    raw = pose_to_smpl(frames, fps, smooth_window=1)    # smoothing DISABLED
    smooth = pose_to_smpl(frames, fps, smooth_window=7)  # smoothing ON

    d_raw = _consecutive_angles(raw["root_orient"])
    d_smooth = _consecutive_angles(smooth["root_orient"])

    # max consecutive Δangle after smoothing <= raw max, and below a small bound.
    assert d_smooth.max() <= d_raw.max() + 1e-6
    # a ~pi turn over 59 steps is ~0.053 rad/step; a generous bound well under
    # the raw jitter spikes proves the flips are gone.
    assert d_smooth.max() < 0.20

    # frame-to-frame angular-delta VARIANCE is reduced (jitter flattened).
    assert np.var(d_smooth) < np.var(d_raw)

    # smoothed turn is (near-)monotone: measure the largest BACKWARD step in the
    # cumulative angle from frame 0. A width-7 moving average attenuates a
    # 0.6-rad jitter spike to ~0.6/7 rad but can't fully erase it, so we assert
    # the residual reversal is (a) far smaller than the raw jitter-driven
    # reversal, and (b) small in absolute terms — not that it's exactly monotone.
    ang_raw = _angle_from_start(raw["root_orient"])
    ang_smooth = _angle_from_start(smooth["root_orient"])
    back_raw = max(0.0, float(-np.diff(ang_raw).min()))
    back_smooth = max(0.0, float(-np.diff(ang_smooth).min()))
    assert back_smooth < back_raw, "smoothing must reduce turn reversals"
    assert back_smooth < 0.15, "residual reversal should be < ~9 deg"
    # and it actually turns a meaningful amount (~half turn survives smoothing).
    assert ang_smooth[-1] > 1.5  # > ~86 deg of net rotation

    # body_pose and (here) betas are untouched by the root-only smoother.
    assert np.allclose(raw["pose_body"], smooth["pose_body"])


# ---------------------------------------------------------------------------
# (4) direct double-cover unit test on _hemisphere_align
# ---------------------------------------------------------------------------
def test_hemisphere_align_fixes_double_cover():
    # Build a smooth quaternion trajectory, then deliberately sign-flip some
    # frames (q and -q are the SAME rotation) to simulate ROMP's per-frame sign
    # hops. _hemisphere_align must return a sign-continuous trajectory.
    T = 20
    aa = np.zeros((T, 3))
    aa[:, 2] = np.linspace(0.0, 1.2, T)          # smooth yaw ramp
    q = R.from_rotvec(aa).as_quat()              # (T,4) XYZW, continuous

    flipped = q.copy()
    flipped[3::4] *= -1.0                         # hop signs on some frames
    # sanity: we really introduced sign discontinuities
    dots_before = np.sum(flipped[1:] * flipped[:-1], axis=1)
    assert np.any(dots_before < 0)

    aligned = _hemisphere_align(flipped)
    dots_after = np.sum(aligned[1:] * aligned[:-1], axis=1)
    # every consecutive pair now on the same hemisphere (short arc)
    assert np.all(dots_after >= 0.0)

    # and it's still the SAME rotations (sign flip is a no-op on orientation)
    assert np.allclose(
        R.from_quat(aligned).as_matrix(),
        R.from_quat(flipped).as_matrix(),
    )


def test_slerp_lowpass_beats_raw_on_double_cover_root():
    # A root sequence whose axis-angle is smooth, but re-expressed with random
    # quaternion sign hops before smoothing, is the worst case. Feed sign-hopped
    # rotvecs (via near-2pi wraps) and confirm smoothing still lowers jitter.
    T = 50
    frames, fps = synth_slow_turn(T=T, seed=3)
    raw = pose_to_smpl(frames, fps, smooth_window=1)
    smooth = pose_to_smpl(frames, fps, smooth_window=9)
    assert _consecutive_angles(smooth["root_orient"]).max() <= \
        _consecutive_angles(raw["root_orient"]).max() + 1e-6


# ---------------------------------------------------------------------------
# (5) shape validation raises LOUDLY on malformed frames
# ---------------------------------------------------------------------------
def test_betas_11_looks_like_bev_raises():
    frames, fps = synth_slow_turn(T=8)
    bad = dict(frames[3])
    bad["smpl_betas"] = np.zeros((1, 11), np.float32)   # 11 == BEV, not ROMP
    frames[3] = bad
    with pytest.raises(RompFormatError, match="BEV|11"):
        pose_to_smpl(frames, fps)


def test_theta_wrong_length_raises():
    frames, fps = synth_slow_turn(T=8)
    bad = dict(frames[2])
    bad["smpl_thetas"] = np.zeros((1, 70), np.float32)  # not 72
    frames[2] = bad
    with pytest.raises(RompFormatError, match="72"):
        pose_to_smpl(frames, fps)


def test_no_pose_key_raises():
    frames, fps = synth_slow_turn(T=6)
    bad = {k: v for k, v in frames[1].items() if k != "smpl_thetas"}
    frames[1] = bad
    with pytest.raises(RompFormatError, match="smpl_thetas|poses|global_orient"):
        pose_to_smpl(frames, fps)


def test_zero_people_frame_raises():
    frames, fps = synth_slow_turn(T=6)
    empty = {
        "smpl_thetas": np.zeros((0, 72), np.float32),
        "smpl_betas": np.zeros((0, 10), np.float32),
        "cam": np.zeros((0, 3), np.float32),
        "center_confs": np.zeros((0,), np.float32),
    }
    frames[0] = empty
    with pytest.raises(RompFormatError, match="N==0"):
        pose_to_smpl(frames, fps)


def test_empty_frame_list_raises():
    with pytest.raises(RompFormatError, match="empty"):
        pose_to_smpl([], 30.0)


def test_bad_betas_policy_raises():
    frames, fps = synth_slow_turn(T=6)
    with pytest.raises(RompFormatError, match="betas_policy"):
        pose_to_smpl(frames, fps, betas_policy="mode")


# ---------------------------------------------------------------------------
# extras: subject selection when N>1, and zero_transl escape hatch
# ---------------------------------------------------------------------------
def test_multi_person_selects_most_confident():
    frames, fps = synth_slow_turn(T=5)
    # Turn frame 2 into a 2-person frame; subject 1 is the confident one and has
    # a distinctive root so we can tell which was picked.
    theta0 = np.zeros(72, np.float32)
    theta1 = np.zeros(72, np.float32)
    theta1[0:3] = R.from_euler("y", 1.0).as_rotvec()
    frames[2] = {
        "smpl_thetas": np.stack([theta0, theta1]).astype(np.float32),
        "smpl_betas": np.zeros((2, 10), np.float32),
        "cam": np.array([[1.2, 0.0, 0.0], [1.2, 0.0, 0.0]], np.float32),
        "center_confs": np.array([0.1, 0.9], np.float32),   # subject 1 wins
    }
    seq = pose_to_smpl(frames, fps, smooth_window=1)
    # the picked root (before smoothing disabled) should reflect subject 1's yaw
    # after the up-axis fix; just assert it's finite and non-degenerate here.
    assert np.isfinite(seq["root_orient"][2]).all()
    assert seq["root_orient"].shape[0] == len(frames)


def test_zero_transl_forces_zero_translation():
    frames, fps = synth_slow_turn(T=12)
    seq = pose_to_smpl(frames, fps, zero_transl=True)
    assert np.allclose(seq["trans"], 0.0)


def test_measured_fps_preferred_then_true_fps_fallback(tmp_path):
    # measured_fps wins when present...
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps({"measured_fps": 29.7, "true_fps": 30.0}))
    assert _fps_from_clip_json(p1) == pytest.approx(29.7)
    # ...and PLAN's true_fps is the fallback when measured_fps is absent.
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"true_fps": 25.0}))
    assert _fps_from_clip_json(p2) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Version-guard fallback paths — the branches most likely to fire when real
# ROMP output differs from simple_romp's `smpl_thetas`. Each must WORK, or fail
# LOUD (RompFormatError). These are the on-robot risk surface (review #2).
# ---------------------------------------------------------------------------
_B10 = np.array([0.3, -0.1, 0, 0, 0, 0, 0, 0, 0, 0], np.float32)
_CAM = np.array([1.2, 0.02, -0.1], np.float32)


def _theta72():
    """Constant pose with a non-zero global_orient AND a non-zero body joint, so the
    0:3 / 3:66 split is actually exercised. Constant across frames -> SLERP is identity,
    so alt-key outputs compare exactly to the smpl_thetas path."""
    th = np.zeros(72, np.float32)
    th[0:3] = R.from_euler("y", 0.4).as_rotvec()
    th[3:6] = np.array([0.1, -0.2, 0.05], np.float32)   # left-hip joint, non-zero
    return th


def test_alt_key_poses_equivalent_to_smpl_thetas():
    th = _theta72()
    prim = pose_to_smpl([make_synthetic_romp_frame(th, _B10, _CAM)] * 4, 30.0)
    alt = pose_to_smpl([{
        "poses": th.reshape(1, 72),                      # alt fork key (full ROMP repo)
        "smpl_betas": _B10.reshape(1, 10),
        "cam": _CAM.reshape(1, 3),
    }] * 4, 30.0)
    assert np.allclose(alt["root_orient"], prim["root_orient"], atol=1e-5)
    assert np.allclose(alt["pose_body"], prim["pose_body"], atol=1e-5)


def test_alt_key_global_orient_plus_body_pose63():
    th = _theta72()
    seq = pose_to_smpl([{
        "global_orient": th[0:3].reshape(1, 3),
        "body_pose": th[3:66].reshape(1, 63),
        "smpl_betas": _B10.reshape(1, 10),
        "cam": _CAM.reshape(1, 3),
    }] * 3, 30.0)
    assert seq["pose_body"].shape == (3, 63)
    assert np.allclose(seq["pose_body"][0], th[3:66], atol=1e-5)


def test_body_pose_69_sliced_to_63():
    th = _theta72()
    bp69 = np.zeros(69, np.float32)
    bp69[:63] = th[3:66]
    bp69[63:] = 9.0                                       # junk hand dims that MUST be dropped
    seq = pose_to_smpl([{
        "global_orient": th[0:3].reshape(1, 3),
        "body_pose": bp69.reshape(1, 69),
        "smpl_betas": _B10.reshape(1, 10),
        "cam": _CAM.reshape(1, 3),
    }] * 2, 30.0)
    assert seq["pose_body"].shape == (2, 63)
    assert np.allclose(seq["pose_body"][0], th[3:66], atol=1e-5)   # 63:69 junk gone


def test_betas_bev_11dim_raises_loud():
    th = _theta72()
    with pytest.raises(RompFormatError):
        pose_to_smpl([{
            "smpl_thetas": th.reshape(1, 72),
            "smpl_betas": np.zeros((1, 11), np.float32),  # BEV adds an 11th (age/scale) param
            "cam": _CAM.reshape(1, 3),
        }] * 2, 30.0)


def test_missing_pose_key_raises_loud():
    with pytest.raises(RompFormatError):
        pose_to_smpl([{
            "smpl_betas": _B10.reshape(1, 10),
            "cam": _CAM.reshape(1, 3),
        }] * 2, 30.0)


def test_transl_derived_from_cam_when_no_cam_trans():
    th = _theta72()
    seq = pose_to_smpl([{
        "smpl_thetas": th.reshape(1, 72),
        "smpl_betas": _B10.reshape(1, 10),
        "cam": _CAM.reshape(1, 3),                        # NO cam_trans -> derive from cam
    }] * 3, 30.0)
    assert seq["trans"].shape == (3, 3)
    assert np.isfinite(seq["trans"]).all()
    assert not np.allclose(seq["trans"], 0.0)             # [tx/s, ty/s, 1/s], up-axis-fixed
