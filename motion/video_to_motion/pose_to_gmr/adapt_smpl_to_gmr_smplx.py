from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


# Same camera-frame -> world-frame (Z-up) axis correction GMR's own
# load_gvhmr_pred_file/get_gvhmr_data_offline_fast applies to GVHMR's
# camera-relative output (see twist_deploy/GMR/general_motion_retargeting/
# utils/smpl.py). HMR2/4D-Humans' global_orient is in the same kind of
# camera-frame (Y-down-ish) space, so it needs the identical fix — without
# it the retargeted robot ends up upside down.
_AXIS_FIX_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64).T
_AXIS_FIX_ROT = R.from_matrix(_AXIS_FIX_MATRIX)


def _fix_root_orient_axes(root_orient_aa: np.ndarray) -> np.ndarray:
    """Apply the camera->world axis correction to a sequence of axis-angle root rotations."""
    rot = R.from_rotvec(root_orient_aa)
    fixed = _AXIS_FIX_ROT * rot
    return fixed.as_rotvec().astype(np.float32)


# SMPL-X body_pose joint indices (0-indexed into the 21-joint array; body_pose
# excludes the pelvis/root, so index = full-skeleton smplx.joint_names index - 1).
# Verified via `from smplx.joint_names import JOINT_NAMES`.
_LEFT_HIP, _RIGHT_HIP, _SPINE1, _LEFT_KNEE, _RIGHT_KNEE = 0, 1, 2, 3, 4

# These five joints showed a large, near-perfectly-consistent rotation across
# every single frame of the source clip (std/mean ratio ~0.03-0.06 — e.g. knees
# bent ~35 deg on literally every frame, std ~1 deg). That consistency is the
# signature of a static estimator bias, not captured motion: a real stand-and-wave
# clip doesn't hold a rigid 35 degree knee bend the whole time. This is a known
# monocular-pose-estimation artifact (HMR2/SMPL fitters often compensate for
# depth/perspective ambiguity on a standing subject with a systematic
# hip-flexion + knee-bend + spine-flexion bias). Left uncorrected, this chain is
# what produced the "C-shaped" forward hunch — not the root orientation, which
# was already fixed and empirically shown to have ~no effect on it.
_BIAS_JOINTS = [_LEFT_HIP, _RIGHT_HIP, _SPINE1, _LEFT_KNEE, _RIGHT_KNEE]


def _debias_joints(pose_body_by_joint: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    """Subtract each joint's per-clip mean rotation (quaternion mean), frame by frame.

    Removes the constant bias while preserving genuine frame-to-frame motion
    (e.g. any real wave-induced wobble in these joints survives; only the
    static offset is removed). pose_body_by_joint is (N, 21, 3) axis-angle.
    """
    out = pose_body_by_joint.copy()
    for j in joint_indices:
        quats = R.from_rotvec(pose_body_by_joint[:, j, :]).as_quat()
        ref = quats[0]
        signs = np.sign(quats @ ref)
        signs[signs == 0] = 1.0
        aligned = quats * signs[:, None]
        mean_q = aligned.mean(axis=0)
        mean_q /= np.linalg.norm(mean_q)
        mean_rot = R.from_quat(mean_q)
        frame_rots = R.from_quat(aligned)
        out[:, j, :] = (mean_rot.inv() * frame_rots).as_rotvec()
    return out.astype(np.float32)


def _smooth_axis_angle_sequence(aa: np.ndarray, window: int) -> np.ndarray:
    """Temporal smoothing for a (N, ..., 3) axis-angle sequence via quaternion averaging.

    HMR2 estimates every frame independently (no world-consistent tracking),
    so per-frame pose is noisy and visibly jittery once retargeted. A small
    sliding-window quaternion average removes high-frequency jitter without
    needing a real motion prior.
    """
    if window <= 1:
        return aa
    orig_shape = aa.shape
    num_frames = orig_shape[0]
    flat = aa.reshape(num_frames, -1, 3)
    num_joints = flat.shape[1]
    smoothed = np.empty_like(flat)
    half = window // 2
    for j in range(num_joints):
        quats = R.from_rotvec(flat[:, j, :]).as_quat()  # (N, 4) x,y,z,w
        for t in range(num_frames):
            lo = max(0, t - half)
            hi = min(num_frames, t + half + 1)
            window_quats = quats[lo:hi]
            # Guard against sign flips (q and -q represent the same rotation)
            # before averaging, otherwise antipodal quaternions cancel out.
            ref = window_quats[0]
            signs = np.sign(window_quats @ ref)
            signs[signs == 0] = 1.0
            aligned = window_quats * signs[:, None]
            mean_q = aligned.mean(axis=0)
            mean_q /= np.linalg.norm(mean_q)
            smoothed[t, j, :] = R.from_quat(mean_q).as_rotvec()
    return smoothed.reshape(orig_shape).astype(np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a stage-1 HMR2/4D-Humans SMPL export into the SMPL-X npz "
            "schema GMR's load_smplx_file()/smplx_to_robot.py expects."
        )
    )
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate the stage-1 export was extracted at (matches run_pipeline.py --fps).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=15,
        help=(
            "Odd-sized sliding window (frames) for quaternion smoothing. 1 disables smoothing. "
            "HMR2 estimates every frame independently (no temporal tracking), so the raw signal "
            "has persistent high-frequency noise — frame-to-frame direction flips ~40-45% of the "
            "time regardless of window size, moving-average smoothing only shrinks the amplitude, "
            "not the flip rate. window=5 left ~1.2mm/~0.9deg residual jitter on knees/feet, "
            "visible as an unnatural 'swinging' even though it never breaks floor contact "
            "(measured, not assumed). window=15 cuts that to ~0.3mm/~0.4deg, below what's visibly "
            "perceptible. Larger still starts eating real motion, this is a practical floor, not a fix."
        ),
    )
    parser.add_argument(
        "--no_static_root",
        dest="static_root",
        action="store_false",
        default=True,
        help=(
            "Use HMR2's raw pred_cam_t as root translation instead of holding "
            "the root static at the origin. Off (i.e. static root) by default, "
            "since pred_cam_t is camera-space, not a real world position."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_npz = Path(args.input_npz).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_npz, allow_pickle=False) as data:
        global_orient = data["global_orient_axis_angle"].astype(np.float32)
        body_pose = data["smpl_pose_21_axis_angle"].astype(np.float32)
        betas = data["betas"].astype(np.float32)
        trans = data["pred_cam_t"].astype(np.float32)

    if global_orient.ndim != 3 or global_orient.shape[1:] != (1, 3):
        raise ValueError(f"Expected global_orient_axis_angle with shape [N,1,3], got {global_orient.shape}")
    if body_pose.ndim != 3 or body_pose.shape[1:] != (21, 3):
        raise ValueError(f"Expected smpl_pose_21_axis_angle with shape [N,21,3], got {body_pose.shape}")
    num_frames = body_pose.shape[0]
    if global_orient.shape[0] != num_frames or betas.shape[0] != num_frames or trans.shape[0] != num_frames:
        raise ValueError("Frame count mismatch between global_orient/body_pose/betas/trans")

    root_orient = global_orient.reshape(num_frames, 3).astype(np.float64)
    pose_body_by_joint = body_pose.reshape(num_frames, 21, 3)

    # Fix 1: camera-frame -> world-frame axis correction on root orientation.
    # Without this the puppet retargets upside down (see _AXIS_FIX_MATRIX doc).
    root_orient = _fix_root_orient_axes(root_orient)

    # Fix 1b: debias the hip/spine1/knee flexion chain. This — not root
    # orientation — was the actual source of the forward hunch (see
    # _BIAS_JOINTS doc). Confirmed by a control test: zeroing root_orient
    # entirely had ~no effect on the hunch, proving it lived in body_pose.
    pose_body_by_joint = _debias_joints(pose_body_by_joint, _BIAS_JOINTS)
    pose_body = pose_body_by_joint.reshape(num_frames, 63)

    # Fix 2: temporal smoothing. HMR2 has no cross-frame consistency, so raw
    # per-frame axis-angles jitter; smooth root orientation and body pose.
    root_orient = _smooth_axis_angle_sequence(
        root_orient.reshape(num_frames, 1, 3), args.smoothing_window
    ).reshape(num_frames, 3)
    pose_body = _smooth_axis_angle_sequence(
        pose_body.reshape(num_frames, 21, 3), args.smoothing_window
    ).reshape(num_frames, 63)

    # SMPL betas are 10-dim; SMPL-X wants 16. Pad with zeros (matches GMR's own
    # scripts/smpl_to_smplx.py convention) and collapse to a single per-sequence
    # shape vector (GMR's load_smplx_file only ever reads betas[0] / a flat (16,)).
    betas_16 = np.zeros((16,), dtype=np.float32)
    betas_16[:10] = betas[0]

    # Fix 3: pred_cam_t is HMR2's camera-space root offset, not a true
    # world-space root trajectory (this pipeline is monocular, single-camera —
    # there is no SLAM/world tracking). Feeding it straight into `trans` made
    # the retargeted robot fly around. Hold the root static horizontally
    # instead — correct for standing/waving clips, wrong for locomotion clips
    # (pass --no-static_root once a real world-space trajectory exists).
    #
    # Z still needs a real value, not 0: SMPL's canonical rest pose has the
    # pelvis joint defined near the body's local origin, not at a
    # floor-referenced world height. trans=[0,0,0] literally placed the
    # pelvis at ground level, so the retarget solved for a deep crouch to
    # keep the (already-planted) feet under it.
    #
    # G1_STANDING_PELVIS_HEIGHT_M is measured directly from the robot's own
    # MJCF geometry (twist_deploy/GMR/assets/unitree_g1/g1_mocap_29dof.xml),
    # not guessed from human anthropometry: with qpos=0 (straight legs) and
    # pelvis at z=0, G1's actual foot-sole collision spheres
    # (left/right_ankle_roll_link, the type=2/contype=1 geoms — not the
    # visual mesh, which has a much larger bounding sphere and overstates
    # foot depth) sit at z=-0.7919. A prior version used a generic human
    # hip-height/stature ratio (0.53 * estimated height) instead, which
    # happened to mostly work only because the (buggy, since-fixed) ~35deg
    # knee bend shortened the effective leg enough to mask a ~9cm mismatch;
    # once Fix 1b straightened the legs, that mismatch surfaced as feet
    # clipping ~2cm through the floor. This value is G1-specific, not
    # human-specific — if this adapter targets a different robot, remeasure.
    G1_STANDING_PELVIS_HEIGHT_M = 0.7919
    if args.static_root:
        trans = np.zeros_like(trans)
        trans[:, 2] = G1_STANDING_PELVIS_HEIGHT_M

    smplx_data = {
        "pose_body": pose_body,
        "betas": betas_16,
        "root_orient": root_orient,
        "trans": trans,
        "gender": np.array(args.gender),
        "mocap_frame_rate": np.array(args.fps),
    }

    np.savez(output_npz, **smplx_data)
    print(f"Wrote GMR-ready SMPL-X sequence to {output_npz}")
    print(
        "Shapes:",
        f"pose_body={pose_body.shape}",
        f"betas={betas_16.shape}",
        f"root_orient={root_orient.shape}",
        f"trans={trans.shape}",
        f"gender={args.gender}",
        f"mocap_frame_rate={args.fps}",
    )


if __name__ == "__main__":
    main()
