#!/usr/bin/env python3
"""CPU-only SMPL-X -> Unitree G1 ref_dof_pos retarget pipeline.

Chains two pieces, same pattern as
motion/motion_builder/combined/walk_and_wave/scripts/kimodo_npz_to_holomotion.py:

  1. GMR (video_to_motion/cpu_pipeline/gmr, recovered from git history at
     7ddc51b^:twist_deploy/GMR) -- GeneralMotionRetargeting IK against the
     Unitree G1 MJCF, driven directly (NOT via scripts/smplx_to_robot.py,
     which unconditionally opens a MuJoCo passive viewer via
     RobotMotionViewer/mjv.launch_passive and therefore needs a display even
     for a --save_path-only run; we skip the viewer entirely so this runs
     headless/CPU-only). Produces the same in-memory schema GMR's own script
     would pickle: fps, root_pos [T,3], root_rot [T,4] xyzw, dof_pos [T,29].
  2. motion/holomotion/scripts/pkl_to_offline_npz.py's existing
     forward_kinematics_all_frames / central_diff / angular_velocity_from_quats
     + convert_legacy_offline_npz -- same FK/schema plumbing every other
     source in this repo goes through, so output matches
     motion/motion_builder/combined/walk_and_wave/converted/waving_holomotion.npz's schema exactly.

Must run in the `4d-humans-cpu` conda env (has GMR installed -e, torch-cpu,
mujoco, scipy).

Usage:
    python retarget_cpu.py <smplx_npz_input> <output_ref_dof_pos.npz> [--fps 30]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE / "gmr"
SMPLX_BODY_MODEL_PATH = GMR_ROOT / "assets" / "body_models"

HOLOMOTION_SCRIPTS = HERE.parents[1] / "holomotion" / "scripts"
sys.path.insert(0, str(HOLOMOTION_SCRIPTS))
from pkl_to_offline_npz import (  # noqa: E402
    angular_velocity_from_quats,
    central_diff,
    convert_legacy_offline_npz,
    forward_kinematics_all_frames,
)

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def retarget_smplx_to_g1(smplx_file: Path, fps: int):
    """Run GMR's IK retarget for unitree_g1, headless (no viewer).

    Returns (aligned_fps, root_pos [T,3], root_rot_xyzw [T,4], dof_pos [T,29]) --
    the same schema GMR's own scripts/smplx_to_robot.py --save_path would pickle.
    """
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        str(smplx_file), str(SMPLX_BODY_MODEL_PATH)
    )
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
    )

    qpos_list = []
    for i in range(len(smplx_data_frames)):
        qpos = retarget.retarget(smplx_data_frames[i])
        qpos_list.append(qpos)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # GMR's internal qpos root quat is wxyz (MuJoCo free-joint convention) --
    # convert to xyzw, matching what scripts/smplx_to_robot.py pickles and
    # what pkl_to_offline_npz.load_pkl_motion/forward_kinematics_all_frames expect.
    root_rot_xyzw = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    return aligned_fps, root_pos, root_rot_xyzw, dof_pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("smplx_input", type=Path, help="SMPL-X motion npz (pose_body/root_orient/trans/betas/gender)")
    ap.add_argument("output", type=Path, help="Final HoloMotion v1.4 offline-tracking .npz (ref_dof_pos schema)")
    ap.add_argument("--fps", type=int, default=30, help="Target retarget fps (default 30)")
    ap.add_argument("--motion-key", default=None)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    fps, root_pos, root_rot_xyzw, dof_pos = retarget_smplx_to_g1(args.smplx_input, args.fps)
    if dof_pos.shape[-1] != 29:
        raise ValueError(f"Expected 29-DOF dof_pos (HoloMotion's G1 convention), got {dof_pos.shape[-1]}")
    dt = 1.0 / fps
    T = dof_pos.shape[0]
    print(f"Retargeted {args.smplx_input.name}: {T} frames @ {fps} fps, 29-DOF")

    global_translation, global_rotation_quat = forward_kinematics_all_frames(root_pos, root_rot_xyzw, dof_pos)
    dof_vel = central_diff(dof_pos, dt)
    global_velocity = central_diff(global_translation, dt)
    global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

    motion_key = args.motion_key or args.smplx_input.stem
    legacy_path = args.output.with_suffix(".legacy.npz")
    metadata = {
        "motion_key": motion_key,
        "motion_fps": fps,
        "original_num_frames": T,
        "source": "GMR smplx->g1 retarget (video_to_motion/cpu_pipeline/retarget_cpu.py)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        legacy_path,
        metadata=np.asarray(json.dumps(metadata)),
        dof_pos=dof_pos.astype(np.float32),
        dof_vels=dof_vel.astype(np.float32),
        global_translation=global_translation.astype(np.float32),
        global_rotation_quat=global_rotation_quat.astype(np.float32),
        global_velocity=global_velocity.astype(np.float32),
        global_angular_velocity=global_angular_velocity.astype(np.float32),
    )
    print(f"Wrote intermediate legacy npz: {legacy_path}")

    result = convert_legacy_offline_npz(legacy_path, args.output, overwrite=args.overwrite)
    legacy_path.unlink()
    print(f"Wrote final v1.4 offline-tracking npz: {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
