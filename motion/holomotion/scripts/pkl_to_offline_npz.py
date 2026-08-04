#!/usr/bin/env python3
"""Convert a TWIST/GMR-schema motion .pkl into a HoloMotion v1.4 offline-tracking .npz.

Why this exists: the .pkl files under twist_deploy/motion_staging/ were produced
by GMR for the TWIST pipeline (root_pos, root_rot[xyzw], dof_pos, local_body_pos,
link_body_list) -- a different schema from what HoloMotion's deploy stack expects
(ref_dof_pos/ref_global_translation/ref_global_rotation_quat/... at 29 DOF, 30
bodies, computed via full forward kinematics). HoloMotion's own conversion tooling
(holomotion_preprocess.py, HumanoidBatch) needs the SMPLSim submodule + open3d --
heavy, x86-side-only tooling we deliberately didn't vendor. This script instead
computes the same forward kinematics with plain MuJoCo (already used throughout
this repo, lightweight, works fine on this Jetson) against HoloMotion's own G1
29-DOF MJCF, so the body ordering/naming is guaranteed to match what deployment
expects -- then hands off to HoloMotion's OWN convert_legacy_offline_npz.py for
the final schema/validation pass, rather than re-deriving that format by hand.

Usage:
    python pkl_to_offline_npz.py <source.pkl> <output.npz> [--motion-key NAME]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# The staged .pkl motions were pickled under numpy>=2.0 (numpy._core layout);
# this venv's numpy predates that rename. Alias the old layout under the new
# name so unpickling doesn't need numpy._core to actually exist.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    for _sub in ("multiarray", "numeric", "numerictypes", "umath", "_multiarray_umath"):
        if hasattr(np.core, _sub):
            sys.modules[f"numpy._core.{_sub}"] = getattr(np.core, _sub)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MJCF_PATH = REPO_ROOT / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml"

# deployment/unitree_g1_ros2_29dof/src/humanoid_policy/offline_motion_conversion.py
# only imports numpy/json/pathlib/tempfile -- safe to import directly without the
# deploy container or its other (heavier) ROS2/ONNX dependencies.
sys.path.insert(
    0, str(REPO_ROOT / "deployment/unitree_g1_ros2_29dof/src")
)
from humanoid_policy.offline_motion_conversion import convert_legacy_offline_npz  # noqa: E402


def load_pkl_motion(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    fps = float(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)      # [T, 3]
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)      # [T, 4] xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)        # [T, 29]
    if dof_pos.shape[-1] != 29:
        raise ValueError(
            f"Expected 29-DOF dof_pos (HoloMotion's G1 convention), got {dof_pos.shape[-1]}. "
            "This script is for the raw GMR retarget .pkl, not a TWIST-reduced 23-DOF one."
        )
    return fps, root_pos, root_rot, dof_pos


def forward_kinematics_all_frames(root_pos, root_rot_xyzw, dof_pos):
    """Returns global_translation [T,30,3] and global_rotation_quat [T,30,4] (xyzw),
    body order matching HoloMotion's own MJCF (bodies 1..30, skipping mj body 0='world')."""
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    assert model.nbody == 31, f"expected 30 bodies + world, got {model.nbody}"

    T = dof_pos.shape[0]
    translation = np.zeros((T, 30, 3), dtype=np.float64)
    rotation = np.zeros((T, 30, 4), dtype=np.float64)

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        # mujoco free-joint quat is wxyz; our pkl stores xyzw
        data.qpos[3:7] = root_rot_xyzw[t][[3, 0, 1, 2]]
        data.qpos[7:36] = dof_pos[t]
        mujoco.mj_forward(model, data)
        translation[t] = data.xpos[1:31]
        # xquat is wxyz -> convert to xyzw for HoloMotion's convention
        rotation[t] = data.xquat[1:31][:, [1, 2, 3, 0]]

    return translation, rotation


def central_diff(x, dt):
    """[T, ...] -> [T, ...] velocity via central differences, edge frames use one-sided diff."""
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def angular_velocity_from_quats(rotation_xyzw, dt):
    """[T,B,4] xyzw -> [T,B,3] world-frame angular velocity via consecutive-frame
    relative rotation (rotvec/dt), central-differenced the same way as linear vel."""
    T, B, _ = rotation_xyzw.shape
    rotvec_step = np.zeros((T, B, 3), dtype=np.float64)
    for b in range(B):
        rots = R.from_quat(rotation_xyzw[:, b, :])
        rel_fwd = (rots[1:] * rots[:-1].inv()).as_rotvec()  # [T-1, 3], frame t -> t+1
        rotvec_step[:-1, b, :] += rel_fwd
        rotvec_step[1:, b, :] += rel_fwd
    # average the two adjacent step-rotations at interior frames (mirrors central_diff
    # weighting), single-sided at the edges
    counts = np.full((T, 1), 2.0)
    counts[0] = 1.0
    counts[-1] = 1.0
    return rotvec_step / counts[:, :, None] / dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="TWIST/GMR-schema motion .pkl (29-DOF)")
    ap.add_argument("output", type=Path, help="Final HoloMotion v1.4 offline-tracking .npz")
    ap.add_argument("--motion-key", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fps, root_pos, root_rot, dof_pos = load_pkl_motion(args.source)
    dt = 1.0 / fps
    T = dof_pos.shape[0]
    print(f"Loaded {args.source.name}: {T} frames @ {fps} fps, 29-DOF")

    global_translation, global_rotation_quat = forward_kinematics_all_frames(
        root_pos, root_rot, dof_pos
    )
    dof_vel = central_diff(dof_pos, dt)
    global_velocity = central_diff(global_translation, dt)
    global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

    motion_key = args.motion_key or args.source.stem
    legacy_path = args.output.with_suffix(".legacy.npz")
    metadata = {
        "motion_key": motion_key,
        "motion_fps": fps,
        "original_num_frames": T,
    }
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

    result = convert_legacy_offline_npz(
        legacy_path, args.output, overwrite=args.overwrite
    )
    legacy_path.unlink()
    print(f"Wrote final v1.4 offline-tracking npz: {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
