#!/usr/bin/env python3
"""Convert a HoloRetarget robot .h5 (ref_root_pos/ref_root_rot/ref_dof_pos)
into a HoloMotion v1.4 offline-tracking .npz, for sim2sim/deploy testing.

HoloRetarget's own output is minimal by design (root pose + dof pos only --
see docs/motion_retargeting.md: "derives joint velocity, root velocity,
projected gravity... through the shared motion-tracking observation module").
The offline-tracking schema needs the fuller per-body FK + velocities, so
this does the same MuJoCo-FK-plus-finite-difference pass used in
pkl_to_offline_npz.py, just reading from this H5 layout instead of a TWIST
pkl, then validates through HoloMotion's own schema checker exactly as before.

Usage:
    python holoretarget_h5_to_offline_npz.py <robot.h5> <output.npz> [--fps 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MJCF_PATH = REPO_ROOT / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml"

sys.path.insert(0, str(REPO_ROOT / "deployment/unitree_g1_ros2_29dof/src"))
from humanoid_policy.offline_motion_conversion import convert_legacy_offline_npz  # noqa: E402


def forward_kinematics_all_frames(root_pos, root_rot_xyzw, dof_pos):
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    assert model.nbody == 31, f"expected 30 bodies + world, got {model.nbody}"

    T = dof_pos.shape[0]
    translation = np.zeros((T, 30, 3), dtype=np.float64)
    rotation = np.zeros((T, 30, 4), dtype=np.float64)

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_rot_xyzw[t][[3, 0, 1, 2]]
        data.qpos[7:36] = dof_pos[t]
        mujoco.mj_forward(model, data)
        translation[t] = data.xpos[1:31]
        rotation[t] = data.xquat[1:31][:, [1, 2, 3, 0]]

    return translation, rotation


def central_diff(x, dt):
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def angular_velocity_from_quats(rotation_xyzw, dt):
    T, B, _ = rotation_xyzw.shape
    rotvec_step = np.zeros((T, B, 3), dtype=np.float64)
    for b in range(B):
        rots = R.from_quat(rotation_xyzw[:, b, :])
        rel_fwd = (rots[1:] * rots[:-1].inv()).as_rotvec()
        rotvec_step[:-1, b, :] += rel_fwd
        rotvec_step[1:, b, :] += rel_fwd
    counts = np.full((T, 1), 2.0)
    counts[0] = 1.0
    counts[-1] = 1.0
    return rotvec_step / counts[:, :, None] / dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="HoloRetarget robot .h5 shard")
    ap.add_argument("output", type=Path)
    ap.add_argument("--fps", type=float, default=50.0, help="HoloRetarget output is always 50Hz")
    ap.add_argument("--motion-key", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.source, "r") as f:
        root_pos = np.asarray(f["ref_root_pos"], dtype=np.float64)
        root_rot = np.asarray(f["ref_root_rot"], dtype=np.float64)  # xyzw, per docs/motion_retargeting.md
        dof_pos = np.asarray(f["ref_dof_pos"], dtype=np.float64)

    T = dof_pos.shape[0]
    dt = 1.0 / args.fps
    print(f"Loaded {args.source.name}: {T} frames @ {args.fps} fps, {dof_pos.shape[-1]}-DOF")

    global_translation, global_rotation_quat = forward_kinematics_all_frames(root_pos, root_rot, dof_pos)
    dof_vel = central_diff(dof_pos, dt)
    global_velocity = central_diff(global_translation, dt)
    global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

    motion_key = args.motion_key or args.source.stem
    legacy_path = args.output.with_suffix(".legacy.npz")
    metadata = {"motion_key": motion_key, "motion_fps": args.fps, "original_num_frames": T}
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

    result = convert_legacy_offline_npz(legacy_path, args.output, overwrite=args.overwrite)
    legacy_path.unlink()
    print(f"Wrote final v1.4 offline-tracking npz: {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
