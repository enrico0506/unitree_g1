#!/usr/bin/env python3
"""Convert a raw Kimodo motion npz (local_rot_mats/root_positions) straight into
a HoloMotion v1.4 offline-tracking .npz (ref_dof_pos/ref_global_*), no CSV hop.

Chains two existing pieces in memory instead of going through a file:
  1. kimodo.exports.mujoco.MujocoQposConverter.dict_to_qpos -- kimodo's own
     local_rot_mats/root_positions -> (T, 36) mujoco qpos array.
  2. holomotion/scripts/pkl_to_offline_npz.py's forward_kinematics_all_frames /
     central_diff / angular_velocity_from_quats + convert_legacy_offline_npz --
     same FK/schema plumbing every other source in this repo goes through, so
     output matches man_waving_holomotion.npz etc exactly.

Must run in the `kimodo` conda env (needs the kimodo package + torch).

fps is NOT stored in the raw npz -- pass it explicitly. If unknown:
fps = (frame count in the npz) / (--duration you passed to kimodo_gen).

Usage:
    conda activate kimodo
    python kimodo_npz_to_holomotion.py <source_raw.npz> <output.npz> --fps 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import G1Skeleton34

HOLOMOTION_SCRIPTS = Path(__file__).resolve().parents[4] / "holomotion" / "scripts"
sys.path.insert(0, str(HOLOMOTION_SCRIPTS))
from pkl_to_offline_npz import (  # noqa: E402
    angular_velocity_from_quats,
    central_diff,
    convert_legacy_offline_npz,
    forward_kinematics_all_frames,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="raw kimodo npz (local_rot_mats + root_positions keys)")
    ap.add_argument("output", type=Path, help="Final HoloMotion v1.4 offline-tracking .npz")
    ap.add_argument("--fps", type=float, required=True, help="see module docstring for how to derive it")
    ap.add_argument("--motion-key", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data = np.load(args.source, allow_pickle=True)
    kimodo_dict = {
        "local_rot_mats": data["local_rot_mats"],
        "root_positions": data["root_positions"],
    }

    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    qpos = converter.dict_to_qpos(kimodo_dict, root_quat_w_first=True)
    qpos = np.asarray(qpos)
    if qpos.ndim == 3:
        assert qpos.shape[0] == 1, f"expected single clip, got batch {qpos.shape[0]}"
        qpos = qpos[0]
    if qpos.shape[-1] != 36:
        raise ValueError(f"Expected 36-col qpos (3 root pos + 4 root quat wxyz + 29 dof), got {qpos.shape[-1]}")

    root_pos = qpos[:, 0:3].astype(np.float64)
    root_quat_wxyz = qpos[:, 3:7].astype(np.float64)
    dof_pos = qpos[:, 7:36].astype(np.float64)
    # forward_kinematics_all_frames expects xyzw (matches pkl_to_offline_npz.py's convention)
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]

    fps = args.fps
    dt = 1.0 / fps
    T = dof_pos.shape[0]
    print(f"Loaded {args.source.name}: {T} frames @ {fps} fps, 29-DOF")

    global_translation, global_rotation_quat = forward_kinematics_all_frames(root_pos, root_quat_xyzw, dof_pos)
    dof_vel = central_diff(dof_pos, dt)
    global_velocity = central_diff(global_translation, dt)
    global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

    motion_key = args.motion_key or args.source.stem
    legacy_path = args.output.with_suffix(".legacy.npz")
    metadata = {
        "motion_key": motion_key,
        "motion_fps": fps,
        "original_num_frames": T,
        "source": "kimodo raw npz (in-memory, no CSV hop)",
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
