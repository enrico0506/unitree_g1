#!/usr/bin/env python3
"""Convert a Kimodo G1 qpos CSV into a HoloMotion v1.4 offline-tracking .npz.

Why this exists: same destination format as pkl_to_offline_npz.py (which handles
the GMR/TWIST .pkl source), but Kimodo's G1 export (`kimodo_gen --model
kimodo-g1-rp`) is a different SOURCE schema -- a raw MuJoCo qpos CSV, one row
per frame, 36 columns: [root_x, root_y, root_z, root_qw, root_qx, root_qy,
root_qz, joint_angle_0, ..., joint_angle_28]. Confirmed directly from
kimodo/exports/mujoco.py's MujocoQposConverter (root_quat_w_first=True is the
default kimodo_gen uses) -- not guessed. Kimodo's G1 skeleton is already the
same 29-DOF convention HoloMotion's G1 MJCF uses, so no joint remapping is
needed here, just repacking + a quaternion convention flip (Kimodo/MuJoCo
qpos is wxyz; the shared forward_kinematics step below expects xyzw, matching
pkl_to_offline_npz.py's convention).

Reuses pkl_to_offline_npz.py's forward-kinematics / central-diff / legacy-npz
plumbing rather than re-deriving it, so both source pipelines stay consistent
by construction instead of by two independently-maintained copies.

IMPORTANT -- fps is NOT stored in the CSV. Kimodo's frame count is derived
from `--duration` and the loaded checkpoint's own `model.fps` at generation
time (see kimodo/scripts/generate.py), which this script has no way to read
back out of a CSV file alone. You must pass --fps explicitly. If you don't
know it: fps = (rows in the CSV) / (the --duration in seconds you passed to
kimodo_gen for that generation).

STATUS: written but not yet run against a real generated CSV -- Kimodo
generation on this dev machine is blocked (8B-param text encoder doesn't fit
in available RAM, see motion/kimodo/SETUP.md). Verify this end-to-end with an
actual CSV before trusting it in a deploy pipeline.

Usage:
    python kimodo_csv_to_holomotion_npz.py <source.csv> <output.npz> --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pkl_to_offline_npz import (  # noqa: E402
    angular_velocity_from_quats,
    central_diff,
    convert_legacy_offline_npz,
    forward_kinematics_all_frames,
)
import json  # noqa: E402


def load_csv_motion(path: Path, fps: float):
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]  # single-frame CSV
    if rows.shape[-1] != 36:
        raise ValueError(
            f"Expected 36 columns (3 root pos + 4 root quat wxyz + 29 dof), got {rows.shape[-1]}. "
            "This script is for kimodo_gen's --model kimodo-g1-rp CSV output specifically -- "
            "other Kimodo skeletons (e.g. kimodo-soma-rp) use a different export shape."
        )
    root_pos = rows[:, 0:3].astype(np.float64)
    root_quat_wxyz = rows[:, 3:7].astype(np.float64)
    dof_pos = rows[:, 7:36].astype(np.float64)
    # Shared forward_kinematics_all_frames (from pkl_to_offline_npz.py) expects
    # xyzw, matching the GMR/TWIST .pkl convention it was written for.
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    return fps, root_pos, root_quat_xyzw, dof_pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="Kimodo G1 qpos CSV (kimodo_gen --model kimodo-g1-rp output)")
    ap.add_argument("output", type=Path, help="Final HoloMotion v1.4 offline-tracking .npz")
    ap.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Not stored in the CSV -- see module docstring for how to derive it.",
    )
    ap.add_argument("--motion-key", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fps, root_pos, root_rot, dof_pos = load_csv_motion(args.source, args.fps)
    dt = 1.0 / fps
    T = dof_pos.shape[0]
    print(f"Loaded {args.source.name}: {T} frames @ {fps} fps, 29-DOF")

    global_translation, global_rotation_quat = forward_kinematics_all_frames(root_pos, root_rot, dof_pos)
    dof_vel = central_diff(dof_pos, dt)
    global_velocity = central_diff(global_translation, dt)
    global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

    motion_key = args.motion_key or args.source.stem
    legacy_path = args.output.with_suffix(".legacy.npz")
    metadata = {
        "motion_key": motion_key,
        "motion_fps": fps,
        "original_num_frames": T,
        "source": "kimodo_gen (kimodo-g1-rp)",
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

    result = convert_legacy_offline_npz(legacy_path, args.output, overwrite=args.overwrite)
    legacy_path.unlink()
    print(f"Wrote final v1.4 offline-tracking npz: {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
