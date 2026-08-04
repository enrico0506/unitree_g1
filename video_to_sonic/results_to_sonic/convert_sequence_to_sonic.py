from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sonic_protocol import pack_pose_message  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a stage-1 HMR2 export to SONIC-ready data.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--packed_output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_npz = Path(args.input_npz).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_npz, allow_pickle=False) as data:
        smpl_pose = data["smpl_pose_21_axis_angle"].astype(np.float32)
        smpl_joints = data["smpl_joints_local"].astype(np.float32)
        frame_index = data["frame_index"].astype(np.int32)

    if smpl_pose.ndim != 3 or smpl_pose.shape[1:] != (21, 3):
        raise ValueError(f"Expected smpl_pose_21_axis_angle with shape [N,21,3], got {smpl_pose.shape}")
    if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (24, 3):
        raise ValueError(f"Expected smpl_joints_local with shape [N,24,3], got {smpl_joints.shape}")
    if smpl_pose.shape[0] != smpl_joints.shape[0]:
        raise ValueError("Frame count mismatch between smpl_pose and smpl_joints")

    num_frames = smpl_pose.shape[0]
    joint_pos = np.zeros((num_frames, 29), dtype=np.float32)
    joint_vel = np.zeros((num_frames, 29), dtype=np.float32)

    sonic_data = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "smpl_joints": smpl_joints,
        "smpl_pose": smpl_pose,
        "frame_index": frame_index,
    }

    np.savez_compressed(output_npz, **sonic_data)
    print(f"Wrote SONIC-ready sequence to {output_npz}")
    print(
        "Shapes:",
        f"joint_pos={joint_pos.shape}",
        f"joint_vel={joint_vel.shape}",
        f"smpl_joints={smpl_joints.shape}",
        f"smpl_pose={smpl_pose.shape}",
    )

    if args.packed_output:
        packed_output = Path(args.packed_output).resolve()
        packed_output.parent.mkdir(parents=True, exist_ok=True)
        packed_payload = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "smpl_joints": smpl_joints,
            "smpl_pose": smpl_pose,
        }
        packed_output.write_bytes(pack_pose_message(packed_payload, topic="pose", version=3))
        print(f"Wrote packed SONIC v3 message to {packed_output}")


if __name__ == "__main__":
    main()
