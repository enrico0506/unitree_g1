"""Splice wave clip's arm joints onto walk clip's legs/root, recompute FK + velocities,
producing one combined walk+wave reference motion for HoloMotion motion_tracking training.

Both source clips are 180 frames @ 30fps (same length, no resampling needed) and the
wave clip already bookends at rest (raise -> wave -> lower), so this is a straight
per-frame column overwrite -- no insertion window / blending needed (see conversation).

Run under `conda activate kimodo` (needs mujoco, installed there for the conversion step).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "motion/holomotion/scripts")
from pkl_to_offline_npz import (  # noqa: E402
    angular_velocity_from_quats,
    central_diff,
    convert_legacy_offline_npz,
    forward_kinematics_all_frames,
)

# ---- load converted clips ----
walk = np.load("motion/motion_builder/combined/walk_and_wave/converted/walking_holomotion.npz", allow_pickle=True)
wave = np.load("motion/motion_builder/combined/walk_and_wave/converted/waving_holomotion.npz", allow_pickle=True)

walk_dof_pos = walk["ref_dof_pos"]
wave_dof_pos = wave["ref_dof_pos"]
print("loaded:", walk_dof_pos.shape, wave_dof_pos.shape)  # expect (180, 29) (180, 29)

# ---- find arm dof column indices from robot config ----
robot_cfg_path = "motion/holomotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
with open(robot_cfg_path) as f:
    robot_cfg = yaml.safe_load(f)

dof_names = robot_cfg["robot"]["dof_names"]
arm_dof_names = robot_cfg["robot"]["arm_dof_names"]
arm_indices = [dof_names.index(n) for n in arm_dof_names]
print("arm_indices:", arm_indices)

# ---- splice: walk's legs/root/waist + wave's arms ----
out_dof_pos = walk_dof_pos.copy()
out_dof_pos[:, arm_indices] = wave_dof_pos[:, arm_indices]
print("combined dof_pos:", out_dof_pos.shape)  # expect (180, 29)

# ---- recompute downstream quantities from the new dof_pos ----
# root/pelvis is body index 0 in ref_global_* -- untouched by the arm swap, reuse walk's own
root_pos = walk["ref_global_translation"][:, 0, :]
root_quat_xyzw = walk["ref_global_rotation_quat"][:, 0, :]

fps = 30.0
dt = 1.0 / fps

global_translation, global_rotation_quat = forward_kinematics_all_frames(
    root_pos.astype(float), root_quat_xyzw.astype(float), out_dof_pos.astype(float)
)
dof_vel = central_diff(out_dof_pos, dt)
global_velocity = central_diff(global_translation, dt)
global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

print("global_translation:", global_translation.shape, "dof_vel:", dof_vel.shape)

# ---- save as legacy npz, then run through HoloMotion's own schema/validation pass ----
output_path = Path("motion/motion_builder/combined/walk_and_wave/combined/walk_wave_holomotion.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)
legacy_path = output_path.with_suffix(".legacy.npz")

metadata = {
    "motion_key": "walk_wave",
    "motion_fps": fps,
    "original_num_frames": out_dof_pos.shape[0],
    "source": "walking_holomotion.npz legs/root + waving_holomotion.npz arms",
}

np.savez(
    legacy_path,
    metadata=np.asarray(json.dumps(metadata)),
    dof_pos=out_dof_pos.astype(np.float32),
    dof_vels=dof_vel.astype(np.float32),
    global_translation=global_translation.astype(np.float32),
    global_rotation_quat=global_rotation_quat.astype(np.float32),
    global_velocity=global_velocity.astype(np.float32),
    global_angular_velocity=global_angular_velocity.astype(np.float32),
)
print(f"wrote {legacy_path}")

result = convert_legacy_offline_npz(legacy_path, output_path, overwrite=True)
legacy_path.unlink()
print(f"wrote {output_path}")
print(json.dumps(result, indent=2))