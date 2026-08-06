"""Splice wave_v2's arm joints onto walk_circle's legs/root, recompute FK + velocities.

Same approach as combine_walk_wave.py, but this time the two source clips are
NOT the same length: walk_circle is 180 frames, wave_v2 is 198 frames (both
@ 30fps, so it's a duration mismatch, not an fps mismatch). Fix: resample
wave_v2's arm columns from 198 -> 180 frames by time (np.interp), so the wave
gesture fits exactly onto walk_circle's timeline (plays back ~9% faster,
same relative shape) rather than mismatching in length.

Run under `conda activate kimodo` (needs mujoco).
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

# ---- load source clips from the library ----
walk = np.load("motion/motion_library/single/walk_circle/walk_circle_holomotion.npz", allow_pickle=True)
wave = np.load("motion/motion_library/single/wave_v2/wave_v2_holomotion.npz", allow_pickle=True)

walk_dof_pos = walk["ref_dof_pos"]
wave_dof_pos = wave["ref_dof_pos"]
print("loaded:", walk_dof_pos.shape, wave_dof_pos.shape)  # (180, 29) (198, 29) -- different lengths

# ---- find arm dof column indices from robot config ----
robot_cfg_path = "motion/holomotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
with open(robot_cfg_path) as f:
    robot_cfg = yaml.safe_load(f)

dof_names = robot_cfg["robot"]["dof_names"]
arm_dof_names = robot_cfg["robot"]["arm_dof_names"]
arm_indices = [dof_names.index(n) for n in arm_dof_names]
print("arm_indices:", arm_indices)

# ---- resample wave_v2's arm columns from 198 -> 180 frames (by time, not truncation) ----
walk_fps = 30.0
wave_fps = 30.0
walk_T = walk_dof_pos.shape[0]
wave_T = wave_dof_pos.shape[0]

wave_duration = wave_T / wave_fps
old_t = np.linspace(0, wave_duration, wave_T)
new_t = np.linspace(0, wave_duration, walk_T)  # squeeze onto walk's frame count

wave_arm_resampled = np.stack(
    [np.interp(new_t, old_t, wave_dof_pos[:, i]) for i in arm_indices], axis=1
)
print("resampled wave arms:", wave_arm_resampled.shape)  # (180, 14)

# ---- splice: walk's legs/root/waist + resampled wave arms ----
out_dof_pos = walk_dof_pos.copy()
out_dof_pos[:, arm_indices] = wave_arm_resampled
print("combined dof_pos:", out_dof_pos.shape)  # (180, 29)

# ---- recompute downstream quantities from the new dof_pos ----
root_pos = walk["ref_global_translation"][:, 0, :]
root_quat_xyzw = walk["ref_global_rotation_quat"][:, 0, :]

fps = walk_fps
dt = 1.0 / fps

global_translation, global_rotation_quat = forward_kinematics_all_frames(
    root_pos.astype(float), root_quat_xyzw.astype(float), out_dof_pos.astype(float)
)
dof_vel = central_diff(out_dof_pos, dt)
global_velocity = central_diff(global_translation, dt)
global_angular_velocity = angular_velocity_from_quats(global_rotation_quat, dt)

print("global_translation:", global_translation.shape, "dof_vel:", dof_vel.shape)

# ---- save as legacy npz, then run through HoloMotion's own schema/validation pass ----
output_path = Path("motion/motion_builder/combined/walk_and_wave/combined/walk_wave_v2_holomotion.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)
legacy_path = output_path.with_suffix(".legacy.npz")

metadata = {
    "motion_key": "walk_wave_v2",
    "motion_fps": fps,
    "original_num_frames": out_dof_pos.shape[0],
    "source": "walk_circle_holomotion.npz legs/root + wave_v2_holomotion.npz arms (resampled 198->180 frames)",
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
