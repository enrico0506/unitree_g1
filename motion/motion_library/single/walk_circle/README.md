# walk_circle

- **fps**: 30, **frames**: 180, **duration**: 6.0s
- **source**: `raw_kimodo.npz` — Kimodo-generated (`kimodo_gen --model kimodo-g1-rp`,
  `--duration 6`), person walking in a curved/circular path
- **pipeline**: `motion/motion_builder/combined/walk_and_wave/scripts/kimodo_npz_to_holomotion.py`
  (MujocoQposConverter -> forward kinematics, no CSV intermediate)
- **source file**: `raw_kimodo.npz` (Kimodo local_rot_mats/root_positions, pre-conversion)
- **visualize**: `python motion/motion_library/view.py walk_circle`
- **shape**: continuous walk gait, curved heading
- **known quirks**: none flagged yet
