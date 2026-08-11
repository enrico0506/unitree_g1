# walk_straight

- **fps**: 30, **frames**: 180, **duration**: 6.0s
- **source**: `raw_kimodo.npz` — Kimodo-generated (`kimodo_gen --model kimodo-g1-rp`,
  `--duration 6`), person walking in a straight line
- **pipeline**: `motion/motion_builder/single/walk_straight/scripts/kimodo_npz_to_holomotion.py`
  (MujocoQposConverter -> forward kinematics, no CSV intermediate)
- **source file**: `raw_kimodo.npz`
- **visualize**: `python motion/motion_library/view.py walk_straight`
- **shape**: straight-line walk, ~6.07m forward over the clip (~1 m/s pace),
  0.34m lateral drift (ratio 0.056 vs forward distance -- essentially
  straight, not curved like `walk_circle`), flat height throughout (no
  climbing)
- **known quirks**: none flagged yet. Built specifically as the straight-path
  counterpart to `walk_circle`, for approach/continuation segments in
  combined motions (e.g. walk toward stairs, climb, continue walking) where
  a curved path doesn't make sense.
