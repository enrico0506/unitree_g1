# wave_v2

- **fps**: 30, **frames**: 198, **duration**: 6.6s
- **source file**: `raw_source.mp4` (video capture, "Hailuo" clip, person standing
  facing camera, waving) — a distinct wave gesture from `wave_standing`, not a
  re-shoot/replacement of it. Named `wave_v2` for pipeline-run order, not motion
  similarity. Intermediate SMPL-X stage output kept at
  `motion/motion_builder/single/wave_v2/intermediate/wave_v2_smplx.npz` (regeneratable
  from `raw_source.mp4`, not needed to use this clip).
- **pipeline**: `motion/video_to_motion/run_pipeline.py` (4D-Humans -> SMPL-X, CPU) then
  `motion/video_to_motion/dispatch_retarget.py` -> `cpu_pipeline/retarget_cpu.py`
  (recovered GMR retarget, CPU) -> `motion/holomotion/scripts/pkl_to_offline_npz.py`
  FK/central-diff plumbing
- **visualize**: `python motion/motion_library/view.py wave_v2`
- **shape**: standing, one-arm wave
- **known quirks**: verified non-degenerate (per-column `ref_dof_pos` std check,
  highest variance on arm/shoulder DOFs as expected). Not yet visually reviewed.
  A separate, distinct wave from `wave_standing` — decide per-use which gesture
  fits, not "which is better."
