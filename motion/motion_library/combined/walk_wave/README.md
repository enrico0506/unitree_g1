# walk_wave

- **fps**: 30, **frames**: 180, **duration**: 6.0s
- **built from**:
  - legs/root/waist: `single/walk_circle` (unchanged, full timeline)
  - arms: `single/wave_standing` (v1, arm dof columns spliced in whole-clip, no
    insertion window needed — both source clips happened to be 180 frames @
    30fps already, and the wave clip is self-bookended at rest)
- **built with**: `motion/motion_builder/combined/walk_and_wave/scripts/combine_walk_wave.py`
- **source file**: n/a — derived clip, no raw source of its own (see "built from" above)
- **visualize**: `python motion/motion_library/view.py walk_wave`
- **shape**: walks continuously; arm raises/waves/lowers partway through while
  legs keep walking, back to rest, walk continues
- **known quirks**: arm dof indices `[15..28]` (`arm_dof_names` from
  `29dof_training_isaaclab.yaml`) hardcoded as the swap boundary — re-derive
  if the robot config's dof ordering ever changes. Built from `wave_standing`,
  judged "not great" by ear. `single/wave_v2` is available in the library as
  a *different* wave gesture (not a fix/upgrade of this one) — a `walk_wave_v2`
  combined clip could be built from it if that gesture fits better, but it's
  a distinct choice, not a strict replacement.
