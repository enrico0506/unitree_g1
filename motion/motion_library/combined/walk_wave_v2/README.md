# walk_wave_v2

- **fps**: 30, **frames**: 180, **duration**: 6.0s
- **built from**:
  - legs/root/waist: `single/walk_circle` (unchanged, full timeline)
  - arms: `single/wave_v2` (resampled by time from 198 -> 180 frames to fit
    walk_circle's timeline — wave_v2 is naturally 6.6s, walk_circle is 6.0s,
    so the gesture plays back ~9% faster here than in the standalone wave_v2
    clip, same relative shape)
- **built with**: `motion/motion_builder/combined/walk_and_wave/scripts/combine_walk_wave_v2.py`
- **source file**: n/a — derived clip, no raw source of its own (see "built from" above)
- **visualize**: `python motion/motion_library/view.py walk_wave_v2`
- **shape**: walks continuously; arm raises/waves/lowers partway through while
  legs keep walking, back to rest, walk continues
- **known quirks**: arm dof indices `[15..28]` (`arm_dof_names` from
  `29dof_training_isaaclab.yaml`) hardcoded as the swap boundary — re-derive
  if the robot config's dof ordering ever changes. Uses `wave_v2`'s gesture,
  a distinct wave from the one in `combined/walk_wave` (which uses
  `wave_standing`) — these are two different combined clips by design, not
  v1/v2 of the same thing.
