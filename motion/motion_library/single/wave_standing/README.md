# wave_standing

- **fps**: 30, **frames**: 155, **duration**: 5.13s
- **source**: `raw_source.webm` (video capture, person standing, waving)
- **pipeline**: `motion/video_to_motion/run_pipeline.py` (4D-Humans -> SMPL-X) then GMR
  retarget -> `motion/holomotion/scripts/pkl_to_offline_npz.py`
- **source file**: `raw_source.webm` (video capture). An earlier, unused
  Kimodo-generated alternate take (raise/wave/lower from rest, never became
  this clip's output) is kept at
  `motion/motion_builder/single/wave_standing/unused_alternates/raw_kimodo_v1.npz`.
- **visualize**: `python motion/motion_library/view.py wave_standing`
- **shape**: standing still throughout, one arm raises, waves, lowers back to rest
- **known quirks**: judged "not great" by ear/eye. Note `wave_v2` in this same
  `single/` directory is a *different* wave gesture, not a fixed re-take of
  this one — pick between them by which gesture fits, not by assuming v2 is
  strictly better.
