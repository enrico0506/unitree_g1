# cartwheel

- **fps**: 30, **frames**: 151, **duration**: 5.0s
- **source**: TWIST's bundled example motion library, retargeted to G1 via GMR
  before that source tool was removed from this repo (pkl artifact was staged
  under the now-removed `twist_deploy/motion_staging/` at the time)
- **pipeline**: `motion/holomotion/scripts/pkl_to_offline_npz.py`
- **source file**: not recoverable — the intermediate .pkl was consumed and
  discarded when this clip was converted, and TWIST's own example library
  (upstream of the .pkl) was never vendored here, only its output
- **visualize**: `python motion/motion_library/view.py cartwheel`
- **shape**: full cartwheel acrobatic motion
- **known quirks**: none flagged yet
