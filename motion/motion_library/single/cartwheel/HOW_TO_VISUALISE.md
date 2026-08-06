# How to visualise: cartwheel

Live interactive MuJoCo window, name-based lookup:

```bash
python motion/motion_library/view.py cartwheel
```

Or point directly at the npz (works from any script, not just this helper):

```bash
python motion/motion_builder/combined/walk_and_wave/scripts/view_npz.py motion/motion_library/single/cartwheel/cartwheel_holomotion.npz
```

List every motion available in the library:

```bash
python motion/motion_library/view.py --list
```
