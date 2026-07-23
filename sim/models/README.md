# sim/models/g1/ — Unitree G1 MJCF model

This directory is **fetched, not committed** (see `.gitignore`: `sim/models/g1/`).
It's the ready-made, correctly-scoped 29-DOF G1 MuJoCo MJCF model from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
which is itself derived from Unitree's own public description
(`unitreerobotics/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml`).

## How to fetch it

```bash
bash sim/fetch_model.sh
```

This shallow-clones `mujoco_menagerie` into a temp dir and copies just the
`unitree_g1/` subdirectory here. Result: `sim/models/g1/g1.xml` (+ `assets/`
meshes, `g1_with_hands.xml`, `scene.xml`, `LICENSE`, etc).

## Why it's fetched-at-setup-time instead of vendored into this repo's git history

Two independent reasons, checked separately:

1. **License** — `sim/models/g1/LICENSE` (after fetching) is Unitree Robotics'
   **BSD-3-Clause** license, which explicitly permits redistribution with
   attribution. No license doubt here — this would be fine to vendor from a
   legal standpoint.
2. **Size/hygiene** — the model + meshes are ~38 MB of binary assets. This
   repo already treats large downloadable model weights as "not source, regenerate/
   re-download" (see the `.gitignore` entries for `*.pt`/`*.onnx`/
   `perception/*/models/`). Bloating this repo's git history 6x+ (it was ~5.6 MB
   before this) for a re-fetchable third-party asset isn't worth it. Same
   pattern, applied to mesh assets instead of trained weights.

## Verified working (2026-07-23)

- Fetched successfully on this machine (network access to github.com confirmed
  available from this Jetson).
- `mujoco.MjModel.from_xml_path("sim/models/g1/g1.xml")` loads cleanly under
  `sim/.venv`'s `mujoco==3.2.3`, and `mujoco.mj_step`/`mj_forward` run without
  creating any display/EGL/GL context (headless, no `DISPLAY` set).

If this directory is missing (e.g. fresh clone, or the fetch is blocked by a
network policy at run time), `scripts/sim_runner.py` prints a clear error
pointing back here rather than crashing obscurely.
