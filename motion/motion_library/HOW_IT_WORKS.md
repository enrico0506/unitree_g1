# How this all fits together

Three roles, kept deliberately separate:

- **`motion/motion_library/`** (this folder) — finished, reusable clips. `single/` =
  one atomic motion. `combined/` = spliced from two or more `single/` clips.
  Everything here is in HoloMotion's `ref_dof_pos` format ([T,29] G1 29-DOF,
  plus `ref_global_*` body poses/velocities) — ready to drop into RL training.
- **`motion/motion_builder/`** — active workspace where a motion gets built (raw
  input, intermediate conversions, the combine/splice script). Messy by
  design. Once a build's result is good, it gets copied into `motion/motion_library/`
  with a README.
- **Tools** (`motion/video_to_motion/`, `motion/kimodo/`, `motion/holomotion/scripts/`) — shared
  pipelines any build calls into. Not owned by any one build.

## The two working source-to-clip paths, right now

Only these two are actually usable end-to-end today. Anything else (live VR
teleop, mocap suits, hand-keyframing) is either unbuilt or a different use
case (live control, not clip authoring) — see `motion/holomotion/deployment/` for
that instead.

### 1. Video → clip

```
your video (.mp4/.webm)
  → motion/video_to_motion/run_pipeline.py            (4D-Humans, CPU, video -> SMPL -> SMPL-X)
  → motion/video_to_motion/dispatch_retarget.py        (SMPL-X -> G1 ref_dof_pos npz)
       ├─ CUDA available?  -> motion/video_to_motion/gpu_pipeline/retarget_gpu.py   (HoloRetarget)
       └─ no CUDA (this machine) -> motion/video_to_motion/cpu_pipeline/retarget_cpu.py   (GMR, recovered from git history)
  → motion/motion_library/single/<name>/<name>_holomotion.npz
```
One command each stage; `dispatch_retarget.py` auto-picks CPU or GPU, no
manual choice needed. Slow on CPU (person-detector + HMR2 per frame, expect
minutes not seconds) — that's inherent to video pose estimation, not a bug.

Real example: `motion/motion_library/single/wave_v2/README.md` documents exactly
this path for a real clip, start to finish.

### 2. Kimodo (text-to-motion) → clip

```
text prompt
  → kimodo_gen --model kimodo-g1-rp   (generates raw kimodo-native npz: local_rot_mats + root_positions)
  → motion/motion_builder/<build>/scripts/kimodo_npz_to_holomotion.py   (kimodo -> qpos -> G1 ref_dof_pos npz, no CSV hop)
  → motion/motion_library/single/<name>/<name>_holomotion.npz
```
Real example: `motion/motion_library/single/walk_circle/README.md`.

**Caveat**: `kimodo_gen` itself (the generation step) is currently blocked on
this dev machine — its text encoder needs ~17GB VRAM this laptop doesn't
have (see `motion/kimodo/SETUP.md`). The *conversion* script
(`kimodo_npz_to_holomotion.py`) works fine once you have a kimodo-generated
npz from somewhere that can actually run generation (the robot's Jetson,
per SETUP.md, or another CUDA machine).

## Combining clips

Two `single/` clips (e.g. a walk + a wave) get spliced into one
`combined/` clip via a build script in `motion/motion_builder/` — see
`motion/motion_builder/combined/walk_and_wave/scripts/combine_walk_wave.py` as the working
example (splices arm dof columns from one clip onto another's legs/root,
recomputes forward kinematics + velocities, resamples by time if the two
clips have different frame counts). `combined/walk_wave/README.md` and
`combined/walk_wave_v2/README.md` document two real results.

## Everything else in this repo (not yet a usable source)

- **Live VR teleop** (`motion/holomotion/deployment/holomotion_teleop/`) — real-time
  control via Pico/XRoboToolkit, not clip authoring. Different use case.
- **HoloRetarget's own CLI path** (needs CUDA) — wired up in
  `motion/video_to_motion/gpu_pipeline/retarget_gpu.py` but never executed end-to-end (this dev
  machine has no CUDA GPU). Should work as-is on a CUDA machine; untested.

## Once you have a clip

- Look at it: `python motion/motion_library/view.py <name>` (see each motion's own
  `HOW_TO_VISUALISE.md`).
- Sim2sim it: `./motion/sim/run_holomotion.sh <name>` runs it through HoloMotion's
  real MuJoCo sim2sim eval against the shared general-purpose tracking policy —
  no training needed, works for any clip in the library out of the box. See
  `motion/sim/README.md`.
- Train on it: point `train_hdf5_roots` in a
  `motion/holomotion/config/training/motion_tracking/*.yaml` at it and run
  `motion/holomotion/scripts/training/train_motion_tracking.sh` — see
  `motion/holomotion/docs/train_motion_tracking.md`.
