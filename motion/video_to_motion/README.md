## `video_to_motion`

Portable pipeline for:

`video -> 4D-Humans SMPL pose sequence -> GMR-ready SMPL-X sequence`

The output feeds `twist_deploy/GMR`'s retargeter (`scripts/smplx_to_robot.py`),
which produces the G1-format `.pkl` that TWIST or HoloMotion consume. This
folder carries the full source needed for that chain, without the heavy
local caches or the multi-gigabyte GR00T clone.

### Bundled example

The example input is:

- `input_videos/man_waving_screen.webm`

The already-generated outputs that belong to that clip are:

- `video_to_pose/exports/man_waving_screen.npz` — raw HMR2 SMPL export
- `pose_to_gmr/gmr_ready/man_waving_screen_smplx.npz` — GMR-ready SMPL-X

### One-command pipeline

Activate your `4d-humans-cpu` environment, then run:

```bash
cd motion/video_to_motion
python run_pipeline.py --video man_waving_screen.webm --device cpu --fps 10
```

If there is exactly one file in `input_videos/`, `--video` is optional. This
runs both stages (HMR2 export, then the SMPL->SMPL-X adapter) in one command
— retargeting onto G1 with GMR is the one remaining manual step, since it
needs a `--robot` choice (see the command the script prints at the end).

Useful flags:

- `--render-demo`
  Writes comparison images with the rendered humanoid.
- `--device cpu|auto|cuda`
  Controls the 4D-Humans inference device.
- `--name my_run`
  Override the output prefix.
- `--gmr_smoothing_window N` / `--gmr_gender male|female|neutral`
  Passed through to the adapter (see its own `--help` for why the smoothing
  default is 15, not something smaller).

The launcher creates:

- `video_to_pose/inputs/<name>_frames/`
- `video_to_pose/exports/<name>.npz`
- `video_to_pose/outputs/<name>_demo/` when `--render-demo` is enabled
- `pose_to_gmr/gmr_ready/<name>_smplx.npz`

### Required manual step

Before stage 1 works on a fresh machine, place the SMPL file at:

- `4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

The HMR2 checkpoint downloads automatically on first run. The SMPL file does not.
The placeholder note for this lives in `4D-Humans/data/README.md`.

GMR's own retarget step (the one manual command printed at the end of the
pipeline) separately needs SMPL-X body models — see
`twist_deploy/GMR/assets/body_models/README.md`.

### Structure

- `4D-Humans/`
  Vendored 4D-Humans source with the CPU fallback patches used here.
- `video_to_pose/`
  Stage 1: frame extraction + HMR2 export code and generated `.npz` files.
- `pose_to_gmr/`
  Stage 2: the SMPL->SMPL-X adapter (`adapt_smpl_to_gmr_smplx.py`) and its
  output. See that script's own module docstring/comments for what each of
  its three fixes does and why (axis correction, hip/knee/spine1 debias,
  static-root pelvis height) — those aren't optional cosmetic choices, they
  were each verified necessary against real retarget output.

### Stage 3: retargeting (`cpu_pipeline/` / `gpu_pipeline/` / `dispatch_retarget.py`)

The final stage turns a GMR-ready SMPL-X npz into the G1 `ref_dof_pos` npz
that HoloMotion consumes, and comes in two interchangeable implementations
behind one dispatcher: `cpu_pipeline/retarget_cpu.py` runs headless via a
vendored GMR tree (works on any machine, no GPU needed) and
`gpu_pipeline/retarget_gpu.py` runs via HoloRetarget (needs CUDA, raises
immediately if none is found). `dispatch_retarget.py` is the entry point you
should actually call — it detects CUDA (`torch.cuda.is_available()`, falling
back to CPU with a warning if `torch` itself isn't importable in the current
env), picks whichever pipeline applies, prints its choice and why, and
forwards the same `<input> <output> [--fps N]` args to it as a subprocess,
propagating its exit code. Use `--force-cpu` / `--force-gpu` to override
auto-detection. The CPU path needs the `4d-humans-cpu` conda env (GMR
installed there via `-e`); run `dispatch_retarget.py` itself with that env's
`python` and the CPU subprocess inherits it automatically.

```bash
/home/enrico/miniconda3/envs/4d-humans-cpu/bin/python motion/video_to_motion/dispatch_retarget.py \
    motion/video_to_motion/pose_to_gmr/gmr_ready/<name>_smplx.npz \
    motion/motion_builder/combined/walk_and_wave/converted/<name>_holomotion.npz --fps 30
```
(paths above assume you're running from the repo root; adjust if `cd`'d into `motion/video_to_motion` per the stage-1 example above)
