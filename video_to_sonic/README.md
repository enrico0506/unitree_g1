## `video_to_sonic`

Portable pipeline for:

`video -> 4D-Humans pose sequence -> SONIC-ready sequence -> ZMQ packed message`

This folder now carries the full source needed for the `man_waving` path without
the heavy local caches or the multi-gigabyte GR00T clone.

### Bundled example

The example input is:

- `input_videos/man_waving_screen.webm`

The already-generated outputs that belong to that clip are:

- `video_to_results/exports/man_waving_screen.npz`
- `results_to_sonic/sonic_ready/man_waving_screen_sonic.npz`
- `results_to_sonic/packed_messages/man_waving_screen_pose_v3.bin`

### One-command pipeline

Activate your `4d-humans-cpu` environment, then run:

```bash
cd video_to_sonic
python run_pipeline.py --video man_waving_screen.webm --device cpu --fps 10
```

If there is exactly one file in `input_videos/`, `--video` is optional.

Useful flags:

- `--render-demo`
  Writes comparison images with the rendered humanoid.
- `--device cpu|auto|cuda`
  Controls the 4D-Humans inference device.
- `--name my_run`
  Override the output prefix.

The launcher creates:

- `video_to_results/inputs/<name>_frames/`
- `video_to_results/exports/<name>.npz`
- `video_to_results/outputs/<name>_demo/` when `--render-demo` is enabled
- `results_to_sonic/sonic_ready/<name>_sonic.npz`
- `results_to_sonic/packed_messages/<name>_pose_v3.bin`

### Required manual step

Before stage 1 works on a fresh machine, place the SMPL file at:

- `4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

The HMR2 checkpoint downloads automatically on first run. The SMPL file does not.
The placeholder note for this lives in `4D-Humans/data/README.md`.

### Structure

- `4D-Humans/`
  Vendored 4D-Humans source with the CPU fallback patches used here.
- `video_to_results/`
  Stage 1 export code and generated HMR2 `.npz` files.
- `results_to_sonic/`
  Stage 2 conversion, validation, local SONIC protocol packer, and ZMQ streaming.

### SONIC runtime note

`results_to_sonic` no longer depends on importing the full `GR00T-WholeBodyControl`
repo just to pack messages. For actual SONIC deployment on the Unitree machine,
clone NVIDIA's upstream runtime separately. The local test reference used here was:

- `NVlabs/GR00T-WholeBodyControl` commit `4141c34280abb67c82e115342a8720f4a83d750d`

That repo is not vendored here because the upstream checkout contains Git LFS assets
and binaries larger than GitHub's normal push limits.

### Stream the converted sequence

```bash
python video_to_sonic/results_to_sonic/stream_sonic_sequence.py \
  --input_npz video_to_sonic/results_to_sonic/sonic_ready/man_waving_screen_sonic.npz \
  --host 127.0.0.1 \
  --port 5556 \
  --fps 2 \
  --window_size 5
```

Use `--send_start` if the receiving SONIC deployment expects the streamed-pose
start command before playback.
