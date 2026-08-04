# Kimodo — Jetson setup notes

Vendored from [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) (`.git/`
stripped, same convention as `motion/holomotion/`). This is source only —
nothing has been installed or run on this dev machine, since this machine
isn't the target hardware (it has an AMD GPU; Kimodo needs an NVIDIA Jetson).
The steps below are written for when this runs on the actual robot's Jetson
Orin NX (JetPack 5.1, 16GB shared RAM/VRAM).

**Future goal**: feed Kimodo-generated motions into HoloMotion
(`motion/holomotion/`) instead of / alongside the video->GMR->TWIST chain in
`video_to_motion/` + `twist_deploy/`. Not wired up yet — this commit is just
getting the source in place.

## The critical hardware constraint

Kimodo wants ~17GB VRAM by default. The Orin NX has 16GB total (shared
CPU/GPU memory) — it will not fit as-is. Fix in Step 4 below brings GPU
usage under 3GB by moving the text encoder to CPU. Without that env var,
expect an OOM on first run, not a slow run.

## Step 1 — Environment

```bash
conda create -n kimodo python=3.10
conda activate kimodo
```

## Step 2 — PyTorch (Jetson-specific build, install BEFORE Kimodo)

Do **not** `pip install torch` — that pulls the generic PyPI wheel, which is
built for x86_64 and will not work on the Orin NX's ARM64 + JetPack CUDA/cuDNN
stack. You need NVIDIA's JetPack 5.1-matched PyTorch wheel from the Jetson
Zoo / NVIDIA developer forums instead. This is the step most likely to cause
trouble — look up the exact wheel URL for JetPack 5.1 + Python 3.10 when
actually doing this on the robot (URLs shift between JetPack releases, don't
hardcode one here without verifying it's current).

## Step 3 — Install Kimodo

Once PyTorch is confirmed working (`python -c "import torch; print(torch.cuda.is_available())"`
should print `True` on the Jetson):

```bash
pip install git+https://github.com/nv-tlabs/kimodo.git
```

Skip the `[all]` extra — that pulls in the interactive Gradio demo, not
needed for batch/offline generation over SSH.

## Step 4 — The VRAM fix (required on this hardware)

```bash
export TEXT_ENCODER_DEVICE=cpu
```

Forces the LLM2Vec text encoder (reads the text prompt) onto CPU instead of
GPU. The diffusion/motion-generation part still runs on GPU. Without this,
expect an OOM — the Orin NX doesn't have the ~17GB Kimodo wants by default.
Slower text-encoding step, but it fits.

## Step 5 — Generate

CLI takes a text prompt, outputs a motion file. For G1-specific checkpoints,
output saves directly as a MuJoCo qpos CSV — check whether the specific
checkpoint used was trained already-retargeted to G1 joint space before
assuming a separate GMR retargeting step is needed; it may not be, unlike
the video->SMPL->GMR path this repo uses elsewhere.

## Docker vs. venv — venv is the right call here, not Docker

Counter to the usual "Docker for isolation" instinct:

- Kimodo's official `Dockerfile` is **x86-only** — built from
  `nvcr.io/nvidia/pytorch:24.10-py3`, a datacenter-GPU image, not
  Jetson/ARM64. It will not run on the Orin NX as-is. Making it work would
  mean rebuilding from an L4T base image (`nvcr.io/nvidia/l4t-pytorch` or
  similar) — i.e. solving the exact same "get PyTorch working on Jetson"
  problem as Step 2, just with an extra layer of Docker complexity on top.
  Not a shortcut.
- This is a one-time, offline generation step, not a persistent service —
  doesn't need the hardened isolation Docker is normally worth the cost for
  (compare: HoloMotion's deploy path *does* use a proper Jetson-native
  Docker image, because that's an actual persistent deployment target).
- `conda create -n kimodo` already isolates Kimodo's Python deps from
  everything else on the robot (ROS2, YOLO, the dashboard) at the dependency
  level — the main thing Docker would otherwise buy here.
- Kimodo has a C++ build step (`MotionCorrection/`, needs cmake/gcc) —
  simpler to debug natively in a venv/conda env than inside a container
  where base-image mismatches are also in play.

If Step 2/3 turn into a genuine dependency fight on the real hardware, only
then consider Docker — with an L4T base image, not the shipped Dockerfile.
