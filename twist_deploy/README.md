# twist_deploy/

Vendored source for the two external repos that make up our
person-dances-in-front-of-camera → G1-mimics motion pipeline:

- **`TWIST/`** — [YanjieZe/TWIST](https://github.com/YanjieZe/TWIST), the
  sim2real deployment/policy repo for TWIST (a general motion-tracking
  controller for the Unitree G1). We only need to *run* the pretrained
  checkpoint on the robot, not train it.
- **`GMR/`** — [YanjieZe/GMR](https://github.com/YanjieZe/GMR) ("General
  Motion Retargeting"), the retargeter (same author) that converts human
  motion capture into the `.pkl` motion files TWIST's policy consumes.

Both were fetched with `git clone --depth 1` (shallow, single commit) on
2026-08-02, then had their `.git/` directories removed so they vendor as
plain files tracked by *this* repo's git — same convention as
`video_to_sonic/4D-Humans/`. No `git submodule`/nested repos are used.

## What this is for

Deployment target: laptop-side, over ethernet, talking to the G1 at
`192.168.123.222`. There is **no Jetson/JetPack involved** in this pipeline —
the laptop runs the policy and streams low-level commands to the robot
directly, the way our other `deploy_real`-style scripts already do.

The thing we actually want to execute is TWIST's pretrained checkpoint:

```
TWIST/assets/twist_general_motion_tracker.pt   (4.3 MB, present, real weights — not a placeholder)
```

Relevant TWIST entry points for sim2real deployment (kept, not stubs):
- `TWIST/deploy_real/server_low_level_g1_real.py` — low-level real-robot control loop
- `TWIST/deploy_real/server_low_level_g1_sim.py` — sim2sim equivalent
- `TWIST/deploy_real/server_high_level_motion_lib.py` — high-level motion server
- `TWIST/deploy_real/server_motion_optitrack_v2 (legacy).py`
- `TWIST/deploy_real/data_utils/`, `TWIST/deploy_real/robot_control/`
- `TWIST/pose/` — pose utilities used at inference time
- `TWIST/play_student.sh`, `TWIST/play_teacher.sh`, `TWIST/to_jit.sh` — deploy-side helper scripts
- `TWIST/assets/g1/`, `TWIST/assets/unitree_g1/` — G1 URDF/MJCF + meshes used by the sim2sim/deploy scripts

GMR's retargeter (`GMR/general_motion_retargeting/`, `GMR/scripts/`) is
vendored in full — it's lightweight (no training assets) and is what
produces the `.pkl` files TWIST's policy expects as input.

## What was excluded, and why

### TWIST/ — removed entirely
- **`TWIST/legged_gym/`** and **`TWIST/rsl_rl/`** — IsaacGym-based RL
  training code (policy/teacher training, environment definitions). Not
  needed to run the pretrained checkpoint, and pulling in IsaacGym is a
  heavy, CUDA/training-specific dependency we explicitly don't want here.
  If training or fine-tuning is ever needed, re-clone these two directories
  from the upstream repo.
- **Note:** `TWIST/train_student.sh` and `TWIST/train_teacher.sh` were left
  in place (they're tiny shell scripts) but will **not run** without the
  removed `legged_gym`/`rsl_rl` directories — they're kept only as
  reference for what training commands looked like upstream.

### GMR/ — assets trimmed to just the G1
GMR ships URDF/MJCF + mesh packs for ~19 different humanoid platforms
(Unitree H1/H1-2, Booster T1/K1, Fourier, Galaxea, PAL Talos, Stanford
Toddy, PND Adam Lite, etc.) under `GMR/assets/`, totaling **~1.2 GB**. We
only target the Unitree G1, so every other robot's asset folder was
deleted from the vendored copy, keeping just:
- `GMR/assets/unitree_g1/` — G1 MJCF/URDF + meshes (retained)
- `GMR/assets/hard_motions/`, `GMR/assets/xsens_bvh_test/` — small test/demo
  motion data (retained, a few MB)
- `GMR/assets/GMR.png`, `GMR/assets/GMR_pipeline.png`, `GMR/assets/optitrack.png` — docs images (retained)

Removed robot asset packs (re-fetch from upstream `GMR/assets/<name>` if a
different robot target is ever needed): `agibot_a2`, `berkeley_humanoid_lite`,
`booster_k1`, `booster_t1`, `booster_t1_29dof`, `engineai_pm01`,
`fourier_gr3v2_1_1`, `fourier_n1`, `galaxea_r1pro`, `hightorque_hi`,
`kuavo_s45`, `openloong`, `pal_talos`, `pnd_adam_lite`, `stanford_toddy`,
`tienkung`, `unitree_h1`, `unitree_h1_2`. This cut GMR from ~1.5 GB to ~59 MB.

### No Git LFS content found
Neither repo uses Git LFS (`.gitattributes` was checked in both, none
found, and no LFS pointer files were present) and no individual file over
20 MB other than the ordinary mesh/asset files under `assets/`. So unlike
`video_to_sonic/4D-Humans/data/` (which needs a manually-downloaded SMPL
`.pkl` the upstream repo can't ship due to license/LFS), there is currently
**no placeholder file needed here** — everything vendored is real content,
already present on disk.

## Sizes after trimming

| Folder | Size |
|---|---|
| `TWIST/` | ~112 MB |
| `GMR/` | ~59 MB |
| **Total** | **~171 MB** |

## Explicitly not done here (by design)

- No `pip install` was run.
- No IsaacGym or other CUDA/training dependency was downloaded.
- Nothing was committed to git — these files are left untracked/staged for
  review.
