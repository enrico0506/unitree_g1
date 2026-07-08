# motion/models — what to download and where it goes

Model weights are **not in git** (large + license-gated — see `.gitignore`). Drop the files
into the folders below and the pipeline finds them via `config/pipeline.yaml`. Ordered by
what you need for each step; you do **not** need all of them at once.

| You want to run… | You need | Gated? |
|---|---|---|
| **Sim render only** (already working) | `models/g1/` G1 MuJoCo model | no |
| **pose→GMR→MuJoCo seam** (next step) | `models/smplx/` + GMR installed | **SMPL-X: yes** |
| Full clip→pose (ROMP) | `models/smpl/` + `simple_romp` | **SMPL: yes** |
| SONIC tracking | `models/sonic/` ONNX + TensorRT 10.7 | on-device |

---

## 1. `models/g1/` — G1 MuJoCo model  (NO registration, already on this machine)

The sim uses `~/Dokumente/g1_bot/unitree_mujoco/unitree_robots/g1/g1_29dof.xml` by default.
To make it self-contained, copy it (with its `meshes/`) here:

```bash
cp -r ~/Dokumente/g1_bot/unitree_mujoco/unitree_robots/g1/* motion/models/g1/
```

Or get the canonical one:
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
cp -r mujoco_menagerie/unitree_g1/* motion/models/g1/     # then point g1_mjcf at the .xml
```

## 2. `models/smplx/` — SMPL-X body model  ⭐ THE ONE THAT UNBLOCKS THE NEXT STEP

GMR needs SMPL-X to turn SMPL parameters into G1 joints. It is **free but registration-gated**:

1. Register (free, ~1 min, non-commercial): **https://smpl-x.is.tue.mpg.de/** → *Download*.
2. Download **"SMPL-X v1.1 (NPZ+PKL, 830 MB)"** (`models_smplx_v1_1.zip`).
3. Unzip and place the neutral/male/female models here:
   ```
   motion/models/smplx/SMPLX_NEUTRAL.npz
   motion/models/smplx/SMPLX_MALE.npz
   motion/models/smplx/SMPLX_FEMALE.npz
   ```
   (NEUTRAL is enough for our use; `pose_to_smpl` writes `gender="neutral"`.)

## 3. GMR — the retargeter  (code, not a model; NO registration)

```bash
git clone https://github.com/YanjieZe/GMR third_party/GMR
conda run -n g1 pip install -e third_party/GMR     # pulls mujoco (have it), mink, smplx
```
GMR ships its own G1 MJCF (`assets/unitree_g1/g1_mocap_29dof.xml`) + meshes — that is the
exact model `joint_maps.G1_MJCF_ORDER` was validated against.

## 4. `models/smpl/` — SMPL body model  (only for the full clip→pose via ROMP)

ROMP needs the **basic SMPL** model (different registration from SMPL-X):
1. Register: **https://smpl.is.tue.mpg.de/** → download **SMPL v1.0.0** (`SMPL_python_v.1.0.0.zip`).
2. Place `basicmodel_{neutral,m,f}_lbs_10_207_0_v1.0.0.pkl` → `motion/models/smpl/`.
3. `conda run -n g1 pip install simple_romp` (downloads its own ROMP checkpoint on first run).

## 5. `models/sonic/` — SONIC whole-body policy  (on-device; hardest)

From **NVlabs GR00T-WholeBodyControl** (Apache-2.0 code, NVIDIA Open Model License weights):
`model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`. Needs **TensorRT 10.7**
on the Orin (JetPack 6). This is Phase 4 / on-device — not needed for the MuJoCo seam.

---

## The moment you have #2 + #3, run the seam validation (no robot, no SONIC):

```bash
MUJOCO_GL=egl conda run -n g1 python motion/sim/validate_pose_gmr.py --out /tmp/g1_pose_gmr
```
It synthesizes a motion, runs `pose_to_smpl → GMR retarget → gmr.pkl`, validates it, and
renders the retargeted result on the G1 in MuJoCo — the last glue seam that can be proven
off-robot. Until the model is present it prints exactly what is missing.
