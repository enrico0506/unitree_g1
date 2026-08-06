#!/usr/bin/env python3
"""CUDA-only retarget wrapper: SMPL-X motion npz -> G1 ref_dof_pos npz.

Wires this repo's video-to-motion `smplx_npz_input` convention (the format
`video_to_motion/pose_to_gmr/adapt_smpl_to_gmr_smplx.py` writes: keys
`pose_body[T,63]`, `betas[16]`, `root_orient[T,3]`, `trans[T,3]`, `gender`,
`mocap_frame_rate`) through HoloRetarget, the repo's canonical GPU retargeter
under `motion/holomotion/holoretarget/`.

CLI contract (must match `retarget_cpu.py` so a dispatcher can call either
interchangeably):

    python retarget_gpu.py <smplx_npz_input> <output_ref_dof_pos.npz> [--fps 30]

`--fps` is the SOURCE motion's frame rate (the rate the input npz was
captured/exported at). HoloRetarget's own pipeline always resamples to its
internal 50 Hz canonical rate and produces output at 50 Hz -- see
`motion/holomotion/holosmpl/cli.py` (`HOLOSMPL_TARGET_FPS = 50.0`, line 31)
and `motion/holomotion/scripts/holoretarget_h5_to_offline_npz.py` (`--fps`
help text: "HoloRetarget output is always 50Hz", line 83). The output npz's
`metadata.motion_fps` is therefore always 50.0, regardless of `--fps`.

HARD REQUIREMENT -- CUDA: HoloRetarget requires a CUDA-capable Newton/Warp
runtime end to end (see `motion/holomotion/holoretarget/README.md`, line 10:
"HoloRetarget requires a CUDA-capable Newton/Warp runtime"). This script
checks `torch.cuda.is_available()` before importing anything else from
HoloSMPL/HoloRetarget/Warp and raises immediately if there is no CUDA GPU.
This machine (AMD GPU, no CUDA) cannot run this script -- it has only been
verified to import and argument-parse correctly, never executed end to end.

PREREQUISITE -- licensed SMPL model: converting SMPL-family training data
(our case) into HoloRetarget's 24-joint body representation needs the
licensed neutral SMPL model at:

    thirdparties/smpl_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl

per `motion/holomotion/holoretarget/README.md` (lines 46-55). Download
`SMPL_python_v.1.1.0.zip` from the official SMPL website and extract it
under `thirdparties/smpl_models/` yourself -- this script does not fetch it.

Pipeline wired up (each stage cited from source read in this repo, nothing
guessed):

1. Repack `smplx_npz_input` (root_orient/pose_body/trans/betas/gender +
   `--fps` as `mocap_frame_rate`) into a temp dir. These are exactly the
   fields `holosmpl`'s built-in "amass" source reads -- see
   `motion/holomotion/holosmpl/converters/smpl_family/amass.py`
   (`convert_amass_sample`, lines 74-150; field discovery in
   `_load_root_orient`/`_load_pose_body`/`_load_betas`/`_load_source_fps`,
   lines 159-216) and the registry entry `"amass"` in
   `motion/holomotion/holosmpl/supported_datasets/registry.py` (lines 19-25:
   `source_glob="*.npz"`, `classify=classify_amass_source`,
   `convert=convert_amass_sample`).

2. `holosmpl.workflows.raw_to_holosmpl.raw_to_holosmpl(source="amass", ...)`
   (`motion/holomotion/holosmpl/workflows/raw_to_holosmpl.py`, lines 13-88)
   runs canonical -> formal_npz -> formal_h5 conversion, matching what
   `python -m holosmpl convert --source amass ...` does
   (`motion/holomotion/holosmpl/cli.py`, `cmd_convert`, lines 404-435).

3. `holomotion.src.training.data_production.robot_h5.retarget_holosmpl_h5_to_robot_h5`
   (`motion/holomotion/holomotion/src/training/data_production/robot_h5.py`,
   lines 159-273) runs the formal H5 through `HoloRetargeter` and writes a
   robot training H5 v2 shard with `ref_root_pos`/`ref_root_rot`/
   `ref_dof_pos`. This is the exact function `python -m holosmpl
   retarget-holoretarget-h5` calls (`holosmpl/cli.py`,
   `cmd_retarget_holoretarget_h5`, lines 305-328, wired to the
   `retarget-holoretarget-h5` subparser at lines 612-623) -- per
   `motion/holomotion/holoretarget/README.md` (line 28-30), this is the
   documented offline SMPL-family-to-robot-H5 conversion path. This script
   imports and calls the function directly rather than shelling out to
   `python -m holosmpl`, since we already hold the Python objects in-process;
   the underlying call is identical to what the CLI subcommand runs.

4. HoloRetarget's own robot H5 output is intentionally minimal (root pose +
   dof pos only -- see `motion/holomotion/holoretarget/README.md` line 4-6
   and `holoretarget_h5_to_offline_npz.py` lines 5-11). It does NOT already
   match this repo's `ref_dof_pos`/`ref_dof_vel`/`ref_global_translation`/
   `ref_global_rotation_quat`/`ref_global_velocity`/
   `ref_global_angular_velocity` schema (confirmed against
   `motion/motion_builder/combined/walk_and_wave/converted/waving_holomotion.npz`'s keys), so this
   script reshapes it: the same MuJoCo forward-kinematics + central-difference
   pass as `motion/holomotion/scripts/holoretarget_h5_to_offline_npz.py`
   (`forward_kinematics_all_frames`/`central_diff`/
   `angular_velocity_from_quats`, lines 37-76), then
   `humanoid_policy.offline_motion_conversion.convert_legacy_offline_npz`
   (`motion/holomotion/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/offline_motion_conversion.py`,
   lines 96-170) to write the final v1.4 schema npz with exactly those six
   array keys plus `metadata`.

Usage:
    python retarget_gpu.py <smplx_npz_input> <output_ref_dof_pos.npz> [--fps 30]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HOLOMOTION_ROOT = REPO_ROOT / "motion" / "holomotion"
HUMANOID_POLICY_SRC = HOLOMOTION_ROOT / "deployment" / "unitree_g1_ros2_29dof" / "src"
MJCF_PATH = HOLOMOTION_ROOT / "assets" / "robots" / "unitree" / "G1" / "29dof" / "g1_29dof_rev_1_0.xml"

# HoloRetarget's own output rate -- see module docstring point 4.
HOLORETARGET_OUTPUT_FPS = 50.0
# HoloSMPL's internal canonical resample target -- holosmpl/cli.py line 31.
HOLOSMPL_TARGET_FPS = 50.0


def _require_cuda() -> None:
    """Fail fast, before any heavy HoloSMPL/HoloRetarget/Warp import."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "retarget_gpu.py requires a CUDA-capable GPU: HoloRetarget needs "
            "a CUDA-capable Newton/Warp runtime end to end "
            "(motion/holomotion/holoretarget/README.md, line 10). "
            "torch.cuda.is_available() returned False on this machine. "
            "Run this script on a CUDA machine, or use retarget_cpu.py here instead."
        )


def _load_smplx_npz_input(path: Path, *, fps: float) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        missing = [k for k in ("pose_body", "betas", "root_orient", "trans") if k not in data.files]
        if missing:
            raise ValueError(
                f"{path} is missing required smplx_npz_input field(s) {missing}. "
                "Expected the schema written by "
                "video_to_motion/pose_to_gmr/adapt_smpl_to_gmr_smplx.py: "
                "pose_body[T,63], betas[16], root_orient[T,3], trans[T,3], "
                "gender, mocap_frame_rate."
            )
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        gender_raw = data["gender"] if "gender" in data.files else "neutral"

    gender_val = gender_raw.item() if getattr(gender_raw, "shape", None) == () else gender_raw
    if isinstance(gender_val, bytes):
        gender_val = gender_val.decode("utf-8")
    gender = str(gender_val)

    if pose_body.ndim != 2 or pose_body.shape[1] not in (63, 69):
        raise ValueError(f"pose_body must be [T,63] or [T,69], got {pose_body.shape}")
    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError(f"root_orient must be [T,3], got {root_orient.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans.shape}")
    if betas.ndim != 1:
        raise ValueError(f"betas must be [B], got {betas.shape}")

    return {
        "pose_body": pose_body,
        "betas": betas,
        "root_orient": root_orient,
        "trans": trans,
        "gender": gender,
        "mocap_frame_rate": float(fps),
    }


def _write_amass_style_npz(fields: dict, out_path: Path) -> None:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        pose_body=fields["pose_body"],
        betas=fields["betas"],
        root_orient=fields["root_orient"],
        trans=fields["trans"],
        gender=np.asarray(fields["gender"]),
        mocap_frame_rate=np.asarray(fields["mocap_frame_rate"]),
    )


def _forward_kinematics_all_frames(root_pos, root_rot_xyzw, dof_pos):
    """Same MuJoCo FK pass as scripts/holoretarget_h5_to_offline_npz.py:37-54."""

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    assert model.nbody == 31, f"expected 30 bodies + world, got {model.nbody}"

    frame_count = dof_pos.shape[0]
    translation = np.zeros((frame_count, 30, 3), dtype=np.float64)
    rotation = np.zeros((frame_count, 30, 4), dtype=np.float64)

    for t in range(frame_count):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_rot_xyzw[t][[3, 0, 1, 2]]
        data.qpos[7:36] = dof_pos[t]
        mujoco.mj_forward(model, data)
        translation[t] = data.xpos[1:31]
        rotation[t] = data.xquat[1:31][:, [1, 2, 3, 0]]

    return translation, rotation


def _central_diff(x, dt: float):
    import numpy as np

    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def _angular_velocity_from_quats(rotation_xyzw, dt: float):
    """Same finite-difference angular velocity as
    scripts/holoretarget_h5_to_offline_npz.py:65-76."""

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    frame_count, body_count, _ = rotation_xyzw.shape
    rotvec_step = np.zeros((frame_count, body_count, 3), dtype=np.float64)
    for b in range(body_count):
        rots = R.from_quat(rotation_xyzw[:, b, :])
        rel_fwd = (rots[1:] * rots[:-1].inv()).as_rotvec()
        rotvec_step[:-1, b, :] += rel_fwd
        rotvec_step[1:, b, :] += rel_fwd
    counts = np.full((frame_count, 1), 2.0)
    counts[0] = 1.0
    counts[-1] = 1.0
    return rotvec_step / counts[:, :, None] / dt


def _read_robot_h5_shard(shard_path: Path):
    import h5py
    import numpy as np

    with h5py.File(shard_path, "r") as handle:
        root_pos = np.asarray(handle["ref_root_pos"], dtype=np.float64)
        root_rot = np.asarray(handle["ref_root_rot"], dtype=np.float64)  # xyzw
        dof_pos = np.asarray(handle["ref_dof_pos"], dtype=np.float64)
    return root_pos, root_rot, dof_pos


def run_gpu_retarget(smplx_npz_input: Path, output_npz: Path, *, fps: float) -> dict:
    """Run the full smplx_npz -> HoloSMPL -> HoloRetarget -> offline-npz chain."""

    _require_cuda()

    # Heavy imports only after the CUDA gate above.
    if str(HOLOMOTION_ROOT) not in sys.path:
        sys.path.insert(0, str(HOLOMOTION_ROOT))
    if str(HUMANOID_POLICY_SRC) not in sys.path:
        sys.path.insert(0, str(HUMANOID_POLICY_SRC))

    from holosmpl.workflows.raw_to_holosmpl import raw_to_holosmpl
    from holomotion.src.training.data_production.robot_h5 import (
        retarget_holosmpl_h5_to_robot_h5,
    )
    from humanoid_policy.offline_motion_conversion import convert_legacy_offline_npz

    import numpy as np

    smplx_npz_input = Path(smplx_npz_input).resolve()
    output_npz = Path(output_npz).resolve()
    fields = _load_smplx_npz_input(smplx_npz_input, fps=fps)

    with tempfile.TemporaryDirectory(prefix="retarget_gpu_") as tmp:
        tmp_root = Path(tmp)
        amass_input_root = tmp_root / "amass_in"
        _write_amass_style_npz(fields, amass_input_root / f"{smplx_npz_input.stem}.npz")

        holosmpl_out_root = tmp_root / "holosmpl_out"
        holosmpl_result = raw_to_holosmpl(
            source="amass",
            input_root=amass_input_root,
            output_root=holosmpl_out_root,
            target_fps=HOLOSMPL_TARGET_FPS,
            overwrite=True,
            write_formal_h5=True,
        )
        formal_h5_root = Path(holosmpl_result["formal_h5_root"])

        robot_h5_out_root = tmp_root / "robot_h5_out"
        robot_manifest = retarget_holosmpl_h5_to_robot_h5(
            holosmpl_h5_root=formal_h5_root,
            output_root=robot_h5_out_root,
            overwrite=True,
        )
        hdf5_shards = robot_manifest["hdf5_shards"]
        if not hdf5_shards:
            raise RuntimeError(f"HoloRetarget produced no output shards for {smplx_npz_input}")
        shard_path = robot_h5_out_root / hdf5_shards[0]["file"]
        root_pos, root_rot_xyzw, dof_pos = _read_robot_h5_shard(shard_path)

        dt = 1.0 / HOLORETARGET_OUTPUT_FPS
        global_translation, global_rotation_quat = _forward_kinematics_all_frames(
            root_pos, root_rot_xyzw, dof_pos
        )
        dof_vel = _central_diff(dof_pos, dt)
        global_velocity = _central_diff(global_translation, dt)
        global_angular_velocity = _angular_velocity_from_quats(global_rotation_quat, dt)

        legacy_path = tmp_root / "legacy.npz"
        motion_key = smplx_npz_input.stem
        legacy_metadata = {
            "motion_key": motion_key,
            "motion_fps": HOLORETARGET_OUTPUT_FPS,
            "original_num_frames": int(dof_pos.shape[0]),
        }
        np.savez(
            legacy_path,
            metadata=np.asarray(json.dumps(legacy_metadata)),
            dof_pos=dof_pos.astype(np.float32),
            dof_vels=dof_vel.astype(np.float32),
            global_translation=global_translation.astype(np.float32),
            global_rotation_quat=global_rotation_quat.astype(np.float32),
            global_velocity=global_velocity.astype(np.float32),
            global_angular_velocity=global_angular_velocity.astype(np.float32),
        )

        result = convert_legacy_offline_npz(
            legacy_path,
            output_npz,
            fps=HOLORETARGET_OUTPUT_FPS,
            overwrite=True,
        )

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smplx_npz_input", type=Path, help="SMPL-X motion npz (this repo's smplx_npz_input schema)")
    parser.add_argument("output_ref_dof_pos_npz", type=Path, help="Output G1 ref_dof_pos npz (HoloMotion v1.4 offline-tracking schema)")
    parser.add_argument("--fps", type=float, default=30.0, help="Source motion frame rate of the input npz (default: 30)")
    args = parser.parse_args(argv)

    result = run_gpu_retarget(args.smplx_npz_input, args.output_ref_dof_pos_npz, fps=args.fps)
    print(f"Wrote {args.output_ref_dof_pos_npz}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
