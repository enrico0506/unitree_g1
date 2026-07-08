#!/usr/bin/env python3
"""validate_pose_gmr.py — SIM validation of the pose_to_smpl -> GMR seam (workstation).

The last glue seam that can be proven OFF-ROBOT (no SONIC, no real ROMP):

    synth ROMP frames -> pose_to_smpl -> smpl.npz
                      -> GMR retarget (SMPL-X -> G1 29-DOF) -> gmr.pkl
                      -> validate gmr.pkl -> render on the G1 in MuJoCo

It closes the one question the pure-numpy tests can't: does GMR actually ACCEPT the .npz
pose_to_smpl writes, and does the retarget produce an upright G1? That needs the two
gated pieces (see motion/models/README.md):

  * GMR installed:   git clone https://github.com/YanjieZe/GMR third_party/GMR
                     conda run -n g1 pip install -e third_party/GMR
  * SMPL-X model:    register at smpl-x.is.tue.mpg.de -> models/smplx/SMPLX_NEUTRAL.npz

Until both are present it prints EXACTLY what is missing and exits non-zero (never a stack
trace). Run:
    MUJOCO_GL=egl conda run -n g1 python motion/sim/validate_pose_gmr.py --out /tmp/g1_pose_gmr
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from motion.pipeline.glue.pose_to_smpl import pose_to_smpl, save_smpl_sequence  # noqa: E402

FPS = 30.0
T = 60


def synth_romp_frames() -> list[dict]:
    """A gentle, valid standing motion (small arm + torso sway) as synthetic ROMP per-frame
    dicts — enough to prove GMR ingests the npz and retargets to an upright G1. Exact SMPL
    joint semantics don't matter for the SEAM check, only that it's a plausible pose."""
    t = np.arange(T) / FPS
    frames = []
    for i in range(T):
        theta = np.zeros(72, np.float32)
        # SMPL body_pose starts at index 3 (joint 1). Nudge shoulders (~jnt 16/17) + spine.
        theta[3 + 15 * 3 + 2] = 0.5 + 0.3 * np.sin(2 * np.pi * 0.8 * t[i])   # L shoulder
        theta[3 + 16 * 3 + 2] = -0.5 - 0.3 * np.sin(2 * np.pi * 0.8 * t[i])  # R shoulder
        theta[3 + 2 * 3 + 0] = 0.05 * np.sin(2 * np.pi * 0.5 * t[i])          # spine sway
        frames.append({
            "smpl_thetas": theta.reshape(1, 72),
            "smpl_betas": np.zeros((1, 10), np.float32),
            "cam": np.array([[1.0, 0.0, 0.2]], np.float32),
        })
    return frames


def _preflight(smplx_dir: pathlib.Path, gmr_dir: pathlib.Path):
    missing = []
    gmr_script = gmr_dir / "scripts" / "smplx_to_robot.py"
    if not gmr_script.exists():
        missing.append(
            f"GMR not cloned at {gmr_dir}:\n"
            "    git clone https://github.com/YanjieZe/GMR third_party/GMR")
    else:
        try:
            import general_motion_retargeting  # noqa: F401  (GMR installed?)
        except Exception:
            missing.append(
                "GMR cloned but not INSTALLED — finish the pip install (CPU torch avoids the\n"
                "    multi-GB CUDA download):\n"
                "    conda run -n g1 pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "    conda run -n g1 pip install -e third_party/GMR")
    smplx_models = list(smplx_dir.glob("SMPLX_*.npz")) + list(smplx_dir.glob("SMPLX_*.pkl"))
    if not smplx_models:
        missing.append(
            f"SMPL-X model missing in {smplx_dir} — register (free) at "
            "https://smpl-x.is.tue.mpg.de/ and drop SMPLX_NEUTRAL.npz there "
            "(see motion/models/README.md).")
    try:
        import mujoco  # noqa: F401
    except Exception:
        missing.append("mujoco not importable — run inside the `g1` conda env "
                       "(conda run -n g1 ...) with MUJOCO_GL=egl.")
    return missing, gmr_script


def run(out_dir: pathlib.Path, smplx_dir: pathlib.Path, gmr_dir: pathlib.Path,
        model_xml: str) -> int:
    missing, gmr_script = _preflight(smplx_dir, gmr_dir)
    if missing:
        print("Cannot run the pose->GMR seam yet. Provide:\n")
        for m in missing:
            print("  - " + m + "\n")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    smpl_npz = out_dir / "smpl.npz"
    gmr_pkl = out_dir / "gmr.pkl"

    # 1) synth -> pose_to_smpl -> smpl.npz
    seq = pose_to_smpl(synth_romp_frames(), FPS)
    save_smpl_sequence(seq, smpl_npz)
    print(f"[1] pose_to_smpl -> {smpl_npz} (T={len(seq['root_orient'])})")

    # 2) GMR retarget. NB: GMR's CLI args are version-dependent (GMR is active dev) —
    # confirm against your installed version; adjust here if it differs.
    cmd = [sys.executable, str(gmr_script),
           "--smplx_file", str(smpl_npz),
           "--robot", "unitree_g1",
           "--save_path", str(gmr_pkl)]
    print(f"[2] GMR retarget: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(gmr_dir), capture_output=True, text=True,
                          env={"SMPLX_DIR": str(smplx_dir), **_env()})
    if proc.returncode != 0 or not gmr_pkl.exists():
        print("    GMR retarget FAILED — likely the SMPL-X model path or a CLI-arg drift.\n"
              f"    stderr tail:\n{(proc.stderr or proc.stdout)[-800:]}")
        return 1

    # 3) validate gmr.pkl
    from motion.pipeline.glue.gmr_pkl import load_gmr_pkl
    g = load_gmr_pkl(str(gmr_pkl))
    print(f"[3] gmr.pkl valid: dof_pos {np.asarray(g['dof_pos']).shape}, "
          f"root_pos {np.asarray(g['root_pos']).shape}")

    # 4) render on the G1 in MuJoCo
    import mujoco
    from motion.sim.validate_in_mujoco import render
    model = mujoco.MjModel.from_xml_path(model_xml)
    render(g, model, out_dir)
    print(f"\nPOSE->GMR SEAM PASS — retargeted motion rendered to {out_dir/'replay.mp4'}")
    return 0


def _env():
    import os
    return dict(os.environ)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/g1_pose_gmr")
    ap.add_argument("--smplx-dir", default=str(_REPO / "motion" / "models" / "smplx"))
    ap.add_argument("--gmr-dir", default=str(_REPO / "motion" / "third_party" / "GMR"))
    # Default to GMR's OWN g1_mocap_29dof.xml — the exact model it retargets to (dof order
    # == joint_maps.G1_MJCF_ORDER), meshes shipped in the clone. Fall back to unitree_mujoco.
    _gmr_model = _REPO / "motion" / "third_party" / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    _fallback = "/home/enrico/Dokumente/g1_bot/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
    ap.add_argument("--model", default=str(_gmr_model) if _gmr_model.exists() else _fallback)
    a = ap.parse_args(argv)
    return run(pathlib.Path(a.out), pathlib.Path(a.smplx_dir), pathlib.Path(a.gmr_dir), a.model)


if __name__ == "__main__":
    raise SystemExit(main())
