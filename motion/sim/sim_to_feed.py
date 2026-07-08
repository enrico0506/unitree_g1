#!/usr/bin/env python3
"""sim_to_feed.py — ON-DEVICE MuJoCo render of a motion -> replay.mp4 (Phase 5).

Renders the reproduced motion on the 29-DOF G1 in MuJoCo (offscreen / EGL on the Orin)
into ``replay.mp4``, which the Motion tab plays next to the original clip. Two inputs:

  --sonic <dir>     the SONIC sim2sim output (tracked qpos trajectory) -> render it.
  --kinematic <pkl> a GMR gmr.pkl -> replay dof_pos directly on the model (the FALLBACK
                    path when SONIC can't cleanly track the reference).

ON-DEVICE ONLY: this imports mujoco + loads the G1 MJCF, neither of which exists on a
workstation. So if mujoco (or the model) is missing it prints guidance and EXITS NON-ZERO,
letting SonicProvider flip the job to ERROR (SONIC path) rather than crash. The pure
orchestration around it (motion/app/replay.py) is unit-tested off-robot.

VERIFY-on-robot:
  * The G1 MJCF path + the exact qpos layout SONIC writes (free-joint 7 + 29 dof?), and the
    dof column order (IsaacLab vs MJCF) — render one known clip and eyeball it upright.
  * fps: render at the motion's control_hz (50) so replay.mp4 plays at real speed.
  * EGL/OSMesa headless GL on the Orin (set MUJOCO_GL=egl).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RENDER_HZ = 50.0          # match the pipeline control_hz so replay plays at real speed
G1_MJCF_ENV = "G1_MJCF"   # override the model path via env; else the GMR/mujoco asset


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
        import imageio  # noqa: F401  (mp4 writer)
    except Exception as exc:  # pragma: no cover - on-device only
        sys.stderr.write(
            "sim_to_feed: mujoco + imageio are required and are ON-DEVICE only "
            f"(import failed: {exc}).\n"
            "  On the Orin: pip install mujoco imageio[ffmpeg]; set MUJOCO_GL=egl.\n")
        raise SystemExit(3)


def _load_g1_model():  # pragma: no cover - on-device only
    import os
    import mujoco
    path = os.environ.get(G1_MJCF_ENV)
    if not path:
        sys.stderr.write(
            f"sim_to_feed: set ${G1_MJCF_ENV} to the G1 MJCF used by GMR "
            "(assets/unitree_g1/g1_mocap_29dof.xml) — VERIFY the qpos/dof order matches.\n")
        raise SystemExit(3)
    return mujoco.MjModel.from_xml_path(path)


def _render_qpos(model, qpos_seq, out_path: Path):  # pragma: no cover - on-device only
    """Step the model through the qpos trajectory, capture frames, write mp4 at RENDER_HZ."""
    import mujoco
    import imageio
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    for qpos in qpos_seq:
        data.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=RENDER_HZ, macro_block_size=None)


def _qpos_from_sonic(sonic_dir: Path):  # pragma: no cover - on-device only
    """Load SONIC's tracked qpos trajectory. VERIFY the exact file/format on-robot."""
    import numpy as np
    # SONIC's sim2sim writes a per-step state log; the exact filename/columns are
    # deploy-version-specific -> confirm on-device (look for qpos/*.csv or a .npy).
    for cand in ("qpos.npy", "states.npy"):
        p = sonic_dir / cand
        if p.exists():
            return np.load(p)
    for cand in ("qpos.csv", "states.csv"):
        p = sonic_dir / cand
        if p.exists():
            return np.loadtxt(p, delimiter=",", skiprows=1)
    sys.stderr.write(f"sim_to_feed: no SONIC qpos trajectory found under {sonic_dir} "
                     "(expected qpos.npy/states.npy/qpos.csv) — VERIFY on-robot.\n")
    raise SystemExit(3)


def _qpos_from_gmr(pkl_path: Path):  # pragma: no cover - on-device only
    """Kinematic fallback: build qpos from a GMR gmr.pkl (root free-joint + 29 dof)."""
    import numpy as np
    from motion.pipeline.glue.gmr_pkl import load_gmr_pkl
    g = load_gmr_pkl(str(pkl_path))
    root_pos = np.asarray(g["root_pos"])          # (T,3)
    root_rot_xyzw = np.asarray(g["root_rot"])     # (T,4) XYZW
    dof = np.asarray(g["dof_pos"])                # (T,29) MJCF order
    # MuJoCo free joint qpos = [pos(3), quat wxyz(4)]; convert XYZW->WXYZ.
    root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([root_pos, root_quat_wxyz, dof], axis=1)   # (T, 7+29)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a motion on the G1 in MuJoCo -> replay.mp4")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sonic", metavar="DIR", help="SONIC sim2sim output dir")
    src.add_argument("--kinematic", metavar="PKL", help="GMR gmr.pkl (kinematic fallback)")
    ap.add_argument("--out", required=True, help="output replay.mp4 path")
    args = ap.parse_args(argv)

    _require_mujoco()                       # exits 3 off-robot (mujoco absent)
    model = _load_g1_model()                # pragma: no cover
    if args.sonic:                          # pragma: no cover
        qpos = _qpos_from_sonic(Path(args.sonic))
    else:                                   # pragma: no cover
        qpos = _qpos_from_gmr(Path(args.kinematic))
    _render_qpos(model, qpos, Path(args.out))   # pragma: no cover
    print(f"sim_to_feed: wrote {args.out} ({len(qpos)} frames @ {RENDER_HZ} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
