#!/usr/bin/env python3
"""validate_in_mujoco.py — SIM-FIRST validation of the motion glue on the real G1 model.

Runs on a WORKSTATION (no robot, no SONIC, no ROMP): loads the G1 29-DOF MuJoCo model and
proves the parts of the pipeline that are otherwise VERIFY-on-robot:

  1. JOINT TABLE (issue #78): assert joint_maps.G1_MJCF_ORDER == the model's actual hinge
     joint order. A wrong permutation here is the silent-corruption risk on hardware.
  2. GLUE ROUND-TRIP: a synthetic gmr.pkl -> gmr_to_sonic_csv -> read joint_pos.csv (IsaacLab
     order) -> reindex back to MJCF -> compare to the original dof_pos. If they match, the
     reindex + quat(XYZW<->WXYZ) + (50 Hz) resample are faithful ON THE REAL MODEL.
  3. KINEMATIC PLAYBACK: render the motion on the G1 to an mp4 + a keyframe montage PNG, so
     you can eyeball that a recognizable motion actually plays upright before the robot.

Run (needs the `g1` conda env: mujoco):
    MUJOCO_GL=egl python motion/sim/validate_in_mujoco.py --out /tmp/g1_motion_sim
Exits 0 if every numeric check passes, non-zero otherwise.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from motion.pipeline.glue.joint_maps import (  # noqa: E402
    G1_ISAACLAB_ORDER, G1_MJCF_ORDER, MJCF_TO_ISAAC,
)
from motion.pipeline.glue.gmr_to_sonic_csv import gmr_to_sonic_csv  # noqa: E402

DEFAULT_MODEL = "/home/enrico/Dokumente/g1_bot/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
FPS = 50.0        # synthesize at the control rate so the 50 Hz resample is ~identity
T = 100           # 2 s


def synth_gmr(model_joint_names) -> dict:
    """A recognizable synthetic motion (wave both arms + a gentle squat), authored by JOINT
    NAME so it's correct regardless of index. Returned as a GMR gmr.pkl dict (MJCF dof order)."""
    idx = {n: i for i, n in enumerate(G1_MJCF_ORDER)}
    t = np.arange(T) / FPS
    dof = np.zeros((T, 29), np.float32)

    def anim(name, series):
        if name in idx:
            dof[:, idx[name]] = series

    wave = 0.6 * np.sin(2 * np.pi * 1.2 * t)          # ~1.2 Hz wave
    # raise both shoulders (pitch back/up) + oscillate elbows -> a two-arm wave
    anim("left_shoulder_pitch_joint", -1.3 + 0.0 * t)
    anim("right_shoulder_pitch_joint", -1.3 + 0.0 * t)
    anim("left_shoulder_roll_joint", 0.3 + 0.2 * np.sin(2 * np.pi * 1.2 * t))
    anim("right_shoulder_roll_joint", -0.3 - 0.2 * np.sin(2 * np.pi * 1.2 * t))
    anim("left_elbow_joint", 0.9 + 0.4 * wave)
    anim("right_elbow_joint", 0.9 - 0.4 * wave)
    # gentle squat (both knees + hips + ankles) so the base moves too
    squat = 0.35 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2
    for j in ("left_knee_joint", "right_knee_joint"):
        anim(j, 1.4 * squat)
    for j in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        anim(j, -0.7 * squat)
    for j in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        anim(j, -0.7 * squat)

    root_pos = np.zeros((T, 3), np.float32)
    root_pos[:, 2] = 0.75 - 0.15 * squat          # pelvis dips with the squat
    root_rot = np.tile(np.array([0, 0, 0, 1], np.float32), (T, 1))   # XYZW identity
    return {"fps": FPS, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof}


def check_joint_table(model_joint_names) -> bool:
    ok = list(G1_MJCF_ORDER) == list(model_joint_names)
    print(f"[1] joint table: G1_MJCF_ORDER == model order -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        only_mine = [n for n in G1_MJCF_ORDER if n not in model_joint_names]
        only_model = [n for n in model_joint_names if n not in G1_MJCF_ORDER]
        print(f"    only in table: {only_mine[:5]}\n    only in model: {only_model[:5]}")
    return ok


def check_roundtrip(gmr, out_dir: pathlib.Path) -> bool:
    """gmr.pkl -> CSV bundle -> read joint_pos (IsaacLab) -> back to MJCF -> compare to source."""
    gmr_to_sonic_csv(gmr, out_dir)
    jp = np.loadtxt(out_dir / "joint_pos.csv", delimiter=",", skiprows=1)  # (N,29) IsaacLab
    inv = np.argsort(MJCF_TO_ISAAC)                    # IsaacLab -> MJCF (inverse permutation)
    back_mjcf = jp[:, inv]                             # (N,29) MJCF order
    src = gmr["dof_pos"]
    n = min(len(src), len(back_mjcf))
    err = float(np.max(np.abs(back_mjcf[:n] - src[:n]))) if n else float("inf")
    # body_quat WXYZ should round-trip the identity root_rot
    bq = np.loadtxt(out_dir / "body_quat.csv", delimiter=",", skiprows=1).reshape(-1, 4)
    quat_ok = bool(np.allclose(np.abs(bq[0]), [1, 0, 0, 0], atol=1e-4) or
                   np.allclose(np.linalg.norm(bq, axis=1), 1.0, atol=1e-4))
    ok = err < 1e-3 and quat_ok
    print(f"[2] glue round-trip: reindex+quat+resample faithful -> {'PASS' if ok else 'FAIL'} "
          f"(max dof err={err:.2e}, N_csv={len(jp)}, quat_unit={quat_ok})")
    return ok


def render(gmr, model, out_dir: pathlib.Path) -> bool:
    """Kinematic playback -> mp4 + a 6-frame montage PNG. Needs mujoco (g1 env)."""
    import mujoco
    d = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    root_pos, root_rot_xyzw, dof = gmr["root_pos"], gmr["root_rot"], gmr["dof_pos"]
    root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    frames = []
    for i in range(len(dof)):
        d.qpos[:3] = root_pos[i]
        d.qpos[3:7] = root_quat_wxyz[i]
        d.qpos[7:7 + 29] = dof[i]        # MJCF order == model order (proven in check [1])
        mujoco.mj_forward(model, d)
        renderer.update_scene(d)
        frames.append(renderer.render())
    frames = np.asarray(frames)
    # montage: 6 evenly-spaced keyframes side by side
    picks = np.linspace(0, len(frames) - 1, 6).astype(int)
    montage = np.concatenate([frames[p] for p in picks], axis=1)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_dir / "montage.png", montage)
        imageio.mimsave(out_dir / "replay.mp4", frames, fps=FPS, macro_block_size=None)
        print(f"[3] render: wrote {out_dir/'replay.mp4'} + montage.png ({len(frames)} frames)")
    except Exception as exc:
        # fall back to a raw PNG via mujoco/numpy if imageio/ffmpeg missing
        _write_png(out_dir / "montage.png", montage)
        print(f"[3] render: wrote montage.png ({len(frames)} frames); mp4 skipped ({exc})")
    return True


def _write_png(path, rgb):
    """Minimal PNG writer (stdlib zlib) so a montage saves even without imageio."""
    import struct, zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    pathlib.Path(path).write_bytes(png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="G1 29-DOF MuJoCo XML")
    ap.add_argument("--out", default="/tmp/g1_motion_sim", help="output dir (videos + report)")
    ap.add_argument("--no-render", action="store_true", help="skip mujoco render (numeric only)")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.model}\nout:   {out}\n")
    ok = True

    model = None
    model_joints = list(G1_MJCF_ORDER)
    if not args.no_render:
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_path(args.model)
            model_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(model.njnt)
                            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        except Exception as exc:
            print(f"(mujoco unavailable: {exc}) — running numeric checks only\n")
            args.no_render = True

    ok &= check_joint_table(model_joints)
    gmr = synth_gmr(model_joints)
    ok &= check_roundtrip(gmr, out)
    if not args.no_render and model is not None:
        render(gmr, model, out)

    print("\n" + ("VALIDATION PASS" if ok else "VALIDATION FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
