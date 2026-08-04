#!/usr/bin/env python3
"""Re-render an already-simulated HoloMotion sim2sim trajectory from a chosen
camera angle, without re-running the (expensive) ONNX policy inference.

Reads the *_robot.npz that eval_mujoco_sim2sim.py dumps (robot_dof_pos,
robot_global_translation/rotation_quat for the root) and replays it through
plain MuJoCo forward kinematics + offscreen render, same approach used
elsewhere in this repo for the TWIST sim2sim view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MJCF_PATH = REPO_ROOT / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=Path)
    ap.add_argument("output_mp4", type=Path)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    ap.add_argument("--distance", type=float, default=2.2)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    dof_pos = d["robot_dof_pos"]
    root_pos = d["robot_global_translation"][:, 0, :]
    root_quat_xyzw = d["robot_global_rotation_quat"][:, 0, :]
    T = dof_pos.shape[0]
    print(f"Replaying {T} frames from {args.npz_path.name}")

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(args.output_mp4), fourcc, args.fps, (args.width, args.height))

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_xyzw[t][[3, 0, 1, 2]]  # xyzw -> wxyz
        data.qpos[7:36] = dof_pos[t]
        mujoco.mj_forward(model, data)
        cam.lookat = data.xpos[16]  # torso_link, matches HoloMotion's own anchor body
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        vw.write(img[:, :, ::-1])  # RGB -> BGR for cv2

    vw.release()
    print(f"Wrote {args.output_mp4}")


if __name__ == "__main__":
    main()
