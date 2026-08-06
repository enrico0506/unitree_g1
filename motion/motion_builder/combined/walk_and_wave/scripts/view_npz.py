#!/usr/bin/env python3
"""Play back any HoloMotion ref_* npz in a live interactive MuJoCo window.

Give it the npz, it plays the motion on the G1 model, looping, until you close
the window. No video file, no rendering wait -- raw interactive viewer.

Usage:
    python view_npz.py <path_to.npz> [--fps 30] [--key-prefix ref_]
"""

import argparse
import os
import time
from pathlib import Path

# Force GLFW onto XWayland instead of native Wayland -- the native path here
# is missing/broken window-decoration plugins (libdecor-gtk fails to init) and
# its viewer thread doesn't respond to teardown, hanging the process (and the
# terminal's tty) on Ctrl-C. XWayland's GLFW backend tears down cleanly.
# Must happen before mujoco.viewer's first import -- GLFW reads this at init.
os.environ.pop("WAYLAND_DISPLAY", None)

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4] / "holomotion"
MJCF_PATH = REPO_ROOT / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_path", type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--key-prefix", default="ref_", help="ref_ for reference motion, robot_ for sim2sim eval output")
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    p = args.key_prefix
    dof_pos = d[f"{p}dof_pos"]
    root_pos = d[f"{p}global_translation"][:, 0, :]
    root_quat_xyzw = d[f"{p}global_rotation_quat"][:, 0, :]
    T = dof_pos.shape[0]
    print(f"Playing {T} frames from {args.npz_path.name} @ {args.fps} fps -- close window to stop")

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    dt = 1.0 / args.fps

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                for t in range(T):
                    if not viewer.is_running():
                        break
                    step_start = time.time()
                    data.qpos[0:3] = root_pos[t]
                    data.qpos[3:7] = root_quat_xyzw[t][[3, 0, 1, 2]]  # xyzw -> wxyz
                    data.qpos[7:36] = dof_pos[t]
                    mujoco.mj_forward(model, data)
                    viewer.cam.lookat[:] = data.xpos[16]  # torso_link, follow camera
                    viewer.sync()
                    elapsed = time.time() - step_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        # Ctrl-C mid-GLFW-frame otherwise kills the viewer thread abruptly and can
        # leave the terminal's tty settings clobbered. Catching it here lets the
        # `with` block's __exit__ attempt a clean close of the viewer/GL context.
        print("\nStopped (Ctrl-C) -- closing viewer...")
    finally:
        # Belt-and-suspenders: mujoco's passive-viewer render thread has been seen
        # to hang on teardown under some Wayland/GLFW setups even after __exit__
        # "returns" -- os._exit forces the process (and its tty) to actually die
        # instead of leaving a zombie thread holding the terminal hostage.
        print("Done.")
        os._exit(0)


if __name__ == "__main__":
    main()
