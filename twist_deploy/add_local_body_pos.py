from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "GMR"))
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the local_body_pos/link_body_list fields TWIST's MotionLib "
            "requires but GMR's smplx_to_robot.py leaves as None. Computes each "
            "body's position relative to the pelvis (root), expressed in the "
            "pelvis's own rotating frame, via forward kinematics on the actual "
            "G1 model — not guessed."
        )
    )
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--output_pkl", default=None, help="Defaults to overwriting --pkl_path")
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    output_pkl = Path(args.output_pkl) if args.output_pkl else pkl_path

    with open(pkl_path, "rb") as f:
        motion = pickle.load(f)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)

    # All real robot bodies except the implicit "world" body (id 0).
    body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]

    num_frames = len(motion["root_pos"])
    local_body_pos = np.empty((num_frames, len(body_names), 3), dtype=np.float32)

    for t in range(num_frames):
        data.qpos[:3] = motion["root_pos"][t]
        data.qpos[3:7] = motion["root_rot"][t][[3, 0, 1, 2]]
        data.qpos[7:] = motion["dof_pos"][t]
        mj.mj_forward(model, data)

        root_pos_world = data.xpos[model.body("pelvis").id]
        root_rot_world = R.from_quat(motion["root_rot"][t])  # x,y,z,w

        for j, name in enumerate(body_names):
            body_pos_world = data.xpos[model.body(name).id]
            local_body_pos[t, j] = root_rot_world.inv().apply(body_pos_world - root_pos_world)

    motion["local_body_pos"] = local_body_pos
    motion["link_body_list"] = body_names

    with open(output_pkl, "wb") as f:
        pickle.dump(motion, f)

    print(f"Computed local_body_pos for {len(body_names)} bodies x {num_frames} frames")
    print(f"Wrote {output_pkl}")


if __name__ == "__main__":
    main()
