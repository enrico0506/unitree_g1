from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "GMR"))
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402


def _foot_contact_geoms(model: mj.MjModel) -> list[tuple[int, float]]:
    """Real physics contact spheres on the feet (contype != 0), not the visual mesh.

    The visual foot mesh has a much larger bounding sphere than the actual
    sole depth and overstates how low the foot really reaches — see the
    comment in adapt_smpl_to_gmr_smplx.py's Fix 3 for how this bit us once
    already trying to precompute a pelvis height analytically.
    """
    geoms = []
    for i in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[i]).name
        if "ankle_roll" in body_name and model.geom_contype[i] != 0:
            geoms.append((i, model.geom_size[i][0]))
    if not geoms:
        raise RuntimeError("No foot contact geoms found (expected *ankle_roll* bodies with contype != 0)")
    return geoms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shift retargeted root_pos so feet sit exactly on the floor (z=0), measured empirically from the model, not predicted from input trans."
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
    foot_geoms = _foot_contact_geoms(model)

    num_frames = len(motion["root_pos"])
    lowest_per_frame = np.empty(num_frames)
    for t in range(num_frames):
        data.qpos[:3] = motion["root_pos"][t]
        data.qpos[3:7] = motion["root_rot"][t][[3, 0, 1, 2]]
        data.qpos[7:] = motion["dof_pos"][t]
        mj.mj_forward(model, data)
        lowest_per_frame[t] = min(data.geom_xpos[i][2] - r for i, r in foot_geoms)

    # Use the median across the clip as the calibration offset rather than the
    # per-frame minimum: a per-frame correction would mask genuine vertical
    # motion (e.g. a real footstep) by yanking every frame to exactly z=0,
    # which is wrong for anything but a static standing clip. Median is a
    # single global calibration shift, robust to a few outlier frames, and
    # doesn't hide real motion within the clip.
    offset = -float(np.median(lowest_per_frame))
    motion["root_pos"] = motion["root_pos"].copy()
    motion["root_pos"][:, 2] += offset

    print(f"Foot contact before shift: min={lowest_per_frame.min():.4f} median={np.median(lowest_per_frame):.4f} max={lowest_per_frame.max():.4f}")
    print(f"Applied uniform root_pos z offset: {offset:+.4f} m")

    with open(output_pkl, "wb") as f:
        pickle.dump(motion, f)
    print(f"Wrote corrected motion to {output_pkl}")


if __name__ == "__main__":
    main()
