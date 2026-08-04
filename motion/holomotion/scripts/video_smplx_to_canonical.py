#!/usr/bin/env python3
"""Wrap video_to_motion's SMPL-X output as a HoloSMPL canonical clip.

video_to_motion/pose_to_gmr/*_smplx.npz already carries root_orient[T,3],
pose_body[T,63], trans[T,3], betas[16], gender, mocap_frame_rate -- almost
exactly HoloSMPL's canonical schema (holosmpl/core/schema/canonical.py). This
just adds the two missing scalars (source_fps/target_fps, split out from
mocap_frame_rate) and a metadata JSON blob with pose_body_layout set, which
canonical_clip_to_formal_clip() requires to know how to expand pose_body into
the 72-dim human_pose_aa downstream.

Usage:
    python video_smplx_to_canonical.py <video_smplx.npz> <canonical_output.npz> [--clip-key NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="video_to_motion *_smplx.npz")
    ap.add_argument("output", type=Path, help="HoloSMPL canonical clip .npz")
    ap.add_argument("--clip-key", default=None)
    ap.add_argument("--target-fps", type=float, default=50.0)
    args = ap.parse_args()

    with np.load(args.source, allow_pickle=True) as d:
        root_orient = np.asarray(d["root_orient"], dtype=np.float32)
        pose_body = np.asarray(d["pose_body"], dtype=np.float32)
        trans = np.asarray(d["trans"], dtype=np.float32)
        betas = np.asarray(d["betas"], dtype=np.float32)
        gender = str(d["gender"])
        source_fps = float(d["mocap_frame_rate"])

    if pose_body.shape[-1] != 63:
        raise ValueError(
            f"Expected pose_body [T,63] (smplx_21_body layout), got shape {pose_body.shape}"
        )

    clip_key = args.clip_key or args.source.stem
    metadata = {
        "dataset": "video_to_motion",
        "clip_id": clip_key,
        "pose_body_layout": "smplx_21_body",
        "source_path": str(args.source),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        betas=betas,
        gender=np.asarray(gender),
        source_fps=np.asarray(source_fps, dtype=np.float32),
        target_fps=np.asarray(args.target_fps, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote canonical clip: {args.output} ({pose_body.shape[0]} frames, {source_fps} -> {args.target_fps} fps)")


if __name__ == "__main__":
    main()
