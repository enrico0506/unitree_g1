#!/usr/bin/env python3
"""Run a motion_library clip through HoloMotion's MuJoCo sim2sim eval.

Runs *inside* the holomotion_sim2sim container (see run_holomotion.sh, the
host-side entrypoint that starts the container and calls this -- that's the
one you actually want to run). Looks a motion up by name in the mounted
motion_library (/workspace/motions), then drives
holomotion/src/evaluation/eval_mujoco_sim2sim.py against the shared v1.4
general motion-tracking policy (/workspace/ckpt/exported/model_14000.onnx) so
any clip in the library can be checked without a per-motion training run.

Output lands at /workspace/ckpt/exported/mujoco_output_<onnx_stem>/, which on
the host is motion/holomotion_ckpt/exported/mujoco_output_<onnx_stem>/ --
<motion>_holomotion.mp4 (rendered rollout) + <motion>_holomotion_robot.npz
(actual robot trajectory, for replay_from_front.py or ref-vs-actual analysis).

Usage (inside the container, or just use ./run_holomotion.sh from the host):
    python sim2sim.py wave_v2
    python sim2sim.py cartwheel --gui           # interactive viewer, needs a display
    python sim2sim.py walk_circle --onnx /workspace/ckpt/exported/model_14000.onnx
    python sim2sim.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MOTIONS_ROOT = Path("/workspace/motions")
HOLOMOTION_ROOT = Path("/workspace/holomotion")
PACKAGE_ROOT = HOLOMOTION_ROOT / "holomotion"
EVAL_SCRIPT = PACKAGE_ROOT / "src" / "evaluation" / "eval_mujoco_sim2sim.py"
ROBOT_XML = HOLOMOTION_ROOT / "assets" / "robots" / "unitree" / "G1" / "29dof" / "scene_29dof.xml"
DEFAULT_ONNX = Path("/workspace/ckpt/exported/model_14000.onnx")
LIVE_FRAME_PATH = Path("/workspace/sim/live/frame.jpg")

# The eval_mujoco_sim2sim.sh wrapper this vendors from sources train.env and
# expects a "holomotion_train" conda env; that env doesn't exist in this
# deploy image (v1.4.0-orin-jp5.1-arm64) -- only holomotion_deploy does, and
# it has everything eval_mujoco_sim2sim.py module-level-imports (mujoco,
# onnxruntime w/ CUDA+TensorRT, ray, hydra, torch, cv2). Confirmed 2026-08-11.
DEPLOY_PYTHON = Path("/root/miniconda3/envs/holomotion_deploy/bin/python")


def find_motions() -> dict[str, Path]:
    """name -> npz path, scanning single/*/ and combined/*/ (mirrors
    motion_library/view.py's find_motions(), duplicated rather than shared
    across the docker mount boundary -- it's 15 lines)."""
    motions: dict[str, Path] = {}
    for category_dir in ("single", "combined"):
        base = MOTIONS_ROOT / category_dir
        if not base.is_dir():
            continue
        for motion_dir in sorted(base.iterdir()):
            if not motion_dir.is_dir():
                continue
            candidates = [
                p for p in motion_dir.glob("*_holomotion.npz") if not p.name.startswith("raw_")
            ]
            if candidates:
                motions[motion_dir.name] = candidates[0]
    return motions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion_name", nargs="?")
    ap.add_argument("--list", action="store_true", help="List motions available in the mounted library")
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Policy checkpoint to evaluate against")
    ap.add_argument("--gui", action="store_true", help="Interactive MuJoCo window instead of headless+video (needs a display)")
    # 50.0, not the clips' authored 30fps: the eval's _resample_motion_data_to_
    # policy_rate (2026-08-11 fix) now writes one video frame per 50Hz policy
    # step regardless of the clip's native rate, so the video container must
    # be tagged at that same 50fps or playback duration comes out wrong
    # (frame_count/30 instead of frame_count/50 -- was stretching every video
    # to 1.67x its correct length after that fix landed).
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    motions = find_motions()

    if args.list or not args.motion_name:
        print("Available motions (from /workspace/motions):")
        for name, path in sorted(motions.items()):
            print(f"  {name:20s} -> {path.relative_to(MOTIONS_ROOT)}")
        if not args.motion_name:
            return

    if args.motion_name not in motions:
        print(f"Unknown motion '{args.motion_name}'. Run with --list to see available names.")
        sys.exit(1)

    if not args.onnx.is_file():
        print(f"ONNX checkpoint not found: {args.onnx}")
        sys.exit(1)

    npz_path = motions[args.motion_name]
    print(f"Evaluating '{args.motion_name}' ({npz_path}) sim2sim against {args.onnx}")

    headless = not args.gui
    cmd = [
        str(DEPLOY_PYTHON),
        str(EVAL_SCRIPT),
        f"headless={'true' if headless else 'false'}",
        f"record_video={'true' if headless else 'false'}",
        "camera_tracking=true",
        "camera_distance=2.5",
        f"video_fps={args.fps}",
        "+model_type=holomotion",
        "use_gpu=true",
        "dump_npzs=true",
        "dump_onnx_io_npy=false",
        "calc_per_clip_metrics=false",
        "generate_report=false",
        "policy_action_delay_step=0",
        "action_delay_type=step",
        f"+ckpt_onnx_path={args.onnx}",
        f"+motion_npz_path={npz_path}",
        f"robot_xml_path={ROBOT_XML}",
    ]

    env = dict(os.environ)
    if headless:
        # osmesa (the vendored eval_mujoco_sim2sim.sh's headless default)
        # needs libOSMesa, which isn't installed in this container and pulls
        # in a heavy apt dependency to add. EGL renders headless straight
        # through the Tegra GPU and just works here (confirmed 2026-08-11).
        env.setdefault("MUJOCO_GL", "egl")
        # Live-view: drop every frame as a JPEG for live_view_server.py to
        # serve as an MJPEG stream, alongside the normal mp4/npz output --
        # purely additive, always on for headless runs (cheap, no flag
        # needed). See motion/sim/README.md's "Watching it live" section.
        LIVE_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
        env["LIVE_STREAM_FRAME_PATH"] = str(LIVE_FRAME_PATH)
    else:
        # --gui opens a real GLFW/GLX window (mujoco.viewer.launch_passive)
        # -- EGL is an offscreen-only context and can't back a window. Needs
        # DISPLAY/XAUTHORITY forwarded in from the host (run_holomotion.sh
        # does this) and an X server on your end of the SSH connection.
        env.setdefault("MUJOCO_GL", "glfw")
        # This image's NVIDIA driver has a proper EGL vendor registration
        # (glvnd/egl_vendor.d) but NOT a GLX one -- and GLX vendor
        # auto-negotiation is server-side anyway, which doesn't apply when
        # the X server on the other end (your laptop's) isn't NVIDIA's.
        # Force GLX through Mesa's software/indirect path explicitly instead
        # of letting GLVND try NVIDIA's driver against a foreign X server,
        # which crashes silently at context creation. Confirmed needed
        # 2026-08-11 -- direct GLFW/GLX over X11 forwarding died with no
        # traceback (just a bare nonzero exit) without this.
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
        env.setdefault("LIBGL_ALWAYS_INDIRECT", "1")
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    # The holomotion package (holomotion/src/...) isn't pip-installed in this
    # deploy image -- no editable install was ever run here. Put the repo
    # root on PYTHONPATH so `import holomotion.src...` resolves as a
    # namespace package instead.
    env["PYTHONPATH"] = str(HOLOMOTION_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    result = subprocess.run(cmd, cwd=str(HOLOMOTION_ROOT), env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    onnx_stem = args.onnx.stem
    out_dir = args.onnx.parent / f"mujoco_output_{onnx_stem}"
    motion_stem = npz_path.stem
    video_path = out_dir / f"{motion_stem}.mp4"
    if headless and video_path.is_file():
        _transcode_to_h264(video_path)

    print()
    print(f"Done. Output (host path: motion/holomotion_ckpt/exported/mujoco_output_{onnx_stem}/):")
    if headless:
        print(f"  video: {video_path}")
    print(f"  robot trajectory npz: {out_dir / f'{motion_stem}_robot.npz'}")


def _transcode_to_h264(video_path: Path) -> None:
    """cv2.VideoWriter (the vendored eval script's video output) encodes
    with the 'mp4v' fourcc, which is MPEG-4 Part 2 (FMP4) -- not H.264.
    Browsers and VS Code's built-in video preview only decode H.264/VP8/9/
    AV1 natively, so every video this pipeline produces fails to open
    ("An error occurred while loading the video file") without this.
    Transcodes in place via a static ffmpeg binary (imageio-ffmpeg, no apt/
    system-package involvement -- this container has a pre-existing broken
    cuda-compat-12-2 dependency that `apt --fix-broken install` would
    resolve by *removing*, which risks breaking CUDA; not touching that).
    Best-effort: leaves the original FMP4 file in place if anything fails,
    doesn't fail the run over it. Confirmed needed 2026-08-11."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print(f"(skipping H.264 transcode -- imageio_ffmpeg not installed; {video_path} won't preview in a browser/VS Code)")
        return
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    # ffmpeg picks its output muxer from the filename's extension -- it must
    # end in .mp4, not just contain "mp4" somewhere in the name.
    tmp_path = video_path.with_name(video_path.stem + ".tmp.mp4")
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp_path.is_file():
        print(f"(H.264 transcode failed, leaving original FMP4 file as-is: {result.stderr.strip()[-500:]})")
        tmp_path.unlink(missing_ok=True)
        return
    tmp_path.replace(video_path)


if __name__ == "__main__":
    main()
