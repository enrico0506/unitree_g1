from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
FOURD_ROOT = THIS_DIR / "4D-Humans"
INPUT_VIDEOS_DIR = THIS_DIR / "input_videos"
VIDEO_TO_POSE_DIR = THIS_DIR / "video_to_pose"
POSE_TO_GMR_DIR = THIS_DIR / "pose_to_gmr"

EXPORT_SCRIPT = VIDEO_TO_POSE_DIR / "export_primary_subject_sequence.py"
ADAPT_SCRIPT = POSE_TO_GMR_DIR / "adapt_smpl_to_gmr_smplx.py"
DEMO_SCRIPT = FOURD_ROOT / "demo.py"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end video_to_motion pipeline: video -> frames -> 4D-Humans -> GMR-ready SMPL-X."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help=(
            "Source video path. If omitted, the script auto-picks the only video found in "
            "video_to_motion/input_videos/."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional run name. Defaults to a slug generated from the video filename.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame extraction rate for ffmpeg. Use 10 for a pragmatic CPU workflow.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for 4D-Humans and detectron2.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="regnety",
        choices=["vitdet", "regnety"],
        help="Person detector backend. regnety is usually faster on CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="4D-Humans batch size.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional cap for the 4D-Humans export stage. 0 means all extracted frames.",
    )
    parser.add_argument(
        "--render-demo",
        action="store_true",
        help="Also run 4D-Humans demo.py to generate visual comparison images.",
    )
    parser.add_argument(
        "--side_view",
        action="store_true",
        help="When --render-demo is enabled, include the side-view render.",
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="When --render-demo is enabled, include the top-view render.",
    )
    parser.add_argument(
        "--no_full_frame",
        action="store_true",
        help="When --render-demo is enabled, skip the full-frame overlay render.",
    )
    parser.add_argument(
        "--gmr_smoothing_window",
        type=int,
        default=15,
        help="Passed through to the SMPL->SMPL-X adapter's --smoothing_window (see its own help for why 15).",
    )
    parser.add_argument(
        "--gmr_gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
        help="Passed through to the adapter's --gender.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "video_to_motion_run"


def _find_candidate_videos() -> list[Path]:
    INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in INPUT_VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def _resolve_video(video_arg: str) -> Path:
    if video_arg:
        candidate = Path(video_arg).expanduser()
        if candidate.exists():
            return candidate.resolve()
        input_dir_candidate = INPUT_VIDEOS_DIR / video_arg
        if input_dir_candidate.exists():
            return input_dir_candidate.resolve()
        raise FileNotFoundError(
            f"Video not found: {video_arg}\n"
            f"Also checked: {input_dir_candidate}"
        )

    candidates = _find_candidate_videos()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "No source video found. Put one file in video_to_motion/input_videos/ "
            "or pass --video explicitly."
        )
    available = "\n".join(f"- {path.name}" for path in candidates)
    raise RuntimeError(
        "Multiple source videos found. Pass --video explicitly.\n"
        f"Available videos:\n{available}"
    )


def _check_dependencies() -> None:
    missing = [tool for tool in ("ffmpeg",) if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required tools on PATH: {', '.join(missing)}")
    if not FOURD_ROOT.exists():
        raise FileNotFoundError(f"Missing 4D-Humans directory: {FOURD_ROOT}")


def _clean_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(subprocess.list2cmdline([part]) for part in command)
    location = f" (cwd={cwd})" if cwd else ""
    print(f"$ {printable}{location}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _count_extracted_frames(frame_dir: Path) -> int:
    return sum(1 for _ in frame_dir.glob("frame_*.jpg"))


def main() -> None:
    args = _parse_args()
    _check_dependencies()

    source_video = _resolve_video(args.video)
    run_name = _slugify(args.name) if args.name else _slugify(source_video.stem)

    frame_dir = VIDEO_TO_POSE_DIR / "inputs" / f"{run_name}_frames"
    export_npz = VIDEO_TO_POSE_DIR / "exports" / f"{run_name}.npz"
    demo_out_dir = VIDEO_TO_POSE_DIR / "outputs" / f"{run_name}_demo"
    smplx_npz = POSE_TO_GMR_DIR / "gmr_ready" / f"{run_name}_smplx.npz"

    for path in (
        INPUT_VIDEOS_DIR,
        VIDEO_TO_POSE_DIR / "inputs",
        VIDEO_TO_POSE_DIR / "exports",
        VIDEO_TO_POSE_DIR / "outputs",
        POSE_TO_GMR_DIR / "gmr_ready",
        FOURD_ROOT / ".local_home" / ".config" / "matplotlib",
    ):
        path.mkdir(parents=True, exist_ok=True)

    for generated_path in (frame_dir, export_npz, demo_out_dir, smplx_npz):
        _clean_path(generated_path)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print("Pipeline configuration:", flush=True)
    print(f"- source_video: {source_video}", flush=True)
    print(f"- run_name: {run_name}", flush=True)
    print(f"- extraction_fps: {args.fps}", flush=True)
    print(f"- device: {args.device}", flush=True)
    print(f"- detector: {args.detector}", flush=True)

    frame_pattern = frame_dir / "frame_%06d.jpg"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-vf",
            f"fps={args.fps}",
            str(frame_pattern),
        ]
    )

    extracted_frames = _count_extracted_frames(frame_dir)
    if extracted_frames == 0:
        raise RuntimeError(f"No frames were extracted into {frame_dir}")
    print(f"Extracted {extracted_frames} frames into {frame_dir}", flush=True)

    fourd_env = os.environ.copy()
    fourd_env["HOME"] = str((FOURD_ROOT / ".local_home").resolve())
    fourd_env["MPLCONFIGDIR"] = str((FOURD_ROOT / ".local_home" / ".config" / "matplotlib").resolve())
    fourd_env["PYTHONUNBUFFERED"] = "1"

    export_command = [
        sys.executable,
        "-u",
        str(EXPORT_SCRIPT),
        "--img_folder",
        str(frame_dir.resolve()),
        "--out_npz",
        str(export_npz.resolve()),
        "--device",
        args.device,
        "--detector",
        args.detector,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.max_frames > 0:
        export_command.extend(["--max_frames", str(args.max_frames)])

    _run(export_command, cwd=FOURD_ROOT, env=fourd_env)

    if args.render_demo:
        demo_command = [
            sys.executable,
            "-u",
            str(DEMO_SCRIPT),
            "--device",
            args.device,
            "--detector",
            args.detector,
            "--img_folder",
            str(frame_dir.resolve()),
            "--out_folder",
            str(demo_out_dir.resolve()),
            "--batch_size",
            str(args.batch_size),
        ]
        if not args.no_full_frame:
            demo_command.append("--full_frame")
        if args.side_view:
            demo_command.append("--side_view")
        if args.top_view:
            demo_command.append("--top_view")
        _run(demo_command, cwd=FOURD_ROOT, env=fourd_env)

    _run(
        [
            sys.executable,
            "-u",
            str(ADAPT_SCRIPT),
            "--input_npz",
            str(export_npz.resolve()),
            "--output_npz",
            str(smplx_npz.resolve()),
            "--fps",
            str(args.fps),
            "--gender",
            args.gmr_gender,
            "--smoothing_window",
            str(args.gmr_smoothing_window),
        ]
    )

    print("\nPipeline outputs:", flush=True)
    print(f"- frames: {frame_dir}", flush=True)
    print(f"- stage1_export: {export_npz}", flush=True)
    if args.render_demo:
        print(f"- rendered_comparisons: {demo_out_dir}", flush=True)
    print(f"- gmr_ready_smplx: {smplx_npz}", flush=True)
    print(
        "\nNext step (not run automatically): retarget onto G1 with GMR, e.g.\n"
        f"  cd ../twist_deploy/GMR && python3 scripts/smplx_to_robot.py "
        f"--smplx_file {smplx_npz.resolve()} --robot unitree_g1 --save_path <out>.pkl",
        flush=True,
    )


if __name__ == "__main__":
    main()
