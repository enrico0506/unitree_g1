#!/usr/bin/env python3
"""Single entry point for SMPL-X -> Unitree G1 ref_dof_pos retargeting.

Picks the GPU pipeline (`gpu_pipeline/retarget_gpu.py`, via HoloRetarget) when
CUDA is available, otherwise falls back to the CPU pipeline
(`cpu_pipeline/retarget_cpu.py`, via GMR). Both wrapper scripts share the same
CLI contract:

    <script> <smplx_npz_input> <output.npz> [--fps N]

so this dispatcher just decides which one to run and forwards args to it
as a subprocess, using the same Python interpreter it was itself invoked
with (`sys.executable`).

Usage:
    python dispatch_retarget.py <smplx_npz_input> <output.npz> [--fps 30] [--force-cpu] [--force-gpu]

Note: the CPU pipeline needs the `4d-humans-cpu` conda env (GMR installed
-e there). If you run dispatch_retarget.py itself with that env's
interpreter, the CPU subprocess inherits it automatically via
`sys.executable`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU_SCRIPT = HERE / "cpu_pipeline" / "retarget_cpu.py"
GPU_SCRIPT = HERE / "gpu_pipeline" / "retarget_gpu.py"


def _cuda_available() -> bool:
    """Best-effort CUDA detection. Never crashes: no torch -> False + warning."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception as exc:  # ImportError or anything torch throws on import
        print(f"[dispatch_retarget] WARNING: could not import torch ({exc}); "
              f"assuming no CUDA, falling back to CPU pipeline.", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smplx_npz_input", type=Path, help="SMPL-X motion npz input")
    parser.add_argument("output", type=Path, help="Output G1 ref_dof_pos npz")
    # Kept as a raw string (not float/int) so it forwards unchanged to whichever
    # wrapper script we dispatch to -- retarget_cpu.py's --fps is type=int and
    # would reject "30.0" if we round-tripped through float here.
    parser.add_argument("--fps", type=str, default="30", help="Source motion frame rate (default: 30)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force-cpu", action="store_true", help="Always use the CPU (GMR) pipeline")
    group.add_argument("--force-gpu", action="store_true", help="Always use the GPU (HoloRetarget) pipeline")
    args = parser.parse_args(argv)

    if args.force_gpu:
        script, reason = GPU_SCRIPT, "--force-gpu passed"
    elif args.force_cpu:
        script, reason = CPU_SCRIPT, "--force-cpu passed"
    elif _cuda_available():
        script, reason = GPU_SCRIPT, "CUDA available"
    else:
        script, reason = CPU_SCRIPT, "CUDA not available"

    pipeline_name = "GPU (HoloRetarget)" if script is GPU_SCRIPT else "CPU (GMR)"
    print(f"[dispatch_retarget] {reason} -> using {pipeline_name} pipeline: {script}")

    cmd = [
        sys.executable,
        str(script),
        str(args.smplx_npz_input),
        str(args.output),
        "--fps",
        args.fps,
    ]
    print(f"[dispatch_retarget] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
