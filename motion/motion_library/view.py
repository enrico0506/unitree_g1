#!/usr/bin/env python3
"""Look up a motion by name in the library (single/ or combined/) and play it
live in an interactive MuJoCo window. Reuses
motion/motion_builder/combined/walk_and_wave/scripts/view_npz.py rather than duplicating
the render loop.

Usage:
    python view.py <motion_name>       # e.g. wave_standing, walk_circle, walk_wave
    python view.py --list              # show every motion name found
"""

import argparse
import runpy
import sys
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parent
VIEWER_SCRIPT = LIBRARY_ROOT.parents[0] / "motion_builder" / "combined" / "walk_and_wave" / "scripts" / "view_npz.py"


def find_motions():
    """name -> npz path, scanning single/*/ and combined/*/ for the one non-raw
    *_holomotion.npz in each motion folder (skips raw_*/ provenance files)."""
    motions = {}
    for category_dir in ("single", "combined"):
        base = LIBRARY_ROOT / category_dir
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion_name", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    motions = find_motions()

    if args.list or not args.motion_name:
        print("Available motions:")
        for name, path in motions.items():
            print(f"  {name:20s} -> {path.relative_to(LIBRARY_ROOT.parents[1])}")
        if not args.motion_name:
            return

    if args.motion_name not in motions:
        print(f"Unknown motion '{args.motion_name}'. Run with --list to see available names.")
        sys.exit(1)

    npz_path = motions[args.motion_name]
    print(f"Viewing {args.motion_name}: {npz_path}")

    sys.argv = [str(VIEWER_SCRIPT), str(npz_path), "--fps", str(args.fps)]
    runpy.run_path(str(VIEWER_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
