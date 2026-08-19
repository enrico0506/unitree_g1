"""Height-grid contract. Shared by training and deployment.

The ONLY definition of the grid. Change here, both sides follow.
numpy only - no isaaclab, no ros. This file must import on the Orin.
"""
import numpy as np

# --- the contract ---------------------------------------------------
EXTENT_X = 1.6        # m, forward/back, centered on base
EXTENT_Y = 1.0        # m, left/right, centered on base
RESOLUTION = 0.05     # m per cell
FRAME = "base"        # robot-centric, yaw-aligned, moves with pelvis

NX = int(round(EXTENT_X / RESOLUTION)) + 1   # 33
NY = int(round(EXTENT_Y / RESOLUTION)) + 1   # 21
NUM_CELLS = NX * NY                          # 693

# obs encoding
CLIP_MIN = -1.0       # m, relative height below base
CLIP_MAX = 1.0
UNKNOWN_FILL = 0.0    # what an unobserved cell becomes AFTER encoding

# --- query points ---------------------------------------------------
# Row-major, x-major: index = ix * NY + iy
# x forward positive, y left positive.
# LOCKED ORDER. Anything that reorders this breaks every trained policy.

def query_points():
    """(NUM_CELLS, 2) array of (x, y) offsets in base frame."""
    xs = np.linspace(-EXTENT_X / 2, EXTENT_X / 2, NX)
    ys = np.linspace(-EXTENT_Y / 2, EXTENT_Y / 2, NY)
    pts = np.zeros((NUM_CELLS, 2), dtype=np.float32)
    i = 0
    for x in xs:
        for y in ys:
            pts[i, 0] = x
            pts[i, 1] = y
            i += 1
    return pts


def spec_dict():
    """For stamping into a saved policy folder, and checking at load time."""
    return {
        "extent_x": EXTENT_X,
        "extent_y": EXTENT_Y,
        "resolution": RESOLUTION,
        "frame": FRAME,
        "nx": NX,
        "ny": NY,
        "num_cells": NUM_CELLS,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "unknown_fill": UNKNOWN_FILL,
    }