"""Stair terrain + curriculum.

Reuses Isaac Lab's own stair sub-terrain classes (isaaclab.terrains) — same ones
every robot's rough_env_cfg.py uses via ROUGH_TERRAINS_CFG
(isaaclab.terrains.config.rough) — instead of hand-rolled mesh code. This is a
stairs-only subset of that same config: pyramid_stairs + its inverted twin (up/
down), tuned to our riser/tread range, plus a flat slice so the policy doesn't
forget how to walk. difficulty (0..1) is interpolated by the generator per
curriculum row automatically — no custom code needed for that.

Needs isaaclab installed; not import-tested on this machine. Verify per the
README block at the bottom.
"""
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)

# --- curriculum bounds -----------------------------------------------------
# Row 0 (easy) -> row N-1 (hard). step_height_range is what Isaac Lab
# curriculum-interpolates across rows; step_width (tread depth) is fixed per
# sub-terrain, so "tread shrinks with difficulty" is approximated by giving
# the up/down variants a mid-range fixed tread rather than a true per-row ramp.
RISER_RANGE = (0.10, 0.25)   # m, easy -> hard
TREAD_DEPTH = 0.28           # m, fixed (midpoint of the 0.35 -> 0.22 target)
PLATFORM_WIDTH = 2.0         # m, flat top landing
BORDER_WIDTH = 1.0           # m, generator-level: flat margin around the WHOLE tiled grid (rows x cols)
SUBTERRAIN_BORDER_WIDTH = 0.5  # m, per-patch: flat margin baked into EACH 8x8 patch, separate scope from
                                # BORDER_WIDTH above. With platform_width=2.0 on an 8m patch, this was
                                # previously also 1.0, leaving only (8 - 2 - 2*1)/2 = 2m of actual stair run
                                # per side; 0.5 leaves 3m — check the viewer for enough steps per flight.
PATCH_SIZE = 8.0             # m, square terrain patch


def make_terrain_generator_cfg(num_rows: int = 10, num_cols: int = 20) -> TerrainGeneratorCfg:
    """Curriculum terrain grid. num_rows = difficulty levels, num_cols = variant repeats."""
    return TerrainGeneratorCfg(
        size=(PATCH_SIZE, PATCH_SIZE),
        border_width=BORDER_WIDTH,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                proportion=0.45,
                step_height_range=RISER_RANGE,
                step_width=TREAD_DEPTH,
                platform_width=PLATFORM_WIDTH,
                border_width=SUBTERRAIN_BORDER_WIDTH,
            ),
            "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.45,
                step_height_range=RISER_RANGE,
                step_width=TREAD_DEPTH,
                platform_width=PLATFORM_WIDTH,
                border_width=SUBTERRAIN_BORDER_WIDTH,
            ),
            # flat, so the policy doesn't forget how to walk
            "flat": MeshPlaneTerrainCfg(proportion=0.10),
        },
    )


# --- open items, not built (drop hand-rolled mesh code per decision to reuse
# stock Isaac Lab terrain classes instead) ----------------------------------
# - uneven risers (+-1cm jitter), whole-staircase yaw, mid-flight landing:
#   none of these exist as stock isaaclab sub-terrain classes. If wanted later,
#   write a custom SubTerrainBaseCfg + function (isaaclab.terrains.trimesh
#   pattern: a module-level function(difficulty, cfg) -> (meshes, origin),
#   referenced via cfg.function) rather than reinventing pyramid_stairs itself.
#
# --- viewer check ------------------------------------------------------
# from isaaclab.terrains import TerrainGenerator
# gen = TerrainGenerator(cfg=make_terrain_generator_cfg(num_rows=10, num_cols=20), device="cpu")
# open the produced mesh/USD in the Isaac Sim viewer, eyeball each row across
# the curriculum (easy row: ~10cm riser; hard row: ~25cm riser, 28cm tread throughout).
