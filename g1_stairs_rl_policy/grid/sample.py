"""Turn ground heights into the policy observation vector.

One implementation. Training calls it with mesh lookups,
deployment calls it with elevation-map lookups.
"""
import numpy as np

from . import spec


def heights_to_obs(ground_z, base_z):
    """
    ground_z : (NUM_CELLS,) absolute ground height at each query point.
               np.nan for unobserved cells.
    base_z   : scalar, absolute height of robot base.

    returns  : (NUM_CELLS,) float32, clipped relative height.
               Negative = ground below base (normal).
    """
    ground_z = np.asarray(ground_z, dtype=np.float32)
    if ground_z.shape != (spec.NUM_CELLS,):
        raise ValueError(
            f"expected {(spec.NUM_CELLS,)}, got {ground_z.shape}"
        )

    rel = ground_z - float(base_z)
    rel = np.clip(rel, spec.CLIP_MIN, spec.CLIP_MAX)

    unknown = np.isnan(ground_z)
    rel[unknown] = spec.UNKNOWN_FILL
    return rel.astype(np.float32)


def obs_to_image(obs):
    """(NUM_CELLS,) -> (NX, NY) for plotting. Debug only, never in the loop."""
    return np.asarray(obs, dtype=np.float32).reshape(spec.NX, spec.NY)