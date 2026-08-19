"""Height-grid noise model. Written now, OFF for run 1, ON for run 2.

Applied to the grid AFTER grid.sample.heights_to_obs() — operates on the encoded
(NUM_CELLS,) obs vector, same shape/contract as grid/spec.py. numpy only, no
isaaclab, so it's usable from both the training env and later from deploy-side
replay/eval scripts.

Models what the real onboard elevation map actually does wrong: noisy height
estimates, dropped cells (occlusion/lidar dropout), slow map-origin drift
(odometry drift), and N-step latency (map builder runs slower than policy).
"""
from collections import deque

import numpy as np

from grid import spec

# --- defaults, tune per curriculum stage ---------------------------------
HEIGHT_NOISE_STD = 0.02        # m, gaussian per-cell
DROPOUT_PROB = 0.05            # fraction of cells set to UNKNOWN per step
ORIGIN_DRIFT_STD = 0.002       # m/step, random-walk on the grid's assumed base pos
LATENCY_STEPS = 2              # policy sees the grid from N control steps ago

ENABLED = False                # run 1 = clean grid; flip True for run 2


class MapNoise:
    """Stateful per-env noise: call reset() at episode start, apply() every step."""

    def __init__(self, enabled: bool = ENABLED, seed: int | None = None):
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)
        self._origin_offset = np.zeros(2, dtype=np.float32)
        self._latency_buf: deque[np.ndarray] = deque(maxlen=max(LATENCY_STEPS, 1))

    def reset(self):
        self._origin_offset[:] = 0.0
        self._latency_buf.clear()

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """obs: (NUM_CELLS,) clean encoded heights -> noisy encoded heights."""
        if not self.enabled:
            return obs

        out = obs.copy()

        # gaussian height noise
        out += self.rng.normal(0.0, HEIGHT_NOISE_STD, size=out.shape).astype(np.float32)
        out = np.clip(out, spec.CLIP_MIN, spec.CLIP_MAX)

        # dropped cells -> UNKNOWN_FILL, same as an unobserved cell in sample.py
        drop_mask = self.rng.uniform(size=out.shape) < DROPOUT_PROB
        out[drop_mask] = spec.UNKNOWN_FILL

        # origin drift: random-walk shift, applied as a resample (nearest-cell shift)
        self._origin_offset += self.rng.normal(0.0, ORIGIN_DRIFT_STD, size=2).astype(np.float32)
        out = _shift_grid(out, self._origin_offset)

        # latency: return what the map looked like N steps ago
        self._latency_buf.append(out)
        if len(self._latency_buf) < self._latency_buf.maxlen:
            return self._latency_buf[0]  # not enough history yet, hold oldest available
        return self._latency_buf[0]


def _shift_grid(obs: np.ndarray, offset_m: np.ndarray) -> np.ndarray:
    """Shift the (NX, NY) grid by offset_m (meters), nearest-cell, edges fill UNKNOWN."""
    img = obs.reshape(spec.NX, spec.NY)
    dx_cells = int(round(offset_m[0] / spec.RESOLUTION))
    dy_cells = int(round(offset_m[1] / spec.RESOLUTION))
    if dx_cells == 0 and dy_cells == 0:
        return obs
    shifted = np.full_like(img, spec.UNKNOWN_FILL)
    xs = slice(max(0, dx_cells), spec.NX + min(0, dx_cells))
    ys = slice(max(0, dy_cells), spec.NY + min(0, dy_cells))
    src_xs = slice(max(0, -dx_cells), spec.NX - max(0, dx_cells))
    src_ys = slice(max(0, -dy_cells), spec.NY - max(0, dy_cells))
    shifted[xs, ys] = img[src_xs, src_ys]
    return shifted.reshape(-1)


# --- wiring ----------------------------------------------------------------
# One MapNoise instance per env in env_stairs.py's observation term, e.g.:
#
#   self._map_noise = [MapNoise(enabled=cfg.map_noise_enabled) for _ in range(num_envs)]
#   ... on reset(env_ids): [self._map_noise[i].reset() for i in env_ids]
#   ... in the height-scan obs term: obs[i] = self._map_noise[i].apply(obs[i])
#
# Flip ENABLED (or cfg.map_noise_enabled) True for run 2 only, once run 1's clean
# policy climbs reliably.
