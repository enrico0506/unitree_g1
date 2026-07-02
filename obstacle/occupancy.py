"""Probabilistic polar log-odds occupancy grid for a 360 deg reactive obstacle ring.

Self-contained, PURE NUMPY (no rclpy/ROS, no hard scipy dependency). Import this
into a ROS2 node later; unit-test it standalone with ``test_occupancy.py``.

WHY
---
Raw per-frame per-sector nearest-distances flicker: a sparse/noisy sector reads
an obstacle one frame and "clear" the next, and a momentarily-missed return reads
as free even when nothing was actually scanned there. This module replaces that
with a recursive Bayesian estimate per (sector, range-bin) cell, giving a stable,
precise, self-clearing 3-state (unknown / free / occupied) ring.

BACKGROUND (binary Bayes filter in log-odds; see refs)
------------------------------------------------------
Each cell holds a log-odds value ``l = log(p / (1 - p))``; recover probability
with ``p = 1 - 1/(1 + exp(l))``. ``l = 0`` is the uninformative prior p = 0.5
("unknown"). The recursive update is simply ``l += inverseSensorModel`` with prior
``l0 = 0`` (OctoMap, Hornung 2013). OctoMap-style defaults used here: an occupied
observation adds ``l_occ = +0.85`` (p ~ 0.7), a free observation adds
``l_free = -0.40`` (p ~ 0.4), and l is clamped to ``[-2.0, +3.5]`` so the estimate
stays responsive.

Inverse sensor model along a beam: cells NEARER than the measured return are FREE
(add l_free), the bin CONTAINING the return is OCCUPIED (add l_occ), and cells
BEYOND the return are UNKNOWN (no change). In a POLAR grid each beam maps to
exactly one angular row, so "ray-casting" collapses to plain range-bin indexing in
that row -- no Bresenham needed. Equivalently: bins with index < return_bin -> free,
index == return_bin -> occupied (this is the intent of "center-range < return", but
index-based avoids double-marking the return bin when the return sits in the upper
half of its own bin).

Anti-flicker measures:
  (a) a sector with NO return this frame gets NO measurement update -- unknown
      persists, it is NOT freed (a missed beam is not evidence of free space);
  (b) leaky exponential decay toward the prior every frame, ``l *= exp(-dt/tau_s)``
      with tau_s ~ 0.7 s (STVL-style time decay), so stale evidence fades and the
      grid self-clears instead of latching forever;
  (c) dual-threshold hysteresis on the derived binary blocked/clear state
      (VFH+): a sector flips to BLOCKED only when its strongest cell l > l_high and
      flips back to CLEAR only when it falls below l_low (or returns to the prior),
      so a cell dithering around one threshold does not toggle every frame.

Refs: OctoMap (Hornung et al., 2013); STVL time-based decay (Macenski);
VFH+ hysteresis (Ulrich & Borenstein, 1998); PSU polar egocentric occupancy grid.

CONVENTIONS
-----------
Angular convention (shared with the rest of the obstacle stack):
``ang_deg = degrees(atan2(y, x))`` -> 0 = straight ahead, + = LEFT (CCW),
range [-180, 180). Sector index of an angle:
``i = floor((ang_deg - start_deg) / bin_deg) % n_sectors``; conversely sector i's
CENTER angle is ``start_deg + (i + 0.5) * bin_deg``. Range bin j spans
``[j*r_bin, (j+1)*r_bin)`` with center ``(j + 0.5) * r_bin``.

Performance: pure vectorized numpy, no per-sector Python loop on the hot path and
no ``np.add.at`` / ``np.minimum.at`` scatter (bincount / boolean-mask adds only).
Target Jetson Orin, ~10-20 Hz; a 180x60 grid updates in well under 1 ms.
"""

import math

import numpy as np

__all__ = ["PolarOccupancyGrid", "UNKNOWN", "FREE", "OCCUPIED"]

# 3-state ring codes (int8).
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class PolarOccupancyGrid:
    """Recursive log-odds occupancy grid over (angular sector, range bin) cells.

    See the module docstring for the model and conventions. The public surface is
    :meth:`update`, :meth:`nearest_occupied`, :meth:`prob_matrix`,
    :meth:`prob_per_sector`, and :meth:`reset`.
    """

    def __init__(
        self,
        n_sectors=180,
        r_max=3.0,
        r_bin=0.05,
        start_deg=-180.0,
        l_occ=0.85,
        l_free=-0.40,
        l_min=-2.0,
        l_max=3.5,
        tau_s=0.7,
        l_high=0.85,
        l_low=0.0,
    ):
        if n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        if r_max <= 0.0 or r_bin <= 0.0:
            raise ValueError("r_max and r_bin must be positive")
        if l_high <= l_low:
            raise ValueError("l_high must be > l_low for hysteresis")

        self.n_sectors = int(n_sectors)
        self.r_max = float(r_max)
        self.r_bin = float(r_bin)
        self.n_rbins = int(math.ceil(r_max / r_bin))
        self.start_deg = float(start_deg)
        self.bin_deg = 360.0 / self.n_sectors

        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.tau_s = float(tau_s)
        self.l_high = float(l_high)
        self.l_low = float(l_low)

        # Small band around l == 0 that counts as "back at the prior" (unknown).
        # Also used as the practical clear tolerance so that a positively-decaying
        # occupied cell (which only approaches 0 from above, never crossing an
        # l_low of 0) still releases the hysteresis latch once it is effectively
        # back at the prior. p(0.05) ~ 0.512, i.e. essentially 0.5.
        self._eps = 0.05

        # Log-odds field, shape (n_sectors, n_rbins). float32 keeps the hot path
        # cache-friendly on the Orin; the arithmetic range [l_min, l_max] is tiny.
        self._grid = np.zeros((self.n_sectors, self.n_rbins), dtype=np.float32)

        # Per-sector latched binary state for hysteresis, and a "has this sector
        # ever received a measurement" flag so a never-observed sector reads
        # UNKNOWN rather than FREE.
        self._blocked = np.zeros(self.n_sectors, dtype=bool)
        self._observed = np.zeros(self.n_sectors, dtype=bool)

        # Precomputed range-bin geometry (built once, reused every frame).
        self._j_idx = np.arange(self.n_rbins, dtype=np.int64)          # (n_rbins,)
        self.range_centers = (self._j_idx.astype(np.float64) + 0.5) * self.r_bin

    # ------------------------------------------------------------------ helpers
    def sector_center_deg(self):
        """Center angle (deg) of every sector; same sign convention as the stack."""
        i = np.arange(self.n_sectors, dtype=np.float64)
        return self.start_deg + (i + 0.5) * self.bin_deg

    def sector_index(self, ang_deg):
        """Sector index for one angle (deg), atan2(y,x): 0=ahead, +=left."""
        return int(math.floor((ang_deg - self.start_deg) / self.bin_deg)) % self.n_sectors

    # ------------------------------------------------------------------- update
    def update(self, sector_range, dt, covered_mask=None):
        """Fold one frame of per-sector nearest-obstacle ranges into the grid.

        Parameters
        ----------
        sector_range : (n_sectors,) array-like of float
            Nearest measured obstacle range (m) per sector this frame. Use
            ``np.nan`` for "no return in this sector" (a missed / empty beam).
        dt : float
            Seconds since the previous update (drives the leaky decay). Values
            <= 0 skip the decay step.
        covered_mask : (n_sectors,) array-like of bool, optional
            OPTIONAL. If given, a sector that is True here but has a NaN return is
            treated as "scanned and clear" and freed out to r_max (adds l_free to
            every bin). Default None is CONSERVATIVE: a NaN sector is left untouched
            (unknown persists), because a missing return is not proof of free space.
        """
        R = np.asarray(sector_range, dtype=np.float64).reshape(-1)
        if R.shape[0] != self.n_sectors:
            raise ValueError(
                "sector_range must have length n_sectors=%d, got %d"
                % (self.n_sectors, R.shape[0])
            )

        grid = self._grid

        # (1) Leaky exponential decay of ALL cells toward the prior (l -> 0).
        #     A truly-unobserved cell (l == 0) is unaffected; stale free/occupied
        #     evidence fades so the grid self-clears (STVL-style time decay).
        if dt is not None and dt > 0.0:
            grid *= math.exp(-float(dt) / self.tau_s)

        finite = np.isfinite(R)                                    # (n_sectors,)

        # (2) Inverse sensor model for sectors WITH a return. In a polar grid the
        #     beam is a single row, so ray-casting is pure index arithmetic:
        #       bins index <  return_bin -> FREE     (+= l_free)
        #       bin  index == return_bin -> OCCUPIED (+= l_occ)
        #       bins index >  return_bin -> unchanged (unknown beyond the hit)
        #     Fully vectorized over sectors via a broadcast index compare.
        R_safe = np.where(finite, R, 0.0)
        occ_bin = np.floor(R_safe / self.r_bin).astype(np.int64)
        occ_bin = np.where(finite, occ_bin, -1)                    # -1 -> matches nothing

        j = self._j_idx[np.newaxis, :]                             # (1, n_rbins)
        ob = occ_bin[:, np.newaxis]                                # (n_sectors, 1)
        fin_col = finite[:, np.newaxis]

        free_mask = fin_col & (j < ob)                             # nearer than the hit
        occ_mask = fin_col & (j == ob)                             # the hit bin itself

        # Boolean-mask adds: each masked position is unique, so this is a safe,
        # fully vectorized in-place accumulate (no np.add.at scatter needed).
        grid[free_mask] += np.float32(self.l_free)
        grid[occ_mask] += np.float32(self.l_occ)

        # (3) Optional: sectors flagged scanned-but-clear (covered & NaN) are freed
        #     out to r_max. Default (covered_mask is None) leaves NaN sectors alone.
        cov = None
        if covered_mask is not None:
            cov = np.asarray(covered_mask, dtype=bool).reshape(-1)
            if cov.shape[0] != self.n_sectors:
                raise ValueError("covered_mask must have length n_sectors")
            cov = cov & (~finite)
            if cov.any():
                grid[cov, :] += np.float32(self.l_free)

        # Clamp once to keep the estimate responsive (OctoMap clamping update).
        np.clip(grid, self.l_min, self.l_max, out=grid)

        # Mark sectors that received any measurement this frame.
        self._observed |= finite
        if cov is not None:
            self._observed |= cov

        # Advance the per-sector hysteresis latch on the (post-update) strongest
        # cell: BLOCK when the peak crosses l_high, CLEAR only when it drops below
        # l_low OR returns to the prior band (|l| <= eps); otherwise HOLD. The
        # prior-band clause lets a decaying occupied sector release its latch even
        # when l_low == 0 (positive decay approaches 0 without ever going negative).
        max_l = grid.max(axis=1)
        newly_blocked = max_l > self.l_high
        newly_clear = (max_l < self.l_low) | (np.abs(max_l) <= self._eps)
        self._blocked = np.where(
            newly_blocked, True, np.where(newly_clear, False, self._blocked)
        ).astype(bool)

    # --------------------------------------------------------- derived queries
    def nearest_occupied(self):
        """Per-sector nearest occupied distance and 3-state code.

        Returns
        -------
        dist : (n_sectors,) float64
            Center range (m) of the NEAREST bin whose log-odds l >= l_high, or
            ``np.nan`` if no bin in the sector reaches l_high.
        state : (n_sectors,) int8
            OCCUPIED (2) if the sector's hysteresis latch is blocked;
            FREE (1) if the sector has been observed and its peak l is below l_low
            (and clear of the unknown band);
            UNKNOWN (0) otherwise (never observed, or peak l back near the prior).
        """
        grid = self._grid

        # Nearest bin at/above l_high. argmax on a bool row returns the first True;
        # guard rows with no True via an explicit any() mask.
        occ_bins = grid >= self.l_high
        has_occ = occ_bins.any(axis=1)
        first_occ = np.argmax(occ_bins, axis=1)                    # 0 where all-False
        dist = np.where(
            has_occ, (first_occ.astype(np.float64) + 0.5) * self.r_bin, np.nan
        )

        max_l = grid.max(axis=1)
        unknown_band = np.abs(max_l) <= self._eps

        state = np.zeros(self.n_sectors, dtype=np.int8)
        # FREE: observed, peak clearly below l_low, not in the prior band, not blocked.
        free_sel = (
            self._observed
            & (max_l < self.l_low)
            & (~unknown_band)
            & (~self._blocked)
        )
        state[free_sel] = FREE
        # OCCUPIED overrides (hysteresis latch wins).
        state[self._blocked] = OCCUPIED
        return dist, state

    def prob_matrix(self):
        """Full occupancy probability field, p = 1 - 1/(1+exp(l)), shape grid."""
        with np.errstate(over="ignore"):
            return 1.0 - 1.0 / (1.0 + np.exp(self._grid.astype(np.float64)))

    def prob_per_sector(self):
        """Max occupancy probability over range bins, per sector (for coloring)."""
        return self.prob_matrix().max(axis=1)

    # log-odds accessors (handy for a ROS node / debugging).
    @property
    def logodds(self):
        return self._grid

    @property
    def blocked(self):
        return self._blocked

    def reset(self):
        """Clear all evidence back to the uninformative prior."""
        self._grid.fill(0.0)
        self._blocked.fill(False)
        self._observed.fill(False)
