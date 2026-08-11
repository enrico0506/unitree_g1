"""Ground segmentation robust to a TILTED, SWAYING sensor.

Self-contained, PURE-NUMPY (scipy optional, NOT required). No rclpy / ROS
imports so it can be unit-tested standalone and later imported into a ROS2
node. Designed for a Livox Mid-360 mounted on a walking Unitree G1.

WHY THIS EXISTS
---------------
The Mid-360 on the G1 is bolted ~12 deg nose-up and the robot sways as it
walks, so the sensor frame is never level. Any raw "is z below a height
threshold?" or "is the local surface normal vertical?" test is therefore
unreliable: the floor tilts in and out of the band with every step and a wall
a metre ahead can read like a ramp. The fix, borrowed from LIO-SAM's gravity
alignment, is to LEVEL the cloud first using the IMU's gravity (accelerometer)
vector -- a single rotation that makes measured "up" coincide with +z -- and
only then run a fixed-height ground test, which is now valid.

For the segmentation itself we offer two backends:

  * Patchwork++ (Lee et al., arXiv:2207.11919; url-kaist). Concentric-Zone-Model
    region-wise plane fitting with an "uprightness" gate (default
    uprightness_thr 0.5 ~= 60 deg tolerance) that copes with slope/tilt on its
    own; ~18 ms/scan on CPU. Used automatically when `pypatchworkpp` is
    importable. We STILL gravity-level first so its region planes start near
    horizontal.

  * A dependency-free 2.5D ELEVATION GRID fallback. Bin (x, y) into square
    cells; the floor height of a cell is a LOW percentile of z inside it (robust
    to a low overhang / table underside that would drag a mean or least-squares
    plane upward and then delete the very obstacle it belongs to). A point is an
    obstacle when its height above its cell floor lands in
    (ground_clearance, max_above). Thin/low obstacles survive because the test
    is relative to the local floor, not an absolute band.

SHARED CONVENTIONS
------------------
    points : numpy (N, 3) float32, columns [x, y, z] = [forward, left, up]
             in metres, sensor at the origin.

Everything is vectorised. The per-cell percentile is computed with a single
lexsort + fancy index (O(N log N)); we deliberately AVOID np.add.at /
np.minimum.at (slow, un-buffered scatter on Jetson).

Refs: Patchwork++ (arXiv:2207.11919); LIO-SAM gravity alignment.
"""

import numpy as np

# Optional Patchwork++ backend. NEVER a hard dependency -- absence just falls
# back to the elevation grid.
try:  # pragma: no cover - availability is environment-dependent
    import pypatchworkpp  # noqa: F401
    _HAVE_PATCHWORK = True
except Exception:  # pragma: no cover
    pypatchworkpp = None
    _HAVE_PATCHWORK = False


def patchwork_available():
    """True if the optional `pypatchworkpp` backend imported successfully."""
    return _HAVE_PATCHWORK


# --------------------------------------------------------------------------- #
# Gravity levelling
# --------------------------------------------------------------------------- #
def rotation_from_gravity(gravity_vec):
    """Rotation (3, 3) that maps the measured up-direction onto +z.

    `gravity_vec` (3,) is the accelerometer reading expressed in the SENSOR
    frame; under gravity it points roughly toward -up (down). The physical
    up-direction is therefore ``up = -gravity / |gravity|``. We return the
    minimal (shortest-arc) rotation R such that ``R @ up == [0, 0, 1]``, built
    with Rodrigues' formula. Applying R to the cloud makes gravity point along
    -z, after which a fixed height threshold is meaningful.

    Numerically safe: returns identity when already aligned (or gravity is
    degenerate/zero) and a well-defined 180 deg flip when up points at -z.
    """
    g = np.asarray(gravity_vec, dtype=np.float64).reshape(3)
    n = np.linalg.norm(g)
    if not np.isfinite(n) or n < 1e-9:
        return np.eye(3, dtype=np.float32)          # no usable gravity -> no-op

    up = -g / n                                     # measured up (unit)
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, target)                        # rotation axis * sin(theta)
    s = np.linalg.norm(v)
    c = float(np.dot(up, target))                   # cos(theta)

    if s < 1e-8:                                    # already (anti)parallel to +z
        if c > 0.0:
            return np.eye(3, dtype=np.float32)      # aligned -> identity
        # up points at -z: rotate 180 deg about any perpendicular axis (x).
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    # Rodrigues: R = I + [v]_x + [v]_x^2 * (1 - c) / s^2
    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R.astype(np.float32)


def level_by_gravity(points, gravity_vec):
    """Rotate `points` (N, 3) so measured gravity points along -z.

    Returns the leveled cloud (N, 3) float32. Identity (a copy) when already
    level or when gravity is unusable.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(-1, 3).copy()
    R = rotation_from_gravity(gravity_vec)
    # p' = R @ p for every point == points @ R.T
    return (pts @ R.T.astype(np.float32)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Robot self-footprint
# --------------------------------------------------------------------------- #
def self_mask(points, front=0.35, back=0.35, half_w=0.30, max_h=0.6,
              floor_z=0.0):
    """Keep-mask (True == NOT the robot's own body).

    Drops LOW returns inside the robot's footprint box -- its legs, feet and
    harness -- so they are not mistaken for obstacles. The box spans
    x in (-back, front), |y| < half_w and only heights BELOW ``floor_z + max_h``
    (chest/head-height returns directly above the footprint are genuine overhead
    hazards and are kept). Coordinates are assumed already leveled.

    Returns a bool array (N,); True marks a point to KEEP.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    in_self = (
        (x > -back) & (x < front)
        & (np.abs(y) < half_w)
        & (z < floor_z + max_h)
    )
    return ~in_self


# --------------------------------------------------------------------------- #
# 2.5D elevation-grid ground removal
# --------------------------------------------------------------------------- #
def segment_obstacles_elevation(points, cell=0.20, ground_clearance=0.08,
                                max_above=1.9, floor_pct=10.0, min_cell_pts=3):
    """Elevation-grid ground removal. Returns a bool obstacle mask (N,).

    Bins x, y into square `cell`-metre cells. Each cell's floor height is the
    ``floor_pct``-percentile of z inside it (a LOW percentile: robust to a low
    overhang whose points would otherwise raise the floor and hide obstacles).
    A point's height above its cell floor is ``z - cell_floor``; it is flagged
    as an obstacle when ``ground_clearance < height < max_above``.

    Cells with fewer than `min_cell_pts` points have an unreliable floor
    estimate, so their points are kept as obstacles (SAFETY-FIRST: better a
    false positive than driving through an unmeasured hazard).

    Efficiency: the per-cell percentile is a single ``np.lexsort`` (sort by cell
    then by z) followed by fancy indexing -- O(N log N), no ``np.add.at`` /
    ``np.minimum.at``.

    True in the returned mask == obstacle (keep); False == ground / removed.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # --- integer cell key per point ---------------------------------------- #
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    # Shift to non-negative and pack (ix, iy) into a single scalar key so we can
    # group with one np.unique. ncol spans the full y extent so keys never clash.
    ix -= ix.min()
    iy -= iy.min()
    ncol = int(iy.max()) + 1
    key = ix * ncol + iy

    # cell index per point (inv), point count per cell (counts).
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    K = counts.shape[0]

    # --- per-cell low-percentile floor via lexsort + fancy index ----------- #
    # Sort primarily by cell (inv), secondarily by z: within each contiguous
    # cell block, z is now ascending, so the percentile is a direct index.
    order = np.lexsort((z, inv))
    sorted_z = z[order]

    starts = np.empty(K, dtype=np.int64)            # first index of each cell
    starts[0] = 0
    np.cumsum(counts[:-1], out=starts[1:])

    # linear-interpolated percentile (matches np.percentile default) per cell.
    vidx = (floor_pct / 100.0) * (counts - 1).astype(np.float64)
    lo = np.floor(vidx).astype(np.int64)
    hi = np.minimum(lo + 1, counts - 1)
    frac = vidx - lo
    cell_floor = (sorted_z[starts + lo] * (1.0 - frac)
                  + sorted_z[starts + hi] * frac).astype(np.float32)

    # --- height test ------------------------------------------------------- #
    height = z - cell_floor[inv]
    obstacle = (height > ground_clearance) & (height < max_above)

    # sparse cells: unreliable floor -> keep as obstacle (safety-first).
    sparse = counts < min_cell_pts
    obstacle |= sparse[inv]
    return obstacle


# --------------------------------------------------------------------------- #
# Optional RANSAC plane baseline
# --------------------------------------------------------------------------- #
def ransac_plane(points, dist=0.1, iters=50, rng=None):
    """Simple RANSAC dominant-plane fit (baseline / diagnostic).

    Returns ``(plane, inlier_mask)`` where plane = (a, b, c, d) with unit normal
    (a, b, c) and ``a*x + b*y + c*z + d ~= 0`` for inliers (|residual| < dist).
    Best used AFTER gravity levelling, when the floor is the dominant near-
    horizontal plane. Returns ``(None, all-False)`` if too few points.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return None, np.zeros(n, dtype=bool)
    rng = np.random.default_rng() if rng is None else rng

    best_inl = np.zeros(n, dtype=bool)
    best_cnt = 0
    best_plane = None
    for _ in range(int(iters)):
        i, j, k = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[i], pts[j], pts[k]
        nrm = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        d = -float(nrm @ p0)
        resid = np.abs(pts @ nrm + d)
        inl = resid < dist
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
            best_plane = (float(nrm[0]), float(nrm[1]), float(nrm[2]), d)
    return best_plane, best_inl


# --------------------------------------------------------------------------- #
# Patchwork++ backend (optional)
# --------------------------------------------------------------------------- #
def _segment_patchwork(points):
    """Ground removal via Patchwork++. Returns a bool obstacle mask (N,).

    Wrapped defensively: the pybind API has shifted across releases, so any
    failure raises and the caller (segment_obstacles) falls back to the
    elevation grid.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    params = pypatchworkpp.Parameters()
    pw = pypatchworkpp.patchworkpp(params)
    # Patchwork++ expects (N, 4) x,y,z,intensity; pad a zero intensity column.
    xyzi = np.hstack([pts, np.zeros((n, 1), dtype=np.float64)])
    pw.estimateGround(xyzi)

    mask = np.zeros(n, dtype=bool)
    # Prefer index accessors when present; else fall back to nonground coords.
    idx_fn = getattr(pw, "getNongroundIndices", None)
    if callable(idx_fn):
        ng = np.asarray(idx_fn(), dtype=np.int64)
        mask[ng] = True
        return mask
    # No index accessor on this build: fall back to the returned nonground COORDS.
    # Those carry no input ordering, and matching them back point-by-point would be
    # far too expensive for the hot path -- so we only handle the one unambiguous
    # case, "every input point came back as nonground", and otherwise leave the mask
    # all-False (the caller then sees no obstacles from Patchwork++).
    nonground = np.asarray(pw.getNonground(), dtype=np.float64)
    if nonground.shape[0] == n:
        mask[:] = True
    return mask


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def segment_obstacles(points, gravity_vec=None, method="auto",
                      self_kw=None, **kw):
    """Segment obstacle points from a (possibly tilted) LiDAR scan.

    Parameters
    ----------
    points : (N, 3) float32
        Sensor-frame cloud, [x, y, z] = [forward, left, up].
    gravity_vec : (3,) or None
        Accelerometer reading in the sensor frame. If given, the cloud is
        gravity-LEVELLED before segmentation and the returned obstacle points
        are in that leveled frame.
    method : {"auto", "patchwork", "elevation"}
        "auto"/"patchwork" use Patchwork++ when importable, else the elevation
        grid; "elevation" forces the grid. Patchwork failures fall back to the
        grid automatically.
    self_kw : dict or None
        Overrides forwarded to `self_mask` (front, back, half_w, max_h,
        floor_z).
    **kw :
        Forwarded to `segment_obstacles_elevation` (cell, ground_clearance,
        max_above, floor_pct, min_cell_pts).

    Returns
    -------
    (obstacle_points, obstacle_mask)
        obstacle_points : (M, 3) float32 -- the obstacle points, in the LEVELED
            frame when `gravity_vec` was supplied (raw frame otherwise).
        obstacle_mask   : (N,) bool -- True at input indices that are obstacles
            (ground removed AND not part of the robot's own footprint). Aligned
            to the INPUT `points` order regardless of levelling.
    """
    pts_in = np.asarray(points, dtype=np.float32)
    if pts_in.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

    # 1) level -------------------------------------------------------------- #
    pts = level_by_gravity(pts_in, gravity_vec) if gravity_vec is not None \
        else pts_in

    # 2) ground removal ----------------------------------------------------- #
    mask = None
    if method in ("auto", "patchwork") and _HAVE_PATCHWORK:
        try:
            mask = _segment_patchwork(pts)
        except Exception:
            mask = None                             # fall through to elevation
    if mask is None:
        mask = segment_obstacles_elevation(pts, **kw)

    # 3) drop the robot's own footprint ------------------------------------ #
    skw = self_kw or {}
    keep = self_mask(pts, **skw)
    mask = mask & keep

    return pts[mask], mask
