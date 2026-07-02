"""filters.py -- self-contained numpy point-cloud filters for the g1 obstacle pipeline.

Three building blocks that stop the current pipeline from silently dropping thin /
sparse hazards (taut cables, thin table / chair legs, railings) while still rejecting
sensor noise:

  1. custommsg_to_numpy() -- fast Livox CustomMsg -> numpy ingest (frombuffer fast path
     + a vectorised fallback for the real ROS2 message object).
  2. dror_filter()        -- Dynamic Radius Outlier Removal: a range-scaled neighbour
     radius so genuinely-sparse *far* returns are not culled as if they were noise,
     but true fliers still are.
  3. sector_kth_nearest() -- per-sector k-th-nearest horizontal distance with a
     range-scaled minimum point count, plus a near-field tripwire override.

PURE NUMPY + scipy.spatial.cKDTree ONLY -- no rclpy / ROS imports, so this file is
unit-testable standalone and can be imported into a ROS2 node later.

--------------------------------------------------------------------------------------
SHARED CONVENTIONS (every function in this module obeys these):

  * Point clouds are numpy (N,3) float32, columns [x, y, z] = [forward, left, up]
    in metres, with the sensor at the origin.
  * Angles: ang_deg = degrees(atan2(y, x)).  0 = straight ahead, + = LEFT (CCW),
    range [-180, 180).  horiz = hypot(x, y) is the horizontal (ground-plane) range.
  * Target platform: Jetson Orin, ~10-20k points/frame at 10-20 Hz.  Everything is
    vectorised; per-sector reductions use np.bincount / lexsort + indexing (NOT
    np.add.at / np.minimum.at, which are slow scatter ops on this numpy).

--------------------------------------------------------------------------------------
BACKGROUND -- Livox CustomMsg (livox_ros_driver2/msg/CustomMsg):

    std_msgs/Header header
    uint64            timebase     # ns, time of the first point
    uint32            point_num    # number of points
    uint8             lidar_id
    uint8[3]          rsvd
    CustomPoint[]     points

  livox_ros_driver2/msg/CustomPoint (VERIFIED against the .msg on this machine --
  note offset_time is the FIRST field, not x/y/z):

    uint32  offset_time    # ns from timebase
    float32 x
    float32 y
    float32 z
    uint8   reflectivity
    uint8   tag
    uint8   line

  Iterating msg.points in Python (`[(p.x, p.y, p.z) for p in msg.points]`) is the slow
  path -- it is a per-point attribute fetch through the rosidl Python bindings.  The
  fast path reinterprets a contiguous record buffer with np.frombuffer(...,
  CUSTOMPOINT_DTYPE).  The real rosidl_generator_py CustomMsg does NOT expose such a
  buffer (`.points` is a list of Python objects), so the fast path only fires when the
  message (or a test fake, or a future zero-copy driver) exposes a raw buffer; the
  real message falls back to a single vectorised list read.  See custommsg_to_numpy().

References:
  * DROR: Charron et al., "De-noising of Lidar Point Clouds Corrupted by Snowfall"
    (IEEE 9860643) -- dynamic search radius r_d = beta * r * alpha.
  * DSOR: Kurup & Bos, arXiv:2109.07078 -- range-scaled statistical outlier removal.
  * pointcloud_to_laserscan -- per-angular-bin nearest-range reduction.
"""

import operator

import numpy as np

try:
    import scipy
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
    _SCIPY_VERSION = scipy.__version__
except Exception:                                   # pragma: no cover
    cKDTree = None
    _HAVE_SCIPY = False
    _SCIPY_VERSION = None

__all__ = [
    "CUSTOMPOINT_DTYPE",
    "custommsg_to_numpy",
    "dror_filter",
    "sector_kth_nearest",
    "sector_tripwire",
    "merge_min",
    "required_points",
    "SCIPY_CAPS",
]

# =====================================================================================
# CustomPoint dtype
# =====================================================================================
#
# Field order and types are taken verbatim from livox_ros_driver2/msg/CustomPoint.msg.
# The corresponding C struct that rosidl generates uses *natural* alignment: the 4-byte
# fields (offset_time, x, y, z) sit at offsets 0/4/8/12, the three uint8s at 16/17/18,
# and the struct is padded up to a multiple of its 4-byte alignment -> itemsize 20 with
# ONE trailing pad byte after `line`.  numpy reproduces exactly this with align=True.
#
# We spell the offsets/itemsize out explicitly (rather than relying on align=True) so
# the padding assumption is documented and greppable.
#
# ---- HOW TO VALIDATE against a real message -----------------------------------------
#   1. Capture one CustomMsg and grab a raw record buffer for its points (from a driver
#      that exposes one, or serialise a few points yourself).
#   2. Assert  len(buffer) == msg.point_num * CUSTOMPOINT_DTYPE.itemsize   (i.e. 20-byte
#      stride).  If it is 19 * point_num instead, the build is TIGHTLY PACKED -- rebuild
#      the dtype with itemsize=19 (or align=False).
#   3. rec = np.frombuffer(buffer, CUSTOMPOINT_DTYPE); compare rec['x'][:5],
#      rec['offset_time'][:5], ... against [ (p.x, p.offset_time, ...) for p in
#      msg.points[:5] ].  They must match exactly; a mismatch means the stride/field
#      order differs on your platform and the offsets below must be adjusted.
#
CUSTOMPOINT_DTYPE = np.dtype(
    {
        "names":    ["offset_time", "x", "y", "z", "reflectivity", "tag", "line"],
        "formats":  ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
        "offsets":  [0, 4, 8, 12, 16, 17, 18],
        "itemsize": 20,   # 19 data bytes + 1 trailing pad byte (4-byte struct alignment)
    }
)
assert CUSTOMPOINT_DTYPE.itemsize == 20, "unexpected CustomPoint stride"


# =====================================================================================
# scipy capability probing (so the module is portable across scipy versions)
# =====================================================================================
def _detect_ball_array_radii():
    """True if cKDTree.query_ball_point accepts a per-point radius array + return_length.

    That signature landed in scipy 1.6.0.  This box's scipy 1.3.3 is patched and also
    supports it, but we probe rather than assume so the module runs on stock 1.3.x too.
    """
    if not _HAVE_SCIPY:
        return False
    try:
        t = cKDTree(np.zeros((2, 3), dtype=np.float64))
        out = t.query_ball_point(np.zeros((2, 3)), np.array([0.5, 0.5]),
                                 return_length=True)
        return np.ndim(out) == 1
    except Exception:
        return False


def _detect_query_jobs():
    """Return the kwarg dict for multi-threaded cKDTree.query ({} if unsupported).

    scipy renamed n_jobs -> workers in 1.6.0.  Try the modern name first.
    """
    if not _HAVE_SCIPY:
        return {}
    for kw in ("workers", "n_jobs"):
        try:
            t = cKDTree(np.zeros((2, 3), dtype=np.float64))
            t.query(np.zeros((2, 3)), k=1, **{kw: -1})
            return {kw: -1}
        except TypeError:
            continue
        except Exception:
            continue
    return {}


_BALL_ARRAY_RADII = _detect_ball_array_radii()
_QUERY_JOBS = _detect_query_jobs()

SCIPY_CAPS = {
    "scipy_version": _SCIPY_VERSION,
    "query_ball_point_array_radii": _BALL_ARRAY_RADII,
    "query_jobs_kwarg": next(iter(_QUERY_JOBS), None),
}


# =====================================================================================
# (a) Fast Livox CustomMsg -> numpy ingest
# =====================================================================================
def _point_records(msg):
    """Duck-typed lookup of a contiguous CustomPoint record buffer, else None.

    Returns a structured ndarray of dtype CUSTOMPOINT_DTYPE (the frombuffer fast path)
    or None if the message only offers a Python list of point objects (the fallback).
    """
    pts = getattr(msg, "points", None)

    # (i) points is already a structured array of our dtype -> zero reinterpretation.
    if isinstance(pts, np.ndarray):
        if pts.dtype == CUSTOMPOINT_DTYPE:
            return pts
        # a raw uint8/byte ndarray of the record bytes -> reinterpret.
        return np.frombuffer(np.ascontiguousarray(pts).tobytes(), dtype=CUSTOMPOINT_DTYPE)

    # (ii) points itself is a bytes-like buffer.
    if isinstance(pts, (bytes, bytearray, memoryview)):
        return np.frombuffer(pts, dtype=CUSTOMPOINT_DTYPE)

    # (iii) an explicit side buffer attribute a fast driver / fake might expose.
    for name in ("point_data", "points_buffer"):
        buf = getattr(msg, name, None)
        if isinstance(buf, (bytes, bytearray, memoryview)):
            return np.frombuffer(buf, dtype=CUSTOMPOINT_DTYPE)
        if isinstance(buf, np.ndarray):
            if buf.dtype == CUSTOMPOINT_DTYPE:
                return buf
            return np.frombuffer(np.ascontiguousarray(buf).tobytes(),
                                 dtype=CUSTOMPOINT_DTYPE)
    return None


def custommsg_to_numpy(msg, decimate=1):
    """Decode a Livox CustomMsg (or a duck-typed fake) into numpy arrays.

    Parameters
    ----------
    msg : object
        Anything exposing ``.timebase`` (uint64 ns) and either
          * ``.points`` as a Python list of objects with .x/.y/.z/.offset_time/
            .reflectivity attributes (the real rosidl message -> vectorised fallback), OR
          * a contiguous record buffer (``.points`` as bytes/ndarray, or a
            ``.point_data`` / ``.points_buffer`` bytes attribute -> frombuffer fast path).
    decimate : int
        Keep every ``decimate``-th point (>=1).  Applied by slicing.

    Returns
    -------
    points : (N,3) float32   -- [x, y, z] = [forward, left, up] metres.
    times  : (N,) float64    -- absolute seconds = (timebase + offset_time) * 1e-9.
    refl   : (N,) uint8       -- reflectivity 0..255.

    Notes
    -----
    * Fast path: np.frombuffer over a contiguous CUSTOMPOINT_DTYPE buffer -- zero
      per-point Python work.  This is what the real driver *should* expose for speed.
    * Fallback: the real rosidl CustomMsg has NO contiguous buffer, so at least one
      Python pass over the point objects is unavoidable.  We do a SINGLE pass with
      operator.attrgetter pulling all needed fields into one list of tuples, then one
      np.array call -- far cheaper than the node's separate per-field list-comps, and
      much cheaper than b"".join(struct.pack ...) per point (which would still be a
      per-point Python loop AND an extra serialise, i.e. strictly worse -- so we do not
      do that).
    * times precision: absolute Livox timestamps are ~1.7e18 ns, which exceed float64's
      53-bit exact-integer range; storing them as seconds keeps ~2e-7 s (~240 ns)
      resolution.  That is ample for ordering points inside one ~0.1 s frame.  For
      full-ns work, subtract ``msg.timebase`` and keep offset_time separately.
    """
    decimate = max(1, int(decimate))
    timebase = int(getattr(msg, "timebase", 0) or 0)

    rec = _point_records(msg)
    if rec is not None:
        # ---- fast path: reinterpret the raw record buffer --------------------------
        if decimate > 1:
            rec = rec[::decimate]
        n = rec.shape[0]
        points = np.empty((n, 3), dtype=np.float32)
        points[:, 0] = rec["x"]
        points[:, 1] = rec["y"]
        points[:, 2] = rec["z"]
        offset = rec["offset_time"].astype(np.float64)
        refl = np.ascontiguousarray(rec["reflectivity"], dtype=np.uint8)
    else:
        # ---- fallback: single vectorised pass over the point-object list -----------
        pts_list = getattr(msg, "points", None)
        if pts_list is None:
            pts_list = []
        if decimate > 1:
            pts_list = pts_list[::decimate]
        n = len(pts_list)
        if n == 0:
            return (np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float64),
                    np.zeros((0,), np.uint8))
        getter = operator.attrgetter("x", "y", "z", "offset_time", "reflectivity")
        rows = [getter(p) for p in pts_list]          # ONE Python pass, all fields
        arr = np.array(rows, dtype=np.float64)         # (n, 5)
        points = np.ascontiguousarray(arr[:, :3], dtype=np.float32)
        offset = arr[:, 3]
        refl = arr[:, 4].astype(np.uint8)

    # absolute seconds; scale the two ns terms independently (see precision note).
    times = timebase * 1e-9 + offset * 1e-9
    return points, times, refl


# =====================================================================================
# (b) Range-scaled outlier rejection -- DROR
# =====================================================================================
def dror_filter(points, alpha_rad=0.004, beta=3.0, r_min=0.08,
                min_neighbors=3, k_for_scale=None):
    """Dynamic Radius Outlier Removal.

    A global SOR/ROR with a fixed radius flags legitimately-sparse far / thin returns
    as noise (point density falls ~1/r^2, so a far cable has few close neighbours).
    DROR instead grows the search radius with range:

        radius_i = max(r_min, beta * r_i * alpha_rad)      # r_i = horizontal range

    where ``alpha_rad`` is the sensor's horizontal angular resolution in radians, so the
    radius tracks the expected neighbour spacing at that range.  A point is KEPT when it
    has at least ``min_neighbors`` points (including itself) inside its own radius.

    Parameters
    ----------
    points : (N,3) float array   -- [x, y, z] metres.
    alpha_rad : float            -- horizontal angular resolution (rad).  Livox Mid-360
                                    is ~0.2-0.4 deg -> ~0.004-0.007 rad.
    beta : float                 -- radius multiplier (paper default ~3).
    r_min : float                -- floor on the radius (m) so near points are not given
                                    an absurdly small radius.
    min_neighbors : int          -- neighbours (incl. self) required within radius_i.
    k_for_scale : int or None    -- optional override for the neighbour-count threshold
                                    used by the density test.  When None the threshold is
                                    ``min_neighbors``.  Provided so a caller can drive
                                    both this and sector_kth_nearest's k from one config
                                    value; set it to reuse the same k=3 floor that DROR
                                    / DBSCAN-3D / the sector logic all default to.

    Returns
    -------
    keep : (N,) bool  -- True for points to keep.

    Implementation / scipy notes
    ----------------------------
    A single cKDTree is built once.  Two equivalent code paths (chosen by probing scipy
    at import, see SCIPY_CAPS):
      * scipy >= 1.6 (array-radii query_ball_point + return_length): count neighbours
        within each point's own radius in one call -> keep = count >= threshold.
      * older scipy: use the k-nearest trick -- query k=threshold neighbours (column 0
        is the point itself at distance 0), and keep iff the (threshold-1)-th nearest
        OTHER point lies within radius_i, i.e. dist[:, threshold-1] <= radius_i.  This
        is EXACTLY equivalent to "count within radius >= threshold" and needs no
        per-point Python loop and no array-radii support.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    thr = int(min_neighbors if k_for_scale is None else max(1, int(k_for_scale)))
    if thr <= 1:
        # every point is its own neighbour -> trivially satisfied.
        return np.ones(n, dtype=bool)

    horiz = np.hypot(points[:, 0], points[:, 1])
    radii = np.maximum(r_min, beta * horiz * alpha_rad)

    if not _HAVE_SCIPY:
        raise RuntimeError("dror_filter requires scipy.spatial.cKDTree")

    tree = cKDTree(points)

    if _BALL_ARRAY_RADII:
        # count includes the point itself (distance 0 always inside radius).
        counts = tree.query_ball_point(points, radii, return_length=True)
        return np.asarray(counts) >= thr

    # ---- portable fallback: k-nearest distance vs per-point radius -----------------
    if n < thr:
        # fewer than `thr` points total -> nobody can reach the threshold.
        return np.zeros(n, dtype=bool)
    dist, _ = tree.query(points, k=thr, **_QUERY_JOBS)
    # dist[:, 0] == 0 (self); dist[:, thr-1] is the (thr-1)-th nearest OTHER point.
    kth = dist[:, thr - 1]
    return kth <= radii


# =====================================================================================
# range-scaled minimum point count (shared by the sector logic)
# =====================================================================================
def required_points(r, k0=3, r0=1.0, r_max=None):
    """Range-scaled minimum point count for a sector to be trusted.

    Point density falls ~1/r^2, so we DEMAND more points up close (where a real
    surface must be dense) and fewer far away (where a genuine object is legitimately
    sparse and we must not drop it):

        required(r) = max(1, round(k0 * (r0 / r)^2))

    ``k0`` is the count expected at the reference range ``r0``.  With k0=3, r0=1 m:
    r=0.5 -> 12, r=1.0 -> 3, r=1.2 -> 2, r>=~1.3 -> 1.  ``r_max`` (if given) caps r in
    the formula, so beyond r_max the requirement stays at its (smallest, floored)
    r_max value rather than being extrapolated -- r_max is the range past which the
    density model is not trusted (typically the obstacle range of interest).

    Accepts a scalar or an array; returns int64 of the same shape.
    """
    r = np.asarray(r, dtype=np.float64)
    if r_max is not None:
        r = np.minimum(r, float(r_max))
    r = np.maximum(r, 1e-6)                     # guard divide-by-zero
    req = np.rint(k0 * (r0 / r) ** 2)
    return np.maximum(1, req).astype(np.int64)


# =====================================================================================
# (c) Range-scaled per-sector k-th-nearest distance
# =====================================================================================
def sector_kth_nearest(ang_deg, horiz, n_sectors=180, start_deg=-180.0, k=3,
                       range_scaled_mincount=True, k0=3, r0=1.0, r_max=3.0):
    """Per-sector k-th-nearest horizontal distance over a full 360 deg ring.

    Each point is binned into one of ``n_sectors`` equal sectors spanning 360 deg from
    ``start_deg`` (bin width = 360 / n_sectors).  For each sector the reported distance
    is the k-th SMALLEST horizontal range (k clamped down to the sector's point count).
    Reporting the k-th nearest instead of the strict minimum makes a sector robust to a
    single near flier (rank-1 = one flier reads as a phantom obstacle); k=3 matches the
    DROR / DBSCAN-3D minimum-cluster floor.

    When ``range_scaled_mincount`` is True a sector is reported only if it holds at
    least ``required_points(nearest_r)`` points (see required_points); otherwise the
    sector reads NaN.  This keeps genuinely-sparse thin/far hazards (few points, but
    more than the far requirement) while rejecting lone fliers (1 point where several
    are required).

    Parameters
    ----------
    ang_deg : (N,) float  -- degrees(atan2(y, x)); 0 ahead, + left, range [-180, 180).
    horiz   : (N,) float  -- hypot(x, y) horizontal range (m).
    n_sectors : int       -- number of equal sectors over the full circle.
    start_deg : float     -- angle of sector 0's leading edge (default -180).
    k : int               -- report the k-th smallest range (clamped to sector count).
    range_scaled_mincount : bool -- apply the required_points() gate.
    k0, r0, r_max : passed to required_points().

    Returns
    -------
    dist   : (n_sectors,) float64  -- k-th-nearest range per sector, NaN where the
                                      sector is empty or fails the min-count gate.
    counts : (n_sectors,) int64    -- points per sector.

    Vectorisation
    -------------
    No Python per-sector loop.  Points are sorted by (sector, horiz) via np.lexsort, so
    each sector occupies a contiguous ascending-range block.  np.bincount gives per-
    sector counts; the block start offsets are the exclusive cumulative sum of counts,
    so the sector minimum is block[start] and the k-th element is block[start + k'-1].
    """
    ang_deg = np.asarray(ang_deg, dtype=np.float64).reshape(-1)
    horiz = np.asarray(horiz, dtype=np.float64).reshape(-1)
    n_sectors = int(n_sectors)

    dist = np.full(n_sectors, np.nan, dtype=np.float64)
    counts = np.zeros(n_sectors, dtype=np.int64)
    if ang_deg.size == 0:
        return dist, counts

    bin_deg = 360.0 / n_sectors
    sector = np.floor((ang_deg - start_deg) / bin_deg).astype(np.int64)
    sector %= n_sectors                                  # wrap the +180 edge into bin 0

    counts = np.bincount(sector, minlength=n_sectors).astype(np.int64)
    populated = counts > 0
    if not populated.any():
        return dist, counts

    # sort primarily by sector, secondarily by range -> contiguous ascending blocks.
    order = np.lexsort((horiz, sector))
    h_sorted = horiz[order]

    # exclusive-cumsum block starts (valid even with interspersed empty sectors).
    starts = np.zeros(n_sectors, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    # nearest range per populated sector = first element of its sorted block.
    nearest = np.full(n_sectors, np.nan, dtype=np.float64)
    nearest[populated] = h_sorted[starts[populated]]

    # k-th smallest (k clamped to the sector's count).
    kth_rank = np.minimum(int(k), counts)                # 0 for empty sectors
    kth_idx = starts + (kth_rank - 1)                    # only indexed where populated
    dist[populated] = h_sorted[kth_idx[populated]]

    if range_scaled_mincount:
        req = required_points(nearest, k0=k0, r0=r0, r_max=r_max)   # NaN nearest -> req 1
        under = populated & (counts < req)
        dist[under] = np.nan

    return dist, counts


def sector_tripwire(ang_deg, horiz, n_sectors=180, start_deg=-180.0,
                    tripwire_range=1.6, tripwire_min=2):
    """Near-field tripwire override, one value per sector.

    Any sector holding at least ``tripwire_min`` points within ``tripwire_range`` metres
    reports its NEAREST such point.  This is the same override the current node uses: it
    catches (a) a wide+sparse near obstacle that gives too few points for the k-th-
    nearest min-count gate (taut cable, barrier tape, thin railing), and (b) a near
    hazard sharing a sector with a farther dense surface, which the k-th-nearest would
    otherwise mask.  tripwire_min>=2 keeps a single floor-noise spike from firing it.

    Returns (n_sectors,) float64 -- nearest range where the tripwire fires, else NaN.
    Merge into a base ring with merge_min().
    """
    ang_deg = np.asarray(ang_deg, dtype=np.float64).reshape(-1)
    horiz = np.asarray(horiz, dtype=np.float64).reshape(-1)
    n_sectors = int(n_sectors)

    out = np.full(n_sectors, np.nan, dtype=np.float64)
    if ang_deg.size == 0:
        return out

    near = horiz <= float(tripwire_range)
    if not near.any():
        return out
    a = ang_deg[near]
    h = horiz[near]

    bin_deg = 360.0 / n_sectors
    sector = np.floor((a - start_deg) / bin_deg).astype(np.int64) % n_sectors
    counts = np.bincount(sector, minlength=n_sectors).astype(np.int64)

    order = np.lexsort((h, sector))
    h_sorted = h[order]
    starts = np.zeros(n_sectors, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    fired = counts >= int(tripwire_min)
    out[fired] = h_sorted[starts[fired]]                 # nearest within tripwire_range
    return out


def merge_min(*rings):
    """NaN-aware element-wise minimum of several (n_sectors,) rings.

    NaN means "no reading" and is ignored (np.fmin semantics): fmin(NaN, x)=x,
    fmin(NaN, NaN)=NaN.  Use to fold a tripwire ring onto a base k-th-nearest ring.
    """
    it = iter(rings)
    acc = np.asarray(next(it), dtype=np.float64).copy()
    for r in it:
        acc = np.fmin(acc, np.asarray(r, dtype=np.float64))
    return acc
