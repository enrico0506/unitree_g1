"""IMU-gyro point-cloud de-skew (rotational motion compensation).

Kills the smear you get when accumulating Livox Mid-360 frames on a swaying /
turning Unitree G1. A single "frame" is really a *sweep*: the sensor samples
points over ~50-100 ms, and if the body rotates during that window every point
is measured in a slightly different sensor orientation. Naively stacking them
smears walls into curved sheets and doubles thin obstacles. De-skew rotates
every point into one common reference orientation (usually the scan end).

Why gyro-only, and why this is cheap
------------------------------------
FAST-LIO2 (arXiv:2107.06829) de-skews by IMU backward-propagation to the
scan-end pose, integrating *both* gyro and accelerometer. For short-range
REACTIVE obstacle avoidance we only need the rotational part: at a few metres
range, rotation dominates the smear and the constant-velocity translation of a
walking robot (< ~1 m/s over a ~0.1 s sweep => < ~10 cm of body travel, and far
less apparent motion for anything past ~2 m) is negligible. So we integrate
only the gyro. This is the same idea as the Livox_cloud_undistortion_ROS2
package, minus the accelerometer bookkeeping.

Geometry / conventions (shared across this repo)
------------------------------------------------
- Point clouds are numpy arrays, shape (N, 3), columns [x, y, z] =
  [forward, left, up] in metres, sensor at the origin.
- Gyro samples are angular velocity in rad/s in the *same* sensor frame as the
  cloud (the Livox /livox/imu stream is co-located: 200 Hz, +/-2000 deg/s).
  wx = roll rate about +x (forward), wy = pitch rate about +y (left),
  wz = yaw rate about +z (up), all right-handed.
- The Mid-360 CustomMsg gives a per-point absolute time
  (timebase + offset_time, offset_time being uint32 ns from frame start); pass
  those absolute seconds as `point_times`. `ref_time` is the orientation you
  want everything expressed in (typically the newest point time / scan end).

Math
----
Let C(t) be the sensor orientation at time t relative to an arbitrary integration
origin (the oldest buffered gyro sample). With the gyro measured in the body
frame, C obeys  C_dot = C [w]_x, so C is built by *right-multiplying* small
incremental rotations (FAST-LIO holds the gyro constant between IMU samples):

    C(t_{k+1}) = C(t_k) . Rodrigues( w_k * (t_{k+1} - t_k) )

A point measured at time t_i in sensor coords p_i lands, in the sensor frame at
the reference time t_r, at

    p_r = R_rel(t_i) . p_i ,   R_rel(t_i) = C(t_r)^T . C(t_i)

The arbitrary integration origin cancels in C(t_r)^T C(t_i), so it does not
matter where we start integrating. R_rel is always a *small* rotation (t_i and
t_r are within one sweep), which is what makes this numerically friendly.

Rather than LERP-ing the cumulative rotation vectors (fine for small angles, but
degrades if the buffer spans a lot of turning), we interpolate for the
piecewise-constant gyro model: precompute the exact cumulative rotation C(t_k) at
every IMU knot (float64), and per point find the bracketing IMU sample to the
left of its time and add the fractional increment `w_left * (t - t_left)` on top.
That increment spans at most one IMU period (~5 ms), so it is a tiny angle and is
applied as a first-order rotation `p + dt (w_left x p)` (error << 1 mm), avoiding
building N rotation matrices. The heavy O(N) work runs in float32; the O(K) gyro
integration runs in float64. Everything past the knot loop is vectorised over N.

Pure numpy (+ optional scipy elsewhere). No rclpy/ROS imports, so it unit-tests
standalone and imports cleanly into a ROS2 node later.
"""

import threading
from collections import deque

import numpy as np


# --------------------------------------------------------------------------- #
#  Free functions: vectorised Rodrigues, its inverse, and gyro integration.
# --------------------------------------------------------------------------- #

def rotvec_to_matrix(rotvecs):
    """Vectorised Rodrigues: rotation vector(s) -> rotation matrix/matrices.

    A rotation vector encodes an axis (its direction) and an angle (its
    magnitude, radians). Accepts shape (3,) -> returns (3, 3), or (N, 3) ->
    returns (N, 3, 3).

    Numerically safe at zero angle: instead of dividing by the angle to get a
    unit axis we use the raw rotvec components with the smooth coefficients
    a = sin(th)/th and b = (1-cos th)/th^2, evaluated via np.sinc so they are
    exact (1 and 1/2) at th = 0 with no branch:

        R = cos(th) I + a [v]_x + b v v^T ,   v = rotvec, [v]_x = skew(v)

    (via the identity [v]_x^2 = v v^T - th^2 I, so I + a[v]_x + b[v]_x^2 reduces
    to the above). This needs only an outer product v v^T, not a batched
    matrix-matrix product -- the latter is markedly slower under this numpy build.
    """
    v = np.atleast_2d(np.asarray(rotvecs, dtype=np.float64))   # (N, 3)
    n = v.shape[0]
    theta = np.linalg.norm(v, axis=1)                          # (N,)

    # sin(th)/th and (1-cos th)/th^2, both smooth and finite at th -> 0.
    # np.sinc(x) = sin(pi x)/(pi x); 1-cos th = 2 sin^2(th/2).
    a = np.sinc(theta / np.pi)                                 # -> 1 as th->0
    b = 0.5 * np.sinc(theta / (2.0 * np.pi)) ** 2              # -> 1/2 as th->0

    # Skew-symmetric matrices [v]_x, stacked (N, 3, 3).
    K = np.zeros((n, 3, 3), dtype=np.float64)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    K[:, 0, 1] = -vz; K[:, 0, 2] = vy
    K[:, 1, 0] = vz;  K[:, 1, 2] = -vx
    K[:, 2, 0] = -vy; K[:, 2, 1] = vx

    vvT = np.einsum('ni,nj->nij', v, v)                        # (N, 3, 3)
    R = (np.cos(theta)[:, None, None] * np.eye(3)
         + a[:, None, None] * K
         + b[:, None, None] * vvT)

    if np.ndim(rotvecs) == 1:
        return R[0]
    return R


def matrix_to_rotvec(mats):
    """Inverse of rotvec_to_matrix (the SO(3) log map), vectorised.

    Accepts (3, 3) -> (3,) or (N, 3, 3) -> (N, 3). Handles the near-zero and
    near-pi degeneracies. Mainly here to round-trip integrate_gyro's output;
    de-skew itself works in matrices and does not call this.
    """
    R = np.asarray(mats, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[None]
    m = R.shape[0]

    tr = np.trace(R, axis1=1, axis2=2)
    cos_t = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)                                   # [0, pi]
    sin_t = np.sin(theta)

    # r = axis * 2 sin(theta), from the skew-symmetric part of R.
    r = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1)            # (m, 3)

    out = np.zeros((m, 3), dtype=np.float64)
    small = theta < 1e-7
    near_pi = (np.pi - theta) < 1e-4
    normal = ~small & ~near_pi

    # General case: rotvec = theta / (2 sin theta) * r
    if np.any(normal):
        scale = np.zeros(m)
        scale[normal] = theta[normal] / (2.0 * sin_t[normal])
        out[normal] = r[normal] * scale[normal, None]
    # Near zero: theta/(2 sin theta) -> 1/2, and r is already ~ 2*theta*axis.
    if np.any(small):
        out[small] = 0.5 * r[small]
    # Near pi: sin(theta) ~ 0 so r is unreliable; recover the axis from
    # (R + I)/2 ~ axis axis^T, take the most reliable column.
    if np.any(near_pi):
        A = 0.5 * (R[near_pi] + np.eye(3))
        diag = np.clip(np.diagonal(A, axis1=1, axis2=2), 0.0, None)
        col = np.argmax(diag, axis=1)
        rows = np.arange(A.shape[0])
        axis = A[rows, :, col]                                 # (p, 3)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        # Fix the sign against the (tiny but directional) skew part.
        s = np.sign(np.sum(axis * r[near_pi], axis=1))
        s[s == 0] = 1.0
        out[near_pi] = axis * (s * theta[near_pi])[:, None]

    return out[0] if single else out


def apply_rotvec(points, rotvecs):
    """Rotate points by rotation vectors, fused -- the Rodrigues *vector* form.

    Applies the rotation directly to the vectors without ever materialising the
    (N, 3, 3) rotation matrices, using
        p_rot = cos(th) p + a (v x p) + b (v.p) v
    with a = sin(th)/th, b = (1-cos th)/th^2 (both via np.sinc, safe at th=0).

    points   : (N, 3)
    rotvecs  : (3,) applied to every point, or (N, 3) one per point.
    returns  : (N, 3) float64
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))    # (N, 3)
    v = np.asarray(rotvecs, dtype=np.float64)
    if v.ndim == 1:
        v = np.broadcast_to(v, p.shape)                        # (N, 3)
    theta = np.sqrt(np.einsum('ni,ni->n', v, v))               # (N,)
    a = np.sinc(theta / np.pi)
    b = 0.5 * np.sinc(theta / (2.0 * np.pi)) ** 2
    dot = np.einsum('ni,ni->n', v, p)                          # (N,)
    return (np.cos(theta)[:, None] * p
            + a[:, None] * np.cross(v, p)
            + (b * dot)[:, None] * v)


def _cumulative_matrices(times, gyros):
    """Integrate piecewise-constant gyro -> cumulative rotation matrices.

    times (K,), gyros (K, 3), assumed sorted by time. Returns C (K, 3, 3) with
    C[0] = I and C[k] = orientation at times[k] relative to times[0], built by
    right-multiplying the incremental rotation Rodrigues(w_k * dt_k) (the gyro
    is held constant on [t_k, t_{k+1}), FAST-LIO style). The K-length loop is
    an unavoidable serial dependency but K is tiny (~100 samples per buffer).
    """
    K = len(times)
    C = np.empty((K, 3, 3), dtype=np.float64)
    C[0] = np.eye(3)
    if K >= 2:
        dt = np.diff(times)                                    # (K-1,)
        dR = rotvec_to_matrix(gyros[:-1] * dt[:, None])        # (K-1, 3, 3)
        for k in range(K - 1):
            C[k + 1] = C[k] @ dR[k]
    return C


def integrate_gyro(times, gyros):
    """Integrate a buffer of gyro samples into cumulative rotation vectors.

    times (K,) seconds, gyros (K, 3) rad/s (sensor frame), sorted by time.
    Returns (K, 3) cumulative rotation vectors: element k is the rotation that
    carries a vector from the orientation at times[0] into the orientation at
    times[k]. (For very large cumulative angles the returned rotvec is the
    canonical [0, pi] representative; rotvec_to_matrix round-trips it exactly.)
    """
    return matrix_to_rotvec(_cumulative_matrices(times, gyros))


# --------------------------------------------------------------------------- #
#  The de-skewer: a rolling gyro buffer + per-point rotational compensation.
# --------------------------------------------------------------------------- #

class ImuGyroDeskewer:
    """Rolling gyro buffer that rotationally de-skews Livox sweeps.

    Typical wiring in a ROS2 node:

        dsk = ImuGyroDeskewer(buffer_s=0.5)
        # in the /livox/imu callback (200 Hz):
        dsk.add_gyro(t_imu, wx, wy, wz)
        # in the /livox/lidar callback, per sweep:
        pts = dsk.deskew(points, point_times, ref_time=point_times.max())
    """

    def __init__(self, buffer_s=0.5):
        self.buffer_s = float(buffer_s)
        self._buf = deque()            # (t, wx, wy, wz), time-ascending
        self._lock = threading.Lock()  # add_gyro and deskew may run on diff threads

    # -- ingestion ---------------------------------------------------------- #

    def add_gyro(self, t, wx, wy, wz):
        """Push one angular-velocity sample (t seconds; w in rad/s, sensor frame).

        Prunes anything older than buffer_s behind the newest sample. Assumes
        samples arrive in (roughly) non-decreasing time order, as ROS delivers
        them; a stray out-of-order sample is still appended and simply widens
        the effective buffer slightly.
        """
        t = float(t)
        with self._lock:
            self._buf.append((t, float(wx), float(wy), float(wz)))
            cutoff = t - self.buffer_s
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def _snapshot(self):
        """Copy the buffer out to sorted numpy arrays (times, gyros)."""
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return np.empty(0), np.empty((0, 3))
        arr = np.asarray(buf, dtype=np.float64)                # (K, 4)
        order = np.argsort(arr[:, 0], kind='stable')           # be robust to reorder
        arr = arr[order]
        return arr[:, 0], arr[:, 1:4]

    def n_samples(self):
        with self._lock:
            return len(self._buf)

    # -- de-skew ------------------------------------------------------------ #

    def deskew(self, points, point_times, ref_time):
        """Rotate every point into the orientation the sensor had at ref_time.

        points      : (N, 3) float, sensor-frame [x, y, z].
        point_times : (N,) absolute seconds, one per point.
        ref_time    : scalar seconds, target orientation (usually the sweep end).

        Returns (N, 3) float32. If fewer than 2 gyro samples are buffered there
        is nothing to integrate, so points are returned unchanged (a copy).

        Query times outside the buffered gyro span are clamped to the nearest
        buffered sample (no extrapolation), so a point slightly ahead of / behind
        the gyro coverage is simply not rotated relative to that boundary.

        Performance: the O(K) gyro integration (K ~ tens-to-hundreds of samples)
        is done once in float64; the O(N) per-point work is done in float32 (the
        rotation matrices are only cm-accurate anyway, and float32 halves the
        memory traffic that dominates on the Jetson). ~3-4 ms for 15k points.
        """
        P = np.ascontiguousarray(points, dtype=np.float32)
        pt = np.asarray(point_times, dtype=np.float64).ravel()
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be (N, 3)")
        if pt.shape[0] != P.shape[0]:
            raise ValueError("point_times must have one entry per point")

        T, W = self._snapshot()
        K = T.shape[0]
        if K < 2 or P.shape[0] == 0:
            return P.copy()

        # --- O(K): integrate the gyro, build per-knot relative rotations. ----
        C = _cumulative_matrices(T, W)                         # (K, 3, 3) f64
        t_lo, t_hi = T[0], T[-1]
        tr = float(np.clip(ref_time, t_lo, t_hi))
        r_idx = int(np.clip(np.searchsorted(T, tr, side='right') - 1, 0, K - 1))
        Cr = C[r_idx] @ rotvec_to_matrix(W[r_idx] * (tr - T[r_idx]))  # (3, 3)
        # Mrel[k] = Cr^T . C[k] carries the knot-k orientation into the ref frame.
        Mrel = np.einsum('ji,njk->nik', Cr, C).astype(np.float32)     # (K, 3, 3)
        Wf = W.astype(np.float32)

        # --- O(N): locate the bracketing (left) knot for every point time. ---
        tq = np.clip(pt, t_lo, t_hi)
        idx = np.searchsorted(T, tq, side='right') - 1
        np.clip(idx, 0, K - 1, out=idx)
        dt = (tq - T[idx]).astype(np.float32)                  # (N,) >= 0

        # Fractional increment from the left knot to the exact point time, held
        # at that knot's gyro (FAST-LIO style). The angle is sub-IMU-period tiny,
        # so a first-order rotation q = p + dt (w x p) is accurate to << 1 mm and
        # far cheaper than building N rotation matrices.
        q = P + dt[:, None] * np.cross(Wf[idx], P)             # (N, 3) f32
        # p_ref = Mrel[idx] . q   (= Cr^T . C(t_i) . p_i).
        return np.einsum('nij,nj->ni', Mrel[idx], q)           # (N, 3) f32

    def deskew_full(self, points, point_times, ref_time, vel):
        """Rotational de-skew PLUS a constant-velocity translation correction.

        Usually UNNECESSARY for short-range reactive avoidance: at a few metres
        the apparent shift from a walking robot's translation over one ~0.1 s
        sweep is well under a centimetre, dwarfed by the rotational smear. This
        stub is provided for completeness / longer-range use.

        vel : (3,) sensor-frame linear velocity (m/s) at ref_time, assumed
              roughly constant across the sweep.

        Adds vel * (t_i - t_r) to each rotated point: a point measured earlier
        (t_i < t_r), when the sensor sat further back, is pulled toward where it
        sits in the ref frame after the body has moved.
        """
        out = self.deskew(points, point_times, ref_time).astype(np.float64)
        T, _ = self._snapshot()
        if T.shape[0] < 2:
            return out.astype(np.float32, copy=False)
        pt = np.asarray(point_times, dtype=np.float64).ravel()
        tq = np.clip(pt, T[0], T[-1])
        tr = float(np.clip(ref_time, T[0], T[-1]))
        vel = np.asarray(vel, dtype=np.float64).reshape(3)
        out += vel[None, :] * (tq - tr)[:, None]
        return out.astype(np.float32, copy=False)
