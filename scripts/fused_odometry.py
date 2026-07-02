"""fused_odometry.py -- SE2 pose fusion for the G1 (leg-odom + IMU + LiDAR-odom).

WHY
    The live pose today (scripts/lidar_source.py::OdomReader) is LEG-ONLY: it reads
    `rt/odommodestate` and reports (position.x, position.y, imu_state.rpy[2]). That is
    dead-reckoning -- it integrates foot contacts, so it is high-rate and locally smooth
    but DRIFTS without bound (a small yaw-rate bias slowly rotates the whole frame and
    the position spirals away). FAST-LIO gives a second, absolute SE2 pose (x, y, yaw)
    from the Mid-360 that does NOT drift, but is lower-rate, jumpy, and can DROP OUT
    (feature-poor rooms, fast rotation, CPU contention). Neither alone is good enough:
    leg-only drifts, lidar-only is jumpy and unavailable at times. This module fuses
    the two (plus the IMU yaw-rate that rides on the SAME odommodestate message) into a
    single pose that is smooth AND non-drifting AND survives lidar dropout by coasting.

WHY A COMPLEMENTARY FILTER (not an EKF)
    For SE2 with an absolute low-rate reference, a full EKF buys little and costs a lot:
    the Jacobians/covariance bookkeeping are easy to get subtly wrong, and the payoff
    over a well-tuned complementary filter is marginal when the "measurement" (lidar) is
    already a full pose in the SAME frame. The complementary structure is instead trivial
    to reason about and to prove correct:

        HIGH FREQUENCY (leg + gyro):  integrate the leg-odom BODY-FRAME delta and the
            IMU yaw-rate to propagate the fused pose every leg tick. This path never
            drops out, so the pose is always fresh and smooth.
        LOW FREQUENCY (lidar):        each time a lidar pose arrives, nudge the fused
            pose a FRACTION of the way toward it (gains k_pos, k_yaw in (0,1]). Small
            gains => trust dead-reckoning short-term, bleed off drift slowly => smooth,
            non-drifting output that is NOT just a copy of the jumpy lidar.

    The one idea that makes the position accurate is applying the leg TRANSLATION delta
    (how far the body moved, in ITS OWN frame -- a quantity that is correct even when the
    heading estimate is wrong) at the FUSED, lidar-corrected HEADING. So correcting the
    heading also corrects the direction of all future translation -- which is exactly the
    error that makes leg-only spiral. See add_leg() for the mechanics.

    Complementary filtering is just the steady-state limit of a Kalman filter with fixed
    gains; we ALSO carry a light scalar covariance proxy (grows while coasting, shrinks on
    a lidar accept) purely to (a) report a confidence and (b) size the outlier gate. That
    is enough Bayesian bookkeeping to be honest without the EKF's fragility.

WHAT IT HANDLES
    * yaw wraparound            -- every angle op goes through wrap(); errors are wrapped
                                   to (-pi, pi] before use, so a +179 -> -179 flip is a
                                   ~2 deg innovation, not a ~358 deg one.
    * lidar dropout             -- with no lidar the filter just keeps integrating
                                   leg+gyro (coasts). Confidence decays; a `lidar_healthy`
                                   flag flips once the last accept is older than a timeout.
    * outlier rejection         -- a lidar pose that jumps further than a gap-scaled gate
                                   is rejected (FAST-LIO relocalization glitch / a bad
                                   scan match). Persistent rejects (a genuine relocalize /
                                   "kidnapped robot") force a re-acquire so we never latch
                                   onto a stale pose forever.
    * source timestamps / rates -- each add_*() carries the source stamp; leg predict uses
                                   the true dt between leg samples, gyro uses the newest
                                   reading, lidar corrects whenever it lands. Rates are
                                   independent and may interleave in any order.

API (the consumer calls these; see G1FusedOdometryAdapter for the on-robot wiring)
    f = FusedOdometry()                       # or FusedOdometry(cfg=dict(...))
    f.add_imu(gyro_z, t)                       # yaw-rate rad/s  @ stamp t  (fast)
    f.add_leg(x, y, yaw, t)                    # leg SE2 pose    @ stamp t  (high-rate, drifts)
    f.add_lidar(x, y, yaw, t)                  # lidar SE2 pose  @ stamp t  (absolute, may drop)
    x, y, yaw = f.get_pose()                   # latest fused SE2 pose
    st = f.get_state()                         # full dict: pose, yaw_rate, gyro_bias,
                                               #   pos_std, yaw_std, confidence, health...

PURE NUMPY / math -- NO ROS, NO unitree_sdk2py, NO robot. Importable and runnable on a
workstation; deterministic (no clocks, no RNG inside the filter). The sim harness
(scripts/sim_odometry.py) proves fused RMSE <= 0.5x leg-only and graceful dropout; the
unit tests live in scripts/test_fused_odometry.py.
"""

import math


# ---------------------------------------------------------------------------
# angle helpers
# ---------------------------------------------------------------------------
def wrap(a):
    """Wrap an angle to (-pi, pi]. Used on every yaw sum and every yaw error so the
    +pi/-pi seam is never crossed as a ~2*pi jump."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def _all_finite(*vals):
    """True iff every value is a finite float (no NaN, no +-inf).

    This guards the top of every add_*(): a single non-finite sensor sample must be
    DROPPED as if it never arrived, never fused. Real FAST-LIO and leg odometry emit
    NaN/inf on init transients, degenerate scan-matches and DDS glitches, and a NaN is
    especially dangerous here because it DEFEATS the outlier gate rather than tripping
    it: every comparison involving NaN is False (`nan > gate` is False, `abs(nan) >
    gate` is False), so a NaN innovation is scored as "inside the gate" and ACCEPTED,
    writing NaN into self.x/self.yaw. From there every later predict/correct propagates
    the NaN and the filter never recovers even after good data resumes. Checking
    finiteness at the door is the only place this can be stopped cheaply and totally."""
    for v in vals:
        if not math.isfinite(v):
            return False
    return True


def yaw_rmse(err_list):
    """Root-mean-square of a list of yaw ERRORS, each wrapped to (-pi, pi] first."""
    if not err_list:
        return float("nan")
    s = 0.0
    for e in err_list:
        w = wrap(e)
        s += w * w
    return math.sqrt(s / len(err_list))


# ---------------------------------------------------------------------------
# defaults -- every gain/threshold the filter reads, in one place (mirror the
# obstacle_node DEFAULTS idiom). Override via FusedOdometry(cfg=...).
# ---------------------------------------------------------------------------
DEFAULTS = {
    # --- complementary blend (high-frequency yaw) ---
    "w_gyro": 0.98,           # weight of IMU gyro vs leg yaw-delta in the fused yaw increment
    #                           (0 = leg only, 1 = gyro only). GYRO WEIGHTED HIGH (~0.98): the
    #                           gyro is the low-drift short-term heading source, so it should
    #                           dominate; the leg yaw-delta is only a fallback for when the IMU
    #                           is stale. Weighting the leg heavily would fold the leg's own
    #                           yaw-rate bias into the fused heading, and the gyro-bias estimator
    #                           (which learns from the lidar yaw residual) would then absorb
    #                           BOTH biases and no longer report the true IMU bias. Keeping the
    #                           leg contribution small isolates the IMU bias so coasting is
    #                           accurate. Both sources still drift; lidar fixes the absolute.
    # --- lidar correction gains (low-frequency pull) ---
    "k_pos_lidar": 0.35,     # fraction of the position innovation applied per lidar accept
    "k_yaw_lidar": 0.30,     # fraction of the yaw innovation applied per lidar accept
    #                           small => smooth + trusts dead-reckoning; large => tracks the
    #                           jumpy lidar tightly. 0.3ish is a good "slow correction".
    # --- gyro-bias estimation (drives coasting accuracy during dropout) ---
    "gyro_bias_gain": 0.03,  # slow learning rate: each lidar yaw correction implies an
    #                           average heading-rate error we attribute (in part) to gyro bias.
    "gyro_bias_max": 0.5,    # rad/s clamp on the estimated bias (sanity bound)
    # --- covariance proxy (confidence + gate sizing only; NOT a full EKF P) ---
    "q_pos": 0.02,           # position variance growth per second while coasting (m^2/s)
    "q_yaw": 0.01,           # yaw variance growth per second while coasting (rad^2/s)
    "pos_meas_var": 0.01,    # lidar position measurement variance floor (m^2) ~ (0.1 m)^2
    "yaw_meas_var": 0.002,   # lidar yaw measurement variance floor (rad^2)
    # --- outlier gate on the lidar innovation ---
    "gate_pos_m": 0.5,       # base position gate (m)
    "gate_pos_rate": 1.5,    # + this * gap_seconds (allow more travel if lidar was gone longer)
    "gate_pos_sigma": 3.0,   # + this * sqrt(Ppos) (allow more when we are already unsure)
    "gate_dt_cap": 2.0,      # cap the gap term so a very long dropout does not open the gate wide
    "gate_yaw_rad": 0.6,     # yaw innovation gate (rad)
    "max_reject": 8,         # consecutive rejects before a forced re-acquire (kidnapped robot)
    # --- health / reporting ---
    "lidar_timeout_s": 0.5,  # last accept older than this => lidar_healthy = False (coasting)
    "imu_timeout_s": 0.2,     # gyro cache older than this (vs the leg stamp being
    #                           integrated) is STALE. A stuck/frozen gyro stays finite so the
    #                           NaN guard never fires, yet at w_gyro~0.98 it would silently pin
    #                           fused heading to a wrong (often zero) rate while the robot really
    #                           turns. Mirrors lidar_timeout_s: past this age we ignore the cached
    #                           gyro and fall back to the leg yaw-delta. ~6 frames @30 Hz of slack
    #                           (leg+gyro ride the SAME rt/odommodestate msg, so fresh age ~= 0).
    "conf_scale": 0.5,       # confidence = exp(-pos_std / conf_scale), in (0, 1]
}


class FusedOdometry:
    """Complementary SE2 pose filter. See module docstring for the design rationale.

    State carried:
        x, y, yaw     fused SE2 pose (world/odom frame; yaw in (-pi, pi])
        yaw_rate      last fused yaw rate (rad/s), for reporting
        gyro_bias     slowly-estimated IMU yaw-rate bias (rad/s)
        Ppos, Pyaw    scalar variance proxies (grow while coasting, shrink on lidar accept)
    """

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        c = self.cfg
        # unpack the hot ones for readability in the update methods
        self.w_gyro = float(c["w_gyro"])
        self.k_pos = float(c["k_pos_lidar"])
        self.k_yaw = float(c["k_yaw_lidar"])
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self, x=0.0, y=0.0, yaw=0.0):
        """Clear all state to a known pose (also the not-yet-initialized default)."""
        self.x = float(x)
        self.y = float(y)
        self.yaw = wrap(float(yaw))
        self.yaw_rate = 0.0
        self.gyro_bias = 0.0
        self.Ppos = float(self.cfg["pos_meas_var"])
        self.Pyaw = float(self.cfg["yaw_meas_var"])
        # source bookkeeping
        self._have_pose = False          # set once leg or lidar bootstraps the pose
        self._leg_prev = None            # (x, y, yaw) of the previous leg sample
        self._t_leg_prev = None
        self._gyro_z = None              # newest IMU yaw-rate reading
        self._t_gyro = None
        self._t_lidar_prev = None        # stamp of previous lidar sample (any, accepted or not)
        self._t_lidar_ok = None          # stamp of last ACCEPTED lidar (for health)
        self._reject_run = 0             # consecutive rejects
        self.n_lidar_reject = 0          # total rejects (reporting)
        self.n_lidar_accept = 0
        self._t_now = 0.0                # max stamp seen across all sources

    # -- high-frequency inputs ----------------------------------------------
    def add_imu(self, gyro_z, t):
        """Feed an IMU yaw-rate sample (rad/s) at stamp t. Cheap: we just cache the newest
        reading; add_leg() consumes it on the next predict. This lets the (fast) IMU refine
        the (slower) leg heading without a second integration path to keep in sync."""
        gyro_z = float(gyro_z); t = float(t)
        if not _all_finite(gyro_z, t):
            return                       # drop a non-finite IMU sample (treat as absent)
        self._gyro_z = gyro_z
        self._t_gyro = t
        if t > self._t_now:
            self._t_now = t

    def add_leg(self, x, y, yaw, t):
        """Feed a leg-odom SE2 pose (x, y, yaw) at stamp t. This is the PREDICT step.

        We do NOT trust the leg's absolute frame (it drifts). We take only the RELATIVE
        motion since the last leg sample, expressed in the leg's own BODY frame -- the
        forward/left distance travelled and the heading change, which are locally correct
        regardless of accumulated frame drift -- then re-apply that motion at the FUSED
        heading. Fusing the heading (via the gyro blend + lidar corrections) therefore
        steers the direction of the translation too, which is what kills the spiral."""
        x = float(x); y = float(y); yaw = float(yaw); t = float(t)
        if not _all_finite(x, y, yaw, t):
            # drop a non-finite leg sample: do NOT touch _leg_prev/_t_leg_prev, so the
            # next good sample differences against the last GOOD reference (a slightly
            # longer dt) instead of against NaN. Treated exactly as if it never arrived.
            return
        if t > self._t_now:
            self._t_now = t

        if not self._have_pose:
            # bootstrap the fused pose from the first leg sample
            self.x, self.y, self.yaw = x, y, wrap(yaw)
            self._have_pose = True
            self._leg_prev = (x, y, yaw)
            self._t_leg_prev = t
            return

        if self._leg_prev is None:
            self._leg_prev = (x, y, yaw)
            self._t_leg_prev = t
            return

        dt = t - self._t_leg_prev
        if dt <= 0.0:
            # out-of-order / duplicate stamp: refresh the reference but do not integrate
            self._leg_prev = (x, y, yaw)
            self._t_leg_prev = t
            return

        xp, yp, yawp = self._leg_prev
        # leg motion in the WORLD then rotated into the leg BODY frame at the previous
        # leg heading (invariant to the leg's absolute-frame drift):
        dpx = x - xp
        dpy = y - yp
        cyp, syp = math.cos(yawp), math.sin(yawp)
        dxb = cyp * dpx + syp * dpy        # forward distance in body frame
        dyb = -syp * dpx + cyp * dpy       # left distance in body frame
        dyaw_leg = wrap(yaw - yawp)

        # complementary heading increment: blend the leg yaw-delta with the (bias-corrected)
        # gyro yaw-rate integrated over dt -- but ONLY while the cached gyro is FRESH. WHY the
        # freshness gate: a stuck/frozen gyro keeps returning a finite value, so _all_finite()
        # (which only screens NaN/inf) never drops it; without an age check we would keep
        # trusting that dead reading at w_gyro~0.98 and let fused heading diverge from a real
        # turn (e.g. robot turns 1.99 rad while gyro sits at 0 -> fused yaw ~0.05: heading fully
        # wrong). We captured _t_gyro in add_imu(); compare it to THIS leg stamp t and treat the
        # gyro as stale past imu_timeout_s, mirroring lidar_timeout_s. Falling back to the leg
        # yaw-delta also means a run with no IMU ever reproduces leg odometry EXACTLY (the
        # graceful no-IMU/no-lidar case).
        gyro_fresh = (self._gyro_z is not None and self._t_gyro is not None
                      and (t - self._t_gyro) <= self.cfg["imu_timeout_s"])
        if gyro_fresh:
            dyaw_gyro = (self._gyro_z - self.gyro_bias) * dt
            dyaw = self.w_gyro * dyaw_gyro + (1.0 - self.w_gyro) * dyaw_leg
        else:
            dyaw = dyaw_leg

        # integrate: rotate the body-frame displacement into the world by the fused heading
        # at the START of the interval, then advance the heading. This is EXACT (not an
        # approximation): (dxb, dyb) is the true chord in the frame aligned with yawp, and
        # self.yaw is the fused estimate of that same physical heading -- the intra-step
        # rotation is already baked into (dxb, dyb), so no midpoint term is needed (adding
        # one would double-count half the turn). With no IMU this reproduces leg odometry
        # bit-for-bit (dyaw == dyaw_leg, self.yaw tracks yawp) -- the graceful fallback.
        c0, s0 = math.cos(self.yaw), math.sin(self.yaw)
        self.x += c0 * dxb - s0 * dyb
        self.y += s0 * dxb + c0 * dyb
        self.yaw = wrap(self.yaw + dyaw)
        self.yaw_rate = dyaw / dt

        # grow the covariance proxy (we just dead-reckoned, so we are a bit less sure)
        self.Ppos += self.cfg["q_pos"] * dt
        self.Pyaw += self.cfg["q_yaw"] * dt

        self._leg_prev = (x, y, yaw)
        self._t_leg_prev = t

    # -- low-frequency input (the correction) --------------------------------
    def add_lidar(self, x, y, yaw, t):
        """Feed a lidar (FAST-LIO) absolute SE2 pose at stamp t. This is the CORRECT step:
        gate for outliers, then nudge the fused pose a fraction toward the lidar. Returns
        True if the sample was accepted, False if rejected as an outlier."""
        x = float(x); y = float(y); yaw = float(yaw); t = float(t)
        if not _all_finite(x, y, yaw, t):
            # drop a non-finite lidar frame BEFORE the gate -- a NaN would otherwise pass
            # the gate (NaN comparisons are all False) and poison the pose. Treated as
            # absent, not as an outlier: no state change, and it does NOT count toward the
            # consecutive-reject / kidnapped-robot counter (a NaN is not a disagreement).
            return False
        if t > self._t_now:
            self._t_now = t

        if not self._have_pose:
            # lidar can also bootstrap the pose if it arrives before any leg sample
            self.x, self.y, self.yaw = x, y, wrap(yaw)
            self._have_pose = True
            self._t_lidar_prev = t
            self._t_lidar_ok = t
            self.n_lidar_accept += 1
            return True

        # innovation (lidar minus fused), yaw wrapped
        ex = x - self.x
        ey = y - self.y
        epos = math.hypot(ex, ey)
        eyaw = wrap(yaw - self.yaw)

        # gap since the previous lidar sample sizes the gate: a longer absence means the
        # robot legitimately travelled further, so a larger innovation is expected.
        if self._t_lidar_prev is None:
            gap = 0.0
        else:
            gap = max(0.0, t - self._t_lidar_prev)
        gap_term = self.cfg["gate_pos_rate"] * min(gap, self.cfg["gate_dt_cap"])
        gate_pos = (self.cfg["gate_pos_m"] + gap_term
                    + self.cfg["gate_pos_sigma"] * math.sqrt(max(self.Ppos, 0.0)))
        gate_yaw = self.cfg["gate_yaw_rad"] + self.cfg["gate_pos_sigma"] * math.sqrt(max(self.Pyaw, 0.0))

        self._t_lidar_prev = t

        if epos > gate_pos or abs(eyaw) > gate_yaw:
            # OUTLIER. Reject, lose a little confidence, and count it.
            self._reject_run += 1
            self.n_lidar_reject += 1
            self.Ppos += self.cfg["pos_meas_var"]   # we are now less sure of the coast
            if self._reject_run >= self.cfg["max_reject"]:
                # persistent disagreement => FAST-LIO relocalized / kidnapped robot.
                # Re-acquire: snap fully to lidar and trust it again.
                self.x, self.y, self.yaw = x, y, wrap(yaw)
                self.Ppos = self.cfg["pos_meas_var"]
                self.Pyaw = self.cfg["yaw_meas_var"]
                self._reject_run = 0
                self._t_lidar_ok = t
                self.n_lidar_accept += 1
                return True
            return False

        # ACCEPT. Complementary pull toward the lidar pose.
        self._reject_run = 0
        # gyro-bias update: our integrated heading was off by eyaw over the interval `gap`;
        # attribute a slow fraction of that average rate error to gyro bias so the coast
        # improves. (Sign: eyaw = truth - ours; if we over-rotated, eyaw<0 -> raise bias.)
        if gap > 1e-3:
            rate_err = eyaw / gap
            self.gyro_bias -= self.cfg["gyro_bias_gain"] * rate_err
            b = self.cfg["gyro_bias_max"]
            self.gyro_bias = max(-b, min(b, self.gyro_bias))

        self.x += self.k_pos * ex
        self.y += self.k_pos * ey
        self.yaw = wrap(self.yaw + self.k_yaw * eyaw)

        # shrink the covariance proxy toward the measurement floor
        self.Ppos = max(self.cfg["pos_meas_var"], self.Ppos * (1.0 - self.k_pos))
        self.Pyaw = max(self.cfg["yaw_meas_var"], self.Pyaw * (1.0 - self.k_yaw))

        self._t_lidar_ok = t
        self.n_lidar_accept += 1
        return True

    # -- outputs -------------------------------------------------------------
    def get_pose(self):
        """Latest fused SE2 pose (x, y, yaw). yaw in (-pi, pi]."""
        return (self.x, self.y, self.yaw)

    def lidar_age(self):
        """Seconds since the last ACCEPTED lidar (inf if never). Uses the max stamp seen."""
        if self._t_lidar_ok is None:
            return float("inf")
        return max(0.0, self._t_now - self._t_lidar_ok)

    def lidar_healthy(self):
        """True while the last accepted lidar is fresher than lidar_timeout_s (else coasting)."""
        return self.lidar_age() <= self.cfg["lidar_timeout_s"]

    def get_state(self):
        """Full state dict for a consumer / dashboard: pose, rates, bias, uncertainty,
        a 0..1 confidence, and lidar health."""
        pos_std = math.sqrt(max(self.Ppos, 0.0))
        yaw_std = math.sqrt(max(self.Pyaw, 0.0))
        return {
            "x": self.x, "y": self.y, "yaw": self.yaw,
            "yaw_rate": self.yaw_rate,
            "gyro_bias": self.gyro_bias,
            "pos_std": pos_std, "yaw_std": yaw_std,
            "confidence": math.exp(-pos_std / self.cfg["conf_scale"]),
            "lidar_healthy": self.lidar_healthy(),
            "lidar_age_s": self.lidar_age(),
            "n_lidar_accept": self.n_lidar_accept,
            "n_lidar_reject": self.n_lidar_reject,
            "have_pose": self._have_pose,
        }


# ---------------------------------------------------------------------------
# ON-ROBOT ADAPTER (documented wiring; NO ROS import at module load)
# ---------------------------------------------------------------------------
def quat_to_yaw(qx, qy, qz, qw):
    """Yaw (rad) from a quaternion -- FAST-LIO publishes orientation as a quaternion in
    nav_msgs/Odometry; we only need the heading about +z."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class G1FusedOdometryAdapter:
    """Bridge the THREE real G1 sources into a FusedOdometry, WITHOUT touching the live
    OdomReader -- a new, parallel consumer.

    ============================ ON-ROBOT WIRING (TODO: validate on the G1) ============
    The two message families live on DIFFERENT middlewares, which is the one wrinkle:

      (1) LEG pose + (2) IMU yaw-rate  -- BOTH ride the SAME Unitree DDS message,
          `rt/odommodestate` (SportModeState_), the exact topic OdomReader already uses:
              leg pose : msg.position[0], msg.position[1], msg.imu_state.rpy[2]   (x, y, yaw)
              IMU rate : msg.imu_state.gyroscope[2]                               (yaw-rate rad/s)
          So a SINGLE subscribe callback feeds add_imu() then add_leg() per message. We
          re-subscribe rather than reach into OdomReader, so OdomReader is untouched and
          this consumer can run alongside it (unitree_sdk2py allows multiple subscribers).
          Domain 0, interface as in the dashboard (ChannelFactoryInitialize(0, "eth0")).

      (3) LiDAR pose -- FAST-LIO publishes `nav_msgs/Odometry` on `/Odometry` (see
          g1_nav/config/nav2_params.yaml: `odom_topic: /Odometry`), frame
          camera_init -> body, on ROS2 domain 99 (CycloneDDS). Convert its quaternion
          with quat_to_yaw() and feed add_lidar(x, y, yaw, stamp).

      TIMESTAMPS: use each message's own stamp (odommodestate has a tick; /Odometry has
          header.stamp) converted to float seconds. If a source lacks a usable stamp, fall
          back to time.monotonic() at receipt -- but a consistent CLOCK across sources is
          what the gap-scaled gate and the coast rely on, so prefer the message stamps and
          confirm both clocks are the same base on the robot (they are both the robot's).

      CROSS-DOMAIN: leg/imu are Unitree-DDS (domain 0); FAST-LIO is ROS2 (domain 99). Run
          the DDS subscribe (leg/imu) and an rclpy subscribe (/Odometry) in one process,
          OR relay /Odometry into a Unitree-DDS topic with a tiny bridge. Both are fine;
          the filter only needs the three add_*() calls in stamp order. VALIDATE that the
          FAST-LIO world frame (camera_init) and the leg frame share an origin/orientation
          at startup -- if not, seed the filter by reset()-ing to the first lidar pose so
          both feeds agree from t0 (leg contributes only deltas, so a frame offset in leg
          is harmless; a frame offset in lidar would be corrected-toward and must match).
    ====================================================================================

    Usage on the robot (pseudo, ROS/SDK omitted here so this module stays import-clean):
        adapter = G1FusedOdometryAdapter()
        adapter.start(interface="eth0")     # subscribes both sources, spins the callbacks
        ...
        x, y, yaw = adapter.get_pose()      # drop-in replacement for OdomReader.get_pose()
    """

    def __init__(self, cfg=None):
        self.fused = FusedOdometry(cfg=cfg)
        self._odom_sub = None
        self._lio_sub = None

    # ---- source callbacks (pure; unit-testable by hand-feeding fake msgs) ----
    def on_odommodestate(self, msg, t):
        """Callback for a Unitree SportModeState_ message. Feeds IMU (yaw-rate) then leg
        pose. `t` is the message stamp in float seconds (see wiring note)."""
        gyro_z = float(msg.imu_state.gyroscope[2])
        self.fused.add_imu(gyro_z, t)
        x = float(msg.position[0])
        y = float(msg.position[1])
        yaw = float(msg.imu_state.rpy[2])
        self.fused.add_leg(x, y, yaw, t)

    def on_lidar_odometry(self, msg, t):
        """Callback for a FAST-LIO nav_msgs/Odometry message. `t` is header.stamp seconds."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.fused.add_lidar(float(p.x), float(p.y), yaw, t)

    def get_pose(self):
        return self.fused.get_pose()

    def get_state(self):
        return self.fused.get_state()

    def start(self, interface="eth0", lio_topic="/Odometry"):
        """Subscribe both feeds. ROS/SDK imports are LAZY (kept out of module load) so the
        filter stays importable/testable on a workstation. TODO: validate on the G1 --
        confirm the FAST-LIO topic/type and the odommodestate field names on THIS robot,
        and confirm the two clocks share a base (see the wiring note above)."""
        # (1)+(2) leg + imu from rt/odommodestate  -- mirror OdomReader's subscribe.
        from unitree_sdk2py.core.channel import ChannelSubscriber           # noqa: F401
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_  # noqa: F401
        self._odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        # NOTE: derive `t` from the message tick; using receipt time is a documented fallback.
        self._odom_sub.Init(lambda m: self.on_odommodestate(m, _stamp_of(m)), 10)

        # (3) lidar from FAST-LIO /Odometry (ROS2 domain 99). TODO on-robot: wire rclpy here
        # (or a DDS bridge). Left as an explicit stub so this module imports without ROS2:
        #   import rclpy; from nav_msgs.msg import Odometry
        #   node.create_subscription(Odometry, lio_topic,
        #       lambda m: self.on_lidar_odometry(m, _ros_stamp(m)), qos)
        raise NotImplementedError(
            "G1FusedOdometryAdapter.start(): finish the FAST-LIO /Odometry subscribe on the "
            "robot per the wiring TODO in the class docstring; leg+imu subscribe is prepared.")


def _stamp_of(msg):
    """Best-effort float-seconds stamp for a SportModeState_ (TODO on-robot: confirm the
    field; unitree messages expose a tick/stamp). Falls back to monotonic time."""
    for attr in ("stamp", "tick"):
        v = getattr(msg, attr, None)
        if v is not None:
            try:
                return float(v) * (1e-9 if attr == "tick" else 1.0)
            except (TypeError, ValueError):
                pass
    import time
    return time.monotonic()


if __name__ == "__main__":
    # smoke test: the robot TRULY drives straight along +x at 0.5 m/s (yaw 0), but leg-odom
    # has a +0.3 rad/s yaw-rate bias, so it integrates its OWN drifting heading and reports a
    # curving, spiralling path (heading and position both wrong -- realistic dead-reckoning).
    # The gyro reads the true rate (0) and lidar reports truth at 10 Hz. Confirm the fused
    # pose tracks truth, not the leg spiral.
    f = FusedOdometry()
    dt = 0.02
    lx = ly = lyaw = 0.0                        # leg dead-reckoned pose (integrates its own bias)
    t = 0.0
    for i in range(200):
        t = i * dt
        # leg believes it is turning at +0.3 rad/s while moving 0.5 m/s forward:
        lyaw_mid = lyaw + 0.5 * (0.3 * dt)
        lx += 0.5 * dt * math.cos(lyaw_mid)
        ly += 0.5 * dt * math.sin(lyaw_mid)
        lyaw = wrap(lyaw + 0.3 * dt)
        f.add_imu(0.0, t)                       # gyro says no rotation (agrees with truth)
        f.add_leg(lx, ly, lyaw, t)             # leg pose spirals away
        if i % 5 == 0:                          # lidar at 10 Hz: truth (straight +x, yaw 0)
            f.add_lidar(0.5 * t, 0.0, 0.0, t)
    x, y, yaw = f.get_pose()
    st = f.get_state()
    print(f"after {t:.1f}s straight run with drifting leg:")
    print(f"  fused pose = ({x:.3f}, {y:.3f}, {yaw:+.3f} rad)   truth ~ ({0.5*t:.3f}, 0.000, +0.000)")
    print(f"  leg-only pose spiralled to ({lx:.3f}, {ly:.3f}, {lyaw:+.3f} rad); fused held near truth")
    print(f"  confidence={st['confidence']:.3f} lidar_healthy={st['lidar_healthy']} "
          f"accepts={st['n_lidar_accept']} rejects={st['n_lidar_reject']}")
    assert abs(yaw) < 0.05, f"fused yaw should be corrected near 0, got {yaw}"
    assert abs(y) < 0.10, f"fused y should stay near 0, got {y}"
    assert abs(x - 0.5 * t) < 0.10, f"fused x should track truth, got {x}"
    print("fused_odometry smoke test OK")
