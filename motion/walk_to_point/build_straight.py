"""Step 2 of walk_to_point (PLAN.md): the seam primitives, plus straight-line
distance assembly via greedy measured cycle-chaining.

align_segment() / crossfade() / rotate_about_z() / append_cycle() are the
module's SHARED seam machinery -- turn_injector, stop_tail and
generate_walk_to_point all build on them. assemble_straight() below is
SELF-TEST SCAFFOLDING ONLY (see its own comment): it exercises the seam
machinery in isolation, with the turn injector and stop tail out of the way.
The production path is generate_walk_to_point.generate(), which runs the fused
turn+distance loop over the same primitives.

Cycles are appended GREEDILY, measuring actual cumulative displacement after
each append -- never computed on paper from cycle_dx/cycle_dy, because each
crossfade seam consumes real travel that the naive k = round(target/cycle_dist)
calculation silently misses (flagged explicitly in PLAN.md).

Run under `conda activate kimodo`, from the repo root.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from common import FPS, smoothstep, yaw_from_quat_xyzw  # noqa: E402
from heel_strike import load_walk_source  # noqa: E402

CROSSFADE_WINDOW_CAP = 7  # plan: 6-8 frames for phase-matched heel-strike seams
                          # (shrunk from walk_climb_walk's 20 -- matched gait
                          # phase means the discontinuity being smoothed is
                          # already tiny). This is an UPPER BOUND, not a width:
                          # crossfade() sizes each bridge to the actual gap and
                          # returns the width it used. See its docstring.
YAW_CORRECTION_TRIGGER_DEG = 1.0  # plan: counter-rotate only if |delta| > ~1 deg
NATURAL_STEP = 0.33  # rad -- source clip's own natural max per-frame dof step
NATURAL_DV = 0.19    # m/s -- source clip's own natural max per-frame root-speed change


# ---- seam primitives (shared by every step) ----

def align_segment(prev_root_pos, prev_root_quat, seg_dof_pos, seg_root_pos, seg_root_quat,
                  align_z=True):
    """Rigidly shift seg_* (rotate+translate in the ground plane, offset height)
    so its frame-0 root pose exactly matches prev_root_pos/quat's last frame.

    POSTCONDITION crossfade() depends on: new_pos[0,:2] == prev_root_pos[-1,:2]
    EXACTLY (zero xy position gap); the z endpoints match exactly only when
    align_z=True.

    align_z=False keeps the segment's own source-frame z (delta_z = 0) instead
    of matching prev[-1]'s z. BUG FOUND BY STRESS TESTING (off-canonical
    targets): when appending a cycle of the SAME source clip on flat ground,
    prev's last frame is gait phase cb-1 while the appended cycle starts at
    phase ca -- one frame apart in phase, and pelvis z is PERIODIC (bobs per
    step), so z(cb-1) != z(ca) by a measured -4.08 mm. Matching z to prev[-1]
    injects that phase difference as a PERMANENT downward shift at every
    append: (25,0) sank -10 cm over 20 cycles (QA #6 fail, and the sink
    poisons contact_from_fk's global-min classifier); even a single stop-cycle
    append produced -5.76 mm penetration at (1.8,0)/(2,0). Same clip + flat
    ground means absolute z is already correct as recorded -- pass
    align_z=False at all same-clip cycle-append sites and let crossfade()'s
    bounded z position-blend absorb the ~4 mm natural endpoint gap (it was
    made a bounded blend for exactly this periodic-not-progressive reason,
    see its root-Z comment). Heterogeneous aligns (the stop stance, whose z
    is floor-snapped afterwards anyway) keep the default True."""
    target_xy = prev_root_pos[-1, :2]
    target_z = prev_root_pos[-1, 2]
    target_heading = yaw_from_quat_xyzw(prev_root_quat[-1])

    seg_start_xy = seg_root_pos[0, :2]
    seg_start_z = seg_root_pos[0, 2]
    seg_start_heading = yaw_from_quat_xyzw(seg_root_quat[0])

    delta_heading = target_heading - seg_start_heading
    delta_z = (target_z - seg_start_z) if align_z else 0.0
    rot2d = R.from_euler("z", delta_heading)

    centered_xy = seg_root_pos[:, :2] - seg_start_xy
    rotated_xy = rot2d.apply(np.concatenate([centered_xy, np.zeros((len(centered_xy), 1))], axis=1))[:, :2]
    new_xy = rotated_xy + target_xy
    new_z = seg_root_pos[:, 2] + delta_z
    new_root_pos = np.concatenate([new_xy, new_z[:, None]], axis=1)

    new_root_quat = (rot2d * R.from_quat(seg_root_quat)).as_quat()

    return seg_dof_pos, new_root_pos, new_root_quat


def crossfade(segA_dof, segA_pos, segA_quat, segB_dof, segB_pos, segB_quat, max_window):
    """INSERT an adaptive-width transition (up to `max_window` frames) between
    segA and segB -- does not remove or overlap any frames from either
    segment. Returns (dof, pos, quat, used_window) -- used_window is how many
    bridge frames were actually inserted (see below for why it varies).

    CALLERS MUST USE THE RETURNED used_window for seam bookkeeping: the bridge
    occupies [len(segA), len(segA)+used_window), and the cap is usually NOT the
    width used.

    Three earlier attempts, all found broken by direct measurement:

    Two earlier attempts (both REMOVE-and-overlap designs: take segA's last
    `window` frames + segB's first `window` frames, replace with `window`
    blended frames) both produced a real, measured ~2-2.5x root-speed spike
    at every seam (up to 2.85 m/s vs ~1.1-1.7 m/s cruising -- confirmed the
    raw unmodified source clip never exceeds 1.33 m/s, so this was 100% a
    blending artifact). Root cause, found by direct measurement: overlap-
    and-remove structurally compresses ~2*window frames of real, independent
    gait motion (segA's approach + segB's departure) into just `window`
    output frames -- an inherent ~2x time compression baked into the design,
    which NO interpolation method (smoothstep-linear, cubic Hermite,
    velocity-domain blending -- all three were tried) can fix, because the
    problem isn't how the frames are blended, it's that too much real motion
    is being squeezed into too little output time.

    Real fix: don't remove anything. Keep every frame of segA and segB, and
    INSERT `window` new frames between them instead:
      - dof_pos / quat: segA's frame -1 and segB's frame 0 genuinely differ
        (align_segment only guarantees ROOT continuity, not dof_pos/quat
        continuity) -- these `window` new frames interpolate (slerp for quat)
        between the two, bridging a real pose gap that otherwise needs to be
        closed instantaneously.
      - root position: align_segment already guarantees segA_pos[-1] ==
        segB_pos[0] exactly (zero position gap) -- the only real problem is
        a VELOCITY kink (segA's incoming speed/direction vs segB's outgoing
        speed/direction may differ). The inserted frames ease from segA's
        exit velocity to segB's entry velocity (linear ramp, preserves
        natural pace -- no compression, since these are ADDED frames, not
        frames that used to hold `2*window` worth of real motion). All of
        segB is then rigidly shifted by however far this velocity-eased
        bridge actually travels, so segB continues seamlessly from wherever
        the bridge ends (same rigid-shift principle as align_segment).

    Third attempt (this function's previous version): insert a FIXED
    `window`-frame bridge always, smoothstep/linear-interpolating dof_pos
    and quat between segA_dof[-1] and segB_dof[0]. BUG FOUND BY USER
    (viewer: "he slides forward on every right step") -- measured directly:
    for the common case (repeating the SAME steady cycle -- align_segment
    already makes the two boundary poses nearly identical, real gap ~0.27
    rad, LESS than the source's own natural per-frame step of 0.33 rad), a
    fixed 7-frame bridge dilutes what should be a single ordinary frame's
    worth of motion across 7 frames -- legs move at ~1/7 natural speed while
    root xy (never diluted, it's velocity-built) keeps accelerating through
    the bridge. Measured effect: both-feet-airborne runs of 9-11 frames
    (0.3-0.37s) at every seam -- real walking never has that; it reads as a
    skate/glide right where a foot should be landing.

    Real fix: WIDTH must track the ACTUAL gap size, not a fixed constant.
    `max_window` is a CAP (upper bound, used when the gap is genuinely large
    -- e.g. after turn injection, where segA/segB may not be phase-matched),
    not the width used for every seam. For the common small-gap case this
    collapses to 1-2 frames -- effectively no dilution, just enough to avoid
    an instantaneous pose jump. Anyone who "simplifies" used_window back to
    a constant reintroduces the slide bug exactly.

    Net effect: total frame count increases by `used_window` per seam (a
    handful of extra frames at most -- negligible clip-duration cost).
    Returns (dof, pos, quat, used_window) with segA and segB each fully
    intact, `used_window` new frames spliced between them.
    """
    gap = float(np.abs(segB_dof[0] - segA_dof[-1]).max())
    # velocity gap also needs its own frame budget -- the pose gap alone
    # missed a real case: BUG FOUND BY USER ("first right step still
    # slides"), measured directly -- the walk-start ramp-up segment exits at
    # ~0.62 m/s while the steady cruise cycle enters at ~1.19 m/s (still
    # accelerating up to cruise speed), but the two poses there happen to be
    # similar (small dof gap) so the pose-only sizing picked used_window=1 --
    # one frame can't ease a 0.57 m/s jump without a visible snap. Later
    # seams don't show this (both sides are the same steady cycle, matching
    # speed) -- only the very first seam does.
    v_in0 = segA_pos[-1, :2] - segA_pos[-2, :2] if len(segA_pos) >= 2 else np.zeros(2)
    v_out0 = segB_pos[1, :2] - segB_pos[0, :2] if len(segB_pos) >= 2 else v_in0
    vel_gap = float(np.linalg.norm(v_out0 - v_in0)) * FPS
    # STILL VISIBLE after a first fix (used a single global NATURAL_DV=0.19
    # constant, the clip's OVERALL max accel anywhere -- mid-stride push-off
    # is that steep, but THIS specific spot, a gentle walk-start ramp-up, is
    # only ~0.06 m/s/frame in the raw source). Using the global max as the
    # bound under-widened the bridge here by ~4x -- still a visible snap
    # compared to how gently the real mocap actually accelerates at this
    # exact point. Fix: measure the LOCAL natural accel from each segment's
    # own last few real frames near the boundary (not a blanket constant),
    # so slow ramp-up seams get a wide gentle bridge and fast mid-stride
    # seams (which don't need it -- their vel_gap is already tiny) don't.
    def local_accel(seg_pos, at_start):
        n = min(4, len(seg_pos) - 1)
        if n < 2:
            return NATURAL_DV
        pts = seg_pos[:n + 1, :2] if at_start else seg_pos[-(n + 1):, :2]
        v = np.linalg.norm(np.diff(pts, axis=0), axis=1) * FPS
        return max(float(np.abs(np.diff(v)).max()), 0.02)  # floor: avoid a near-zero bound blowing up window

    natural_dv_local = max(local_accel(segA_pos, at_start=False), local_accel(segB_pos, at_start=True))
    used_window = int(np.clip(
        max(round(gap / NATURAL_STEP), round(vel_gap / natural_dv_local)), 1, max_window))

    w = smoothstep(used_window)[:, None]

    # dof_pos / quat: genuine pose gap -- bridge it with new interpolated
    # frames, width sized to the actual gap (see docstring). Two more layers
    # found and fixed by direct measurement, in order:
    #   1. smoothstep(w) has ZERO DERIVATIVE at t=0/t=1 -- blending POSITION
    #      directly with it forces dof/quat velocity to hit exactly zero at
    #      both bridge edges even though the real gait velocity there isn't.
    #   2. Sampling the position-blend weight AT t=0 makes bridge_dof[0]
    #      literally EQUAL segA_dof[-1] (duplicated pose for one extra
    #      frame), and likewise bridge_dof[-1] duplicates segB_dof[0]. Root
    #      xy never has this (it's velocity-cumsum built, genuinely new
    #      position each frame), not a position blend with a t=0/t=1 sample.
    # Fix: linear ramp (constant velocity, no smoothstep zero-derivative) AND
    # sample only INTERIOR fractions (never exactly 0 or 1), so no frame in
    # the bridge duplicates either boundary segment's edge frame.
    w_lin = np.linspace(1.0 / (used_window + 1), used_window / (used_window + 1), used_window)[:, None]
    bridge_dof = (1 - w_lin) * segA_dof[-1] + w_lin * segB_dof[0]
    rot_a_end = R.from_quat(segA_quat[-1])
    rot_b_start = R.from_quat(segB_quat[0])
    bridge_quat = np.stack(
        [Slerp([0, 1], R.concatenate([rot_a_end, rot_b_start]))(w_lin[i, 0]).as_quat() for i in range(used_window)]
    )

    # root XY: zero position gap (align_segment guarantees it) -- only the
    # velocity needs easing. v_in/v_out from each segment's OWN data. This is
    # legitimate forward progress (walking keeps moving through the bridge),
    # so segB is rigidly shifted in XY to continue from wherever the bridge
    # ends -- same principle as align_segment. (Velocity blend, so w=smoothstep
    # here is fine -- see the dof/quat comment above for why position vs
    # velocity blending need different treatment.)
    v_in, v_out = v_in0, v_out0  # already measured above, sizing the window
    bridge_vel_xy = (1 - w) * v_in + w * v_out  # linear ease, no compression -- these are new frames
    bridge_xy = segA_pos[-1, :2] + np.cumsum(bridge_vel_xy, axis=0)
    shift_xy = bridge_xy[-1] + bridge_vel_xy[-1] - segB_pos[0, :2]
    segB_pos_shifted = segB_pos.copy()
    segB_pos_shifted[:, :2] += shift_xy

    # root Z: NOT velocity-cumsum'd like XY -- pelvis height is periodic
    # (bobs per step, doesn't progress like forward walking does), and
    # segA_pos[-1,2] == segB_pos[0,2] already (align_segment matched it
    # exactly). BUG FOUND BY QA (8m target crashed the stop-tail's foot-
    # adjacency search): treating z the same as xy -- extrapolating a
    # velocity-eased z and then permanently shifting all of segB's z to
    # match -- baked in a small net z drift at every seam (v_in/v_out z
    # rarely cancel exactly), compounding to +4cm of "floating up" over 8
    # crossfade seams (measured, monotonic). Since both ends are already
    # at the same height, a bounded position blend (like dof_pos above)
    # is correct and cannot drift.
    bridge_z = (1 - w[:, 0]) * segA_pos[-1, 2] + w[:, 0] * segB_pos[0, 2]
    bridge_pos = np.concatenate([bridge_xy, bridge_z[:, None]], axis=1)

    dof = np.concatenate([segA_dof, bridge_dof, segB_dof])
    pos = np.concatenate([segA_pos, bridge_pos, segB_pos_shifted])
    quat = np.concatenate([segA_quat, bridge_quat, segB_quat])
    return dof, pos, quat, used_window


def rotate_about_z(root_pos, root_quat, angle, pivot_xy):
    """Rigid rotation of the whole segment about the vertical axis through
    pivot_xy -- root pos AND quat together (yaw-drift counter-rotation)."""
    rot = R.from_euler("z", angle)
    xy3 = np.concatenate([root_pos[:, :2] - pivot_xy, np.zeros((len(root_pos), 1))], axis=1)
    new_xy = rot.apply(xy3)[:, :2] + pivot_xy
    new_pos = np.concatenate([new_xy, root_pos[:, 2:3]], axis=1)
    new_quat = (rot * R.from_quat(root_quat)).as_quat()
    return new_pos, new_quat


def append_cycle(seq, cycle, delta, correct_yaw):
    """ONE greedy append: align the steady cycle onto seq's MEASURED end,
    optionally counter-rotate the per-cycle yaw drift, then crossfade-bridge
    the seam. Shared by assemble_straight() and generate()'s production loop
    (the two loops differ only in their stop rules, which stay separate).

    Returns (seq, used_window, seam_index, raw_pose_mismatch), where
    seam_index == len(seq_in[0]) is where crossfade INSERTED the bridge."""
    # align fresh off the measured end-of-sequence-so-far (never pre-composed
    # analytically -- structural anti-drift guard per plan).
    # align_z=False: same clip, flat ground -- re-anchoring z to prev[-1] bakes
    # in a -4.08 mm/seam sink (see align_segment's docstring)
    seg_dof, seg_pos, seg_quat = align_segment(seq[1], seq[2], *cycle, align_z=False)
    if correct_yaw:
        # counter-rotate this repetition by -delta about the seam point
        seg_pos, seg_quat = rotate_about_z(seg_pos, seg_quat, -delta, seg_pos[0, :2])
    raw_mismatch = float(np.abs(seq[0][-1] - seg_dof[0]).max())
    seam = len(seq[0])
    *out, used_w = crossfade(*seq, seg_dof, seg_pos, seg_quat, CROSSFADE_WINDOW_CAP)
    return tuple(out), used_w, seam, raw_mismatch


# ---- Step 2 proper: the isolated straight-line self-test scaffold ----

def assemble_straight(target_distance):
    """SELF-TEST SCAFFOLDING ONLY -- nothing on the production path calls this.
    It is invoked by the __main__ blocks of build_straight, turn_injector and
    stop_tail, and it is the only isolated exercise of the seam machinery with
    the turn injector and stop tail out of the way. Its greedy loop is a
    DISTANCE-ONLY sibling of generate()'s loop (whose stop rule predicts the
    stop tail's vector landing point) -- two genuinely different stop rules;
    merging them would be an algorithm change.

    Greedy straight-line assembly: start segment, then one measured gait cycle
    at a time until within one cycle's distance of target_distance. Crude end
    (no decel/stance blend -- that's Step 3).

    Returns (dof_pos, root_pos, root_quat, stats) -- stats holds the self-test
    numbers (cycles used, measured displacement, per-seam dof discontinuities).
    """
    src = load_walk_source()
    a, b = src.seg["start_segment"]      # [0, 18): frame 0 -> first (R) heel strike
    ca, cb = src.seg["steady_cycle"]     # [52, 82): R-strike -> R-strike, phase-matched to b
    # use [a, ca) not [a, b) as the start block -- BUG FOUND BY USER ("first
    # right step slides"): b only reaches the FIRST heel strike, still
    # mid-acceleration; skipping [b, ca) and crossfading straight into the
    # already-cruising steady_cycle asks the bridge to invent a real ~0.6 m/s
    # acceleration out of nothing. Using the clip's own uncut ramp-up all the
    # way to ca makes the handoff two ADJACENT real frames (see
    # generate_walk_to_point.py's matching comment for the full story).
    cycle = (src.dof[ca:cb], src.pos[ca:cb], src.quat[ca:cb])
    cycle_dist = float(np.hypot(*src.seg["cycle_net_dxy"]))
    delta = src.seg["cycle_yaw_drift_rad"]  # per-cycle yaw drift from Step 0
    correct_yaw = abs(np.degrees(delta)) > YAW_CORRECTION_TRIGGER_DEG

    seq = (src.dof[a:ca].copy(), src.pos[a:ca].copy(), src.quat[a:ca].copy())
    origin_xy = seq[1][0, :2]

    # MEASURED FACT vs the plan's literal "< ~0.05 rad seam discontinuity":
    # walk_straight's same-phase heel-strike poses differ 0.06-0.28 rad between
    # cycles (start-segment end vs steady cycle: 0.28 rad, adjacent steady
    # cycles: 0.06-0.17 rad -- the recorded gait just isn't that periodic), and
    # the clip's own natural per-frame dof step reaches 0.33 rad mid-swing. So
    # a raw <0.05 rad pose-mismatch check is unsatisfiable by the source data
    # itself. Smallest faithful adaptation (documented, not a redesign): what
    # the shrunk window must guarantee is that the assembled output moves no
    # faster than the source gait ever does -- assert the max per-frame dof
    # step within each seam region exceeds the source clip's own natural max
    # per-frame step by < 0.05 rad. Raw mismatch is still logged. QA #1 and #5
    # apply the same excess-over-natural substitution.
    natural_max_step = float(np.abs(np.diff(src.dof, axis=0)).max())

    def displacement():
        return float(np.hypot(*(seq[1][-1, :2] - origin_xy)))

    n_cycles = 0
    seam_disc = []
    # greedy: stop appending full cycles once within one cycle's distance of
    # target. Displacement is MEASURED off the assembled sequence after every
    # append (crossfade seams consume real travel).
    while target_distance - displacement() > cycle_dist:
        seq, used_w, len_a, raw_mismatch = append_cycle(seq, cycle, delta, correct_yaw)
        # per-seam discontinuity: max per-frame dof step across the bridge
        # region (+/-2 frames margin). crossfade() INSERTS the bridge at
        # indices [len_a, len_a+used_w) (nothing removed, unlike the old
        # overlap-and-remove design this replaced; used_w is adaptive, see
        # crossfade()'s docstring) -- vs the source clip's own natural max step.
        lo = max(len_a - 2, 1)
        hi = min(len_a + used_w + 2, len(seq[0]))
        seam_step = float(np.abs(np.diff(seq[0][lo - 1:hi], axis=0)).max())
        seam_disc.append({"raw_mismatch": raw_mismatch, "seam_max_step": seam_step,
                          "excess_over_natural": seam_step - natural_max_step})
        n_cycles += 1

    stats = {
        "n_cycles": n_cycles,
        "n_frames": len(seq[0]),
        "target": target_distance,
        "measured_displacement": displacement(),
        "cycle_dist": cycle_dist,
        "yaw_correction_active": correct_yaw,
        "natural_max_step": natural_max_step,
        "seam_dof_discontinuity": seam_disc,
    }
    return seq[0], seq[1], seq[2], stats


if __name__ == "__main__":
    for target in (3.0, 8.0, 15.0):
        dof, pos, quat, st = assemble_straight(target)
        disc = st["seam_dof_discontinuity"]
        finite = all(np.isfinite(x).all() for x in (dof, pos, quat))
        print(f"\ntarget {target:5.1f} m: {st['n_cycles']} cycles, {st['n_frames']} frames "
              f"({st['n_frames']/30.0:.2f}s)")
        print(f"  measured displacement: {st['measured_displacement']:.3f} m "
              f"(undershoot {target - st['measured_displacement']:.3f} m, "
              f"cycle_dist {st['cycle_dist']:.3f} m)")
        print(f"  yaw correction active: {st['yaw_correction_active']}")
        print(f"  source natural max per-frame dof step: {st['natural_max_step']:.4f} rad")
        for i, s in enumerate(disc):
            print(f"  seam {i}: raw pose mismatch {s['raw_mismatch']:.4f} rad, "
                  f"seam max step {s['seam_max_step']:.4f} rad, "
                  f"excess over natural {s['excess_over_natural']:+.4f} rad (expect < 0.05)")
        print(f"  finite (no NaN/Inf): {finite}")
        assert finite
        # tiny overshoot possible now (start segment is longer -- covers more
        # real distance on its own -- see the [a,ca) fix above), lower bound
        # loosened from 0 to -0.1 to allow it; still catches a genuinely
        # broken greedy loop (large overshoot or a huge undershoot)
        assert -0.1 <= target - st["measured_displacement"] < st["cycle_dist"] + 0.05, "greedy stop condition violated"
        worst = max(s["excess_over_natural"] for s in disc)
        assert worst < 0.05, f"seam excess step {worst:.4f} >= 0.05 rad over natural"
    print("\nall self-tests passed")
