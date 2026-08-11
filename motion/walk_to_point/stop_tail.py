"""Step 3 of walk_to_point (PLAN.md): the FIXED decelerate-and-stop tail.

The v3 bug being fixed: truncating AT the final heel-strike (moment of maximum
~0.6m fore-aft foot split) and crossfading straight into a parallel stance
drags the rear foot ~0.5m along the ground during the blend at >1 m/s -- a
guaranteed QA-#2 (skate) failure. Fix per plan: let the motion continue about
half a swing PAST the final heel-strike (append one more aligned cycle to swing
through), timewarp_decelerate() through that final swing, and cut at FOOT
ADJACENCY (swing foot beside the stance foot, low, post-deceleration slow) --
from there the blend into the parallel stance only lowers a foot a few cm in
place. Without this paragraph the appended-cycle-then-cut structure looks like
pointless indirection.

append_stop() reads as: append a cycle -> find the adjacency cut -> truncate ->
timewarp -> settled yaw -> place the stop stance -> stance-blend -> region
bookkeeping. Each phase is a named helper below.

Run under `conda activate kimodo`, from the repo root.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from common import (  # noqa: E402
    FK_TRIM, FPS, STAIRS_HOLO, fk_contacts, foot_positions, foot_speed_xy,
    load_walk_npz, settled_heading, smoothstep, wrap, yaw_from_quat_xyzw,
)
from build_straight import (  # noqa: E402
    CROSSFADE_WINDOW_CAP, align_segment, crossfade, rotate_about_z,
)
from heel_strike import load_walk_source  # noqa: E402

STOP_CROSS_W = 12          # 0.4s -- heterogeneous seam (decel tail -> stance), plan: 0.3-0.5s
HOLD_FRAMES = 15           # ~0.5s final stance hold
SETTLE_FRAMES = 15         # ~0.5s window for the circular-mean settled yaw
DECEL_TAIL = 32            # ~1 cycle of pre-cut motion fed to the timewarp
DECEL_EXTEND = 2.2         # tail replayed over 2.2x frames (ease-out warp)
KNEE_DOF = {"L": 3, "R": 9}  # G1 29-dof order (left/right knee_joint)
# z tolerance ladder for the foot-adjacency cut search, then a best-score
# fallback -- see _find_adjacency_cut for the measurements behind both.
ADJACENCY_Z_LADDER = (0.05, 0.06, 0.08, 0.09, 0.11, 0.13)


def timewarp_decelerate(dof_pos, root_pos, root_quat, tail_frames, extend_factor):
    """Stretch the last `tail_frames` with an ease-out time warp -- same
    motion, replayed over more frames near the end, so cadence and forward
    speed both smoothly decrease.

    (Originally proven in the walk_climb_walk-era builder; the sqrt warp that
    version used is the bug fixed below.)"""
    T = dof_pos.shape[0]
    head_end = T - tail_frames
    head_dof, head_pos, head_quat = dof_pos[:head_end], root_pos[:head_end], root_quat[:head_end]
    tail_dof, tail_pos, tail_quat = dof_pos[head_end:], root_pos[head_end:], root_quat[head_end:]

    new_tail_len = int(round(tail_frames * extend_factor))
    old_t = np.arange(tail_frames, dtype=float)
    s = np.linspace(0.0, 1.0, new_tail_len)
    # INTEGRATION BUG FOUND BY QA (generate_walk_to_point): the original
    # sqrt(s) warp has infinite slope at s=0 -- the first warped frame
    # compressed ~3.7 source frames into one (measured: root speed 1.16 ->
    # 8.83 m/s, 0.80 rad dof step, at exactly the warp start, in every
    # generated clip). Replaced with 1-(1-s)^k, k=extend_factor: slope at
    # s=0 is k, so playback speed at the warp start is (tail-1)*k/(new-1)
    # ~= 1.0 (seamless with the unwarped head), easing out to a full stop
    # (slope 0) at the end -- strictly better for a decelerate-to-stand.
    # QA #3's bound text cites the 7.67 m/s/frame this produced; the two
    # references must stay consistent.
    new_t = (tail_frames - 1) * (1.0 - (1.0 - s) ** extend_factor)

    warped_dof = np.stack([np.interp(new_t, old_t, tail_dof[:, i]) for i in range(tail_dof.shape[1])], axis=1)
    warped_pos = np.stack([np.interp(new_t, old_t, tail_pos[:, i]) for i in range(3)], axis=1)
    slerp = Slerp(old_t, R.from_quat(tail_quat))
    warped_quat = slerp(new_t).as_quat()

    return (
        np.concatenate([head_dof, warped_dof]),
        np.concatenate([head_pos, warped_pos]),
        np.concatenate([head_quat, warped_quat]),
    )


def _stance_blend(dof_a, pos_a, quat_a, dof_b, pos_b, quat_b, window):
    """Dedicated OVERLAP-and-replace blend for the ONE tail->stance transition
    (the general straight-cycle seams use crossfade()'s INSERT-based bridge
    instead -- see build_straight.py's crossfade docstring). The two look like
    the same "blend two segments" function but solve opposite problems; here
    `window` is the exact width, in crossfade it is a cap.

    Why this needs its own logic: the stance foot must stay exactly PINNED
    throughout this transition (not just match at the two endpoints) --
    dof_b/pos_b are already set up so the stance foot's absolute position
    matches the tail's planted foot exactly (see _place_stop_stance's
    foot-pinning step). crossfade()'s velocity-eased root bridge doesn't know
    about that constraint -- it eases PELVIS velocity smoothly, but the stance
    leg's OWN joint angles are also interpolating at the same time (decel pose
    -> stance pose), and pelvis motion + leg motion don't cancel out to keep
    the foot fixed -- measured a real 23 cm/s stance-foot drag from this
    combination.

    Since timewarp_decelerate() has already brought speed down near zero by
    this point, a PLAIN smoothstep position blend (the thing that caused
    the ghost-step bug for actively-walking seams) is actually fine here --
    both curves are nearly stationary, so there's no "two things
    progressing" compression to create a speed spike. Blending two
    near-static curves keeps the result near-static too."""
    w = smoothstep(window)[:, None]
    dof = (1 - w) * dof_a[-window:] + w * dof_b[:window]
    pos = (1 - w) * pos_a[-window:] + w * pos_b[:window]
    rots_a = R.from_quat(quat_a[-window:])
    rots_b = R.from_quat(quat_b[:window])
    quat = np.stack(
        [Slerp([0, 1], R.concatenate([rots_a[i], rots_b[i]]))(w[i, 0]).as_quat() for i in range(window)]
    )
    return (
        np.concatenate([dof_a[:-window], dof, dof_b[window:]]),
        np.concatenate([pos_a[:-window], pos, pos_b[window:]]),
        np.concatenate([quat_a[:-window], quat, quat_b[window:]]),
    )


def pick_stop_stance():
    """Plan's preference order: walk_straight's own frame 0 IF genuinely
    standing/idle (Step 0 measured it is NOT: double-support but root already
    translating 0.151 m/s), else stair_climbing frame 0 (already used as the
    planted stance by build_walk_climb_walk). Checked live, not hardcoded."""
    walk, raw = load_walk_npz()
    fc0 = raw["foot_contacts"][0]
    idle = (
        bool(np.all(fc0))
        and np.linalg.norm(walk["ref_global_velocity"][0, 0, :2]) < 0.05
        and np.abs(walk["ref_dof_vel"][0]).max() < 0.1
    )
    src = walk if idle else np.load(STAIRS_HOLO, allow_pickle=True)
    name = "walk_straight[0]" if idle else "stair_climbing[0]"
    return (src["ref_dof_pos"][0], src["ref_global_translation"][0, 0, :],
            src["ref_global_rotation_quat"][0, 0, :], name)


def stance_sanity_check(stance_dof, stance_pos, stance_quat, name):
    """ONE-TIME check (plan Step 3), invoked from __main__ only -- not per
    generation: stop-stance lateral width, pelvis height and knee angles vs
    walk_straight's own double-support frames. A crouched stance (pelvis/knee
    mismatch) would visibly sink during the stop blend."""
    walk, raw = load_walk_npz()
    ds = np.flatnonzero(raw["foot_contacts"].all(axis=1))  # double-support frames (0-3 per Step 0)

    def metrics(dof, pos, quat, fl, fr):
        h = yaw_from_quat_xyzw(quat)
        lat = np.array([-np.sin(h), np.cos(h)])
        width = abs((fl[:2] - fr[:2]) @ lat)
        pelvis_h = pos[2] - min(fl[2], fr[2])
        return width, pelvis_h, dof[KNEE_DOF["L"]], dof[KNEE_DOF["R"]]

    src = load_walk_source()  # stored FK: (T, 30, 3), ankle_roll links = 6/12
    walk_m = np.array([
        metrics(src.dof[f], src.pos[f], src.quat[f], src.foot_l[f], src.foot_r[f])
        for f in ds
    ]).mean(axis=0)
    sl, sr = foot_positions(stance_dof[None], stance_pos[None], stance_quat[None])
    stance_m = np.array(metrics(stance_dof, stance_pos, stance_quat, sl[0], sr[0]))

    labels = ["stance width (m)", "pelvis height (m)", "L knee (rad)", "R knee (rad)"]
    print(f"one-time stop-stance sanity check: {name} vs walk_straight double-support (frames {ds.tolist()}):")
    flags = []
    for lab, sv, wv in zip(labels, stance_m, walk_m):
        print(f"  {lab:18s} stance {sv:+.4f}  walk {wv:+.4f}  delta {sv - wv:+.4f}")
    if abs(stance_m[0] - walk_m[0]) > 0.03:
        flags.append("stance width differs > 3cm")
    if abs(stance_m[1] - walk_m[1]) > 0.03:
        flags.append("pelvis height differs > 3cm (visible sink/rise during blend)")
    if max(abs(stance_m[2] - walk_m[2]), abs(stance_m[3] - walk_m[3])) > 0.2:
        flags.append("knee angle differs > 0.2 rad (crouched-stance mismatch)")
    print(f"  flags: {flags if flags else 'none'}")
    return dict(zip(labels, (stance_m - walk_m).tolist())), flags


# ---- append_stop's phases ----

def _append_stop_cycle(dof, pos, quat):
    """Continue past the final heel strike: append one more aligned steady
    cycle. The input ends AT a strike (guaranteed by the greedy loops, which
    append whole strike->strike cycles); the appended cycle provides the
    following swing to cut inside -- the whole point of the fix is that the cut
    is never at the strike.

    Returns (dof_e, pos_e, quat_e, seam_start, used_w, ground, (fl, fr), con)."""
    src = load_walk_source()
    ca, cb = src.seg["steady_cycle"]
    cycle = (src.dof[ca:cb], src.pos[ca:cb], src.quat[ca:cb])

    # crossfade() INSERTS the bridge starting exactly at len(dof) (no frames
    # removed from dof's own trailing end, unlike the old overlap-and-remove
    # design this replaced -- see build_straight.py's crossfade docstring).
    seam_start = len(dof)
    # align_z=False: same-clip cycle append -- matching z to the sequence's
    # last frame (a different gait phase) shifts the whole stop cycle down by
    # the measured -4.08 mm phase gap; enough on its own to produce -5.76 mm
    # ground penetration at short targets like (1.8,0)/(2,0) (QA #6 fail,
    # bound -5 mm). See align_segment's docstring.
    seg = align_segment(pos, quat, *cycle, align_z=False)
    dof_e, pos_e, quat_e, used_w = crossfade(dof, pos, quat, *seg, CROSSFADE_WINDOW_CAP)

    fl, fr = foot_positions(dof_e, pos_e, quat_e)
    ground = float(min(fl[:seam_start, 2].min(), fr[:seam_start, 2].min()))
    return dof_e, pos_e, quat_e, seam_start, used_w, ground, (fl, fr), fk_contacts(fl, fr)


def _find_adjacency_cut(quat_e, feet, con, ground, n_frames, seam_start, used_w, search_from):
    """Cut at FOOT ADJACENCY: the first single-support frame past the bridge
    where the swing foot's fore-aft separation from the stance foot is < 10cm
    and it is low. Returns (cut_frame, stance_foot).

    z tolerance 0.06 vs the plan's literal 2-5cm: we measure the ankle_roll
    LINK CENTER (sits higher than the contact point and rocks -- same
    measurement-point adaptation documented in common.py), strict pass tried
    first, minimal relaxation only if no frame qualifies. Ladder extended to
    0.13: with the insert-based crossfade the appended cycle's timing shifts
    slightly vs the old overlap-based one -- measured the best
    (lowest-separation) adjacency point now sits at swing height ~10.7-12cm
    (was 7.7-8.9cm under the old bridge), same link-center rock adaptation,
    just a different point in the swing.

    The search must SKIP PAST THE BRIDGE (pure interpolated frames, not real
    captured motion): crossfade() inserts at [seam_start, seam_start+used_w),
    so the appended cycle's own real frames start at seam_start+used_w -- and
    used_w is adaptive, not the CROSSFADE_WINDOW_CAP.

    The ladder handles the common case; heavy turn injection (many degrees of
    accumulated turn) measurably raises the swing height at the nearest-
    adjacency point (measured 13.2cm for a 75 deg total-turn case, vs 10.7-12cm
    untouched) -- past the ladder's max rung. Rather than keep chasing the
    ladder upward for every possible turn magnitude, fall back to the
    single-support frame with the best (lowest) combined separation+height
    score if no rung's threshold is met by anything. The fallback is not
    defensive padding -- it is what makes heavily-turned targets generate at
    all."""
    fl, fr = feet
    cut = stance_foot = None
    best = None  # (score, t, foot) fallback tracked across the whole search
    for z_tol in ADJACENCY_Z_LADDER:
        for t in range(seam_start + used_w + search_from, n_frames):
            L_, R_ = bool(con["L"][t]), bool(con["R"][t])
            if L_ == R_:
                continue  # need single support
            stance_xy, swing_p = (fr[t, :2], fl[t]) if R_ else (fl[t, :2], fr[t])
            h = yaw_from_quat_xyzw(quat_e[t])
            sep = abs((swing_p[:2] - stance_xy) @ np.array([np.cos(h), np.sin(h)]))
            z = swing_p[2] - ground
            if z_tol == ADJACENCY_Z_LADDER[0]:  # only need to build the fallback ranking once
                score = sep + max(z - 0.13, 0.0)  # penalize height only past the last rung
                if sep < 0.15 and (best is None or score < best[0]):
                    best = (score, t, "R" if R_ else "L")
            if sep < 0.10 and z < z_tol:
                cut, stance_foot = t, ("R" if R_ else "L")
                break
        if cut is not None:
            break
    if cut is None and best is not None:
        print(f"  note: no adjacency frame met the z_tol ladder; falling back to best "
              f"candidate t={best[1]} (score {best[0]:.3f})")
        cut, stance_foot = best[1], best[2]
    assert cut is not None, "no foot-adjacency frame found past the final heel strike"
    return cut, stance_foot


def _place_stop_stance(dof_w, pos_w, quat_w, stance_foot, settled, ground):
    """Place the parallel stop stance: xy via align_segment, yaw corrected to
    the settled facing, z via FK floor-snap, then the planted foot pinned.
    Returns (seg_dof, seg_pos, seg_quat, source_name)."""
    s_dof, s_pos, s_quat, source = pick_stop_stance()
    n = STOP_CROSS_W + HOLD_FRAMES
    seg_dof = np.repeat(s_dof[None], n, axis=0)
    seg_pos = np.repeat(s_pos[None], n, axis=0)
    seg_quat = np.repeat(s_quat[None], n, axis=0)
    # heterogeneous align (different source clip), so align_z stays True here;
    # z is floor-snapped just below anyway
    seg_dof, seg_pos, seg_quat = align_segment(pos_w, quat_w, seg_dof, seg_pos, seg_quat)
    seg_pos, seg_quat = rotate_about_z(
        seg_pos, seg_quat, wrap(settled - yaw_from_quat_xyzw(quat_w[-1])), seg_pos[0, :2])
    sl, sr = foot_positions(seg_dof[:1], seg_pos[:1], seg_quat[:1])
    seg_pos = seg_pos.copy()
    seg_pos[:, 2] += ground - min(sl[0, 2], sr[0, 2])  # lowest contact at actual ground level
    # -- pin the PLANTED stance foot: align_segment matches root xy, but the
    # stop-stance pose's foot sits ~5cm off in the root frame vs the tail's
    # planted foot -- measured to drag the planted foot 5.5cm (up to 22 cm/s)
    # through the blend. Translate the whole stance segment so its
    # corresponding foot xy coincides with the tail's planted foot instead;
    # the ~5cm root shift is invisible, the foot drag is not.
    # (sl/sr are read pre-shift on purpose: the z snap doesn't touch xy, and
    # the xy pin doesn't touch z.)
    tl, tr = foot_positions(dof_w[-1:], pos_w[-1:], quat_w[-1:])
    tail_foot = {"L": tl, "R": tr}[stance_foot][0, :2]
    seg_foot = {"L": sl, "R": sr}[stance_foot][0, :2]
    seg_pos[:, :2] += tail_foot - seg_foot
    return seg_dof, seg_pos, seg_quat, source


def append_stop(dof, pos, quat, search_from=2):
    """Append the decelerate-and-stop tail to an assembled walking sequence
    (which must END AT A HEEL STRIKE, per build_straight). Returns
    (dof, pos, quat, info).

    search_from: frames past the END OF THE CROSSFADE BRIDGE where the cut
    search starts (i.e. offset from seam_start + used_window, NOT from the
    seam). The appended cycle has TWO foot-adjacency moments (one per
    half-cycle, at ~+8 and ~+23 frames, advancing ~0.31 m resp. ~0.87 m --
    the two values generate_walk_to_point's STOP_TRAVELS encodes): the default
    2 finds the first; passing ~16 selects the second -- generate_walk_to_point
    picks whichever lands closer to the target (half-cycle stop granularity)."""
    dof_e, pos_e, quat_e, seam_start, used_w, ground, feet, con = _append_stop_cycle(dof, pos, quat)
    cut, stance_foot = _find_adjacency_cut(quat_e, feet, con, ground, len(dof_e),
                                           seam_start, used_w, search_from)

    dof_c, pos_c, quat_c = dof_e[:cut + 1], pos_e[:cut + 1], quat_e[:cut + 1]

    # -- decelerate through the final swing (tail spans ~1 cycle up to the cut)
    tail = min(DECEL_TAIL, len(dof_c) // 3)
    head_end = len(dof_c) - tail
    dof_w, pos_w, quat_w = timewarp_decelerate(dof_c, pos_c, quat_c, tail, DECEL_EXTEND)

    # -- settled facing over the last ~0.5s of the decelerated tail (see
    # common.settled_heading for why never the final frame alone)
    settled = settled_heading(quat_w, SETTLE_FRAMES)

    seg_dof, seg_pos, seg_quat, source = _place_stop_stance(
        dof_w, pos_w, quat_w, stance_foot, settled, ground)

    # _stance_blend (not the general crossfade()) -- this ONE transition
    # needs the stance foot to stay exactly pinned throughout, which plain
    # smoothstep position blending satisfies (both curves near-static after
    # decel) but crossfade()'s velocity-eased bridge does not -- see
    # _stance_blend's docstring.
    out_dof, out_pos, out_quat = _stance_blend(dof_w, pos_w, quat_w, seg_dof, seg_pos, seg_quat, STOP_CROSS_W)

    len_w = len(dof_w)
    # _stance_blend OVERLAPS/replaces the last STOP_CROSS_W frames of dof_w
    # (unlike crossfade()'s insert-based bridge) -- region bookkeeping
    # matches that: blend occupies [len_w-STOP_CROSS_W, len_w). qa_check reads
    # these exact indices by name.
    info = {
        "cut_frame": cut, "frames_past_seam": cut - seam_start, "seam_start": seam_start,
        "decel_tail_frames": tail, "decel_region": (head_end, len_w),
        "blend_region": (len_w - STOP_CROSS_W, len_w),
        "hold_region": (len_w, len_w + HOLD_FRAMES),
        "settled_yaw_deg": float(np.degrees(settled)), "stance_source": source,
        "stance_foot": stance_foot,
        "ground_z": ground,
    }
    return out_dof, out_pos, out_quat, info


def stop_skate_metrics(foot_l, foot_r, ground, blend_start, stance_foot):
    """The two contact-classification-INDEPENDENT stop checks, shared verbatim
    by this module's self-test and QA #2 so the two can never drift apart:

    - v_pin: planted stance foot's max horizontal speed at ANY frame from the
      blend to the end. A dragged foot can exceed contact_from_fk's speed gate
      and thereby vanish from the gated (contact-core) metric -- this catches
      that.
    - glide: swing foot's total xy travel while below 2cm height ("lower a foot
      a few cm IN PLACE", plan Step 3). MEASURED IRREDUCIBLE FLOOR: the gait's
      swing trajectory never passes closer than 9.7cm (at the chosen cut, its
      global minimum) to the stance's landing spot -- mid-swing lateral width
      20.8cm vs stance width 30.1cm -- so the blend must carry the foot ~10cm,
      of which the sub-2cm-height share is ~2.5cm, easing to zero speed (a
      settle, not a drag). Hence the 3.5cm bound and nothing tighter; the v3
      heel-strike-truncation bug measured ~50cm at 85-100 cm/s.

    Returns (v_pin, glide, swing_foot)."""
    speed = {"L": foot_speed_xy(foot_l), "R": foot_speed_xy(foot_r)}
    v_pin = float(speed[stance_foot][blend_start:].max())
    swing = "L" if stance_foot == "R" else "R"
    sf = {"L": foot_l, "R": foot_r}[swing]
    low = (sf[blend_start:, 2] - ground) < 0.02
    glide = float((speed[swing][blend_start:][low] / FPS).sum()) if low.any() else 0.0
    return v_pin, glide, swing


if __name__ == "__main__":
    from build_straight import assemble_straight

    s_dof, s_pos, s_quat, src = pick_stop_stance()
    print(f"stop stance source: {src}")
    stance_sanity_check(s_dof, s_pos, s_quat, src)

    dof, pos, quat, st = assemble_straight(5.0)
    print(f"\nbase sequence: {st['n_cycles']} cycles, {len(dof)} frames, "
          f"{st['measured_displacement']:.3f} m")
    dof2, pos2, quat2, info = append_stop(dof, pos, quat)
    print(f"with stop tail: {len(dof2)} frames ({len(dof2)/FPS:.2f}s); cut "
          f"{info['frames_past_seam']} frames past the final heel-strike seam "
          f"(~half a swing), decel tail {info['decel_tail_frames']} frames -> "
          f"x{DECEL_EXTEND}, settled yaw {info['settled_yaw_deg']:+.2f} deg, "
          f"stance {info['stance_source']}")

    # -- THE check that proves the fix: stance-foot skate through the stop
    # regions (heel-strike-truncation would have shown ~50-100 cm/s here)
    fl, fr = foot_positions(dof2, pos2, quat2)
    ground = info["ground_z"]
    con = fk_contacts(fl, fr)
    speed = {"L": foot_speed_xy(fl), "R": foot_speed_xy(fr)}
    # erode contact mask by FK_TRIM (3) frames -- same interval-core trimming
    # intent as heel_strike's FK cross-check: the ankle LINK rocks/settles
    # during the 1-3 touchdown transition frames while the contact point is
    # planted. NOTE np.roll WRAPS around the array, so this is NOT the same
    # operation as common.interval_core at the clip edges -- left exactly as
    # written (see PLAN.md's deviations note).
    # (The v3 drag bug this test guards stays detectable: a dragged foot fails
    # the two ungated checks below -- planted-foot any-frame speed and
    # swing-foot touchdown glide -- which don't depend on classification. That
    # is why there are three overlapping skate checks and not one.)
    core = {}
    for f in "LR":
        m = con[f].copy()
        for k in range(1, FK_TRIM + 1):
            m = m & np.roll(con[f], k) & np.roll(con[f], -k)
        core[f] = m

    def region_max_skate(a, b):
        vals = [speed[f][t] for f in "LR" for t in range(a, min(b, len(dof2))) if core[f][t]]
        return max(vals) if vals else 0.0

    da, db = info["decel_region"]
    ba, bb = info["blend_region"]
    ha, hb = info["hold_region"]
    v_decel, v_blend, v_hold = region_max_skate(da, db), region_max_skate(ba, bb), region_max_skate(ha, hb)
    print(f"\nstance-foot skate (max xy link speed while in contact):")
    print(f"  decel region  [{da},{db}):  {v_decel*100:6.2f} cm/s (walking, warped down)")
    print(f"  blend region  [{ba},{bb}):  {v_blend*100:6.2f} cm/s  <-- the fixed-bug region")
    print(f"  hold region   [{ha},{hb}):  {v_hold*100:6.2f} cm/s")
    v_stop = max(v_blend, v_hold)
    assert v_stop < 0.03, f"stop-blend skate {v_stop*100:.2f} cm/s >= 3 cm/s"
    v_pin, glide, swing = stop_skate_metrics(fl, fr, ground, ba, info["stance_foot"])
    print(f"  planted {info['stance_foot']} foot, ANY frame in blend..end: {v_pin*100:6.2f} cm/s")
    assert v_pin < 0.03, f"planted-foot drag {v_pin*100:.2f} cm/s >= 3 cm/s"
    print(f"  swing {swing} foot touchdown glide below 2cm: {glide*100:6.2f} cm total")
    assert glide < 0.035, f"swing-foot touchdown glide {glide*100:.2f} cm >= 3.5 cm"

    # -- ground penetration through decel + blend + hold specifically
    min_z = float(min(fl[da:, 2].min(), fr[da:, 2].min()))
    print(f"ground penetration (decel..end): min foot z - ground = "
          f"{(min_z - ground)*1000:+.2f} mm (bound -5 mm)")
    assert min_z - ground >= -0.005

    assert all(np.isfinite(x).all() for x in (dof2, pos2, quat2))
    print("\nall self-tests passed")
