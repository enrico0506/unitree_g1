"""walk_to_point end-to-end generator (PLAN.md v5, full assembly).

    python motion/walk_to_point/generate_walk_to_point.py --dx 5 --dy 2 --output out.npz

(dx, dy) is defined in the robot's START BODY FRAME (x forward, y left) --
pinned per the plan's "Critical finding": deployment re-anchors the clip's
heading to the robot's real heading at playback start, so clip frame-0 pose
== the robot's actual pose, and only relative motion survives.

Flow: start segment -> unified greedy loop (append one measured gait cycle,
then if the settled heading is outside a trim band of the bearing-to-target
inject up to 2 capped stance-pivot turn steps into that cycle's single-support
windows -- measure, never precompute; this also self-corrects the gait's
sub-trigger yaw drift while walking) -> append_stop() -> shared FK pipeline.

Feasibility (plan: "not a naive hypot check"): the greedy loop IS the
feasibility check along the actual assembled path -- a target the turn's own
path length can spiral to is fine (walking U-turn); a target the machinery
cannot converge on (orbit inside the turning radius / inside the minimum
start+stop footprint) hits the iteration cap or the post-assembly miss bound
and is rejected explicitly instead of producing a degenerate clip.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from common import (  # noqa: E402
    FPS, circ_mean, rot2, settled_heading, wrap, yaw_from_quat_xyzw,
)
from heel_strike import load_walk_source  # noqa: E402
from build_straight import append_cycle  # noqa: E402
from turn_injector import (  # noqa: E402
    TURN_CAP_DEG, foot_positions, inject_turn, single_support_windows,
)
from stop_tail import append_stop  # noqa: E402
from pkl_to_offline_npz import (  # noqa: E402
    angular_velocity_from_quats, central_diff, convert_legacy_offline_npz,
    forward_kinematics_all_frames,
)

TRIM_BAND_DEG = 2.0   # stop turning once |heading error| is inside this band
HEADING_SAMPLE = 20   # frames (~2/3 cycle) averaged for the settled heading
STOP_GATE_DEG = 30.0  # vector stop rule only inside this heading error (see below)
RESIDUAL_TURN_GUARD_M = 1.2  # no turns closer than this to the target (see below)
# append_stop() advances by one of two measured amounts, depending on which
# foot-adjacency the cut lands on (first or second half-cycle of the appended
# stop cycle) -- half-cycle stop granularity, picked to minimize arrival miss
STOP_TRAVELS = (0.315, 0.859)   # (search_from=2, search_from=16)
STOP_SEARCH_FROM = (2, 16)
MAX_ITERS = 60        # feasibility cap: non-converging (orbiting) targets rejected here
MISS_REJECT = 0.40    # post-assembly hard bound == QA #7's arrival bound: a bigger
                      # miss means the target sits inside the minimum start+stop
                      # footprint -- reject explicitly per the plan, don't emit it.
                      # Was 0.45 ("just above" QA #7's 0.4) -- MEASURED GAP: (1.2,0)
                      # emitted a clip with miss 0.401 m that then failed its own QA
                      # #7. The 0.40-0.45 band is reachable (worst-case stop-lattice
                      # quantization miss is ~0.35-0.40 m), so the bounds must match
                      # EXACTLY -- generate() must never emit a clip its own QA
                      # rejects. Any future edit to either bound must move both.


def _stop_landing_miss(tvec, tdir, travel):
    """Predicted 2D arrival miss if the stop tail runs now and carries the
    robot `travel` metres along its current travel direction.

    BUG FOUND BY STRESS TESTING: modelling the stop as a SCALAR
    |residual - stop_travel| mispredicts the landing point -- the stop actually
    travels a VECTOR along the current travel direction. Measured on (5,5): the
    scalar rule broke at residual 0.111 m predicting a 0.204 m miss; the actual
    miss was 0.396 m (0.374 along-track -- the stop carried the robot 0.3+ m
    PAST a target it was nearly on top of)."""
    return float(np.linalg.norm(tvec - travel * tdir))


def generate(dx, dy, verbose=True):
    """Assemble the full walk-to-(dx,dy) motion. Returns (dof, pos, quat, info)."""
    # upfront gate: a target closer than the arrival tolerance itself is below
    # the machinery's resolution -- the post-assembly miss bound can NOT catch
    # it, because a loop-around path lands back near the start and therefore
    # "arrives" (measured: (0.1, 0) emitted a 533-frame, 17.8 s wander that
    # ends 0.21 m from a point 0.1 m ahead). Reject explicitly instead.
    if float(np.hypot(dx, dy)) <= MISS_REJECT:
        raise SystemExit(
            f"infeasible target ({dx}, {dy}): {np.hypot(dx, dy):.2f} m from the start is at or "
            f"below the arrival tolerance ({MISS_REJECT} m) -- inside the machinery's minimum "
            f"start+stop footprint (the robot is effectively already there; any assembled path "
            f"would be a loop-around wander, not a walk to the point)")

    src = load_walk_source()
    a, b = src.seg["start_segment"]
    ca, cb = src.seg["steady_cycle"]
    cycle = (src.dof[ca:cb], src.pos[ca:cb], src.quat[ca:cb])
    delta = src.seg["cycle_yaw_drift_rad"]
    correct_yaw = abs(np.degrees(delta)) > 1.0  # build_straight's trigger (inactive for this clip)

    # the gait travels slightly off its facing (~-3 deg for this clip): aim the
    # HEADING at bearing - travel_offset so the TRAVEL points at the target.
    # Removing this makes every target land systematically off to one side
    # while all six local-smoothness checks still pass.
    yaws_src = R.from_quat(src.quat).as_euler("xyz")[:, 2]
    dxy_c = src.seg["cycle_net_dxy"]
    travel_off = wrap(np.arctan2(dxy_c[1], dxy_c[0]) - circ_mean(yaws_src[ca:cb]))

    # BUG FOUND BY USER (viewer: "first right step slides", still visible
    # after 3 rounds of tuning the crossfade bridge math): start_segment
    # only runs to the FIRST heel strike (frame 18), still mid-acceleration
    # (raw clip keeps speeding up smoothly to frame ~50+), while steady_cycle
    # begins at frame 52 already at cruise speed -- crossfade was being asked
    # to invent, out of nothing, a real ~0.6 m/s acceleration that the person
    # actually performed over the SKIPPED frames [18,52). No amount of
    # interpolation math can make synthetic frames look like that real
    # acceleration. Real fix: use those frames instead of skipping them --
    # b -> ca means the "start" block runs the clip's own uncut ramp-up all
    # the way to where the steady cycle begins, so the handoff into the
    # first cycle copy is two ADJACENT frames of the same real recording
    # (already continuous, no bridge needed) instead of two different
    # strides ~1.1s apart in different phases of acceleration.
    seq = (src.dof[a:ca].copy(), src.pos[a:ca].copy(), src.quat[a:ca].copy())
    inj = np.zeros(ca - a)  # cumulative injected turn yaw per frame (QA #4 bookkeeping)
    seams, turn_log = [], []
    yaw0 = yaw_from_quat_xyzw(seq[2][0])
    start_xy = seq[1][0, :2].copy()
    target = start_xy + rot2(yaw0) @ np.array([dx, dy], dtype=float)

    cap = np.radians(TURN_CAP_DEG)
    append_net = 0.85  # updated with the measured net of each append
    converged = False
    for it in range(MAX_ITERS):
        tvec = target - seq[1][-1, :2]
        residual = float(np.linalg.norm(tvec))
        # greedy stop rule: stop as soon as no further append can land a stop
        # variant closer to the target than the best variant lands right now.
        # The vector prediction (see _stop_landing_miss) is GATED to
        # |heading err| < 30 deg -- ungated, it fires the break while still
        # turning toward the target (measured: (0,5) rejected at miss 5.28 m);
        # while turning, the scalar rule is the safe fallback.
        head_now = settled_heading(seq[2], HEADING_SAMPLE)
        err_now = wrap(np.arctan2(tvec[1], tvec[0]) - travel_off - head_now)
        tdir = np.array([np.cos(head_now + travel_off), np.sin(head_now + travel_off)])
        if abs(err_now) < np.radians(STOP_GATE_DEG):
            best_now = min(_stop_landing_miss(tvec, tdir, tv) for tv in STOP_TRAVELS)
            best_next = min(_stop_landing_miss(tvec, tdir, append_net + tv)
                            for tv in STOP_TRAVELS)
        else:
            best_now = min(abs(residual - tv) for tv in STOP_TRAVELS)
            best_next = min(abs(residual - append_net - tv) for tv in STOP_TRAVELS)
        if best_now <= best_next:
            converged = True
            break

        old_end = seq[1][-1, :2].copy()
        # crossfade() INSERTS the bridge at [old_len, old_len+used_w) rather
        # than overlapping/removing frames (see build_straight.py's crossfade
        # docstring for why) -- the seam marker is old_len, not old_len-W.
        # used_w is adaptive (usually much less than the CROSSFADE_WINDOW_CAP
        # -- see crossfade()'s docstring for the dilution bug this fixes).
        seq, used_w, old_len, _ = append_cycle(seq, cycle, delta, correct_yaw)
        seams.append(old_len)
        inj = np.concatenate([inj, np.full(len(seq[0]) - len(inj), inj[-1])])
        append_net = float(np.linalg.norm(seq[1][-1, :2] - old_end))

        # greedy re-aim: settled heading vs bearing-to-target, both from the
        # MEASURED state of the sequence so far
        head = settled_heading(seq[2], HEADING_SAMPLE)
        tvec = target - seq[1][-1, :2]
        err = wrap(np.arctan2(tvec[1], tvec[0]) - travel_off - head)
        # residual guard: within ~1.2 m the bearing is ill-conditioned (it
        # swings wildly as the robot passes abeam) and the stop is imminent --
        # a turn there can't improve arrival, only twist the settled stance
        if abs(err) > np.radians(TRIM_BAND_DEG) and np.linalg.norm(tvec) > RESIDUAL_TURN_GUARD_M:
            fl, fr = foot_positions(*seq)
            windows = single_support_windows(fl, fr)
            # windows inside the just-appended cycle. crossfade() inserts the
            # bridge at [old_len, old_len+used_w), so the new cycle's own
            # frames start at old_len+used_w (not old_len -- that was the old
            # overlap-and-remove design's indexing; used_w not the cap, since
            # it's now adaptive). The clip-edge window is usable too: the
            # smoothstep ramp has zero angular rate at its end, so the next
            # append's crossfade blends over a near-constant heading
            usable = [i for i, (s, e, f) in enumerate(windows)
                      if s >= old_len + used_w]
            n_av = min(2, len(usable))
            if n_av:
                dpsi = np.sign(err) * min(abs(err), n_av * cap * 0.999)
                d_, p_, q_, tinfo = inject_turn(*seq, dpsi, skip_windows=usable[0])
                seq = (d_, p_, q_)
                inj = inj + tinfo["theta_profile"]
                turn_log.append({"iter": it, "dpsi_deg": float(np.degrees(dpsi)),
                                 "windows": tinfo["ramp_windows"]})
                if verbose:
                    print(f"  iter {it}: residual {residual:.2f} m, heading err "
                          f"{np.degrees(err):+.1f} deg -> turn {np.degrees(dpsi):+.1f} deg")
    if not converged:
        raise SystemExit(
            f"infeasible target ({dx}, {dy}): greedy assembly did not converge in "
            f"{MAX_ITERS} cycles (target unreachable along any path this machinery produces)")

    pre_stop_end = seq[1][-1, :2].copy()
    n_cycles = len(seams)
    # variant selection is ALWAYS vector (unlike the break rule it needs no
    # gate -- picking the variant whose predicted 2D landing point is closest
    # is safe at any heading). Measured: (1.8,0) picked variant 1 under the
    # scalar rule (miss 0.366); vector selection picks variant 0 (miss 0.203).
    # LOAD-BEARING: tvec/tdir here hold the BREAK ITERATION's values (nothing
    # mutates the sequence in that iteration). A refactor that recomputes them
    # elsewhere must prove the result is bit-identical.
    variant = int(np.argmin([_stop_landing_miss(tvec, tdir, tv) for tv in STOP_TRAVELS]))
    dof, pos, quat, sinfo = append_stop(*seq, search_from=STOP_SEARCH_FROM[variant])
    sinfo["stop_variant"] = variant
    inj = np.concatenate([inj, np.full(len(dof) - len(inj), inj[-1])])

    miss_vec = pos[-1, :2] - target
    miss = float(np.linalg.norm(miss_vec))
    if miss > MISS_REJECT:
        raise SystemExit(
            f"infeasible target ({dx}, {dy}): assembled path ends {miss:.2f} m from the "
            f"target (> {MISS_REJECT} m) -- inside the machinery's minimum start+stop footprint")

    # info carries exactly what qa_check needs. "inj" must stay a PER-FRAME
    # cumulative injected-yaw array padded to the FULL final length -- QA #4
    # indexes it by seam frame.
    info = {
        "target_world": target, "start_xy": start_xy, "yaw0": yaw0,
        "seams": seams, "inj": inj, "turn_log": turn_log, "stop_info": sinfo,
        "n_cycles": n_cycles, "residual_at_break": residual,
        "pre_stop_end": pre_stop_end, "miss_m": miss, "miss_vec": miss_vec,
        "total_turn_deg": float(np.degrees(inj[-1])),
        "travel_off_deg": float(np.degrees(travel_off)),
    }
    if verbose:
        print(f"assembled ({dx:+.1f},{dy:+.1f}): {n_cycles} cycles + stop = {len(dof)} frames "
              f"({len(dof)/FPS:.1f}s), {len(turn_log)} turn injections totalling "
              f"{info['total_turn_deg']:+.1f} deg, arrival miss {miss:.3f} m")
    return dof, pos, quat, info


def write_npz(dof, pos, quat, out_path, motion_key, source_desc):
    """Shared FK/central-diff/legacy-npz output pipeline (Step 4; same as every
    other tool this session). All stages produce frames on the same
    index-uniform 30fps grid (timewarp_decelerate RESAMPLES onto uniform frames
    via interp, it carries no timestamps), so uniform dt is structural -- the
    assertion below is the only place the npz format's silent assumption is
    checked anywhere in the pipeline."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / FPS
    gt, gr = forward_kinematics_all_frames(pos.astype(float), quat.astype(float),
                                           dof.astype(float))
    dof_vel = central_diff(dof, dt)
    gv = central_diff(gt, dt)
    gav = angular_velocity_from_quats(gr, dt)
    T = len(dof)
    assert all(len(x) == T for x in (pos, quat, gt, gr, dof_vel, gv, gav)), \
        "output arrays not frame-aligned (uniform 30fps dt violated)"
    assert all(np.isfinite(x).all() for x in (dof, pos, quat, gt, gv, gav))

    legacy = out_path.with_suffix(".legacy.npz")
    metadata = {"motion_key": motion_key, "motion_fps": FPS,
                "original_num_frames": T, "source": source_desc}
    np.savez(
        legacy,
        metadata=np.asarray(json.dumps(metadata)),
        dof_pos=dof.astype(np.float32),
        dof_vels=dof_vel.astype(np.float32),
        global_translation=gt.astype(np.float32),
        global_rotation_quat=gr.astype(np.float32),
        global_velocity=gv.astype(np.float32),
        global_angular_velocity=gav.astype(np.float32),
    )
    result = convert_legacy_offline_npz(legacy, out_path, overwrite=True)
    legacy.unlink()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, required=True, help="forward meters, start body frame")
    ap.add_argument("--dy", type=float, required=True, help="left meters, start body frame")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    dof, pos, quat, info = generate(args.dx, args.dy)
    write_npz(dof, pos, quat, args.output, f"walk_to_{args.dx}_{args.dy}",
              f"walk_to_point generator: target ({args.dx}, {args.dy}) m in start body frame, "
              f"{info['n_cycles']} cycles, {info['total_turn_deg']:+.1f} deg turned, "
              f"miss {info['miss_m']:.3f} m")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
