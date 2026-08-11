"""walk_to_point QA gate (PLAN.md, all 7 checks) run against 3 real generated
targets: (5,0) straight, (5,5) 45-deg left, (8,-3) shallower right.

Where the plan's literal thresholds are unsatisfiable by the SOURCE clip
itself (documented measured facts from Steps 0-3, not fudges), the check
bounds the assembled output's EXCESS over the source clip's own natural
values instead, and both numbers are printed:
- #1/#5: source's own per-frame dof step reaches 0.33 rad (vs plan's 0.15) ->
  bound excess over natural max by 0.05 rad.
- #2: the ankle_roll LINK CENTER rocks 5-11.5 cm/s during contact cores in
  the untouched source (vs plan's 2-3 cm/s point-contact ideal) -> bound
  excess over source baseline by 1.5 cm/s; the stop blend/hold (near-static)
  IS still held to the absolute 3 cm/s.
This paragraph is what keeps a future reader from "restoring" the plan's
literal thresholds; PLAN.md's DEVIATIONS section lists all of them.

Run under kimodo env, from the repo root.
"""

import sys
from pathlib import Path

import numpy as np

from common import (  # noqa: E402
    FPS, contact_core_max_speed, foot_positions, wrap, yaw_from_quat_xyzw,
)
from heel_strike import load_walk_source  # noqa: E402
from build_straight import CROSSFADE_WINDOW_CAP  # noqa: E402
from stop_tail import stop_skate_metrics  # noqa: E402
from generate_walk_to_point import generate, write_npz  # noqa: E402

OUT_DIR = Path("motion/walk_to_point/generated")


def source_reference():
    """Source clip's own natural values -- the baselines the assembled output
    must not exceed (see module docstring)."""
    src = load_walk_source()
    fl, fr = src.foot_l, src.foot_r
    v = np.linalg.norm(np.diff(src.pos[:, :2], axis=0), axis=1) * FPS
    ground = min(fl[:, 2].min(), fr[:, 2].min())
    return {
        "dof_step": float(np.abs(np.diff(src.dof, axis=0)).max()),
        "v_jump": float(np.abs(np.diff(v)).max()),
        "skate": contact_core_max_speed(fl, fr),
        "clearance": float(max(fl[:, 2].max(), fr[:, 2].max()) - ground),
    }


def run_qa(dof, pos, quat, info, src):
    results = []
    si = info["stop_info"]
    fl, fr = foot_positions(dof, pos, quat)
    ground = si["ground_z"]
    head_end = si["decel_region"][0]
    ba, bb = si["blend_region"]
    ha, hb = si["hold_region"]
    seams = [s for s in info["seams"] if s + CROSSFADE_WINDOW_CAP + 3 < head_end]

    # 1 -- max per-frame dof delta
    steps = np.abs(np.diff(dof, axis=0))
    mx, l2 = float(steps.max()), float(np.linalg.norm(np.diff(dof, axis=0), axis=1).max())
    ok = mx <= src["dof_step"] + 0.05
    results.append(("1 dof delta", ok,
                    f"max/joint {mx:.4f} rad (source natural {src['dof_step']:.4f}, "
                    f"excess {mx - src['dof_step']:+.4f} < 0.05; plan literal 0.15; L2 {l2:.3f})"))

    # 2 -- stance-foot skate, FK-derived contacts on the assembled output.
    # The gated contact-core metric plus the two contact-classification-
    # independent stop metrics (shared verbatim with stop_tail's self-test via
    # stop_skate_metrics -- see its docstring for the measured glide floor).
    skate = contact_core_max_speed(fl, fr)
    v_pin, glide, _swing = stop_skate_metrics(fl, fr, ground, ba, si["stance_foot"])
    ok = (skate <= src["skate"] + 0.015) and v_pin < 0.03 and glide < 0.035
    results.append(("2 skate", ok,
                    f"contact-core max {skate*100:.2f} cm/s (source baseline {src['skate']*100:.2f}, "
                    f"excess {(skate-src['skate'])*100:+.2f} < 1.5); planted {si['stance_foot']} foot "
                    f"blend..end {v_pin*100:.2f} cm/s (< 3 abs); swing touchdown glide "
                    f"{glide*100:.2f} cm (< 3.5)"))

    # 3 -- root-speed continuity: no step discontinuity anywhere. MEASURED
    # STRUCTURAL FACT: each crossfade seam consumes real travel (the plan's
    # own seam-travel model, ~0.03 m/frame x window) which plays out as a
    # C1-smooth smoothstep speed surge inside the blend (peak dv/frame =
    # 6*G*FPS/W^2 with G = W*v/FPS the overlapped travel; feet stay planted
    # through it -- check #2). A DISCONTINUITY is a dv exceeding that modeled
    # surge: the sqrt-warp bug this caught measured 7.7 m/s/frame vs the
    # model's ~1.2 bound. Checked globally, not just at seams.
    v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * FPS
    jumps = np.abs(np.diff(v))
    G = CROSSFADE_WINDOW_CAP * float(np.median(v[:head_end])) / FPS
    bound = 6.0 * G * FPS / CROSSFADE_WINDOW_CAP ** 2 + src["v_jump"]
    seam_jumps = [float(jumps[max(s - 3, 0):s + CROSSFADE_WINDOW_CAP + 3].max()) for s in seams]
    seam_jumps.append(float(jumps[max(ba - 3, 0):min(bb + 3, len(jumps))].max()))
    worst_j, glob_j = max(seam_jumps), float(jumps.max())
    ok = glob_j <= bound
    results.append(("3 root-speed cont.", ok,
                    f"max dv {glob_j:.3f} m/s/frame global, {worst_j:.3f} at seams, vs modeled "
                    f"seam-surge bound {bound:.3f} (natural {src['v_jump']:.3f}; a pop like the "
                    f"fixed sqrt-warp bug measured 7.67)"))

    # 4 -- cumulative root yaw at each seam, minus the INJECTED turn, stays flat
    # (residual = the gait's own drift; a ramp steeper than the 1-deg/cycle
    # correction trigger means delta was measured wrong)
    yaw_res = [np.degrees(wrap(yaw_from_quat_xyzw(quat[s]) - info["inj"][s] - info["yaw0"]))
               for s in seams]
    # slope fit over SAME-PHASE seams only: seams[0] ends the start segment at
    # a different gait phase, and root-yaw sway is phase-locked (+/-3-4 deg) --
    # including it fakes a ramp with few seams. All values still printed.
    fit = yaw_res[1:] if len(yaw_res) > 2 else yaw_res
    slope = float(np.polyfit(range(len(fit)), fit, 1)[0]) if len(fit) > 1 else 0.0
    ok = abs(slope) <= 1.2
    results.append(("4 yaw flat at seams", ok,
                    f"residual yaw at seams {['%+.1f' % y for y in yaw_res]} deg, "
                    f"slope {slope:+.2f} deg/seam (|slope| < 1.2 = the correction trigger)"))

    # 5 -- per-seam dof discontinuity (excess over source natural, per build_straight)
    seam_steps = [float(steps[max(s - 3, 0):s + CROSSFADE_WINDOW_CAP + 2].max()) for s in seams]
    seam_steps.append(float(steps[max(ba - 3, 0):min(bb + 2, len(steps))].max()))
    worst_s = max(seam_steps)
    ok = worst_s - src["dof_step"] < 0.05
    results.append(("5 seam dof disc.", ok,
                    f"worst seam max dof step {worst_s:.4f} rad, excess over natural "
                    f"{worst_s - src['dof_step']:+.4f} < 0.05 (plan literal 0.05 abs)"))

    # 6 -- ground penetration + swing clearance, stop regions specifically too
    min_z = float(min(fl[:, 2].min(), fr[:, 2].min())) - ground
    stop_min = float(min(fl[si["decel_region"][0]:, 2].min(),
                         fr[si["decel_region"][0]:, 2].min())) - ground
    clear = float(max(fl[:, 2].max(), fr[:, 2].max())) - ground
    ok = min_z >= -0.005 and clear <= src["clearance"] + 0.02
    results.append(("6 ground/clearance", ok,
                    f"min foot z-ground {min_z*1000:+.2f} mm global, {stop_min*1000:+.2f} mm in "
                    f"decel..end (bound -5); max clearance {clear*100:.1f} cm "
                    f"(source {src['clearance']*100:.1f} + 2)"))

    # 7 -- arrival: end position vs target, settled heading vs travel direction.
    # "Travel direction" = the root's displacement over the DECEL region (the
    # motion the robot performs while actually coming to rest) -- previously a
    # 30-frame pre-decel chord, which is curvature-BIASED: when the final cycle
    # carries a full-cap injected turn (measured on (0,5): 22.6 deg inside the
    # chord window + 5.3 deg after head_end), the chord lags the end tangent by
    # ~half the in-window turn + all post-window turn = 16.6 deg, flunking
    # motions whose settled yaw is within 0.4-1.7 deg of their real final
    # travel (measured on all of (0,5)/(-0.707,-0.707)/(-0.707,0.707)/(5,5)).
    # The decel-travel estimator is curvature-honest and still catches the
    # bugs this check exists for: a sidle or a skate-pivot stance twist shows
    # up as translation direction != settled yaw regardless of window.
    miss = info["miss_m"]
    trav = pos[ba, :2] - pos[head_end, :2]
    if np.linalg.norm(trav) < 0.05:  # degenerate decel travel: fall back to pre-decel chord
        trav = pos[head_end, :2] - pos[max(head_end - 30, 0), :2]
    travel_dir = np.degrees(np.arctan2(trav[1], trav[0]))
    settled = si["settled_yaw_deg"]
    hd_err = float(np.degrees(wrap(np.radians(settled - travel_dir))))
    ok = miss < 0.4 and abs(hd_err) < 15.0
    results.append(("7 arrival", ok,
                    f"miss {miss:.3f} m (< 0.4); settled yaw {settled:+.1f} vs decel travel dir "
                    f"{travel_dir:+.1f} deg -> diff {hd_err:+.1f} (< 15)"))
    return results


if __name__ == "__main__":
    src = source_reference()
    print(f"source references: dof step {src['dof_step']:.4f} rad, root-speed jump "
          f"{src['v_jump']:.3f} m/s, contact-core skate {src['skate']*100:.2f} cm/s, "
          f"swing clearance {src['clearance']*100:.1f} cm")
    all_ok = True
    for dx, dy in ((5.0, 0.0), (5.0, 5.0), (8.0, -3.0)):
        print(f"\n=== target ({dx:+.1f}, {dy:+.1f}) ===")
        dof, pos, quat, info = generate(dx, dy)
        out = OUT_DIR / f"wtp_{dx:+.0f}_{dy:+.0f}_holomotion.npz".replace("+", "p").replace("-", "m")
        write_npz(dof, pos, quat, out, out.stem, f"qa_check test target ({dx},{dy})")
        for name, ok, detail in run_qa(dof, pos, quat, info, src):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            all_ok &= ok
        print(f"  wrote {out}")
    print(f"\nQA gate: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)
