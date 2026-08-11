"""Step 1 of walk_to_point (PLAN.md): the stance-pivot turn injector.

Injects a heading change into an assembled walking sequence by RIGIDLY rotating
root pos AND root quat together about the vertical axis through the MEASURED
stance-foot contact point of each single-support interval -- never by adding yaw
to the quat alone (that sweeps the planted foot sideways: instant skate, the
failure mode PLAN.md Step 1 explicitly warns against; the self-test below runs
that naive variant too, to show it fail the skate metric loudly).

Per plan spec:
- injection gated to single-support intervals ONLY (FK-derived contacts on the
  assembled clip -- the source foot_contacts array doesn't survive assembly),
  smoothstep ramp 0->delta_psi inside each window, held constant outside;
- total turn capped at ~10-15 deg/step, distributed evenly over ceil(theta/cap)
  consecutive single-support windows.

Run under `conda activate kimodo`, from the repo root.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from common import (  # noqa: E402
    FPS, contact_skate_intervals, fk_contacts, foot_positions, interval_core,
    rot2, run_boundaries, smoothstep,
)

TURN_CAP_DEG = 12.0  # plan: ~10-15 deg per step


def single_support_windows(foot_l, foot_r):
    """FK-derived contacts -> list of (start, end, stance_foot 'L'/'R') runs
    where exactly one foot is planted. Abutting L-only/R-only runs (this gait
    has ~zero double support, Step 0) split at the stance-foot change."""
    con = fk_contacts(foot_l, foot_r)
    state = np.where(con["L"] & ~con["R"], 1, np.where(con["R"] & ~con["L"], 2, 0))
    starts, ends = run_boundaries(state)
    return [(int(s), int(e), "LR"[state[s] - 1])
            for s, e in zip(starts, ends) if state[s] != 0]


def inject_turn(dof_pos, root_pos, root_quat, total_angle, cap_deg=TURN_CAP_DEG,
                skip_windows=1):
    """Rigid stance-pivot turn injection. total_angle in radians (+ = CCW/left).
    dof_pos is untouched (returned for signature symmetry) -- and that is
    exactly WHY the planted foot doesn't skate: the whole body rotates rigidly
    about the contact point, so no joint has to absorb the turn.
    skip_windows: leave the first window(s) unturned -- the very first
    single-support run starts at the clip edge / start segment, not a clean
    full stance.

    Returns (dof_pos, root_pos, root_quat, info)."""
    foot_l, foot_r = foot_positions(dof_pos, root_pos, root_quat)
    windows = single_support_windows(foot_l, foot_r)
    n_steps = int(np.ceil(abs(total_angle) / np.radians(cap_deg)))
    ramp_windows = windows[skip_windows:skip_windows + n_steps]
    assert len(ramp_windows) == n_steps, (
        f"need {n_steps} single-support windows after skipping {skip_windows}, "
        f"clip has {len(windows)}")
    dpsi = total_angle / n_steps
    foot = {"L": foot_l, "R": foot_r}

    # cumulative rigid ground-plane transform  x' = R(theta) x + d,  applied
    # per frame; within ramp window k it is Rot(psi(t)-psi_k, pivot_k) composed
    # onto the transform frozen at the window start. pivot_k = the measured
    # stance-foot contact point (mean FK xy over the window core, in the
    # ALREADY-TRANSFORMED coordinates -- the planted foot has been moved by the
    # previous turn steps too. Pivoting about the untransformed FK position
    # instead produces a slowly-accumulating error that QA #7 can still pass at
    # small turn angles, so it is not self-checking -- keep the rot2(th)@... + d).
    T = len(root_pos)
    theta = np.zeros(T)
    disp = np.zeros((T, 2))

    th, d = 0.0, np.zeros(2)
    prev_end = 0
    for s, e, f in ramp_windows:
        theta[prev_end:s] = th
        disp[prev_end:s] = d
        cs, ce = interval_core(s, e)
        pivot = rot2(th) @ foot[f][cs:ce, :2].mean(axis=0) + d
        phi = smoothstep(e - s) * dpsi
        for i, t in enumerate(range(s, e)):
            theta[t] = th + phi[i]
            disp[t] = rot2(phi[i]) @ (d - pivot) + pivot
        th, d = theta[e - 1], disp[e - 1].copy()
        prev_end = e
    theta[prev_end:] = th
    disp[prev_end:] = d

    new_xy = np.einsum("tij,tj->ti", np.stack([rot2(a) for a in theta]),
                       root_pos[:, :2]) + disp
    new_pos = np.concatenate([new_xy, root_pos[:, 2:3]], axis=1)
    new_quat = (R.from_euler("z", theta) * R.from_quat(root_quat)).as_quat()
    info = {"n_steps": n_steps, "dpsi_deg": float(np.degrees(dpsi)),
            "ramp_windows": ramp_windows, "theta_profile": theta}
    return dof_pos, new_pos, new_quat, info


def skate_report(dof_pos, root_pos, root_quat, contacts=None):
    """Max horizontal ankle-link speed during (core of) each contact interval.
    Contacts FK-derived from THIS clip unless passed in (baseline's reused for
    the turned clip so identical frames are compared).

    qa_check's global skate number is exactly max() over these per-interval
    values -- both go through common.contact_skate_intervals so the self-test
    and the gate cannot drift apart."""
    foot_l, foot_r = foot_positions(dof_pos, root_pos, root_quat)
    return contact_skate_intervals(foot_l, foot_r, contacts)


if __name__ == "__main__":
    from build_straight import assemble_straight
    from common import yaw_from_quat_xyzw

    TURN_DEG = 45.0
    dof, pos, quat, _ = assemble_straight(8.0)
    print(f"base sequence: {len(dof)} frames ({len(dof)/FPS:.2f}s)")

    base_rep, base_con = skate_report(dof, pos, quat)
    dof_t, pos_t, quat_t, info = inject_turn(dof, pos, quat, np.radians(TURN_DEG))
    print(f"\ninjected {TURN_DEG} deg over {info['n_steps']} steps of "
          f"{info['dpsi_deg']:.2f} deg, ramp windows {info['ramp_windows']}")
    turn_rep, _ = skate_report(dof_t, pos_t, quat_t, base_con)

    # naive "just add yaw to the quat, leave root xy alone" -- the rejected
    # approach, run purely to demonstrate the failure mode being guarded against
    quat_n = (R.from_euler("z", info["theta_profile"]) * R.from_quat(quat)).as_quat()
    naive_rep, _ = skate_report(dof, pos, quat_n, base_con)

    turn_frames = set()
    for s, e, _f in info["ramp_windows"]:
        turn_frames.update(range(s, e))
    print("\nstance-foot skate during contact-interval cores (max xy speed, cm/s):")
    print(f"{'interval':>14} {'baseline':>9} {'stance-pivot':>13} {'naive-yaw':>10}")
    worst = {"base_t": 0.0, "turn_t": 0.0, "naive_t": 0.0, "base_all": 0.0, "turn_all": 0.0}
    for f in "LR":
        for (s, e, vb), (_, _, vt), (_, _, vn) in zip(base_rep[f], turn_rep[f], naive_rep[f]):
            in_turn = bool(turn_frames & set(range(s, e)))
            tag = "TURN" if in_turn else "    "
            print(f"  {f} [{s:3d},{e:3d}){tag} {vb*100:8.2f} {vt*100:12.2f} {vn*100:9.2f}")
            worst["base_all"] = max(worst["base_all"], vb)
            worst["turn_all"] = max(worst["turn_all"], vt)
            if in_turn:
                worst["base_t"] = max(worst["base_t"], vb)
                worst["turn_t"] = max(worst["turn_t"], vt)
                worst["naive_t"] = max(worst["naive_t"], vn)

    # MEASURED FACT (Step 0): the ankle_roll LINK CENTER already moves
    # 5-12 cm/s during contact cores in the untouched source clip (it rocks
    # while the contact point stays planted), so the plan's literal 2-3 cm/s
    # skate threshold cannot be applied to the absolute link speed. What the
    # injector must guarantee -- and what is asserted -- is that the turn adds
    # essentially nothing on top of that baseline: excess < 3 cm/s. (Same
    # excess-over-natural substitution as QA #2.)
    excess = worst["turn_t"] - worst["base_t"]
    print(f"\nturning-step intervals: baseline max {worst['base_t']*100:.2f} cm/s, "
          f"stance-pivot max {worst['turn_t']*100:.2f} cm/s "
          f"(excess {excess*100:+.2f} cm/s, threshold 3), "
          f"naive-yaw max {worst['naive_t']*100:.2f} cm/s")
    print(f"whole clip: baseline max {worst['base_all']*100:.2f} cm/s, "
          f"turned max {worst['turn_all']*100:.2f} cm/s")
    assert excess < 0.03, f"stance-pivot turn adds {excess*100:.2f} cm/s skate"
    # NEGATIVE CONTROL, not dead test code: the naive variant is run precisely
    # so this assert can prove the skate metric has teeth (measured 89.62 cm/s
    # naive vs 10.39 for the stance pivot). Do not delete it as "testing a code
    # path we don't use".
    assert worst["naive_t"] > 0.10, "naive variant should fail loudly (sanity of the metric)"

    # achieved vs commanded heading change (turn effect isolated from the
    # gait's own yaw sway/drift by differencing against the baseline clip)
    dyaw_base = yaw_from_quat_xyzw(quat[-1]) - yaw_from_quat_xyzw(quat[0])
    dyaw_turn = yaw_from_quat_xyzw(quat_t[-1]) - yaw_from_quat_xyzw(quat_t[0])
    achieved = np.degrees((dyaw_turn - dyaw_base + np.pi) % (2 * np.pi) - np.pi)
    print(f"\nheading: commanded {TURN_DEG:.2f} deg, achieved (vs baseline) "
          f"{achieved:.2f} deg, sum of per-step deltas "
          f"{info['n_steps'] * info['dpsi_deg']:.2f} deg")
    assert abs(achieved - TURN_DEG) < 1.0

    assert all(np.isfinite(x).all() for x in (dof_t, pos_t, quat_t))
    print("\nall self-tests passed")
