"""Shared vocabulary for walk_to_point (PLAN.md Steps 0-4).

Only things used by MORE THAN ONE step live here: the source-clip paths, the
30 fps / body-index / FK-tolerance constants, the small numeric helpers that
used to be retyped in four or five files, and the FK wrapper that turns an
assembled (dof, pos, quat) clip into world foot positions.

No algorithm lives here. Every expression below is a literal transcription of
what the step modules used to spell out inline -- the point is that "which
copy is authoritative" now has exactly one answer.

Module map (each step keeps its own runnable __main__ self-test):
    heel_strike.py            Step 0  source-clip segmentation
    build_straight.py         Step 2  seam primitives + straight assembly
    turn_injector.py          Step 1  stance-pivot turn injection
    stop_tail.py              Step 3  decelerate-and-stop tail
    generate_walk_to_point.py Steps 1+2 fused, + Step 4 output
    qa_check.py               the 7-check QA gate

Run under `conda activate kimodo`, from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# one bootstrap for the whole module: this directory (so the step modules can
# import each other by bare name) + the shared holomotion FK pipeline
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "holomotion/scripts"))
from pkl_to_offline_npz import forward_kinematics_all_frames  # noqa: E402

FPS = 30.0

# body indices in the 30-body holomotion arrays (world dropped, pelvis = 0)
LEFT_FOOT_BODY = 6    # left_ankle_roll_link
RIGHT_FOOT_BODY = 12  # right_ankle_roll_link

# ---- the module's source clips ----
WALK_DIR = Path("motion/motion_library/single/walk_straight")
WALK_HOLO_NPZ = WALK_DIR / "walk_straight_holomotion.npz"
WALK_RAW_NPZ = WALK_DIR / "raw_kimodo.npz"
STAIRS_HOLO = Path("motion/motion_library/single/stair_climbing/stair_climbing_holomotion.npz")

# ---- FK contact tolerances ----
# NOTE on thresholds vs the plan's literal "~1cm z / <5cm/s": we measure the
# ankle_roll LINK CENTER, not the foot's contact point. The link rocks 2-6cm
# vertically during heel-strike (toe-up) and toe-off (heel-up) while the
# contact point itself stays planted, and moves 10-50cm/s during those 1-3
# transition frames. The plan's numbers implicitly assume a point contact.
# Smallest reasonable adaptation (documented per instructions, not a redesign):
# run the check on the interval CORE (trim FK_TRIM frames each side) with
# z-range < 2.5cm and speed < 12cm/s. FK still wins on disagreement (see
# heel_strike.detect_heel_strikes' regeneration fallback).
# This is the SINGLE documented origin of the measurement-point adaptation --
# turn_injector, stop_tail and qa_check all cite it.
FK_TRIM = 3
FK_Z_TOL = 0.025
FK_SPEED_TOL = 0.12


# ---- small angle / weight helpers ----

def wrap(a):
    """Wrap an angle (or array of angles) into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot2(a):
    """2x2 ground-plane rotation matrix."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def circ_mean(a):
    """Circular mean of a set of angles (never averages across the pi wrap)."""
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def smoothstep(n):
    """n smoothstep weights from 0 to 1 inclusive (3t^2 - 2t^3)."""
    t = np.linspace(0.0, 1.0, n)
    return 3 * t**2 - 2 * t**3


def yaw_from_quat_xyzw(q):
    """Extract heading (rotation about world z) from a single xyzw quaternion."""
    return R.from_quat(q).as_euler("xyz")[2]


def settled_heading(root_quat, n):
    """Circular mean of root yaw over the last n frames.

    Never the final frame alone: root yaw sways +/-2-4 deg per step, and a
    single last-frame sample can land on a sway extreme (which plants the stop
    stance visibly twisted, and makes the greedy loop's re-aim jitter)."""
    return circ_mean(R.from_quat(root_quat[-n:]).as_euler("xyz")[:, 2])


# ---- run-length / interval helpers on 0-1 signals ----

def run_boundaries(sig):
    """Run-length decomposition of a 1-D signal -> (starts, ends) index arrays,
    covering the whole signal (gaps included)."""
    idx = np.flatnonzero(np.diff(np.asarray(sig).astype(int))) + 1
    return np.concatenate([[0], idx]), np.concatenate([idx, [len(sig)]])


def contact_intervals(sig):
    """[(start, end), ...] for every run where `sig` is truthy."""
    starts, ends = run_boundaries(sig)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if sig[s]]


def interval_core(s, e):
    """Trim FK_TRIM frames off each end of [s, e) -- but only when the interval
    is long enough to survive it. See the FK tolerance note above for why the
    edges of a contact interval are not measurable."""
    return (s + FK_TRIM, e - FK_TRIM) if e - s > 2 * FK_TRIM + 1 else (s, e)


# ---- foot kinematics ----

def foot_speed_xy(foot):
    """Horizontal speed (m/s) of a (T, >=2) foot track.

    np.gradient (CENTRED difference) deliberately, not np.diff -- root speed
    elsewhere (build_straight's velocity estimates, QA #3) uses np.diff and has
    different endpoint behaviour. These are two different quantities; do not
    route them through one helper."""
    return np.linalg.norm(np.gradient(foot[:, :2], axis=0), axis=1) * FPS


def foot_positions(dof_pos, root_pos, root_quat):
    """FK -> (T,3) world positions of both ankle_roll links."""
    trans, _ = forward_kinematics_all_frames(root_pos, root_quat, dof_pos)
    return trans[:, LEFT_FOOT_BODY, :], trans[:, RIGHT_FOOT_BODY, :]


# ---- FK-derived ground contact ----

def debounce(sig, min_run=3):
    """Merge any run (contact or gap) shorter than min_run into its neighbors.
    Repeats until stable, since merging can create new short runs."""
    sig = sig.copy()
    while True:
        changed = False
        starts, ends = run_boundaries(sig)
        for s, e in zip(starts, ends):
            # boundary runs merge into their single neighbor; only skip if the
            # run IS the whole signal
            if e - s < min_run and not (s == 0 and e == len(sig)):
                sig[s:e] = ~sig[s:e]
                changed = True
                break
        if not changed:
            return sig


def contact_from_fk(foot, z_tol=0.04, speed_tol=0.15):
    """Contact signal derived purely from FK -- foot low (within z_tol of its
    global minimum, flat ground) AND horizontally slow. Tolerances sized for
    the ankle-link rocking described in the FK tolerance note above.

    INVARIANT this relies on: flat ground and no cumulative z drift in the
    clip. It thresholds against the clip's OWN GLOBAL foot-z minimum, so any
    per-seam sink silently reclassifies contacts everywhere -- which is exactly
    why every same-clip cycle append passes align_z=False (see
    build_straight.align_segment's docstring)."""
    z = foot[:, 2]
    return debounce((z < z.min() + z_tol) & (foot_speed_xy(foot) < speed_tol))


def fk_contacts(foot_l, foot_r):
    """{'L': contact signal, 'R': ...} for an assembled clip's two foot tracks.
    (The raw foot_contacts array does not survive assembly/blending, so every
    post-assembly contact question is answered from FK -- plan QA #2.)"""
    return {"L": contact_from_fk(foot_l), "R": contact_from_fk(foot_r)}


def contact_skate_intervals(foot_l, foot_r, contacts=None):
    """Per foot: [(start, end, max horizontal link speed over the interval
    CORE), ...] for every contact interval -- the skate metric, per interval.

    contacts: pass a precomputed mask to compare the SAME frames between two
    clips (the turn injector's self-test reuses the baseline clip's mask);
    otherwise derived from this clip via FK.

    Returns (intervals, contacts)."""
    foot = {"L": foot_l, "R": foot_r}
    if contacts is None:
        contacts = fk_contacts(foot_l, foot_r)
    out = {}
    for f in "LR":
        speed = foot_speed_xy(foot[f])
        ivals = []
        for s, e in contact_intervals(contacts[f]):
            cs, ce = interval_core(s, e)
            ivals.append((s, e, float(speed[cs:ce].max())))
        out[f] = ivals
    return out, contacts


def contact_core_max_speed(foot_l, foot_r, contacts=None):
    """Worst contact-core skate over the whole clip -- exactly the max over
    contact_skate_intervals' per-interval values, so the turn injector's
    self-test and QA #2 can never drift apart."""
    ivals, _ = contact_skate_intervals(foot_l, foot_r, contacts)
    return max((v for f in "LR" for _, _, v in ivals[f]), default=0.0)


def load_walk_npz():
    """The module's single source clip as (holomotion, raw_kimodo) npz handles.
    One place that knows where the files are."""
    return (np.load(WALK_HOLO_NPZ, allow_pickle=True),
            np.load(WALK_RAW_NPZ, allow_pickle=True))
