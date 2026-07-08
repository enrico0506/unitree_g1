"""joint_maps — Phase 3/4 glue: G1 29-DOF joint NAME tables + a name-based
MJCF<->IsaacLab permutation builder (the issue-#78 silent-corruption guard).

Pipeline seat: ``clip -> ROMP -> pose_to_smpl -> GMR -> [gmr_to_sonic_csv
uses THIS] -> SONIC -> MuJoCo G1``.

This module is the SINGLE SOURCE OF TRUTH for the 29-joint reordering that sits
between GMR's MJCF/mujoco dof order (the columns of ``gmr.pkl['dof_pos']``) and
SONIC's IsaacLab CSV column order.  The permutation is BUILT BY NAME from the
two frozen lists at import time — never pasted as a literal — because the one
dangerous failure here (GMR issue #78) is a *silently* wrong permutation that
places one joint's angle onto a different joint three stages downstream, on the
real robot.  A by-name build is deterministic from the two lists and is
cross-checked (set-equality + mutual-inverse) at import; a mis-transcribed or
asset-swapped list raises LOUDLY instead of corrupting motion.

Imports **numpy ONLY** (no scipy / torch / mujoco / isaaclab / GMR).  Verified
env: numpy 2.4.4.

FROZEN CONTRACT (motion/PLAN.md Phase 4):
  G1_MJCF_ORDER      29 names — gmr.pkl dof_pos column order (SOURCE OF TRUTH #1)
  G1_ISAACLAB_ORDER  29 names — SONIC CSV column order       (SOURCE OF TRUTH #2)
  build_permutation(src, dst) -> perm s.t. dst[k] == src[perm[k]]  (raises loud)
  MJCF_TO_ISAAC = build_permutation(G1_MJCF_ORDER, G1_ISAACLAB_ORDER)  # forward gather
  ISAAC_TO_MJCF = build_permutation(G1_ISAACLAB_ORDER, G1_MJCF_ORDER)  # inverse (replay/debug)
  assert_runtime_joint_order(names)  # VERIFY-on-robot hook
  xyzw_to_wxyz(q)                    # quaternion scalar-last -> scalar-first
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "JointOrderError",
    "G1_MJCF_ORDER",
    "G1_ISAACLAB_ORDER",
    "build_permutation",
    "MJCF_TO_ISAAC",
    "ISAAC_TO_MJCF",
    "assert_runtime_joint_order",
    "xyzw_to_wxyz",
]

_NDOF = 29


class JointOrderError(ValueError):
    """Raised LOUDLY on any joint-name / permutation problem: length mismatch,
    a duplicate name in the source list (ambiguous ``.index``), or a
    destination name absent from the source.  Never fail silently — a wrong
    permutation is invisible in the numbers but moves the WRONG joint on the
    robot three stages downstream (issue #78)."""


# ---------------------------------------------------------------------------
# The two frozen 29-joint NAME lists.
#
# Kept as bare stems (the human-auditable transcription — compare these against
# the repos) and suffixed with ``_joint`` to match the actual asset joint names
# (GMR's g1_mocap_29dof.xml and IsaacLab's G1 USD both declare ``*_joint``).
# The permutation is invariant to the suffix, but the suffixed names are what
# ``robot.data.joint_names`` / ``robot.robot_dof_names`` report, so keeping them
# real makes ``assert_runtime_joint_order`` a direct equality on device.
# ---------------------------------------------------------------------------

# SOURCE OF TRUTH #1 — gmr.pkl dof_pos columns, GMR g1_mocap_29dof.xml
# declaration order (legs, then waist, then left arm, then right arm).
_MJCF_STEMS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# SOURCE OF TRUTH #2 — SONIC CSV columns, SONIC joint_utils.py G1_ISAACLab_ORDER
# (left/right interleaved, IsaacLab USD traversal order).
_ISAAC_STEMS = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]

G1_MJCF_ORDER: list[str] = [s + "_joint" for s in _MJCF_STEMS]
G1_ISAACLAB_ORDER: list[str] = [s + "_joint" for s in _ISAAC_STEMS]


# ---------------------------------------------------------------------------
# Name-based permutation builder — the frozen rule: build by name, never paste
# a literal (the pasted literal is exactly how issue #78 happens).
# ---------------------------------------------------------------------------
def build_permutation(src_names: list[str], dst_names: list[str]) -> np.ndarray:
    """Return ``perm`` (int64, length N) such that ``dst[k] == src[perm[k]]``.

    Apply as ``dst_ordered = src_ordered[:, perm]`` (a forward *gather*: output
    column ``k`` pulls source column ``perm[k]``).

    RAISES ``JointOrderError`` on:
      (a) ``len(src) != len(dst)``,
      (b) a duplicate name in ``src`` (``.index`` would be ambiguous),
      (c) ANY ``dst`` name not present in ``src`` — the exact failure a
          mis-transcribed or asset-swapped list produces.
    """
    src = list(src_names)
    dst = list(dst_names)
    if len(src) != len(dst):
        raise JointOrderError(
            f"length mismatch: len(src)={len(src)} != len(dst)={len(dst)}"
        )
    if len(set(src)) != len(src):
        dups = sorted({n for n in src if src.count(n) > 1})
        raise JointOrderError(f"duplicate name(s) in src (ambiguous index): {dups}")
    perm = []
    for n in dst:
        if n not in src:
            raise JointOrderError(
                f"dst name {n!r} not found in src "
                f"(mis-transcribed / asset-swapped joint list?)"
            )
        perm.append(src.index(n))
    # Bijection guard: a duplicate NAME in dst maps two outputs to one src column,
    # so perm would repeat -> not a permutation (silent corruption). Catch it loud.
    if len(set(perm)) != len(perm):
        dups = sorted({n for n in dst if dst.count(n) > 1})
        raise JointOrderError(f"duplicate name(s) in dst (perm not a bijection): {dups}")
    return np.asarray(perm, dtype=np.int64)


# FORWARD gather the glue uses:  joint_pos_isaac = dof_pos_mjcf[:, MJCF_TO_ISAAC]
# PROVEN: G1_MJCF_ORDER[MJCF_TO_ISAAC] == G1_ISAACLAB_ORDER (exact).  This value
# equals the hpp constant labeled ``mujoco_to_isaaclab`` =
#   [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
MJCF_TO_ISAAC: np.ndarray = build_permutation(G1_MJCF_ORDER, G1_ISAACLAB_ORDER)

# INVERSE — replay/debug ONLY (e.g. IsaacLab->MJCF to replay SONIC output back
# through the GMR/MuJoCo model).  Equals the hpp constant labeled
# ``isaaclab_to_mujoco`` = [0,3,6,9,13,17,...].
#
# CRITICAL TRAP (issue #78 made concrete): that hpp ``isaaclab_to_mujoco``
# literal is the INVERSE, NOT the forward gather.  Using it as
# ``dof_pos[:, isaaclab_to_mujoco]`` yields garbage — it places left_knee where
# right_hip_pitch belongs.  We never paste literals; both directions are built
# BY NAME and cross-checked mutual-inverse at import (below).
ISAAC_TO_MJCF: np.ndarray = build_permutation(G1_ISAACLAB_ORDER, G1_MJCF_ORDER)


def assert_runtime_joint_order(runtime_names: list[str]) -> None:
    """VERIFY-on-robot hook.  Compare the LIVE robot/sim joint-name order
    (IsaacLab ``robot.data.joint_names``, or a GMR-reindexed name list) against
    the frozen ``G1_ISAACLAB_ORDER`` that this glue writes its CSV columns in.

    Raises ``JointOrderError`` on ANY mismatch.  Call on-device BEFORE trusting
    a CSV bundle::

        assert_runtime_joint_order(robot.data.joint_names)

    A disagreement means the shipped asset's joint order differs from the pinned
    table — the permutation would be silently wrong.  STOP if it fires.
    """
    runtime = list(runtime_names)
    if len(runtime) != len(G1_ISAACLAB_ORDER):
        raise JointOrderError(
            f"runtime joint count {len(runtime)} != {len(G1_ISAACLAB_ORDER)} "
            f"(expected G1 29-DOF IsaacLab order)"
        )
    for i, (got, exp) in enumerate(zip(runtime, G1_ISAACLAB_ORDER)):
        if got != exp:
            raise JointOrderError(
                f"runtime joint[{i}]={got!r} != expected {exp!r}; "
                f"CSV columns would be mis-assigned (issue #78). STOP."
            )


# ---------------------------------------------------------------------------
# Quaternion channel reindex.  GMR .pkl stores XYZW (scalar-last, scipy
# convention); SONIC CSVs expect WXYZ (scalar-first).  Pure index gather —
# numpy only, so this stays importable without scipy.
# ---------------------------------------------------------------------------
def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reindex a quaternion array scalar-LAST (XYZW) -> scalar-FIRST (WXYZ).

    ``q`` : (..., 4).  Returns ``q[..., [3, 0, 1, 2]]`` (a fresh array, never a
    view onto the input).  This is the SAME reindex GMR's own ``data_loader.py``
    uses in reverse.
    """
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(
            f"quaternion last dimension must be 4 (XYZW), got shape {q.shape}"
        )
    return q[..., [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# Module-load cross-checks — the independent guard.  Raise JointOrderError at
# IMPORT on any failure.  We do NOT assert equality against the hpp literal
# (that literal is the inverse; asserting == would BE the bug) — equality is
# checked structurally: set-equality + mutual-inverse + the proven name identity.
# ---------------------------------------------------------------------------
def _self_check() -> None:
    if len(G1_MJCF_ORDER) != _NDOF or len(G1_ISAACLAB_ORDER) != _NDOF:
        raise JointOrderError(
            f"G1 name lists must both be length {_NDOF} "
            f"(got MJCF={len(G1_MJCF_ORDER)}, IsaacLab={len(G1_ISAACLAB_ORDER)})"
        )
    if len(set(G1_MJCF_ORDER)) != _NDOF or len(set(G1_ISAACLAB_ORDER)) != _NDOF:
        raise JointOrderError("G1 name lists contain duplicate names")
    if set(G1_MJCF_ORDER) != set(G1_ISAACLAB_ORDER):
        missing = set(G1_MJCF_ORDER) ^ set(G1_ISAACLAB_ORDER)
        raise JointOrderError(
            f"G1_MJCF_ORDER and G1_ISAACLAB_ORDER name sets differ: {sorted(missing)}"
        )
    # Mutual inverse (PROVEN): ISAAC_TO_MJCF[MJCF_TO_ISAAC] == arange(29).
    if not np.array_equal(ISAAC_TO_MJCF[MJCF_TO_ISAAC], np.arange(_NDOF)):
        raise JointOrderError(
            "MJCF_TO_ISAAC and ISAAC_TO_MJCF are not mutual inverses"
        )
    # Proven name identity: the forward gather turns MJCF names into IsaacLab names.
    if [G1_MJCF_ORDER[i] for i in MJCF_TO_ISAAC] != G1_ISAACLAB_ORDER:
        raise JointOrderError("G1_MJCF_ORDER[MJCF_TO_ISAAC] != G1_ISAACLAB_ORDER")


_self_check()
