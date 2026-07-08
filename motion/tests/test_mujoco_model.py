"""Skip-gated MuJoCo validation, captured as a test.

Proves — against the REAL G1 29-DOF MuJoCo model — the two things that are otherwise
VERIFY-on-robot: (1) joint_maps.G1_MJCF_ORDER matches the model's actual hinge-joint order
(the issue-#78 silent-corruption risk), and (2) the gmr→SONIC-CSV glue round-trips
(reindex + quat + resample) faithfully on that model.

Skips cleanly where mujoco or the model file is absent (base env / CI); runs where both are
present (the ``g1`` conda env on this workstation). Run there with:
    MUJOCO_GL=egl conda run -n g1 pytest motion/tests/test_mujoco_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

mujoco = pytest.importorskip("mujoco")            # skip whole module if mujoco absent

from motion.pipeline.glue.joint_maps import G1_MJCF_ORDER   # noqa: E402
from motion.sim.validate_in_mujoco import (                 # noqa: E402
    DEFAULT_MODEL, check_roundtrip, synth_gmr,
)

if not os.path.exists(DEFAULT_MODEL):
    pytest.skip(f"G1 model not present: {DEFAULT_MODEL}", allow_module_level=True)


def _model_joints():
    m = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(m.njnt)
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


def test_mjcf_order_matches_real_model():
    # The issue-#78 VERIFY, proven off-robot against the actual model.
    assert list(G1_MJCF_ORDER) == _model_joints()


def test_glue_roundtrips_on_model(tmp_path):
    # gmr.pkl -> CSV (IsaacLab) -> reindex back to MJCF == source, to machine precision.
    gmr = synth_gmr(_model_joints())
    assert check_roundtrip(gmr, tmp_path)
