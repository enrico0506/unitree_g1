"""motion.pipeline — Phase 2+ processing stages (pose -> GMR -> SONIC).

Namespace-light package so ``python -m pipeline.glue.pose_to_smpl`` works when
invoked from inside ``motion/`` (the runner's cwd) AND
``from motion.pipeline.glue.pose_to_smpl import ...`` works from the repo root
(the test's cwd). Kept empty on purpose — no import side effects.
"""
