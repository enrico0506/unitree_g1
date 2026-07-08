"""motion.pipeline.glue — pure-Python array reshapers between heavy stages.

The glue modules import with numpy+scipy ONLY (no torch / smplx / ROMP / GMR),
so they can be unit-tested on the workstation while the heavy estimators run
on the Orin. Empty on purpose.
"""
