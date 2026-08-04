Place the manually downloaded SMPL-X body models here, structured as:

- `smplx/SMPLX_NEUTRAL.npz` (or `.pkl`)
- `smplx/SMPLX_MALE.npz`
- `smplx/SMPLX_FEMALE.npz`

Download from https://smpl-x.is.tue.mpg.de/ (registration required, license-gated —
same reason `video_to_motion/4D-Humans/data/` needs a manual SMPL download).

GMR's retarget scripts (`scripts/smplx_to_robot.py`, `adapt_smpl_to_gmr_smplx.py`
in `video_to_motion/pose_to_gmr/`) will fail at startup until these exist.

If using the `.pkl` variant instead of `.npz`, change `ext` in
`smplx/body_models.py` (inside your smplx pip install) from `npz` to `pkl`,
per GMR's README.
