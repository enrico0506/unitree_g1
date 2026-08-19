"""Episode reset sampling: where the robot starts, every reset.

This is what makes "climb stairs approached from any position" actually true.
Skip this file and the policy only ever sees head-on approaches from the bottom.

Pure numpy, no isaaclab import required for the sampling math itself (keeps it
testable standalone); event-manager wiring at the bottom needs isaaclab.
"""
import numpy as np

# --- ranges --------------------------------------------------------------
DIST_TO_STEP_RANGE = (0.5, 2.5)     # m, distance from base to first riser
LATERAL_OFFSET_RANGE = (-0.4, 0.4)  # m, perpendicular to stair-facing direction
APPROACH_YAW_RANGE = (-30.0, 30.0)  # deg, relative to stair-facing direction

MID_FLIGHT_PROB = 0.30              # fraction of resets that start ON a step
MID_FLIGHT_STEP_FRAC_RANGE = (0.15, 0.85)  # how far up the flight, as a fraction

JOINT_POS_JITTER = 0.05             # rad, per-joint uniform jitter
BASE_LIN_VEL_JITTER = 0.1           # m/s
BASE_ANG_VEL_JITTER = 0.1           # rad/s


def sample_reset_pose(rng: np.random.Generator, stair_origin, stair_yaw, num_steps,
                       step_height, step_depth):
    """One reset target, in world frame.

    stair_origin : (3,) world position of the bottom of the first riser.
    stair_yaw    : rad, world heading the stairs face (climbing direction).

    Returns dict: base_pos (3,), base_yaw (rad, world), on_step (bool),
    step_index (int or None).
    """
    on_step = rng.uniform() < MID_FLIGHT_PROB
    approach_yaw = stair_yaw + np.deg2rad(rng.uniform(*APPROACH_YAW_RANGE))
    lateral = rng.uniform(*LATERAL_OFFSET_RANGE)

    if on_step:
        frac = rng.uniform(*MID_FLIGHT_STEP_FRAC_RANGE)
        step_index = int(round(frac * (num_steps - 1)))
        forward = step_index * step_depth
        height = step_index * step_height
    else:
        step_index = None
        forward = -rng.uniform(*DIST_TO_STEP_RANGE)  # negative = before the stairs
        height = 0.0

    # stair-local -> world: forward along stair_yaw, lateral perpendicular to it
    dx = forward * np.cos(stair_yaw) - lateral * np.sin(stair_yaw)
    dy = forward * np.sin(stair_yaw) + lateral * np.cos(stair_yaw)
    base_pos = np.array([
        stair_origin[0] + dx,
        stair_origin[1] + dy,
        stair_origin[2] + height,
    ])

    return {
        "base_pos": base_pos,
        "base_yaw": approach_yaw,
        "on_step": on_step,
        "step_index": step_index,
    }


def apply_joint_jitter(rng: np.random.Generator, default_joint_pos: np.ndarray) -> np.ndarray:
    """default_joint_pos (num_joints,) -> jittered copy."""
    noise = rng.uniform(-JOINT_POS_JITTER, JOINT_POS_JITTER, size=default_joint_pos.shape)
    return default_joint_pos + noise


def sample_base_vel_jitter(rng: np.random.Generator):
    lin = rng.uniform(-BASE_LIN_VEL_JITTER, BASE_LIN_VEL_JITTER, size=3)
    ang = rng.uniform(-BASE_ANG_VEL_JITTER, BASE_ANG_VEL_JITTER, size=3)
    return lin, ang


# --- isaaclab event-manager term ------------------------------------------
# Wire as a reset-mode EventTerm in env_stairs.py's EventCfg, e.g.:
#
#   from isaaclab.managers import EventTermCfg as EventTerm
#   from isaaclab.envs import mdp as base_mdp
#
#   def reset_root_state_on_stairs(env, env_ids, terrain_stair_origins, terrain_stair_yaws):
#       rng = np.random.default_rng()  # per-call; env sets torch/np seed globally at launch
#       for env_id in env_ids:
#           pose = sample_reset_pose(
#               rng, terrain_stair_origins[env_id], terrain_stair_yaws[env_id],
#               num_steps=..., step_height=..., step_depth=...,
#           )
#           # write pose["base_pos"] / pose["base_yaw"] into env.scene["robot"].data
#           # root_state via write_root_pose_to_sim(), same call apply_joint_jitter()
#           # onto default_joint_pos and sample_base_vel_jitter() onto root velocities.
#
#   reset_on_stairs = EventTerm(func=reset_root_state_on_stairs, mode="reset")
#
# Standalone test (no isaaclab needed): sample N poses, histogram base_pos / on_step
# fraction, confirm ~30% land on a step and the rest spread over 0.5-2.5m.
