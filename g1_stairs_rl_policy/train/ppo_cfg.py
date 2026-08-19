"""rsl_rl PPO config for the stairs task.

Subclasses Isaac Lab's own G1RoughPPORunnerCfg (isaaclab_tasks' G1 rough-terrain
PPO hyperparams — network sizes, clip/entropy/learning-rate — verified against
real source, fetched 2026-08-19), not written from scratch. Stairs is a harder
skill than generic rough terrain (same height-scan obs, but curriculum runs to
a steeper endpoint and the reward has more terms to balance), so the only
things overridden below are training LENGTH and bookkeeping name — the network
architecture and PPO hyperparams (clip, entropy, lr, gamma, lam) are left as
Isaac Lab's own tuned defaults. Change those only if training curves show a
specific problem (e.g. KL blowing past desired_kl -> lower learning_rate).

Needs isaaclab_rl installed; not import-tested on this machine.
"""
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class G1StairsPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_stairs"
        # more iterations than flat/rough: harder skill + a 10-row curriculum
        # to climb through before the policy ever sees the hard end.
        self.max_iterations = 8000
        self.save_interval = 100


@configclass
class G1StairsPPORunnerCfg_PLAY(G1StairsPPORunnerCfg):
    """Only meaningful field for play/eval: load, don't train, so nothing to
    override here beyond inheriting the same architecture. Kept as a separate
    class so register.py has a play-time rsl_rl entry point to point at,
    matching the env's Play-v0 / non-Play-v0 split."""


# --- verify on the training box -----------------------------------------
# `from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.
#  rsl_rl_ppo_cfg import G1RoughPPORunnerCfg` — confirm exact module path
# (isaaclab_tasks version-dependent, same caveat as env_stairs.py).
#
# launch (Isaac Lab's own generic rsl_rl train script, reused as-is):
#   python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#     --task Unitree-G1-29dof-Stairs-v0 --headless
