"""Gym registration for the stairs task.

Depends on env_stairs.py for G1StairsEnvCfg / G1StairsPlayEnvCfg and ppo_cfg.py
for the rsl_rl runner cfg.
"""
import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_stairs_rl_policy.train.env_stairs:G1StairsEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_stairs_rl_policy.train.ppo_cfg:G1StairsPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # few envs, no randomization/curriculum/map-noise -> visual inspection only
        "env_cfg_entry_point": "g1_stairs_rl_policy.train.env_stairs:G1StairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_stairs_rl_policy.train.ppo_cfg:G1StairsPPORunnerCfg_PLAY",
    },
)

# sanity check once isaaclab is installed:
#   python -c "import gymnasium as gym, g1_stairs_rl_policy.train.register; \
#              env = gym.make('Unitree-G1-29dof-Stairs-Play-v0'); print(env)"
