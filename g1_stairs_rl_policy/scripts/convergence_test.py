"""Convergence smoke test: does PPO actually learn something, or is a reward
term degenerate/dominating. Minutes, not hours — a handful of PPO iterations
on a small env count, not a real training run.

NOT "does it climb stairs" (that needs the real training run, thousands of
iterations). This only checks the reward signal isn't broken: mean reward
over the back of the run should beat the front by more than noise. If it
doesn't, look at reward weights before spending GPU time on a full run —
most likely culprit is one term dominating and producing a degenerate gait
(stand still to avoid penalties, etc).

Deliberately does NOT rely on rsl_rl's internal logging/buffer attributes
(those vary across rsl_rl versions) — wraps env.step() directly instead, so
this only depends on the stable VecEnv (obs, reward, dones, extras) contract.

Run on the training box (needs isaaclab + isaaclab_rl + rsl_rl installed):
    python g1_stairs_rl_policy/scripts/convergence_test.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 stairs PPO convergence smoke test.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=40)
parser.add_argument("--min_improvement", type=float, default=0.0,
                     help="require back-of-run mean reward to exceed front-of-run by at least this much")
parser.add_argument("--terrain_rows", type=int, default=2,
                     help="curriculum rows to use for this quick run (default: just the 2 easiest — "
                          "row 0 is ~10cm riser/35cm tread, the easy end of the full curriculum. Full "
                          "training later uses all 10.")
parser.add_argument("--terrain_cols", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab / rsl_rl imports must come AFTER AppLauncher starts the sim app
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_stairs_rl_policy.train.register  # noqa: F401 — side effect: gym.register(...)
from g1_stairs_rl_policy.train.env_stairs import G1StairsEnvCfg
from g1_stairs_rl_policy.train.ppo_cfg import G1StairsPPORunnerCfg


class RewardTracker:
    """Wraps env.step() to log mean scalar reward every call, independent of
    whatever internal stats rsl_rl's OnPolicyRunner keeps (version-fragile)."""

    def __init__(self, env):
        self.history = []
        self._env = env
        self._orig_step = env.step
        env.step = self._step

    def _step(self, actions):
        obs, reward, dones, extras = self._orig_step(actions)
        self.history.append(reward.mean().item())
        return obs, reward, dones, extras


def main():
    # small terrain grid for a fast, easy-first pass — not the full 10x20
    # curriculum used by real training. Row 0 is already the easy end
    # (~10cm riser/35cm tread), so this is "can it learn the easiest stairs
    # at all", not a shrunk version of the hard end.
    env_cfg = G1StairsEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = args.terrain_rows
        env_cfg.scene.terrain.terrain_generator.num_cols = args.terrain_cols

    env = gym.make("Unitree-G1-29dof-Stairs-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    tracker = RewardTracker(env)

    agent_cfg = G1StairsPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()

    history = tracker.history
    if not history:
        raise SystemExit("CONVERGENCE TEST FAILED: no steps recorded — check the env.step() wrap took effect")

    front = history[: max(1, len(history) // 5)]
    back = history[-max(1, len(history) // 5):]
    front_mean, back_mean = sum(front) / len(front), sum(back) / len(back)
    improvement = back_mean - front_mean

    print("\n--- convergence test results ---")
    print(f"  steps recorded:      {len(history)}")
    print(f"  front-of-run mean:   {front_mean:.4f}")
    print(f"  back-of-run mean:    {back_mean:.4f}")
    print(f"  improvement:         {improvement:.4f}  (require > {args.min_improvement})")
    print(f"  any NaN/inf reward:  {any(not (r == r and abs(r) < float('inf')) for r in history)}")

    if improvement <= args.min_improvement:
        raise SystemExit(
            "CONVERGENCE TEST FAILED: reward did not improve over the run. "
            "Check for a dominating/degenerate reward term before a full training run."
        )
    print("CONVERGENCE TEST PASSED")


if __name__ == "__main__":
    main()


# --- unverified, same as everything else in this repo (no isaaclab/rsl_rl
# install to test against) ---------------------------------------------
# - RslRlVecEnvWrapper.step()'s exact (obs, reward, dones, extras) return
#   shape/order, OnPolicyRunner's constructor signature (log_dir=None
#   accepted?, device kwarg name), and RslRlOnPolicyRunnerCfg.to_dict() are
#   all assumed from the standard isaaclab_rl/rsl_rl pattern, not confirmed.
# - `gym.make(task, cfg=env_cfg)` as the override mechanism for a custom cfg
#   instance (rather than the registered default) is isaaclab's standard
#   pattern in its own scripts/train.py — not confirmed against this repo's
#   gym version.
# - "cuda:0" hardcoded — change if the training box's device differs.
# - Run smoke_test.py first; this is wasted GPU time if the env doesn't even
#   instantiate cleanly.
