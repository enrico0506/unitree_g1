"""Smoke test: does the env even come up. Seconds, not minutes.

Catches import errors, cfg-construction crashes (e.g. the TerminationTermCfg
bug fixed earlier), and NaN/inf in the first few steps (e.g. the height-scan
ray-miss bug fixed earlier) — BEFORE spending any GPU time on real training.

Run on the training box (needs isaaclab installed):
    python g1_stairs_rl_policy/scripts/smoke_test.py --headless

Does NOT check that the policy climbs stairs, or that rewards are sane over
time — that's convergence_test.py's job. This only checks "does it run".
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 stairs env smoke test.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab imports must come AFTER AppLauncher starts the sim app
import gymnasium as gym
import torch

import g1_stairs_rl_policy.train.register  # noqa: F401 — side effect: gym.register(...)


def main():
    env = gym.make("Unitree-G1-29dof-Stairs-Play-v0", num_envs=args.num_envs)
    obs, extras = env.reset()

    checks = {
        "reset obs finite": torch.isfinite(_flatten(obs)).all().item(),
    }

    for step in range(args.steps):
        actions = torch.rand(env.action_space.shape, device=env.unwrapped.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, extras = env.step(actions)

        if not torch.isfinite(_flatten(obs)).all():
            checks[f"step {step}: obs finite"] = False
            break
        if not torch.isfinite(reward).all():
            checks[f"step {step}: reward finite"] = False
            break
    else:
        checks[f"ran {args.steps} steps without NaN/inf"] = True

    env.close()
    simulation_app.close()

    failed = [name for name, ok in checks.items() if not ok]
    print("\n--- smoke test results ---")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit(f"SMOKE TEST FAILED: {failed}")
    print("SMOKE TEST PASSED")


def _flatten(obs):
    """obs may be a dict of tensors (policy/critic groups) or a single tensor."""
    if isinstance(obs, dict):
        return torch.cat([v.reshape(-1) for v in obs.values()])
    return obs.reshape(-1)


if __name__ == "__main__":
    main()
