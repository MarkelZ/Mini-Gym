"""Load a saved agent and run it for one trajectory in the environment."""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from minigym.task.safety_point_goal import SafetyPointGoal0
from minigym.algorithm.ppolag import load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to a saved .pt checkpoint")
    parser.add_argument("--steps", type=int, default=1000, help="number of steps to run (k)")
    parser.add_argument("--deltat", type=float, default=1.0 / 60.0)
    parser.add_argument("--stochastic", action="store_true", help="sample instead of using the mean action")
    args = parser.parse_args()

    ac, ckpt = load_checkpoint(args.checkpoint)
    low = np.array(ckpt["action_low"], dtype=np.float32)
    high = np.array(ckpt["action_high"], dtype=np.float32)

    env = SafetyPointGoal0()
    total_reward, total_cost = 0.0, 0.0

    for t in range(args.steps):
        obs = torch.as_tensor(np.array(env.obs(), dtype=np.float32))
        act = ac.act(obs, deterministic=not args.stochastic)
        act = np.clip(act, low, high)

        env.act(act.tolist())
        env.step(args.deltat)

        r, c = env.reward(), env.cost()
        total_reward += r
        total_cost += c
        print(f"step {t:4d} | action {act} | reward {r:.2f} | cost {c:.2f}")

    print(f"\nTrajectory finished: total_reward={total_reward:.2f} total_cost={total_cost:.2f}")
