"""Train a PPO-Lagrangian agent on the SafetyPointGoal0 environment."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no display needed for training

from minigym.task.safety_point_goal import SafetyPointGoal0
from minigym.algorithm.ppolag import PPOLagConfig, train

if __name__ == "__main__":
    cfg = PPOLagConfig(
        env_fn=SafetyPointGoal0,
        act_dim=2,                      # [throttle, steer]
        action_low=[0.0, -1.0],
        action_high=[1.0, 1.0],
        max_ep_len=1000,                # k: steps before the env is discarded and recreated
        steps_per_epoch=4000,
        epochs=50,
        cost_limit=25.0,
        save_dir="checkpoints",
        save_every=10,
        log_path="logs/train_log.csv",
        seed=0,
    )
    train(cfg)
