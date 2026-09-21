"""PPO-Lagrangian"""

import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from torch.optim import Adam

from minigym.algorithm.buffer import PPOLagBuffer
from minigym.algorithm.logger import CSVLogger
from minigym.algorithm.policy import ActorCritic


@dataclass
class PPOLagConfig:
    env_fn: Callable  # () -> Environment instance (fresh episode)
    act_dim: int
    action_low: List[float]
    action_high: List[float]
    deltat: float = 1.0 / 60.0  # physics timestep used for env.step()
    max_ep_len: int = 1000  # k: steps before the env object is discarded/reset
    steps_per_epoch: int = 4000
    epochs: int = 50
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    cost_limit: float = 25.0  # target average cost per episode
    lambda_lr: float = 0.05  # step size for the Lagrange multiplier
    hidden_sizes: tuple = (64, 64)
    seed: int = 0
    save_dir: str = "checkpoints"
    save_every: int = 10  # epochs between checkpoints
    log_path: str = "logs/train_log.csv"


def _clip_action(action, low, high):
    return np.clip(action, low, high)


def train(cfg: PPOLagConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    low = np.array(cfg.action_low, dtype=np.float32)
    high = np.array(cfg.action_high, dtype=np.float32)

    # Probe observation dimension from a throwaway environment instance.
    probe_env = cfg.env_fn()
    obs_dim = len(probe_env.obs())
    del probe_env

    ac = ActorCritic(obs_dim, cfg.act_dim, cfg.hidden_sizes)
    pi_optimizer = Adam(ac.pi.parameters(), lr=cfg.pi_lr)
    v_optimizer = Adam(list(ac.v.parameters()) + list(ac.vc.parameters()), lr=cfg.vf_lr)

    buf = PPOLagBuffer(obs_dim, cfg.act_dim, cfg.steps_per_epoch, cfg.gamma, cfg.lam)
    logger = CSVLogger(
        cfg.log_path,
        fieldnames=[
            "epoch",
            "total_steps",
            "avg_reward",
            "avg_cost",
            "avg_ep_len",
            "lagrange_lambda",
            "pi_loss",
            "v_loss",
            "cv_loss",
        ],
    )
    os.makedirs(cfg.save_dir, exist_ok=True)

    lagrange_lambda = 0.0
    total_steps = 0
    env = cfg.env_fn()
    ep_len, ep_ret, ep_cost = 0, 0.0, 0.0
    ep_returns, ep_costs, ep_lens = [], [], []

    for epoch in range(cfg.epochs):
        for t in range(cfg.steps_per_epoch):
            obs = np.array(env.obs(), dtype=np.float32)
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            act, val, cval, logp = ac.step(obs_t)
            clipped_act = _clip_action(act, low, high)

            env.act(clipped_act.tolist())
            env.step(cfg.deltat)
            rew, cost = env.reward(), env.cost()

            buf.store(obs, act, rew, cost, val, cval, logp)
            ep_len += 1
            ep_ret += rew
            ep_cost += cost
            total_steps += 1

            timeout = ep_len == cfg.max_ep_len
            epoch_ended = t == cfg.steps_per_epoch - 1

            if timeout or epoch_ended:
                # Every episode here ends by hitting the step limit k (never a true
                # terminal state), so we always bootstrap with the value of the
                # last-seen observation rather than zero.
                next_obs = torch.as_tensor(np.array(env.obs(), dtype=np.float32))
                with torch.no_grad():
                    last_val = ac.v(next_obs).item()
                    last_cval = ac.vc(next_obs).item()
                buf.finish_path(last_val, last_cval)

                ep_returns.append(ep_ret)
                ep_costs.append(ep_cost)
                ep_lens.append(ep_len)

                # Discard the environment object and create a fresh one.
                env = cfg.env_fn()
                ep_len, ep_ret, ep_cost = 0, 0.0, 0.0

        # ----- Update the Lagrange multiplier using the epoch's average episode cost -----
        avg_cost = float(np.mean(ep_costs)) if ep_costs else 0.0
        avg_reward = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        lagrange_lambda = max(
            0.0, lagrange_lambda + cfg.lambda_lr * (avg_cost - cfg.cost_limit)
        )
        ep_returns, ep_costs, ep_lens = [], [], []

        # ----- PPO update -----
        data = buf.get()
        obs, act, ret, cret, adv, cadv, logp_old = (
            data["obs"],
            data["act"],
            data["ret"],
            data["cret"],
            data["adv"],
            data["cadv"],
            data["logp"],
        )
        # Combine reward advantage with the (Lagrange-weighted) cost advantage, then
        # renormalize so the multiplier's scale doesn't distort the PPO clip ratio.
        adv_total = adv - lagrange_lambda * cadv
        adv_total = adv_total / (1.0 + lagrange_lambda)

        pi_loss_val = 0.0
        for _ in range(cfg.train_pi_iters):
            pi_optimizer.zero_grad()
            _, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = (
                torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_total
            )
            loss_pi = -torch.min(ratio * adv_total, clip_adv).mean()
            loss_pi.backward()
            pi_optimizer.step()
            pi_loss_val = loss_pi.item()

        v_loss_val, cv_loss_val = 0.0, 0.0
        for _ in range(cfg.train_v_iters):
            v_optimizer.zero_grad()
            loss_v = ((ac.v(obs) - ret) ** 2).mean()
            loss_cv = ((ac.vc(obs) - cret) ** 2).mean()
            (loss_v + loss_cv).backward()
            v_optimizer.step()
            v_loss_val, cv_loss_val = loss_v.item(), loss_cv.item()

        logger.log(
            epoch=epoch,
            total_steps=total_steps,
            avg_reward=avg_reward,
            avg_cost=avg_cost,
            avg_ep_len=avg_ep_len,
            lagrange_lambda=lagrange_lambda,
            pi_loss=pi_loss_val,
            v_loss=v_loss_val,
            cv_loss=cv_loss_val,
        )
        print(
            f"epoch {epoch:4d} | steps {total_steps:8d} | reward {avg_reward:8.3f} | "
            f"cost {avg_cost:8.3f} | lambda {lagrange_lambda:6.3f}"
        )

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(ac, lagrange_lambda, obs_dim, cfg, epoch + 1)

    logger.close()
    return ac


def save_checkpoint(
    ac: ActorCritic, lagrange_lambda: float, obs_dim: int, cfg: PPOLagConfig, epoch: int
):
    path = os.path.join(cfg.save_dir, f"epoch_{epoch}.pt")
    torch.save(
        {
            "model_state_dict": ac.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": cfg.act_dim,
            "hidden_sizes": cfg.hidden_sizes,
            "action_low": cfg.action_low,
            "action_high": cfg.action_high,
            "lagrange_lambda": lagrange_lambda,
            "epoch": epoch,
        },
        path,
    )
    print(f"saved checkpoint to {path}")


def load_checkpoint(path: str) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu")
    ac = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], tuple(ckpt["hidden_sizes"]))
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    return ac, ckpt
