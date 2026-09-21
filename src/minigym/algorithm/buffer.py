"""On-policy rollout buffer with GAE-Lambda for both reward and cost."""

import numpy as np
import torch


def discounted_cumsum(x, discount):
    """x[i] = x[i] + discount*x[i+1] + discount^2*x[i+2] + ..."""
    out = np.zeros_like(x)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


class PPOLagBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, cost, val, cval, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0, last_cval=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # Reward GAE-Lambda advantage and reward-to-go.
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discounted_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]

        # Cost GAE-Lambda advantage and cost-to-go.
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discounted_cumsum(cdeltas, self.gamma * self.lam)
        self.cret_buf[path_slice] = discounted_cumsum(costs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages (helps optimization; keep scale of returns as-is).
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() + 1e-8
        adv_norm = (self.adv_buf - adv_mean) / adv_std
        cadv_mean, cadv_std = self.cadv_buf.mean(), self.cadv_buf.std() + 1e-8
        cadv_norm = (self.cadv_buf - cadv_mean) / cadv_std

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            cret=self.cret_buf,
            adv=adv_norm,
            cadv=cadv_norm,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
