"""MLP actor-critic policy."""

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal-Gaussian MLP policy for continuous action spaces."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden_sizes, act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs, act=None):
        """Returns the action distribution, and optionally the log-prob of `act`."""
        dist = self.distribution(obs)
        logp = None
        if act is not None:
            logp = dist.log_prob(act).sum(axis=-1)
        return dist, logp


class MLPCritic(nn.Module):
    """Plain MLP state-value function, used for both reward and cost critics."""

    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.v_net = mlp([obs_dim, *hidden_sizes, 1])

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Bundles the policy with a reward critic and a cost critic."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pi = GaussianPolicy(obs_dim, act_dim, hidden_sizes)
        self.v = MLPCritic(obs_dim, hidden_sizes)
        self.vc = MLPCritic(obs_dim, hidden_sizes)

    @torch.no_grad()
    def step(self, obs):
        """Sample an action and return (action, value, cost_value, log_prob) as numpy/floats."""
        dist = self.pi.distribution(obs)
        act = dist.sample()
        logp = dist.log_prob(act).sum(axis=-1)
        val = self.v(obs)
        cval = self.vc(obs)
        return act.numpy(), val.item(), cval.item(), logp.item()

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        """Return an action for evaluation/rollout (mean action if deterministic)."""
        dist = self.pi.distribution(obs)
        act = dist.mean if deterministic else dist.sample()
        return act.numpy()
