"""Networks for the faithful PyTorch port of MAC-Flow (arXiv:2511.05005).

Ported 1:1 from the official JAX implementation (agents/macflow.py,
utils/networks.py in https://github.com/dongsuleetech/mac-flow), not from this
repo's earlier online variant in mac_flow_learner.py -- the two differ
structurally, see macflow_paper_learner.py's docstring.

Three modules, all per-agent MLPs applied elementwise over the agent axis (the
official code has no RNN and no joint-action network):

    ActorVectorField  v_phi(o_i, x_t, t) -> velocity      (the BC flow)
    OneStepActor      mu_w(o_i, z_i)     -> action        (the distilled policy)
    TwinValue         Q_k(o_i, a_i)      -> scalar, k=1,2 (per-agent critics)

Official defaults carried over: hidden (512,512,512,512), GELU, LayerNorm ON
for the critics (the paper calls this crucial for the Lipschitz condition its
Proposition 4.3 needs) and OFF for the actors, 2-critic ensemble aggregated by
mean (not min -- deliberate, to avoid offline pessimism).
"""

import torch as th
import torch.nn as nn


def mlp(in_dim, hidden_dims, out_dim, layer_norm):
    """(*hidden_dims, out_dim) MLP with GELU, matching utils/networks.py::MLP.
    LayerNorm is applied after the activation of every hidden layer."""
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.GELU())
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActorVectorField(nn.Module):
    """v_phi(o, x_t, t). Also used, with times=None, as the one-step policy
    mu_w(o, z) -- exactly as the official code reuses ActorVectorField for
    `actor_onestep_flow`."""

    def __init__(self, obs_dim, action_dim, hidden_dims, layer_norm=False, with_time=True):
        super().__init__()
        self.with_time = with_time
        in_dim = obs_dim + action_dim + (1 if with_time else 0)
        self.net = mlp(in_dim, hidden_dims, action_dim, layer_norm)

    def forward(self, obs, x, t=None):
        inp = [obs, x] + ([t] if self.with_time else [])
        return self.net(th.cat(inp, dim=-1))


class TwinValue(nn.Module):
    """Two independent Q_k(o_i, a_i). Returns [2, ...] so the learner can pick
    mean (official default, q_agg='mean') or min."""

    def __init__(self, obs_dim, action_dim, hidden_dims, layer_norm=True):
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, hidden_dims, 1, layer_norm)
        self.q2 = mlp(obs_dim + action_dim, hidden_dims, 1, layer_norm)

    def forward(self, obs, actions):
        inp = th.cat([obs, actions], dim=-1)
        return th.stack([self.q1(inp).squeeze(-1), self.q2(inp).squeeze(-1)], dim=0)
