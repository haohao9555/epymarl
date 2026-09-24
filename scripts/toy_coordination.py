"""Toy: joint-action multimodality, where each architectural piece is necessary.

Two agents, two landmarks at m1 and m2, one agent per landmark. The joint reward
is the best of the two assignments:

    r(a1, a2) = max( K(a1-m1) K(a2-m2),  K(a1-m2) K(a2-m1) ),   K = Gaussian bump

so the joint action space has exactly two equally good modes, (m1,m2) and
(m2,m1), and the average of the two -- both agents in the middle -- pays nothing.

The agents are **parameter-shared and see an identical observation** (there is no
agent id). That is what makes this a multi-agent problem rather than a
single-agent one: with an id, "agent 1 always takes the left landmark" is a
deterministic factorised optimum and a Gaussian MAPPO solves it outright. Without
one, a deterministic policy makes both agents do the same thing and collide, so
the only way to split the landmarks is through each agent's own noise.

That gives a three-way comparison in which each piece is load-bearing:

  gauss   one shared Gaussian, agents sample independently
          -> unimodal marginal: it cannot put mass on both landmarks at once
  flow    shared endpoint-parameterised flow, agents sample eps independently
          -> the marginal CAN be bimodal (eps<0 -> m1, eps>0 -> m2), but the two
             agents choose independently, so they collide about half the time
  attn    the same flow with inter-agent attention inside every Euler step
          -> each agent sees where the other is currently heading and can take
             the other landmark

Unlike the single-agent two-peak toy, PPO is *not* indifferent to multimodality
here: a unimodal policy scores ~0 and a bimodal one ~0.5, so keeping both modes
is strictly better and the gradient says so.

    /venv/MPE/bin/python scripts/toy_coordination.py --policy gauss
    /venv/MPE/bin/python scripts/toy_coordination.py --policy flow
    /venv/MPE/bin/python scripts/toy_coordination.py --policy attn
"""

import argparse
import math

import numpy as np
import torch as th
import torch.nn as nn


N_AGENTS = 2
DEV = th.device('cuda' if th.cuda.is_available() else 'cpu')


def reward(a, m1, m2, w):
    """a: [B, 2] actions in (0,1) -> [B] reward. Best of the two assignments."""
    bump = lambda x, m: th.exp(-((x - m) ** 2) / (2 * w ** 2))
    r12 = bump(a[:, 0], m1) * bump(a[:, 1], m2)
    r21 = bump(a[:, 0], m2) * bump(a[:, 1], m1)
    return th.maximum(r12, r21)


def embed_t(t):
    freqs = 2.0 ** th.arange(4, device=t.device, dtype=t.dtype)
    return th.cat([th.cos(t * freqs), th.sin(t * freqs)], dim=-1)


class GaussPolicy(nn.Module):
    """One shared Gaussian; both agents draw from it. Unimodal by construction.

    With attention=True the mean is produced by a network that runs one round of
    inter-agent self-attention over the agents' observation embeddings. It is the
    control for "does the flow matter, or would attention alone do?" -- and it is
    degenerate on purpose: the observations are identical, so every agent feeds
    attention the same token, gets the same context back, and ends up with the
    same mean. The only asymmetry available, the sampling noise, enters *after*
    the mean, where attention can no longer see it. That is exactly what the flow
    changes: it moves the noise inside the computation of the mean."""

    def __init__(self, sigma_init=1.0, attention=False, hidden=64, heads=4, **_):
        super().__init__()
        self.attention = attention
        if attention:
            self.tok = nn.Parameter(th.zeros(1, 1, hidden))     # identical obs -> one token
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        else:
            self.mu = nn.Parameter(th.zeros(1))
        self.log_std = nn.Parameter(th.full((1,), math.log(sigma_init)))

    def mean(self, eps):                       # eps: [B, N, 1] -> [B, N, 1]
        if not self.attention:
            return self.mu.expand_as(eps).contiguous()
        B, N = eps.shape[0], eps.shape[1]
        z = self.tok.expand(B, N, -1)
        z = z + self.attn(z, z, z, need_weights=False)[0]
        return self.out(z)

    def sigma(self):
        return th.exp(self.log_std)


class FlowPolicy(nn.Module):
    """Shared endpoint-parameterised flow, x <- x + (g - x)/(K - k).
    With attention=True one round of inter-agent self-attention runs per step."""

    def __init__(self, K=5, hidden=64, sigma_init=1.0, attention=False, heads=4,
                 init_scale=0.0, skip=0.0):
        super().__init__()
        self.K, self.attention = K, attention
        self.fc1 = nn.Linear(1 + 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # init_scale=0 is the zero-init used on MuJoCo: it makes mu == 0 for every
        # eps, so the flow starts as a CONSTANT map. That is not just a slow start
        # -- the gradient that would break the agents' symmetry is exactly zero in
        # expectation there (the reward depends only on the terminal noise, which is
        # independent of eps), so PPO can only escape it by finite-batch luck. A
        # small non-zero scale gives mu a genuine eps-dependence to amplify.
        nn.init.normal_(self.fc2.weight, std=init_scale)
        nn.init.zeros_(self.fc2.bias)
        # A fixed skip g = skip*x + net(x). With the endpoint recursion
        # x <- x + (g-x)/(K-k) this makes mu = alpha * eps at init, where
        #     alpha = prod_{j=1..K} (1 + (skip-1)/j).
        # skip=0 -> alpha=0: the flow is the CONSTANT map mu==0 (current zero-init).
        # skip=1 -> alpha=1: the flow is the IDENTITY, i.e. the policy is exactly a
        #   Gaussian with std sqrt(1+sigma^2) -- so MAFPO starts as MAPPO and can
        #   only deform away from it, and eps carries real action variance from
        #   step one instead of the 0.6% we measured on MuJoCo.
        self.skip = skip
        self.alpha = float(np.prod([1 + (skip - 1) / j for j in range(1, K + 1)]))
        if attention:
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
        self.log_std = nn.Parameter(th.full((1,), math.log(sigma_init)))

    def _g(self, x, t):                        # x,t: [B, N, 1] -> [B, N, 1]
        z = th.relu(self.fc1(th.cat([x, embed_t(t)], dim=-1)))
        if self.attention:
            z = z + self.attn(z, z, z, need_weights=False)[0]
        return self.skip * x + self.fc2(z)

    def mean(self, eps):
        x = eps
        for k in range(self.K):
            t = x.new_full(x.shape, k / self.K)
            x = x + (self._g(x, t) - x) / (self.K - k)
        return x

    def sigma(self):
        return th.exp(self.log_std)


def train(policy, steps, batch, lr, clip, epochs, m1, m2, w, log_every, seed, ent_coef=0.0):
    th.manual_seed(seed)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    hist = []
    for it in range(1, steps + 1):
        with th.no_grad():
            eps = th.randn(batch, N_AGENTS, 1, device=DEV)
            mu_old = policy.mean(eps).clone()
            s_old = policy.sigma().clone()
            u = mu_old + s_old * th.randn(batch, N_AGENTS, 1, device=DEV)
            a = th.sigmoid(u)
            r = reward(a.squeeze(-1), m1, m2, w)                 # [B]
            adv = ((r - r.mean()) / (r.std() + 1e-8)).unsqueeze(-1)   # shared reward

        for _ in range(epochs):
            mu_new = policy.mean(eps)
            s_new = policy.sigma()
            # per-agent Gaussian ratio, summed over agents (as in the real learner)
            logr = (-0.5 * ((u - mu_new) / s_new) ** 2
                    + 0.5 * ((u - mu_old) / s_old) ** 2
                    - th.log(s_new / s_old)).squeeze(-1).sum(-1)
            ratio = th.exp(logr.clamp(-3, 3))
            adv_f = adv.squeeze(-1)
            loss = -th.min(ratio * adv_f, ratio.clamp(1 - clip, 1 + clip) * adv_f).mean()
            if ent_coef:
                loss = loss - ent_coef * th.log(s_new).sum()
            opt.zero_grad(); loss.backward(); opt.step()

        if it % log_every == 0 or it == 1:
            hist.append((it, *evaluate(policy, m1, m2, w)))
    return hist


@th.no_grad()
def evaluate(policy, m1, m2, w, n=20000):
    # A dedicated generator: evaluation must not consume the training RNG, or
    # changing --log-every silently changes the trajectory (it did: the same
    # seed reached 0.96 with 4 evaluations and 0.24 with 30).
    g = th.Generator(device=DEV); g.manual_seed(12345)
    eps = th.randn(n, N_AGENTS, 1, device=DEV, generator=g)
    mu = policy.mean(eps)
    a = th.sigmoid(mu + policy.sigma() * th.randn(n, N_AGENTS, 1, device=DEV, generator=g)).squeeze(-1)
    r = float(reward(a, m1, m2, w).mean())

    def coverage(x):
        """fraction of draws where the two agents sit on DIFFERENT landmarks,
        and where they sit on the SAME one."""
        n1 = (x - m1).abs() < w
        n2 = (x - m2).abs() < w
        sp = float(((n1[:, 0] & n2[:, 1]) | (n2[:, 0] & n1[:, 1])).float().mean())
        co = float(((n1[:, 0] & n1[:, 1]) | (n2[:, 0] & n2[:, 1])).float().mean())
        return sp, co, n1, n2

    # executed actions (with the terminal noise) -- the fair cross-architecture
    # measure: a Gaussian's mean is identical for both agents, so anything
    # computed on the mean alone is 0 for it by definition, while its reward
    # comes entirely from sigma smearing the two draws apart.
    split_exec, collide_exec, _, _ = coverage(a)
    # the mean map alone -- what the FLOW carries before any noise is added
    sp_mu, co_mu, n1, n2 = coverage(th.sigmoid(mu).squeeze(-1))
    bimodal = float((n1[:, 0].float().mean() > 0.1) and (n2[:, 0].float().mean() > 0.1))
    return r, split_exec, collide_exec, sp_mu, bimodal, float(policy.sigma())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["gauss", "gauss_attn", "flow", "attn"], default="attn")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--m1", type=float, default=0.2)
    p.add_argument("--m2", type=float, default=0.8)
    p.add_argument("--w", type=float, default=0.08)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--init-scale", type=float, default=0.0)
    p.add_argument("--skip", type=float, default=0.0)
    a = p.parse_args()

    print(f"2 agents (shared params, identical observation), landmarks {a.m1}/{a.m2}, "
          f"width {a.w}; policy={a.policy}, {a.seeds} seeds")
    print(f"{'seed':>5}{'iter':>7}{'reward':>9}{'split_ex':>10}{'collide':>9}{'split_mu':>10}{'bimod':>7}{'sigma':>8}")
    finals = []
    for s in range(a.seeds):
        th.manual_seed(s)
        if a.policy in ("gauss", "gauss_attn"):
            pol = GaussPolicy(sigma_init=a.sigma_init, attention=(a.policy == "gauss_attn"))
        else:
            pol = FlowPolicy(K=a.K, sigma_init=a.sigma_init, attention=(a.policy == "attn"),
                             init_scale=a.init_scale, skip=a.skip)
        pol = pol.to(DEV)
        h = train(pol, a.steps, a.batch, a.lr, a.clip, a.epochs, a.m1, a.m2, a.w, a.log_every, s, a.ent_coef)
        for it, r, spe, coe, spm, bi, sg in h:
            print(f"{s:5d}{it:7d}{r:9.3f}{spe:10.3f}{coe:9.3f}{spm:10.3f}{bi:7.0f}{sg:8.3f}")
        finals.append(h[-1]); print()
    f = np.array([[x[1], x[2], x[3], x[4], x[5]] for x in finals])
    print(f"across {a.seeds} seeds -- reward {f[:,0].mean():.3f}+-{f[:,0].std():.3f}  "
          f"split(executed) {f[:,1].mean():.3f}  collide {f[:,2].mean():.3f}  "
          f"split(mean-map) {f[:,3].mean():.3f}  bimodal {int(f[:,4].sum())}/{a.seeds}")


if __name__ == "__main__":
    main()
