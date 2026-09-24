"""Toy: can a flow policy trained by PPO keep two modes, when a Gaussian cannot?

One state, one continuous action a in (0,1). The reward has two equally good
peaks and a valley between them, so "the average of the two right answers" is
the worst answer:

    r(a) = max( N(a; m1, w), N(a; m2, w) )          (peaks normalised to 1)

A diagonal Gaussian policy is structurally unable to cover both peaks: its only
options are to sit on one peak or to straddle the valley. A flow policy
a = sigmoid(mu(eps)) can in principle map different eps to different peaks.

The question this file answers is NOT "which gets more reward" -- both peaks pay
the same, so a single-peak policy is already optimal in expectation. It is
whether the flow *keeps* both modes while PPO optimises, since PPO's objective
is indifferent between covering one peak and covering two.

Everything is deliberately minimal and self-contained (no env, no critic, no
GAE): a bandit, an advantage that is just the centred reward, and the same
clipped surrogate and endpoint-parameterised flow the real learner uses.

    /venv/MPE/bin/python scripts/toy_multimodal.py --policy flow
    /venv/MPE/bin/python scripts/toy_multimodal.py --policy gauss
"""

import argparse
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── the task ────────────────────────────────────────────────────────────────
def reward(a, m1, m2, w, h1=1.0, h2=1.0, w2=None):
    """Two Gaussian bumps. Equal height/width by default (the mode-averaging
    test); give h1 < h2 and a narrower w2 to turn it into the trap version --
    a wide, easy, low-paying peak A against a narrow, harder, high-paying B."""
    w2 = w if w2 is None else w2
    r1 = h1 * th.exp(-((a - m1) ** 2) / (2 * w ** 2))
    r2 = h2 * th.exp(-((a - m2) ** 2) / (2 * w2 ** 2))
    return th.maximum(r1, r2)


# ── policies ────────────────────────────────────────────────────────────────
class GaussPolicy(nn.Module):
    """a = sigmoid(mu + sigma * xi); mu is a free scalar (single state)."""

    def __init__(self, sigma_init=1.0):
        super().__init__()
        self.mu = nn.Parameter(th.zeros(1))
        self.log_std = nn.Parameter(th.full((1,), math.log(sigma_init)))

    def mean(self, eps):                       # ignores eps -- unimodal by construction
        return self.mu.expand_as(eps).contiguous()

    def sigma(self):
        return th.exp(self.log_std)


class FlowPolicy(nn.Module):
    """a = sigmoid(mu(eps) + sigma * xi) with mu from the endpoint-parameterised
    K-step flow used by the real actor: x <- x + (g(x,t) - x)/(K - k)."""

    def __init__(self, K=5, hidden=64, sigma_init=1.0, zero_init=True):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(1 + 8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(th.full((1,), math.log(sigma_init)))

    def _embed_t(self, t):
        freqs = 2.0 ** th.arange(4, device=t.device, dtype=t.dtype)
        return th.cat([th.cos(t * freqs), th.sin(t * freqs)], dim=-1)

    def mean(self, eps):
        x = eps
        for k in range(self.K):
            t = x.new_full(x.shape, k / self.K)
            g = self.net(th.cat([x, self._embed_t(t)], dim=-1))
            x = x + (g - x) / (self.K - k)
        return x

    def sigma(self):
        return th.exp(self.log_std)


# ── PPO on a bandit ─────────────────────────────────────────────────────────
def train(policy, steps, batch, lr, clip, epochs, m1, m2, w, log_every, seed, rk=None):
    th.manual_seed(seed)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    hist = []
    for it in range(1, steps + 1):
        with th.no_grad():
            eps = th.randn(batch, 1)
            mu_old = policy.mean(eps).clone()
            s_old = policy.sigma().clone()
            xi = th.randn(batch, 1)
            u = mu_old + s_old * xi
            a = th.sigmoid(u)
            r = reward(a, m1, m2, w, **(rk or {}))
            adv = (r - r.mean()) / (r.std() + 1e-8)

        for _ in range(epochs):
            mu_new = policy.mean(eps)
            s_new = policy.sigma()
            logr = (-0.5 * ((u - mu_new) / s_new) ** 2
                    + 0.5 * ((u - mu_old) / s_old) ** 2
                    - th.log(s_new / s_old)).sum(-1)
            ratio = th.exp(logr.clamp(-3, 3))
            loss = -th.min(ratio * adv.squeeze(-1),
                           ratio.clamp(1 - clip, 1 + clip) * adv.squeeze(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        if it % log_every == 0 or it == 1:
            hist.append((it, *evaluate(policy, m1, m2, w, rk=rk)))
    return hist


@th.no_grad()
def evaluate(policy, m1, m2, w, n=20000, rk=None):
    """Mode coverage: sample eps (no terminal noise, so this measures what the
    FLOW carries, not what sigma smears) and see how the actions split between
    the two peaks."""
    eps = th.randn(n, 1)
    a = th.sigmoid(policy.mean(eps)).squeeze(-1)
    rk = rk or {}
    r_sampled = reward(th.sigmoid(policy.mean(eps) + policy.sigma() * th.randn(n, 1)), m1, m2, w, **rk).mean()
    w2 = rk.get('w2', w)
    near1 = (a - m1).abs() < 2 * w
    near2 = (a - m2).abs() < 2 * w2
    p1, p2 = float(near1.float().mean()), float(near2.float().mean())
    minor = min(p1, p2)                       # 0 => collapsed to one mode
    return float(r_sampled), float(a.std()), p1, p2, minor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["flow", "gauss"], default="flow")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--m1", type=float, default=0.2)
    p.add_argument("--m2", type=float, default=0.8)
    p.add_argument("--w", type=float, default=0.05)
    p.add_argument("--h1", type=float, default=1.0)
    p.add_argument("--h2", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=None)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--log-every", type=int, default=500)
    a = p.parse_args()

    print(f"task: peaks at {a.m1} and {a.m2}, width {a.w}; policy={a.policy}, "
          f"K={a.K}, sigma_init={a.sigma_init}, {a.seeds} seeds\n")
    print(f"{'seed':>5}{'iter':>7}{'reward':>9}{'std(a)':>9}{'p(peak1)':>10}{'p(peak2)':>10}{'minor':>8}")
    finals = []
    for s in range(a.seeds):
        th.manual_seed(s)
        pol = (FlowPolicy(K=a.K, sigma_init=a.sigma_init) if a.policy == "flow"
               else GaussPolicy(sigma_init=a.sigma_init))
        rk = dict(h1=a.h1, h2=a.h2, w2=a.w2)
        h = train(pol, a.steps, a.batch, a.lr, a.clip, a.epochs, a.m1, a.m2, a.w, a.log_every, s, rk=rk)
        for it, r, sd, p1, p2, mi in h:
            print(f"{s:5d}{it:7d}{r:9.3f}{sd:9.3f}{p1:10.3f}{p2:10.3f}{mi:8.3f}")
        finals.append(h[-1])
        print()
    f = np.array([[x[1], x[2], x[5]] for x in finals])
    print(f"across {a.seeds} seeds -- reward {f[:,0].mean():.3f}+-{f[:,0].std():.3f}  "
          f"std(a) {f[:,1].mean():.3f}  minor-mode mass {f[:,2].mean():.3f}+-{f[:,2].std():.3f}")
    print(f"seeds that kept BOTH modes (minor > 0.1): {int((f[:,2] > 0.1).sum())}/{a.seeds}")
    pk = np.array([[x[3], x[4]] for x in finals])
    print(f"seeds ending on peak1(easy/low): {int((pk[:,0] > 0.5).sum())}/{a.seeds}   "
          f"on peak2(hard/high): {int((pk[:,1] > 0.5).sum())}/{a.seeds}")


if __name__ == "__main__":
    main()
