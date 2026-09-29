"""Mechanism diagnostics for the lever-pulling task (src/envs/lever_env.py).

Loads one mafpo_gauss checkpoint (CommFlow / MAFPO / MAPPO / MAPPO+attn) and measures, on the
trained policy and without any training or return curves, whether coordination comes from
exchanging each agent's own noise:

  M1  distinct-lever fraction, from mu alone (n = 0) and as executed (n ~ N(0, sigma^2)),
      against the cap 1 - (1 - 1/m)^m of any policy whose agents are independent given the state
  M2  per ODE round: distinct fraction of the intents x_k and of the endpoint guesses g_k
      (negotiation should resolve conflicts round by round)       [flow + endpoint only]
  M3  eps swap: swap agent 0's and agent 1's eps -- do their levers swap too? ("noise = identity")
  M4  |d mu_i / d eps_i| vs |d mu_i / d eps_j|, j != i: does an agent's action depend on its
      teammates' noise?
  M5  attention switched off at test time (gate -> 0): what is left of M1
  M6  sigma sweep: M1 with the terminal noise replaced by N(0, s^2) for several s

Usage:
  python scripts/lever_diag.py results/sacred/mafpo_gauss/lever/<id> [--step N] [--n 4096]
"""

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import torch as th

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from controllers.mafpo_gauss_mac import MAFPOGaussMAC  # noqa: E402


def load(sacred_dir, step=None):
    cfg = json.load(open(os.path.join(sacred_dir, "config.json")))
    ea = cfg["env_args"]
    assert ea["key"] == "lever", ea["key"]
    n_agents = int(ea["n_agents"])
    m = int(ea.get("n_levers") or n_agents)
    obs_dim = 1 if int(ea.get("steps", 1)) == 1 else 2
    args = SimpleNamespace(**cfg)
    args.n_agents, args.n_actions, args.use_cuda = n_agents, 1, False
    scheme = {"obs": {"vshape": obs_dim}, "state": {"vshape": n_agents * obs_dim},
              "actions": {"vshape": (1,)}}
    mac = MAFPOGaussMAC(scheme, None, args)

    runs = sorted(glob.glob(os.path.join(REPO, "results/models", f"{cfg['name']}_seed{cfg['seed']}_lever_*")))
    assert runs, f"no checkpoint for {cfg['name']} seed {cfg['seed']}"
    steps = sorted(int(s) for s in os.listdir(runs[-1]) if s.isdigit())
    step = steps[-1] if step is None else min(steps, key=lambda s: abs(s - step))
    mac.load_models(os.path.join(runs[-1], str(step)))
    mac.agent.eval()
    return cfg, args, mac, m, obs_dim, step


def levers(u, m):
    """pre-sigmoid action -> lever index, exactly as LeverEnv.lever."""
    return th.clamp((th.sigmoid(u) * m).floor().long(), max=m - 1)


def distinct_frac(L, m):
    """L [B, N] lever indices -> mean over episodes of (#distinct levers) / m."""
    s, _ = th.sort(L, dim=1)
    return ((s[:, 1:] != s[:, :-1]).sum(1) + 1).float().mean().item() / m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sacred_dir")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    th.manual_seed(a.seed)

    cfg, args, mac, m, obs_dim, step = load(a.sacred_dir, a.step)
    agent, N, B = mac.agent, args.n_agents, a.n
    kind = cfg.get("gauss_mu_source", "flow")
    attn = bool(cfg.get("flow_attention")) if kind == "flow" else bool(cfg.get("mu_attention"))
    name = f"{cfg['name']} @ {step}"

    with th.no_grad():
        obs = th.ones(B, N, obs_dim)
        inputs = [mac.obs_normalizer.normalize_obs(obs)]
        if getattr(args, "obs_agent_id", False):
            inputs.append(th.eye(N).unsqueeze(0).expand(B, -1, -1))
        inputs = th.cat(inputs, dim=-1).reshape(B * N, -1)
        hid = agent.init_hidden().unsqueeze(0).expand(B, N, -1)
        h = agent.encode(inputs, hid).view(B, N, -1)
        eps = th.randn(B, N, 1)
        sigma = agent.sigma()                                        # [N, 1]
        mu = agent.mean(h, eps)                                      # [B, N, 1]
        n = th.randn_like(mu) * sigma

    cap = 1 - (1 - 1 / m) ** m
    print(f"== {name}   kind={kind} attention={attn} m={m} N={N} K={cfg.get('cfm_rollout_steps')} "
          f"sigma={sigma.mean().item():.3f}  cap(independent)={cap:.3f}")

    # M1
    L_mu, L_exe = levers(mu, m)[..., 0], levers(mu + n, m)[..., 0]
    print(f"M1 distinct frac: mu-only {distinct_frac(L_mu, m):.3f} | executed {distinct_frac(L_exe, m):.3f}"
          f" | n flips a lever in {(L_mu != L_exe).float().mean().item():.3f} of agent-steps")

    # M2
    if kind == "flow" and cfg.get("flow_param") == "endpoint":
        K = int(cfg["cfm_rollout_steps"])
        x, row_x, row_g = eps.clone(), [], []
        with th.no_grad():
            for i in range(K):
                t = x.new_full(x.shape[:-1] + (1,), i / K)
                g = agent._head(h, x, t)
                row_x.append(distinct_frac(levers(x, m)[..., 0], m))
                row_g.append(distinct_frac(levers(g, m)[..., 0], m))
                x = x + (g - x) / (K - i)
        print("M2 per round  intent x_k: " + " ".join(f"{v:.3f}" for v in row_x)
              + f"  -> mu {distinct_frac(levers(x, m)[..., 0], m):.3f}")
        print("             guess  g_k: " + " ".join(f"{v:.3f}" for v in row_g))

    # M3
    if kind == "flow":
        with th.no_grad():
            eps_sw = eps.clone()
            eps_sw[:, [0, 1]] = eps[:, [1, 0]]
            L_sw = levers(agent.mean(h, eps_sw), m)[..., 0]
        diff = L_mu[:, 0] != L_mu[:, 1]
        follow = ((L_sw[:, 0] == L_mu[:, 1]) & (L_sw[:, 1] == L_mu[:, 0]))[diff].float().mean().item()
        others = (L_sw[:, 2:] == L_mu[:, 2:]).float().mean().item() if N > 2 else float("nan")
        print(f"M3 eps swap (episodes where agents 0,1 differ, {diff.float().mean().item():.2f} of all): "
              f"levers swap too {follow:.3f} | other agents unchanged {others:.3f}")

    # M4
    if kind == "flow":
        Bs = min(B, 512)
        e = eps[:Bs].clone().requires_grad_(True)
        mu_g = agent.mean(h[:Bs], e)
        self_s, cross_s = [], []
        for i in range(N):
            (gr,) = th.autograd.grad(mu_g[:, i, 0].sum(), e, retain_graph=True)
            gr = gr[..., 0].abs()                                    # [Bs, N]
            self_s.append(gr[:, i].mean().item())
            cross_s.append(th.cat([gr[:, :i], gr[:, i + 1:]], 1).mean().item())
        print(f"M4 |dmu_i/deps_i| {sum(self_s) / N:.4f} | |dmu_i/deps_j| (j!=i) {sum(cross_s) / N:.4f}")

    # M5
    gate = getattr(agent, "attn_gate", None) if kind == "flow" else getattr(agent, "mu_attn_gate", None)
    if attn and gate is not None:
        saved = gate.detach().clone()
        with th.no_grad():
            gate.zero_()
            mu0 = agent.mean(h, eps)
            gate.copy_(saved)
        print(f"M5 attention off (gate=0): mu-only {distinct_frac(levers(mu0, m)[..., 0], m):.3f}"
              f" | executed {distinct_frac(levers(mu0 + n, m)[..., 0], m):.3f}")

    # M6
    row = []
    for s in (0.0, 0.05, 0.1, 0.3, 0.6, 1.0):
        row.append(f"s={s}:{distinct_frac(levers(mu + th.randn_like(mu) * s, m)[..., 0], m):.3f}")
    print("M6 sigma sweep (executed): " + "  ".join(row))


if __name__ == "__main__":
    main()
