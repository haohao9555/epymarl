"""Runs the four conditions of the coordination toy and draws the figure.

    /venv/MPE/bin/python scripts/toy_coordination_figure.py --seeds 5
"""
import argparse
import json
import math
import os
import sys

import matplotlib
import numpy as np
import torch as th

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_coordination import DEV, FlowPolicy, GaussPolicy, train

SUR, TXT, MUT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COND = [("MAPPO (Gaussian)", "gauss", "#52514e"),
        ("MAPPO + attention", "gauss_attn", "#eda100"),
        ("Flow policy", "flow", "#eb6834"),
        ("Flow + attention (ours)", "attn", "#2a78d6")]


def build(tag, K, sig, skip):
    """All four conditions start from the SAME action distribution.

    With skip=1 the flow is the identity at init (mu == eps), so a flow policy
    with terminal noise `sig` emits sigmoid(N(0, 1 + sig^2)) -- exactly a Gaussian
    policy of std sqrt(1 + sig^2). Giving the Gaussian baselines that std makes the
    comparison start from one common point: the flow begins AS the baseline and can
    only deform away from it. (skip=0 is the zero-init variant, where mu == 0 for
    every eps and the flow starts as a constant map.)"""
    if tag in ("gauss", "gauss_attn"):
        alpha = np.prod([1 + (skip - 1) / j for j in range(1, K + 1)])   # std of mu at init
        return GaussPolicy(sigma_init=math.sqrt(alpha ** 2 + sig ** 2),
                           attention=(tag == "gauss_attn")).to(DEV)
    return FlowPolicy(K=K, sigma_init=sig, attention=(tag == "attn"), skip=skip).to(DEV)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w", type=float, default=0.15)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--m1", type=float, default=0.2)
    p.add_argument("--m2", type=float, default=0.8)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--skip", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--out", default="figs/toy_coordination.png")
    a = p.parse_args()

    res = {}
    for name, tag, _ in COND:
        curves, finals = [], []
        for s in range(a.seeds):
            th.manual_seed(s)
            h = train(build(tag, a.K, a.sigma_init, a.skip), a.steps, a.batch, a.lr, 0.2, 4,
                      a.m1, a.m2, a.w, a.every, s, a.ent_coef)
            curves.append([(it, r) for it, r, *_ in h])
            finals.append(h[-1])
        res[tag] = dict(curves=curves, finals=finals)
        f = np.array([[x[1], x[2], x[3]] for x in finals])
        print(f"{name:26s} reward {f[:,0].mean():.3f}+-{f[:,0].std():.3f}  "
              f"split(exec) {f[:,1].mean():.3f}  collide {f[:,2].mean():.3f}")
    json.dump({k: {"finals": v["finals"]} for k, v in res.items()},
              open("figs/toy_coordination.json", "w"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw=dict(width_ratios=[1.55, 1]), facecolor=SUR)
    for A in (ax, ax2):
        A.set_facecolor(SUR)
        [A.spines[k].set_visible(False) for k in ("top", "right")]
        A.spines["left"].set_color("#d5d4cf")
        A.spines["bottom"].set_color("#d5d4cf")
        A.grid(axis="y", color=GRID, lw=0.8)
        A.tick_params(colors=MUT)

    for y, lab in [(1.0, "perfect coordination"), (0.5, "independent random assignment")]:
        ax.axhline(y, color="#c8c7c1", ls="--", lw=1.2)
        ax.text(a.steps * 0.99, y + 0.015, lab, color=MUT, fontsize=8, ha="right")
    for name, tag, c in COND:
        cur = np.array(res[tag]["curves"], dtype=float)          # [S, T, 2]
        it, m, sd = cur[0, :, 0], cur[:, :, 1].mean(0), cur[:, :, 1].std(0)
        ax.fill_between(it, m - sd, m + sd, color=c, alpha=0.14, lw=0)
        ax.plot(it, m, color=c, lw=2.2, label=name)
    ax.set_xlim(0, a.steps)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("PPO updates", color=MUT)
    ax.set_ylabel("joint reward", color=MUT)
    ax.set_title("Two symmetric agents, two landmarks, two equivalent assignments",
                 color=TXT, fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="center right", fontsize=9)

    x, wdt = np.arange(len(COND)), 0.38
    sp = [np.array([f[1] for f in res[t]["finals"]]) for _, t, _ in COND]
    co = [np.array([f[2] for f in res[t]["finals"]]) for _, t, _ in COND]
    ax2.bar(x - wdt / 2, [v.mean() for v in sp], wdt, yerr=[v.std() for v in sp],
            color=[c for *_, c in COND], label="split (one agent each)", capsize=3)
    ax2.bar(x + wdt / 2, [v.mean() for v in co], wdt, yerr=[v.std() for v in co],
            color="#d9d8d2", label="collide (both on one)", capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["MAPPO", "MAPPO\n+attn", "Flow", "Flow\n+attn (ours)"], fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("fraction of executed joint actions", color=MUT)
    ax2.set_title("Who ends up where", color=TXT, fontsize=11.5, loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.text(0.012, 0.015,
             "Agents share parameters and receive an identical observation, so a deterministic policy makes both do the same thing: the only way to split "
             "the landmarks is through each agent's own sampling noise.\nA Gaussian policy adds that noise after its mean, where attention cannot see it -- "
             "which is why attention leaves it unchanged. A flow policy passes the noise through the network, so the agents differ internally and attention "
             f"can assign them.\n{a.seeds} seeds, shaded band is +-1 s.d.; identical hyper-parameters across all four conditions.",
             color=MUT, fontsize=7.6)
    plt.tight_layout(rect=(0, 0.085, 1, 1))
    plt.savefig(a.out, dpi=160)
    print("->", a.out)


if __name__ == "__main__":
    main()
