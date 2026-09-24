"""Offline MAC-Flow (PyTorch) on an exported OMIGA/OG-MARL dataset.

This is the setting the paper actually runs: no environment interaction, a fixed
dataset, periodic evaluation in the environment. It reuses the ported networks
and losses from macflow/paper_nets.py + paper_learner.py's formulation, so a run
here against the same vault the official JAX code used is the check that the
port is faithful (target: ~3,900-4,000 on OMIGA 6halfcheetah/Expert).

Standalone: it does not touch main.py / run.py, because epymarl's loop is built
around an environment runner and an episode buffer that an offline dataset does
not need.

    /venv/MPE/bin/python src/macflow/train_offline.py \
        --data /data/omiga_6halfcheetah_Expert.npz \
        --scenario HalfCheetah --agent-conf 6x1 \
        --steps 500000 --eval-interval 50000 --alpha 3.0

Evaluation reproduces the OMIGA observation convention exactly (every agent sees
the full single-agent state, one-hot id appended, each vector normalised), which
is what the dataset was produced under; the physics is gymnasium MaMuJoCo on the
installed MuJoCo, not the MuJoCo 2.0 the dataset was collected on.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch as th
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ on the path
from macflow.paper_nets import ActorVectorField, TwinValue


# ── evaluation environment (OMIGA observation convention) ────────────────────
class OmigaEvalEnv:
    def __init__(self, scenario, agent_conf, seed=0):
        from gymnasium_robotics import mamujoco_v1
        self.env = mamujoco_v1.parallel_env(scenario, agent_conf, agent_obsk=1)
        self.env.reset(seed=seed)
        self.agents = list(self.env.possible_agents)
        self.n = len(self.agents)
        self.low = np.asarray(self.env.action_space(self.agents[0]).low)
        self.high = np.asarray(self.env.action_space(self.agents[0]).high)

    def _obs(self):
        full = np.asarray(self.env.single_agent_env.unwrapped._get_obs(), dtype=np.float32)
        out = np.zeros((self.n, full.size + self.n), dtype=np.float32)
        for i in range(self.n):
            v = np.concatenate([full, np.eye(self.n, dtype=np.float32)[i]])
            out[i] = (v - v.mean()) / v.std()
        return out

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self._obs()

    def step(self, u):
        """u: [N, A] in [-1, 1] (MuJoCo clamps to the actuator range anyway)."""
        acts = {a: np.clip(u[i], self.low, self.high).astype("float32")
                for i, a in enumerate(self.agents)}
        _, rew, term, trunc, _ = self.env.step(acts)
        r = float(np.mean(list(rew.values())))
        return self._obs(), r, any(term.values()) or any(trunc.values())


@th.no_grad()
def evaluate(onestep, env, n_eps, action_dim, device):
    rets = []
    for k in range(n_eps):
        obs = env.reset(seed=1000 + k)
        R = 0.0
        for _ in range(1000):
            o = th.as_tensor(obs, device=device)
            z = th.randn(o.shape[0], action_dim, device=device)
            u = onestep(o, z).clamp(-1, 1).cpu().numpy()
            obs, r, done = env.step(u)
            R += r
            if done:
                break
        rets.append(R)
    return float(np.mean(rets)), float(np.max(rets)), float(np.min(rets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--scenario", default="HalfCheetah")
    p.add_argument("--agent-conf", default="6x1")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--discount", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512, 512])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/macflow_offline")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-name", default="MACFlow-paper-pytorch-offline")
    args = p.parse_args()

    th.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if th.cuda.is_available() else "cpu"

    d = np.load(args.data)
    obs = th.as_tensor(d["observations"])                      # [T,N,O] cpu
    act = th.as_tensor(d["actions"])                           # [T,N,A]
    rew = th.as_tensor(d["rewards"])                           # [T,N]
    term = th.as_tensor(d["terminals"])                        # [T,N]
    starts, lengths = d["ep_starts"], d["ep_lengths"]
    N, O, A = obs.shape[1], obs.shape[2], act.shape[2]
    L = args.seq_len + 1
    # window start offsets that stay inside a single episode
    valid = np.concatenate([s + np.arange(l - L + 1) for s, l in zip(starts, lengths)])
    print(f"data {args.data}: T={len(obs)} N={N} obs={O} act={A}  windows={len(valid):,}")

    bc_flow = ActorVectorField(O, A, args.hidden, layer_norm=False, with_time=True).to(dev)
    onestep = ActorVectorField(O, A, args.hidden, layer_norm=False, with_time=False).to(dev)
    critic = TwinValue(O, A, args.hidden, layer_norm=True).to(dev)
    target_critic = TwinValue(O, A, args.hidden, layer_norm=True).to(dev)
    target_critic.load_state_dict(critic.state_dict())
    for q in target_critic.parameters():
        q.requires_grad_(False)
    params = list(bc_flow.parameters()) + list(onestep.parameters()) + list(critic.parameters())
    opt = Adam(params, lr=args.lr)

    env = OmigaEvalEnv(args.scenario, args.agent_conf, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    ev = csv.writer(open(os.path.join(args.out, "eval.csv"), "w", newline="", buffering=1))
    ev.writerow(["step", "mean_return", "max_return", "min_return"])
    tr = csv.writer(open(os.path.join(args.out, "train.csv"), "w", newline="", buffering=1))
    tr.writerow(["step", "critic_loss", "bc_flow_loss", "distill_loss", "q_loss", "q_mean"])
    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    def flow_action(o, z):
        x = z
        for i in range(args.flow_steps):
            t = x.new_full(x.shape[:-1] + (1,), i / args.flow_steps)
            x = x + bc_flow(o, x, t) / args.flow_steps
        return x.clamp(-1, 1)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        w = np.random.randint(0, len(valid), size=args.batch_size)
        idx = valid[w][:, None] + np.arange(L)[None, :]          # [B,L]
        i = th.as_tensor(idx)
        o = obs[i].to(dev, non_blocking=True)                    # [B,L,N,O]
        a = act[i][:, :-1].to(dev)                               # [B,T,N,A]
        r = rew[i][:, :-1].to(dev)                               # [B,T,N]
        dn = term[i][:, 1:].to(dev)                              # [B,T,N]

        with th.no_grad():
            z_n = th.randn(*o[:, 1:].shape[:-1], A, device=dev)
            a_n = onestep(o[:, 1:], z_n).clamp(-1, 1)
            nq = target_critic(o[:, 1:], a_n).mean(dim=0)        # mean over the 2-ensemble
            tgt = (r + args.discount * (1.0 - dn) * nq).mean(dim=-1)   # IGM average mixer
        q = critic(o[:, :-1], a).mean(dim=-1)                    # [2,B,T]
        critic_loss = ((q - tgt.unsqueeze(0)) ** 2).mean()

        x0 = th.randn_like(a)
        t = th.rand(*a.shape[:-1], 1, device=dev)
        bc = ((bc_flow(o[:, :-1], (1 - t) * x0 + t * a, t) - (a - x0)) ** 2).mean()

        z = th.randn_like(a)
        with th.no_grad():
            tgt_flow = flow_action(o[:, :-1], z)
        pi = onestep(o[:, :-1], z)
        distill = ((pi - tgt_flow) ** 2).mean()

        mixed_q = critic(o[:, :-1], pi.clamp(-1, 1)).mean(dim=0).mean(dim=-1)
        q_loss = -mixed_q.mean() * (1.0 / mixed_q.detach().abs().mean().clamp(min=1e-8))
        loss = critic_loss + bc + args.alpha * distill + q_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with th.no_grad():
            for tp, pp in zip(target_critic.parameters(), critic.parameters()):
                tp.data.mul_(1 - args.tau).add_(args.tau * pp.data)

        if step % args.log_interval == 0:
            row = [step, float(critic_loss), float(bc), float(distill), float(q_loss), float(mixed_q.mean())]
            tr.writerow(row)
            if wb:
                wb.log(dict(zip(["critic_loss", "bc_flow_loss", "distill_loss", "q_loss", "q_mean"], row[1:])), step=step)
        if step == 1 or step % args.eval_interval == 0:
            m, mx, mn = evaluate(onestep, env, args.eval_episodes, A, dev)
            ev.writerow([step, m, mx, mn])
            print(f"[{step:>7}] eval {m:8.1f} (max {mx:.0f} min {mn:.0f})  "
                  f"critic {float(critic_loss):.3f} bc {float(bc):.3f} distill {float(distill):.3f}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            if wb:
                wb.log({"test_return_per_agent": m, "test_return_mean": m * N,
                        "eval_max": mx, "eval_min": mn}, step=step)
            th.save({"onestep": onestep.state_dict(), "bc_flow": bc_flow.state_dict(),
                     "critic": critic.state_dict()}, os.path.join(args.out, "ckpt.pt"))
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
