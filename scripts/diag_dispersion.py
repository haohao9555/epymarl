"""Read-only checkpoint rollouts for dispersion direction/target diagnostics.

Uses the production MAC, normalization, and action transform. No optimizer or
training runner is constructed. Target labels are conservative geometric
proxies, not observations of an explicit goal variable in the actor.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components.episode_buffer import EpisodeBatch
from controllers.mafpo_gauss_mac import MAFPOGaussMAC
import vmas


def masked_mean(x, mask):
    return float(np.asarray(x)[mask].mean()) if np.any(mask) else None


def angles(v):
    norm = np.linalg.norm(v, axis=-1)
    dot = (v[:, :-1] * v[:, 1:]).sum(-1)
    cosine = dot / np.maximum(norm[:, :-1] * norm[:, 1:], 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1))), norm


def summarize(d, pos_range):
    p, vel, eaten, alive = (d[k] for k in ("pos", "vel", "eaten", "alive"))
    b, t, n = d["action"].shape[:3]
    valid = np.broadcast_to(alive[..., None], (b, t, n))
    # Any food consumption can justify a change of direction. Exclude such
    # transitions from both turning and inferred-target-switch statistics.
    no_food_event = ~(eaten[:, 1:t] != eaten[:, :t-1]).any(-1)
    transition = alive[:, :-1] & alive[:, 1:] & no_food_event
    transition = np.broadcast_to(transition[..., None], (b, t-1, n))
    out = {}
    for name, vectors, threshold in (
        ("mean_action", d["mean_action"], 0.05),
        ("executed_action", d["action"], 0.05),
        ("velocity", vel[:, :t], 0.05),
    ):
        angle, norm = angles(vectors)
        eligible = transition & (norm[:, :-1] > threshold) & (norm[:, 1:] > threshold)
        out[name + "_turn_deg_mean"] = masked_mean(angle, eligible)
        out[name + "_turn_gt45_fraction"] = masked_mean(angle > 45, eligible)
        out[name + "_turn_gt90_fraction"] = masked_mean(angle > 90, eligible)
        out[name + "_eligible_transitions"] = int(eligible.sum())
        out[name + "_norm_mean"] = masked_mean(norm, valid)

    delta = np.diff(p, axis=1)
    path = (np.linalg.norm(delta, axis=-1) * valid).sum(1)
    end_t = alive.sum(1)
    end_pos = p[np.arange(b), end_t]
    displacement = np.linalg.norm(end_pos - p[:, 0], axis=-1)
    out["path_length_mean"] = float(path.mean())
    out["straightness_mean"] = masked_mean(displacement / np.maximum(path, 1e-12), path > 0.05)
    out["boundary_fraction"] = masked_mean((np.abs(p[:, :t]) >= 0.9 * pos_range).any(-1), valid)
    food_count = eaten[np.arange(b), end_t].sum(-1)
    out["foods_mean"] = float(food_count.mean())
    out["foods_histogram"] = np.bincount(food_count, minlength=eaten.shape[-1]+1).tolist()
    out["complete_fraction"] = float((food_count == eaten.shape[-1]).mean())
    out["moving_fraction"] = masked_mean(np.linalg.norm(vel[:, :t], axis=-1) > 0.05, valid)

    # Which uneaten food lies distinctly in the direction of motion? Ignore
    # near-zero speeds, poor alignment, close angular ties, and captured food.
    goal_vec = d["food"][:, None, None, :, :] - p[:, :t, :, None, :]
    distance = np.linalg.norm(goal_vec, axis=-1)
    speed = np.linalg.norm(vel[:, :t], axis=-1)
    cosine = (vel[:, :t, :, None, :] * goal_vec).sum(-1) / np.maximum(speed[..., None] * distance, 1e-12)
    available = ~eaten[:, :t, None, :]
    cosine = np.where(available & (distance > 0.085), cosine, -2.0)
    ordered = np.sort(cosine, axis=-1)
    best = ordered[..., -1]
    margin = best - ordered[..., -2]
    confident = valid & (speed > 0.05) & (best >= np.cos(np.pi/6)) & (margin >= 0.15)
    label = np.where(confident, cosine.argmax(-1), -1)
    eligible = transition & confident[:, :-1] & confident[:, 1:]
    switched = label[:, :-1] != label[:, 1:]
    out["target_proxy_confident_fraction"] = masked_mean(confident, valid)
    out["target_proxy_switch_fraction"] = masked_mean(switched, eligible)
    out["target_proxy_eligible_transitions"] = int(eligible.sum())
    out["target_proxy_switch_count"] = int((switched & eligible).sum())
    out["moving_away_from_all_food_fraction"] = masked_mean(best <= 0, valid & (speed > 0.05) & available.any(-1))
    out["heading_within30deg_of_food_fraction"] = masked_mean(best >= np.cos(np.pi/6), valid & (speed > 0.05) & available.any(-1))
    out["initial_mean_action_xy"] = d["mean_action"][:, 0].mean(axis=(0, 1)).tolist()
    out["initial_mean_action_up_fraction"] = float((d["mean_action"][:, 0, :, 1] > 0.05).mean())
    out["moving_up_fraction"] = masked_mean(vel[:, :t, :, 1] > 0, valid & (speed > 0.05))
    out["final_above_origin_fraction"] = float((end_pos[..., 1] > 0.05).mean())
    out["final_xy_mean"] = end_pos.mean(axis=(0, 1)).tolist()
    below = d["food"][..., 1] < 0
    out["foods_below_origin_total"] = int(below.sum())
    out["foods_below_origin_eaten"] = int((below & eaten[np.arange(b), end_t]).sum())
    pi, pj = np.triu_indices(n, k=1)
    separation = np.linalg.norm(p[:, :, pi] - p[:, :, pj], axis=-1).mean(-1)
    out["agent_separation_at10_mean"] = float(separation[:, min(10, t)].mean())
    out["agent_separation_final_mean"] = float(separation[np.arange(b), end_t].mean())
    out["mean_action_between_agents_std"] = masked_mean(d["mean_action"].std(axis=2).mean(-1), alive)
    d["target_proxy"] = label
    d["foods_per_episode"] = food_count
    d["straightness_per_agent"] = displacement / np.maximum(path, 1e-12)
    return out


@th.no_grad()
def rollout(run_id, episodes, seed, mode):
    sacred = ROOT / "results/sacred/mafpo_gauss/vmas-dispersion" / str(run_id)
    cfg = json.loads((sacred / "config.json").read_text())
    roots = sorted((ROOT / "results/models").glob(cfg["name"] + "_*"))
    if len(roots) != 1:
        raise ValueError(f"Expected one checkpoint directory for {cfg['name']}: {roots}")
    step = max(int(x.name) for x in roots[0].iterdir() if x.name.isdigit())
    checkpoint = roots[0] / str(step)
    env_args = dict(cfg["env_args"])
    horizon = int(env_args.pop("time_limit"))
    env_args.pop("key")
    env_args.pop("pretrained_wrapper", None)
    env_args.pop("continuous_actions", None)
    env = vmas.make_env("dispersion", num_envs=episodes, device="cpu",
                        continuous_actions=True, seed=seed, **env_args)
    observations = env.reset(seed=seed)
    n, obs_dim = len(observations), observations[0].shape[-1]
    action_dim = env.action_space.spaces[0].shape[0]
    cfg.update(use_cuda=False, n_agents=n, n_actions=action_dim)
    args = SimpleNamespace(**cfg)
    scheme = {"obs": {"vshape": obs_dim, "group": "agents"},
              "state": {"vshape": n * obs_dim},
              "actions": {"vshape": (action_dim,), "group": "agents"}}
    mac = MAFPOGaussMAC(scheme, {"agents": n}, args)
    mac.load_models(str(checkpoint))
    mac.agent.eval()
    mac.init_hidden(episodes)
    batch = EpisodeBatch(scheme, {"agents": n}, episodes, horizon+1, device="cpu")
    active = th.ones(episodes, dtype=th.bool)
    hist = {k: [] for k in ("pos", "vel", "eaten", "alive", "action", "mean_action", "eps", "noise", "reward")}
    food = th.stack([x.state.pos for x in env.world.landmarks], 1).numpy().copy()

    def record_state():
        hist["pos"].append(th.stack([x.state.pos for x in env.world.agents], 1).numpy().copy())
        hist["vel"].append(th.stack([x.state.vel for x in env.world.agents], 1).numpy().copy())
        hist["eaten"].append(th.stack([x.eaten for x in env.world.landmarks], 1).numpy().copy())

    record_state()
    for t in range(horizon):
        obs = th.stack(observations, 1)
        batch.update({"obs": obs}, ts=t)
        # Pair initial layouts and random draws across checkpoints/noise modes.
        # Re-seeding per step prevents skipped terminal-noise draws in test mode
        # from changing the epsilon sequence in the following step.
        th.manual_seed(seed + 100000 + t)
        normalized = mac.select_actions(batch, t, 0, test_mode=(mode == "test"))
        batch.update({"actions": normalized}, ts=t)
        mean_action = 2 * th.sigmoid(mac._last_x1_raw) - 1
        action = 2 * normalized - 1
        hist["alive"].append(active.numpy().copy())
        for key, value in (("action", action), ("mean_action", mean_action),
                           ("eps", mac._last_eps), ("noise", mac._last_noise.reshape(episodes, n, action_dim))):
            hist[key].append(value.numpy().copy())
        observations, rewards, done, _ = env.step([action[:, i] for i in range(n)])
        hist["reward"].append((rewards[0] * active).numpy().copy())
        active &= ~done
        record_state()
    data = {k: np.stack(v, axis=1) for k, v in hist.items()}
    data["food"] = food
    metrics = summarize(data, env_args.get("pos_range", 1.0))
    assert np.allclose(data["reward"].sum(1), data["foods_per_episode"]), "Food/reward accounting mismatch"
    metadata = {"run": run_id, "name": cfg["name"], "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_step": step, "training_seed": cfg["seed"], "mode": mode,
                "episodes": episodes, "horizon": horizon, "diagnostic_seed": seed,
                "eps_per_episode": cfg.get("eps_per_episode", False),
                "actor_last_action": cfg.get("actor_last_action", False),
                "test_eps_mode": cfg.get("test_eps_mode", "zero"),
                "sigma": mac.agent.sigma().numpy().tolist(), **metrics}
    return metadata, data


def make_plot(all_data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = all_data.get("run1_test")
    if base is None:
        return
    # Deterministic representative selection: first episode at each observed
    # food-count level, not a search for especially good/bad looking paths.
    indices = []
    for count in np.unique(base["foods_per_episode"]):
        indices.append(int(np.flatnonzero(base["foods_per_episode"] == count)[0]))
    for ep in range(base["pos"].shape[0]):
        if len(indices) >= 3:
            break
        if ep not in indices:
            indices.append(ep)
    indices = indices[:3]
    fig, axs = plt.subplots(2, len(indices), figsize=(4.3*len(indices), 8.0), squeeze=False)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for row, mode in enumerate(("test", "sample")):
        d = all_data[f"run1_{mode}"]
        for col, ep in enumerate(indices):
            ax = axs[row, col]
            end = int(d["alive"][ep].sum())
            for i, color in enumerate(colors):
                path = d["pos"][ep, :end+1, i]
                ax.plot(*path.T, color=color, linewidth=1.8, alpha=0.8, label=f"agent {i+1}")
                ax.scatter(*path[-1], color=color, marker="s", s=25)
                for t in (10, 20, 30):
                    if t < end:
                        ax.annotate("", xy=path[t+1], xytext=path[t-1],
                                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
            eaten = d["eaten"][ep, end]
            for j, f in enumerate(d["food"][ep]):
                ax.add_patch(plt.Circle(f, 0.085, fill=False, color="#999999", lw=0.7))
                ax.scatter(*f, marker="*", s=170, color="#222222" if eaten[j] else "#E6AB02", zorder=5)
                ax.annotate(f"F{j+1}", f, xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax.scatter(0, 0, marker="o", facecolors="none", edgecolors="black", s=70, zorder=6)
            ax.set(xlim=(-1.06, 1.06), ylim=(-1.06, 1.06), aspect="equal",
                   title=f"Episode {ep} | food {d['foods_per_episode'][ep]}/4")
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel("Current test mode: n = 0" if mode == "test" else "Sampling policy: n sampled")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("CommFlow 5.05M checkpoint | same layouts and epsilon draws\nStars: yellow = uneaten, black = eaten; squares = final agent positions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "commflow_trajectories.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=int, default=[1, 5, 6, 2])
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--resume", action="store_true", help="Reuse completed diagnostics in --out")
    p.add_argument("--out", type=Path, default=ROOT / "results/diagnostics" / ("dispersion_direction_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    args = p.parse_args()
    th.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=args.resume)
    summary_path = args.out / "summary.json"
    results = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else []
    trajectories = {}
    for run in args.runs:
        for mode in ("test", "sample"):
            key = f"run{run}_{mode}"
            previous = [r for r in results if r["run"] == run and r["mode"] == mode]
            if previous:
                assert previous[0]["episodes"] == args.episodes and previous[0]["diagnostic_seed"] == args.seed
                with np.load(args.out / (key + ".npz")) as saved:
                    trajectories[key] = dict(saved)
                print("REUSED", key, flush=True)
                continue
            result, data = rollout(run, args.episodes, args.seed, mode)
            np.savez_compressed(args.out / (key + ".npz"), **data)
            results.append(result)
            trajectories[key] = data
            (args.out / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))
            print(json.dumps(result, allow_nan=False), flush=True)
    make_plot(trajectories, args.out)
    print("OUTPUT", args.out, flush=True)


if __name__ == "__main__":
    main()
