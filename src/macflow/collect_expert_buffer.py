"""Collect an "expert" transition buffer for MAC-Flow's flow-BC stage by
rolling out an already-trained checkpoint (default: the mappo_continuous
run's 15M-step checkpoint already sitting in results/models/) in test_mode
(deterministic / distribution-mean actions -- same convention as
BetaActionSelector's test_mode and fpo_actor's own test_mode branch).

Standalone script, not touching run.py or any MAFPO file: it builds just
enough of run_sequential's setup (env -> scheme/groups -> runner -> mac) to
run episodes and dump them to a single .pt file of raw buffer tensors. That
file is what train_online.py's --expert_buffer_path loads to pre-fill the
online replay buffer before online training starts.

Usage:
    python macflow/collect_expert_buffer.py \
        --checkpoint "results/models/mappo_continuous_seed0_pz-mpe-simple-spread-v3_2026-08-10 07:08:52.865354/15000250" \
        --episodes 200 \
        --out macflow/expert_data/pz-mpe-simple-spread-v3_mappo_expert.pt
"""
import argparse
import os
import sys
from types import SimpleNamespace as SN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th

from components.episode_buffer import ReplayBuffer
from controllers import REGISTRY as mac_REGISTRY
from envs import REGISTRY as env_REGISTRY  # noqa: F401 (registers gymma)
from runners import REGISTRY as r_REGISTRY


def build_args(env_key, expert_agent="rnn_continuous", expert_mac="continuous_mac",
                expert_action_selector="beta", expert_agent_output_type="beta"):
    # Minimal args needed to construct env + ContinuousMAC + RNNContinuousAgent
    # matching how mappo_continuous.yaml trained the checkpoint being loaded.
    return SN(
        env="gymma",
        env_args={
            "key": env_key,
            "time_limit": 100,
            "pretrained_wrapper": None,
            "continuous_actions": True,
            "seed": 0,
            "common_reward": True,
            "reward_scalarisation": "sum",
        },
        common_reward=True,
        reward_scalarisation="sum",
        batch_size_run=1,
        mac=expert_mac,
        agent=expert_agent,
        agent_output_type=expert_agent_output_type,
        action_selector=expert_action_selector,
        hidden_dim=128,
        use_rnn=True,
        obs_agent_id=True,
        obs_last_action=False,
        obs_individual_obs=False,
        use_cuda=False,
        buffer_cpu_only=True,
        device="cpu",
        test_greedy=True,
        render=False,
        test_nepisode=1,
        runner_log_interval=10**9,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a saved model dir (contains agent.th)")
    p.add_argument("--env-key", default="pz-mpe-simple-spread-v3")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--expert-agent", default="rnn_continuous")
    p.add_argument("--expert-mac", default="continuous_mac")
    p.add_argument("--expert-action-selector", default="beta")
    p.add_argument("--expert-agent-output-type", default="beta")
    cli = p.parse_args()

    args = build_args(
        cli.env_key, cli.expert_agent, cli.expert_mac,
        cli.expert_action_selector, cli.expert_agent_output_type,
    )

    class NullLogger:
        def log_stat(self, *a, **k):
            pass

    runner = r_REGISTRY["parallel"](args=args, logger=NullLogger())
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Same base scheme run.py builds for any continuous algorithm (MAC-Flow's
    # own scheme is identical to this -- it only grows extra fields for the
    # FPO learners, which this isn't).
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float32},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}

    buffer = ReplayBuffer(
        scheme, groups, cli.episodes, env_info["episode_limit"] + 1,
        preprocess={}, device="cpu",
    )

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    mac.load_models(cli.checkpoint)
    runner.setup(scheme=scheme, groups=groups, preprocess={}, mac=mac)

    returns = []
    for ep in range(cli.episodes):
        episode_batch = runner.run(test_mode=True)
        buffer.insert_episode_batch(episode_batch)
        ep_return = episode_batch["reward"][:, :-1].sum().item()
        returns.append(ep_return)
        if (ep + 1) % 20 == 0:
            print(f"[{ep+1}/{cli.episodes}] running mean return so far: "
                  f"{sum(returns)/len(returns):.2f}")

    runner.close_env()

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    payload = {
        "scheme": scheme,
        "groups": groups,
        "max_seq_length": env_info["episode_limit"] + 1,
        "episodes_in_buffer": buffer.episodes_in_buffer,
        "transition_data": {k: v[: buffer.episodes_in_buffer].clone()
                             for k, v in buffer.data.transition_data.items()},
    }
    th.save(payload, cli.out)
    print(f"Saved {buffer.episodes_in_buffer} expert episodes to {cli.out}")
    print(f"Expert mean return over {cli.episodes} episodes: {sum(returns)/len(returns):.2f}")


if __name__ == "__main__":
    main()
