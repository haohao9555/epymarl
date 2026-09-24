"""Standalone MAC-Flow online training entrypoint that can optionally
pre-fill the replay buffer with an expert transition dataset (collected via
collect_expert_buffer.py) before the usual "collect with the current policy,
insert into a growing buffer, train" loop starts.

Does NOT touch run.py or main.py -- this duplicates the minimal slice of
run_sequential's setup needed (env -> scheme/groups -> buffer -> mac ->
learner -> loop) rather than modifying the shared entrypoint, per the same
"new folder, don't touch existing files" constraint the rest of macflow/
follows. Skips checkpoint-resume/record_mov/tensorboard/sacred -- this is a
validation/experiment script, not a replacement for main.py's full feature
set.

Usage:
    python macflow/train_online.py --config mac_flow_continuous \
        --env-key pz-mpe-simple-spread-v3 --t-max 500000 \
        --expert-buffer-path macflow/expert_data/pz-mpe-simple-spread-v3_mappo_expert.pt
"""
import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace as SN

import yaml

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import torch as th

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d[k], v)
        else:
            d[k] = v
    return d


def build_config(alg_config, env_config, env_key, overrides):
    cfg = load_yaml(os.path.join(SRC_DIR, "config", "default.yaml"))
    cfg = recursive_update(cfg, load_yaml(os.path.join(SRC_DIR, "config", "envs", f"{env_config}.yaml")))
    cfg = recursive_update(cfg, load_yaml(os.path.join(SRC_DIR, "config", "algs", f"{alg_config}.yaml")))
    if env_key is not None:
        cfg["env_args"]["key"] = env_key
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def make_console_logger():
    logger = logging.getLogger("mac_flow_train_online")
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("INFO")
    return logger


def load_expert_buffer(path, target_scheme, target_groups, device):
    payload = th.load(path, map_location="cpu")
    n_ep = payload["episodes_in_buffer"]
    data = SN(transition_data={k: v.to(device) for k, v in payload["transition_data"].items()},
              episode_data={})
    return EpisodeBatch(target_scheme, target_groups, n_ep, payload["max_seq_length"],
                         data=data, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="mac_flow_continuous")
    p.add_argument("--env-config", default="gymma")
    p.add_argument("--env-key", default="pz-mpe-simple-spread-v3")
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--expert-buffer-path", default=None)
    cli = p.parse_args()

    overrides = {"seed": cli.seed, "use_wandb": cli.use_wandb, "use_tensorboard": False}
    if cli.t_max is not None:
        overrides["t_max"] = cli.t_max
    config_dict = build_config(cli.config, cli.env_config, cli.env_key, overrides)
    args = SN(**config_dict)
    args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"
    args.env_args["seed"] = cli.seed

    console_logger = make_console_logger()
    logger = Logger(console_logger)
    if args.use_wandb:
        logger.setup_wandb(config_dict, args.wandb_team, args.wandb_project, args.wandb_mode)

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float32},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,) if args.common_reward else (args.n_agents,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                           preprocess=preprocess,
                           device="cpu" if args.buffer_cpu_only else args.device)

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    if cli.expert_buffer_path:
        expert_batch = load_expert_buffer(cli.expert_buffer_path, scheme, groups,
                                           device="cpu" if args.buffer_cpu_only else args.device)
        buffer.insert_episode_batch(expert_batch)
        console_logger.info(
            "Seeded replay buffer with %d expert episodes from %s (episodes_in_buffer=%d)",
            expert_batch.batch_size, cli.expert_buffer_path, buffer.episodes_in_buffer,
        )

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    start_time = time.time()
    last_time = start_time
    console_logger.info("Beginning training for %d timesteps", args.t_max)

    while runner.t_env <= args.t_max:
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            last_test_T = runner.t_env
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            console_logger.info("t_env: %d / %d", runner.t_env, args.t_max)
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.finish()
    console_logger.info("Finished Training")


if __name__ == "__main__":
    main()
