"""
Smoke test: 验证连续动作 MAPPO 改动在 MPE simple_spread 上能跑通
不依赖 Sacred，直接实例化各组件走一遍完整流程
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch as th
from types import SimpleNamespace as SN

# ── 1. 创建连续动作 MPE 环境 ──────────────────────────────────────────────
print("=" * 60)
print("[1/5] 创建连续动作 MPE 环境")

from envs.gymma import GymmaWrapper

env = GymmaWrapper(
    key="pz-mpe-simple-spread-v3",
    time_limit=25,
    pretrained_wrapper=None,
    seed=42,
    common_reward=True,
    reward_scalarisation="sum",
    continuous_actions=True,   # 传给 gym.make，让 MPE 切换到连续模式
)

env_info = env.get_env_info()
print(f"  n_agents         : {env_info['n_agents']}")
print(f"  n_actions        : {env_info['n_actions']}")
print(f"  obs_shape        : {env_info['obs_shape']}")
print(f"  continuous_actions: {env_info['continuous_actions']}")

assert env_info["continuous_actions"], "环境应检测为连续动作！"
assert isinstance(env.longest_action_space.shape, tuple), "continuous 下 longest_action_space 应为 Box"

# ── 2. reset + step，验证动作格式 ─────────────────────────────────────────
print("\n[2/5] 验证 reset / step 动作格式")

obs, info = env.reset()
print(f"  obs 数量: {len(obs)}, 每个 obs shape: {obs[0].shape}")

avail = env.get_avail_actions()
print(f"  avail_actions[0]: {avail[0]}  (应全为 1)")
assert all(a == 1 for a in avail[0]), "连续动作 avail 应全为 1"

# 随机连续动作（每个 agent 一个 n_actions 维向量）
n_agents  = env_info["n_agents"]
n_actions = env_info["n_actions"]
dummy_actions = [np.random.uniform(0, 1, n_actions).astype(np.float32)
                 for _ in range(n_agents)]
obs2, rew, done, trunc, info2 = env.step(dummy_actions)
print(f"  step 返回 reward: {rew}, done: {done}")

# ── 3. 构建 args 和 scheme ────────────────────────────────────────────────
print("\n[3/5] 构建 scheme / args")

from components.transforms import OneHot

args = SN(
    n_agents       = env_info["n_agents"],
    n_actions      = env_info["n_actions"],
    state_shape    = env_info["state_shape"],
    hidden_dim     = 64,
    use_rnn        = True,
    obs_last_action= False,
    obs_agent_id   = True,
    agent          = "rnn_continuous",
    action_selector= "beta",
    agent_output_type = "beta",
    device         = "cpu",
)

_continuous = env_info["continuous_actions"]

scheme = {
    "state"       : {"vshape": env_info["state_shape"]},
    "obs"         : {"vshape": env_info["obs_shape"], "group": "agents"},
    "actions"     : {"vshape": (n_actions,), "group": "agents", "dtype": th.float32}
                    if _continuous else
                    {"vshape": (1,), "group": "agents", "dtype": th.long},
    "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": th.int},
    "reward"      : {"vshape": (1,)},
    "terminated"  : {"vshape": (1,), "dtype": th.uint8},
}
preprocess = {} if _continuous else {"actions": ("actions_onehot", [OneHot(out_dim=n_actions)])}
groups = {"agents": n_agents}

print(f"  actions vshape : {scheme['actions']['vshape']}")
print(f"  actions dtype  : {scheme['actions']['dtype']}")
print(f"  preprocess     : {preprocess}")

# ── 4. 实例化 ContinuousMAC 并做一次 select_actions ──────────────────────
print("\n[4/5] 实例化 ContinuousMAC，验证 select_actions")

from components.episode_buffer import EpisodeBatch
from controllers.continuous_mac import ContinuousMAC

mac = ContinuousMAC(scheme, groups, args)
mac.init_hidden(batch_size=1)

# 构造一个最小的 EpisodeBatch
batch = EpisodeBatch(scheme, groups, batch_size=1, max_seq_length=2,
                     preprocess=preprocess, device="cpu")

obs_now, _ = env.reset()
batch.update({
    "state"       : [env.get_state()],
    "avail_actions": [env.get_avail_actions()],
    "obs"         : [obs_now],
}, ts=0)

actions = mac.select_actions(batch, t_ep=0, t_env=0, test_mode=False)
print(f"  actions shape  : {actions.shape}  (应为 [1, {n_agents}, {n_actions}])")
print(f"  actions dtype  : {actions.dtype}  (应为 float32)")
print(f"  actions sample : {actions[0, 0].tolist()}")

assert actions.shape == (1, n_agents, n_actions), f"shape 错误: {actions.shape}"
assert actions.dtype == th.float32, f"dtype 错误: {actions.dtype}"

# ── 5. 跑 3 步训练，验证 PPOContinuousLearner 不报错 ─────────────────────
print("\n[5/5] 运行 PPOContinuousLearner.train() 验证不报错")

from components.episode_buffer import ReplayBuffer
from learners.ppo_continuous_learner import PPOContinuousLearner
from utils.logging import get_logger

# 补充 learner 需要的 args 字段
args.lr                          = 3e-4
args.grad_norm_clip              = 10
args.epochs                      = 2
args.eps_clip                    = 0.2
args.entropy_coef                = 0.001
args.gamma                       = 0.99
args.q_nstep                     = 3
args.add_value_last_step         = True
args.standardise_returns         = False
args.standardise_rewards         = False
args.common_reward               = True
args.target_update_interval_or_tau = 200
args.learner_log_interval        = 99999
args.critic_type                 = "cv_critic"
args.use_cuda                    = False
args.batch_size_run              = 1
args.obs_individual_obs          = False

# deepcopy 要求 hidden_states=None（未 init），先重置再建 learner
mac.hidden_states = None
from utils.logging import Logger
logger = Logger(get_logger())   # 项目自定义 Logger，有 log_stat 方法
learner = PPOContinuousLearner(mac, scheme, logger, args)

# 手动滚一个 episode 塞进 buffer
episode_limit = env_info["episode_limit"]
buffer = ReplayBuffer(scheme, groups, buffer_size=4,
                      max_seq_length=episode_limit + 1,
                      preprocess=preprocess, device="cpu")

# 收集一条 episode
ep_batch = EpisodeBatch(scheme, groups, batch_size=1,
                        max_seq_length=episode_limit + 1,
                        preprocess=preprocess, device="cpu")
obs_now, _ = env.reset()
mac.init_hidden(1)

for t in range(episode_limit):
    ep_batch.update({
        "state"        : [env.get_state()],
        "avail_actions": [env.get_avail_actions()],
        "obs"          : [obs_now],
    }, ts=t)

    acts = mac.select_actions(ep_batch, t_ep=t, t_env=t, test_mode=False)
    obs_now, rew, done, trunc, _ = env.step(acts[0].detach().numpy())

    ep_batch.update({
        "actions"   : acts,
        "reward"    : [[(rew,)]],
        "terminated": [[(done,)]],
    }, ts=t)

    if done or trunc:
        break

# 补最后一步
ep_batch.update({
    "state"        : [env.get_state()],
    "avail_actions": [env.get_avail_actions()],
    "obs"          : [obs_now],
}, ts=t + 1)
acts_last = mac.select_actions(ep_batch, t_ep=t + 1, t_env=t + 1, test_mode=False)
ep_batch.update({"actions": acts_last}, ts=t + 1)

# 插入 buffer 并 train
buffer.insert_episode_batch(ep_batch)

if buffer.can_sample(1):
    sample = buffer.sample(1)
    max_t  = sample.max_t_filled()
    sample = sample[:, :max_t]
    learner.train(sample, t_env=100, episode_num=1)
    print("  learner.train() PASSED")

env.close()

print("\n" + "=" * 60)
print("Smoke test PASSED - continuous MAPPO pipeline OK")
print("=" * 60)
