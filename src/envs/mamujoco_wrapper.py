"""MaMuJoCo（Gymnasium-Robotics `mamujoco_v1`）接入 gymma 的 wrapper。

跟 pz_wrapper.PettingZooWrapper 同一套约定（Tuple 动作/观测空间、按 agent
顺序的 tuple obs、list reward、all() 合并 done/truncated），区别只有两点：

1. 动作范围：MaMuJoCo 每个 agent 的动作是 Box(-1, 1)，而这个仓库里的连续
   动作算法（FPO++ 的 _x1_to_action、MAPPO 的 Beta selector）都按 Box(0, 1)
   出动作。这里对外暴露 Box(0, 1)，step() 里线性映射到真实 [low, high]，
   算法侧一行不用改。映射只发生在这一层，buffer 里存的仍然是 [0,1] 动作。
2. info：MuJoCo 每步返回 x_position / x_velocity / reward_ctrl 等 float，
   parallel_runner 会把 final info 里所有 key 求和后当 stats 记，这里把它
   们丢掉，只留空 dict，免得 wandb 里多出一堆无意义的 *_mean。

时间上限：mamujoco 内部的单体 gym env 自带 1000 步 truncation，gymma 外面
再包一层 TimeLimit(time_limit)，env config 里 time_limit 设成 1000 与之一致
即可（更小则以外层为准）。

用法（gym id 在文件底部注册）：
    env_args.key=mamujoco-HalfCheetah-2x3   # scenario-agent_conf
    env_args.key=mamujoco-Ant-4x2 / mamujoco-Walker2d-2x3 / mamujoco-Hopper-3x1 ...
额外 kwargs（agent_obsk、local_categories 等）原样透传给 mamujoco_v1.parallel_env。
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Tuple

from gymnasium_robotics import mamujoco_v1


class MaMuJoCoWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, scenario, agent_conf, agent_obsk=1, render_mode=None, **kwargs):
        self._env = mamujoco_v1.parallel_env(
            scenario, agent_conf, agent_obsk=agent_obsk, render_mode=render_mode, **kwargs
        )
        self._env.reset()
        self.agents = list(self._env.possible_agents)
        self.n_agents = len(self.agents)
        self.last_obs = None

        real_spaces = [self._env.action_space(a) for a in self.agents]
        self._act_low = [np.asarray(s.low, dtype=np.float32) for s in real_spaces]
        self._act_high = [np.asarray(s.high, dtype=np.float32) for s in real_spaces]
        self.action_space = Tuple(
            tuple(Box(0.0, 1.0, shape=s.shape, dtype=np.float32) for s in real_spaces)
        )
        self.observation_space = Tuple(
            tuple(self._env.observation_space(a) for a in self.agents)
        )
        # 故意不设 unwrapped.state_size：gymma.get_state() 固定返回各 agent
        # obs 的拼接，get_state_size() 若读到这里的 state_size（mamujoco 全局
        # state 是 17 维，拼接 obs 是 2x12=24 维）就会和实际 state 长度对不
        # 上，buffer 建表直接崩。要用真正的全局 state 得同时改 gymma。

    def _to_real_action(self, i, a):
        # gymma sizes the action scheme by the LONGEST action space (n_actions =
        # max_i dim(A_i)), so factorisations with unequal splits -- Humanoid
        # "9|8" is the only registered one -- hand every agent that many
        # numbers. Drop the padding here, mirroring _pad_observation() on the
        # observation side: agent i only ever uses its own first dim(A_i)
        # entries and the trailing ones are ignored.
        a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)[: self._act_low[i].shape[0]]
        return self._act_low[i] + a * (self._act_high[i] - self._act_low[i])

    def reset(self, seed=None, options=None):
        obs, _info = self._env.reset(seed=seed, options=options)
        obs = tuple(obs[a] for a in self.agents)
        self.last_obs = obs
        return obs, {}

    def step(self, actions):
        dict_actions = {
            a: self._to_real_action(i, act) for i, (a, act) in enumerate(zip(self.agents, actions))
        }
        observations, rewards, dones, truncated, _infos = self._env.step(dict_actions)
        obs = tuple(observations[a] for a in self.agents)
        rewards = [float(rewards[a]) for a in self.agents]
        done = all(dones[a] for a in self.agents)
        trunc = all(truncated[a] for a in self.agents)
        self.last_obs = obs
        return obs, rewards, done, trunc, {}

    def state(self):
        return self._env.state()

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


# 注册常用的 scenario / agent_conf 组合成 gym id：mamujoco-<Scenario>-<conf>
_CONFS = {
    "HalfCheetah": ["2x3", "6x1"],
    "Ant": ["2x4", "2x4d", "4x2"],
    "Walker2d": ["2x3"],
    "Hopper": ["3x1"],
    "Humanoid": ["9|8"],           # id: mamujoco-Humanoid-9p8
    "HumanoidStandup": ["9|8"],    # id: mamujoco-HumanoidStandup-9p8
    "Swimmer": ["2x1"],
    "Reacher": ["2x1"],
    "InvertedPendulum": ["2x1"],
    "ManySegmentSwimmer": ["10x2"],
    "CoupledHalfCheetah": ["1p1"],
}
for _scenario, _confs in _CONFS.items():
    for _conf in _confs:
        # gym id 里不允许 "|"，Humanoid 的 "9|8" 注册成 "9p8"（p = pipe）。
        gym.register(
            f"mamujoco-{_scenario}-{_conf.replace('|', 'p')}",
            entry_point="envs.mamujoco_wrapper:MaMuJoCoWrapper",
            kwargs={"scenario": _scenario, "agent_conf": _conf},
        )
