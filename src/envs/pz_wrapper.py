from pathlib import Path
import importlib

import gymnasium as gym
from gymnasium.spaces import Tuple

import pettingzoo


class PettingZooWrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, lib_name, env_name, **kwargs):
        env = importlib.import_module(f"pettingzoo.{lib_name}.{env_name}")
        self._env = env.parallel_env(**kwargs)
        self._env.reset()

        self.n_agents = self._env.num_agents
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple([obs[k] for k in self._env.agents])
        self.last_obs = obs
        return obs, info

    def render(self):
        return self._env.render()

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        # ------修复：终止/截断被误判 + 最后一步 reward 丢失 ----------
        # PettingZoo 的 ParallelEnv 有个自己的 API 约定：处理完"最后一步"
        # （不管是真正 terminated 还是仅仅 truncated/超时）之后，会把
        # self._env.agents 清空成 []——这不代表这一步没有产生真实的
        # obs/reward，observations/rewards/dones/truncated/infos 这几个
        # dict 本身仍然是按"这一步之前"的 agent 名正确索引的真实数据。
        #
        # 旧代码在 self._env.step() 返回之后才 `for k in self._env.agents`
        # 取值，这时 self._env.agents 已经被清空，于是在最后一步：
        #   - obs/rewards 变成空 tuple/list（真实的最后一步数据被丢弃）
        #   - done  = all([...]) 对空列表求值恒为 True——哪怕这一步只是
        #     truncated 超时，不是真正 terminated，也会被判成 True
        #   - truncated 同理恒为 True
        #   - info 变成 {}
        # 而 `if done:` 分支还会把已经丢空的 rewards 强行置零——真正的最后
        # 一步 reward 就这么没了。下游 parallel_runner.py 把
        # terminated/truncated 合并成一个标志喂给 GAE，于是 simple_spread
        # 每 25 步一次的 time-limit truncation 被当成真正的 episode 终止，
        # GAE 从不对 episode 尾部 bootstrap，整条轨迹尾部的 advantage 系统性
        # 偏置（这个 bug 不只影响 FPO++，MAPPO 等所有跑在这个 wrapper 上的
        # 算法都受影响）。
        #
        # 修法：调用 self._env.step() 之前先把 self._env.agents 存一份快照
        # （agents_before），之后一律用这份快照去索引返回的几个 dict——数据
        # 从未丢失，丢失的只是"用来索引数据的列表"，两者不是一回事。
        agents_before = list(self._env.agents)

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in agents_before])
        rewards = [rewards[k] for k in agents_before]
        done = all([dones[k] for k in agents_before]) if agents_before else True
        truncated = (
            all([truncated[k] for k in agents_before]) if agents_before else False
        )
        info = {
            f"{k}_{key}": value
            for k in agents_before
            for key, value in infos[k].items()
        }
        self.last_obs = obs
        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()


# import all files within the pettingzoo library that match "**/*_v?.py" underneath library of pettingzoo
envs = Path(pettingzoo.__path__[0]).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    gym.register(
        gymkey,
        entry_point="envs.pz_wrapper:PettingZooWrapper",
        kwargs={
            "lib_name": lib,
            "env_name": filename,
        },
    )
