from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Tuple

import vmas


class VMASWrapper(gym.Env):
    """VMAS scenarios behind the gymma interface.

    continuous_actions=False (default) is the upstream discrete setting. With
    continuous_actions=True each agent's real action space is VMAS's Box(low,
    high) (Box(-1, 1) for dispersion); like mamujoco_wrapper, the wrapper exposes
    Box(0, 1) instead -- the repo's continuous algorithms act through a sigmoid --
    and maps affinely to [low, high] in step(). The buffer keeps the [0, 1]
    actions.

    Scenario kwargs (n_agents, share_reward, ...) pass straight through to
    vmas.make_env, which rejects unknown keys -- so use an env config without
    the MPE-only N.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_name, continuous_actions=False, **kwargs):
        # One single-env VMAS world per epymarl env worker; a full torch thread
        # pool in each of the 8 workers would only oversubscribe the CPU.
        torch.set_num_threads(1)
        self._env = vmas.make_env(
            env_name,
            num_envs=1,
            continuous_actions=continuous_actions,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            **kwargs,
        )

        self.n_agents = self._env.unwrapped.n_agents
        self._continuous = continuous_actions

        if continuous_actions:
            real_spaces = list(self._env.action_space)
            self._act_low = [np.asarray(s.low, dtype=np.float32) for s in real_spaces]
            self._act_high = [np.asarray(s.high, dtype=np.float32) for s in real_spaces]
            self.action_space = Tuple(
                tuple(Box(0.0, 1.0, shape=s.shape, dtype=np.float32) for s in real_spaces)
            )
        else:
            self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _compress_info(self, info):
        if any(isinstance(i, dict) for i in info.values()):
            # info is nested dict --> flatten
            return {f"{key}/{k}": v for key, i in info.items() for k, v in i.items()}
        else:
            return info

    def reset(self, *args, **kwargs):
        obss, info = self._env.reset(*args, **kwargs)
        return obss, self._compress_info(info)

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def step(self, actions):
        if self._continuous:
            actions = [
                lo + np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) * (hi - lo)
                for a, lo, hi in zip(actions, self._act_low, self._act_high)
            ]
        obss, rews, done, truncated, info = self._env.step(actions)
        return obss, rews, done, truncated, self._compress_info(info)

    def close(self):
        return self._env.close()


# import all files within the pettingzoo library that match "**/*_v?.py" underneath library of pettingzoo
envs = Path(vmas.__path__[0]).glob("scenarios/**/*.py")
for env in envs:
    if "__" in env.stem:
        continue
    name = env.stem
    gym.register(
        f"vmas-{name}",
        entry_point="envs.vmas_wrapper:VMASWrapper",
        kwargs={
            "env_name": name,
        },
    )
