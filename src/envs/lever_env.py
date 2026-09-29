"""Continuous lever pulling, after the lever task of CommNet (Sukhbaatar et al., 2016).

n_agents agents pull one of n_levers levers at the same time; the shared reward is the
fraction of distinct levers pulled. Every agent receives the SAME constant observation and
no identity, so the only way to split up is randomness that the agents coordinate. That
isolates the question CommFlow is about:

  * a policy whose agents are independent given the state (MAPPO, MAPPO+attn, MAFPO) can
    reach at most E[distinct]/m = 1 - (1 - 1/m)^m  (~0.67 for m = 5), however dense the
    reward and however long it trains -- a representational cap, not an exploration one;
  * a policy that exchanges each agent's own noise before acting (CommFlow) can reach 1.

Actions are the repo's continuous [0, 1] actions (sigmoid of the Gaussian sample); [0, 1]
is cut into n_levers equal bins and the bin index is the lever. steps = 1 is the one-shot
task; steps > 1 repeats it with the step index appended to the observation (a hook for
testing role persistence across time, e.g. eps_rho).

Reward: every agent gets distinct / n_levers, so with the default reward_scalarisation
"sum" the logged return is n_agents * distinct / n_levers per step -- the number of
distinct levers when n_agents == n_levers -- summed over the episode. info["distinct_frac"] (terminal step, averaged over the episode's steps)
is logged by the runner as distinct_frac_mean / test_distinct_frac_mean.

Use time_limit > steps in the env config: the episode must end by termination, not by the
TimeLimit truncation (which the learner would bootstrap through).
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Tuple


class LeverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_agents=5, n_levers=None, steps=1, **kwargs):
        self.n_agents = int(n_agents)
        self.n_levers = int(n_levers) if n_levers else self.n_agents
        self.steps = int(steps)
        assert self.n_agents >= 1 and self.n_levers >= 1 and self.steps >= 1
        obs_dim = 1 if self.steps == 1 else 2
        self.observation_space = Tuple(
            tuple(Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32) for _ in range(self.n_agents))
        )
        self.action_space = Tuple(
            tuple(Box(0.0, 1.0, shape=(1,), dtype=np.float32) for _ in range(self.n_agents))
        )
        self._t = 0
        self._distinct_sum = 0.0

    def _obs(self):
        o = [1.0] if self.steps == 1 else [1.0, self._t / self.steps]
        return tuple(np.asarray(o, dtype=np.float32) for _ in range(self.n_agents))

    def lever(self, a):
        """[0, 1] action -> lever index (the last bin is closed on the right)."""
        a = float(np.clip(np.asarray(a, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        return min(int(a * self.n_levers), self.n_levers - 1)

    def reset(self, seed=None, options=None):
        self._t = 0
        self._distinct_sum = 0.0
        return self._obs(), {}

    def step(self, actions):
        levers = [self.lever(a) for a in actions]
        frac = len(set(levers)) / self.n_levers
        self._t += 1
        self._distinct_sum += frac
        terminated = self._t >= self.steps
        info = {"distinct_frac": self._distinct_sum / self._t} if terminated else {}
        return self._obs(), [frac] * self.n_agents, terminated, False, info

    def seed(self, seed=None):
        # Deterministic environment: nothing to seed.
        return [seed]

    def close(self):
        pass


gym.register("lever", entry_point="envs.lever_env:LeverEnv")
