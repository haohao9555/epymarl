import torch as th

from components.standarize_stream import RunningMeanStd


class ObsNormalizer:
    """obs / state 的运行均值-方差归一化（MAPPO 风格）。

    一个实例同时被 MAC（actor 输入的 obs）和 critic（state，以及
    obs_individual_obs 时的 obs）共用，两边看到的统计量永远一致。

    更新时机：learner 在 train() **末尾**用本批 rollout 数据 update()，而不是
    开头——这样同一批数据在训练时用的统计量跟 rollout 时一模一样，训练里重
    建的 hidden state / mu_new 在 theta=theta_old 处能精确复现 rollout 存下
    来的 mu_old（ratio 恰好为 1），不会因为归一化参数变了而凭空出现一个假的
    策略变化。下一批 rollout 才用上新统计量。

    归一化后 clip 到 [-clip, clip]（默认 10），防止早期方差估计过小时个别维
    度爆掉。enabled=False 时所有 normalize_* 都是恒等映射，方便 A/B。
    """

    def __init__(self, scheme, args, device):
        self.enabled = bool(getattr(args, "obs_normalise", True))
        self.clip = float(getattr(args, "obs_normalise_clip", 10.0))
        self.obs_ms = RunningMeanStd(shape=(scheme["obs"]["vshape"],), device=device)
        self.state_ms = RunningMeanStd(shape=(scheme["state"]["vshape"],), device=device)

    def _apply(self, ms, x):
        if not self.enabled:
            return x
        mean = ms.mean.to(x.device)
        std = th.sqrt(ms.var.to(x.device) + 1e-8)
        return th.clamp((x - mean) / std, -self.clip, self.clip)

    def normalize_obs(self, obs):
        return self._apply(self.obs_ms, obs)

    def normalize_state(self, state):
        return self._apply(self.state_ms, state)

    @th.no_grad()
    def update(self, batch, mask_bt):
        """batch: EpisodeBatch；mask_bt: [B,T]（bool/0-1）有效 timestep。"""
        if not self.enabled:
            return
        valid = mask_bt.bool()
        obs = batch["obs"][:, :-1]                 # [B,T,N,obs]
        self.obs_ms.update(obs[valid].reshape(-1, obs.shape[-1]).float())
        state = batch["state"][:, :-1]             # [B,T,state]
        self.state_ms.update(state[valid].float())

    def state_dict(self):
        return {
            "obs_mean": self.obs_ms.mean, "obs_var": self.obs_ms.var, "obs_count": self.obs_ms.count,
            "state_mean": self.state_ms.mean, "state_var": self.state_ms.var,
            "state_count": self.state_ms.count,
        }

    def load_state_dict(self, sd):
        self.obs_ms.mean, self.obs_ms.var, self.obs_ms.count = sd["obs_mean"], sd["obs_var"], sd["obs_count"]
        self.state_ms.mean, self.state_ms.var, self.state_ms.count = (
            sd["state_mean"], sd["state_var"], sd["state_count"]
        )
