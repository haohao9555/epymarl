import torch as th

from .policyflow_critic import CentralVCritic


class NormalisedCentralVCritic(CentralVCritic):
    """policyflow_critic.CentralVCritic + 输入归一化：state（和
    obs_individual_obs 时的 obs）先过 learner 挂上来的 self.normalizer
    （与 MAC 共用同一个 ObsNormalizer 实例）。normalizer 为 None 时退化成
    原样。"""

    normalizer = None

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        state = batch["state"][:, ts]
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
        inputs = [state.unsqueeze(2).repeat(1, 1, self.n_agents, 1)]
        if self.args.obs_individual_obs:
            obs = batch["obs"][:, ts]
            if self.normalizer is not None:
                obs = self.normalizer.normalize_obs(obs)
            inputs.append(obs.reshape(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        if self.args.obs_last_action:
            raise NotImplementedError("mafpo_gauss_critic 不支持 obs_last_action（连续动作没有 actions_onehot）")
        inputs.append(
            th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        )
        return th.cat(inputs, dim=-1), bs, max_t
