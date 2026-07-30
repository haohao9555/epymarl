import copy

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPODiscreteLearner:
    """离散 FPO learner。

    与 FPOContinuousLearner 结构对齐：rollout 时存储 CFM eps/t/initial_loss，
    训练时在相同插值点上重新计算 CFM loss，用 loss 变化量构造 FPO ratio：

        rho_s = exp(mean(clamp(L_old - L_new, -rho_clip, rho_clip)))

    CFM 在 n_actions 维 one-hot 空间中操作：
        x_t = (1-t)*eps + t*one_hot(a)
        velocity target = one_hot(a) - eps
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.actor_params = list(mac.parameters())
        self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        initial_cfm_loss = batch["initial_cfm_loss"][:, :-1]    # [B,T,N,cfm_n,1]
        rho_clip = getattr(self.args, "cfm_rho_clip", 3.0)

        valid_time_indices = th.nonzero(
            mask[:, :, 0].reshape(-1) > 0, as_tuple=False
        ).squeeze(1)
        minibatch_size = getattr(self.args, "fpo_minibatch_size", 256)
        entropy_coef = getattr(self.args, 'entropy_coef', 0.0)
        actor_stats = {
            "advantage_mean": [], "advantage_std": [],
            "pg_loss": [], "cfm_loss_mean": [],
            "rho_s_mean": [], "rho_s_std": [],
            "clip_fraction": [], "actor_grad_norm": [],
            "entropy_mean": [],
        }
        critic_train_stats = {
            k: [] for k in ["critic_loss", "critic_grad_norm", "td_error_abs",
                            "target_mean", "value_mean"]
        }

        for _ in range(self.args.epochs):
            # 每个 epoch 先用完整序列更新一次 critic，得到 advantage；
            # actor 在同一 epoch 内的 minibatch 中共享这份 advantage。
            advantages, epoch_critic_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            for key, values in epoch_critic_stats.items():
                critic_train_stats[key].extend(values)

            # 每个 epoch 对有效时间步重新打乱，PPO-style minibatch 遍历。
            permutation = valid_time_indices[
                th.randperm(valid_time_indices.numel(), device=valid_time_indices.device)
            ]

            for start in range(0, permutation.numel(), minibatch_size):
                mb_time_idx = permutation[start:start + minibatch_size]

                # 每个 minibatch 都重建隐状态序列，确保 hidden state 与当前参数一致。
                h_seq = self._build_actor_hidden_sequence(batch)
                mb_h = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])[mb_time_idx]  # [M, N, H]
                mb_cfm_loss = self._compute_cfm_loss_for_time_indices(
                    batch, h_seq, mb_time_idx
                )   # [M, N, cfm_n, 1]  当前参数下的 CFM loss（L_new）
                mb_initial_cfm_loss = initial_cfm_loss.reshape(
                    -1, self.n_agents, initial_cfm_loss.size(-2), initial_cfm_loss.size(-1)
                )[mb_time_idx]   # [M, N, cfm_n, 1]  rollout 时冻结的 CFM loss（L_old）

                advantages_by_time = advantages.reshape(-1, self.n_agents)
                mb_advantages_2d = advantages_by_time[mb_time_idx]   # [M, N]

                # 对称截断：L_old 和 L_new 使用相同上界。
                # 若只截 L_new 而不截 L_old，则初期 initial_cfm_loss 可能远大于 5，
                # 导致 diff 恒为正、rho 恒打满上界（exp(rho_clip)≈20），clip_fraction→1。
                cfm_loss_clamp = getattr(self.args, "cfm_loss_clamp", 5.0)
                mb_cfm_loss = mb_cfm_loss.clamp(max=cfm_loss_clamp)
                mb_initial_cfm_loss = mb_initial_cfm_loss.clamp(max=cfm_loss_clamp)

                # FPO ratio: rho = exp(clamp(mean(L_old - L_new), ±rho_clip))
                # 先对 cfm_n 个采样点取均值，再 clamp，避免单点极端值放大方差。
                # L_old > L_new → rho > 1（当前策略对该动作的"亲和性"提升）
                # L_old < L_new → rho < 1（当前策略偏离了 rollout 时的动作分布）
                diff = mb_initial_cfm_loss - mb_cfm_loss              # [M, N, cfm_n, 1]
                diff_mean = diff.mean(dim=(-2, -1))                   # [M, N]
                mb_rho_s = th.exp(th.clamp(diff_mean, -rho_clip, rho_clip))  # [M, N]

                mb_rho_s = mb_rho_s.reshape(-1)
                mb_advantages = mb_advantages_2d.reshape(-1)
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(
                    mb_rho_s, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                ) * mb_advantages
                pg_loss = -th.min(surr1, surr2).mean()

                # Entropy bonus: softmax over flow logits = K-step Euler integration
                # of the same ODE used at rollout (sample_action). Using the same
                # integration depth keeps this entropy estimate consistent with the
                # actual action-sampling distribution; encourages action diversity,
                # prevents collapse to no-op.
                if entropy_coef > 0.0:
                    M, N, H = mb_h.shape
                    ent_eps = th.randn(M, N, self.n_actions, device=mb_h.device)
                    n_steps = getattr(self.args, "cfm_rollout_steps", 1)
                    logits = self.mac.agent.integrate(
                        mb_h.reshape(M * N, H),
                        ent_eps.reshape(M * N, self.n_actions),
                        n_steps,
                    )
                    log_p = F.log_softmax(logits, dim=-1)
                    entropy = -(log_p.exp() * log_p).sum(-1).mean()
                else:
                    entropy = th.tensor(0.0)

                actor_loss = pg_loss - entropy_coef * entropy

                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.actor_params, self.args.grad_norm_clip
                )
                self.actor_optimiser.step()

                actor_stats["advantage_mean"].append(mb_advantages.mean().item())
                actor_stats["advantage_std"].append(mb_advantages.std(unbiased=False).item())
                actor_stats["pg_loss"].append(pg_loss.item())
                actor_stats["cfm_loss_mean"].append(mb_cfm_loss.mean(dim=(-2, -1)).mean().item())
                actor_stats["rho_s_mean"].append(mb_rho_s.mean().item())
                actor_stats["rho_s_std"].append(mb_rho_s.std(unbiased=False).item())
                actor_stats["clip_fraction"].append(
                    ((mb_rho_s > 1 + self.args.eps_clip) | (mb_rho_s < 1 - self.args.eps_clip))
                    .float().mean().item()
                )
                actor_stats["actor_grad_norm"].append(grad_norm.item())
                actor_stats["entropy_mean"].append(entropy.item())

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs",
                        "value_mean", "target_mean"]:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            self.logger.log_stat("advantage_mean", self._mean_stat(actor_stats["advantage_mean"]), t_env)
            self.logger.log_stat("advantage_std", self._mean_stat(actor_stats["advantage_std"]), t_env)
            self.logger.log_stat("pg_loss", self._mean_stat(actor_stats["pg_loss"]), t_env)
            self.logger.log_stat("cfm_loss_mean", self._mean_stat(actor_stats["cfm_loss_mean"]), t_env)

            self.logger.log_stat("rho_s_mean", self._mean_stat(actor_stats["rho_s_mean"]), t_env)
            self.logger.log_stat("rho_s_std", self._mean_stat(actor_stats["rho_s_std"]), t_env)
            self.logger.log_stat("clip_fraction", self._mean_stat(actor_stats["clip_fraction"]), t_env)
            self.logger.log_stat("actor_grad_norm", self._mean_stat(actor_stats["actor_grad_norm"]), t_env)
            self.logger.log_stat("entropy_mean", self._mean_stat(actor_stats["entropy_mean"]), t_env)
            self.logger.log_stat("fpo_valid_transitions", valid_time_indices.numel(), t_env)
            self.log_stats_t = t_env

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        h_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = self.mac.forward(batch, t=t)
            h_list.append(h)
        return th.stack(h_list, dim=1)                # [B, T, N, hidden_dim]

    def _compute_cfm_loss_for_time_indices(
        self, batch: EpisodeBatch, h_seq: th.Tensor, flat_time_indices: th.Tensor
    ) -> th.Tensor:
        # 从 buffer 中取出 rollout 时存储的离散动作、噪声和插值时间。
        actions = batch["actions"][:, :-1].reshape(
            -1, self.n_agents, 1
        )   # [B*T, N, 1] long
        eps = batch["cfm_eps"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, self.args.cfm_action_dim
        )   # [B*T, N, cfm_n, A]
        cfm_t = batch["cfm_t"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, 1
        )   # [B*T, N, cfm_n, 1]
        h = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])

        mb_action = actions[flat_time_indices]        # [M, N, 1]
        mb_eps    = eps[flat_time_indices]            # [M, N, cfm_n, A]
        mb_cfm_t  = cfm_t[flat_time_indices]          # [M, N, cfm_n, 1]
        mb_h      = h[flat_time_indices]              # [M, N, H]

        # 离散动作 → one-hot，作为 CFM 的 x_1（数据端）。
        one_hot_a = F.one_hot(
            mb_action.squeeze(-1), self.n_actions
        ).float()                                               # [M, N, A]
        act_exp = one_hot_a.unsqueeze(2).expand_as(mb_eps)     # [M, N, cfm_n, A]

        # 标准 Flow Matching 插值：x_t = (1-t)*eps + t*one_hot(a)
        x_t   = (1 - mb_cfm_t) * mb_eps + mb_cfm_t * act_exp
        h_exp = mb_h.unsqueeze(2).expand(-1, -1, mb_eps.size(2), -1)

        flat_h   = h_exp.reshape(-1, h_exp.shape[-1])
        flat_x_t = x_t.reshape(-1, x_t.shape[-1])
        flat_t   = mb_cfm_t.reshape(-1, 1)

        v_pred = self.mac.agent.velocity(flat_h, flat_x_t, flat_t)
        v_pred = v_pred.reshape_as(mb_eps)

        target = act_exp - mb_eps    # velocity target: dx_t/dt = one_hot(a) - eps

        return ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)  # [M, N, cfm_n, 1]

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = target_critic(batch).squeeze(3)   # [B, T+1, N]

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        terminated = batch["terminated"][:, :-1].float()   # [B, T, 1]
        gae_lambda = getattr(self.args, "gae_lambda", 0.95)
        advantages = self.compute_gae(
            rewards, mask, target_vals, terminated, self.args.gamma, gae_lambda
        )                                                   # [B, T, N], masked
        target_returns = advantages + target_vals[:, :-1]  # λ-returns for critic

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {k: [] for k in ["critic_loss", "critic_grad_norm",
                                        "td_error_abs", "target_mean", "value_mean"]}

        v = critic(batch)[:, :-1].squeeze(3)               # [B, T, N]
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()
        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        running_log["td_error_abs"].append(masked_td_error.abs().sum().item() / mask_elems)
        running_log["value_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)
        return advantages, running_log

    def compute_gae(self, rewards, mask, values, terminated, gamma, gae_lambda):
        """GAE advantage estimation (backward pass).

        rewards:    [B, T, N]
        mask:       [B, T, N]
        values:     [B, T+1, N]  target critic values (includes bootstrap at T)
        terminated: [B, T, 1]    1 if episode ended at step t
        Returns:    advantages [B, T, N], zeroed at invalid steps
        """
        T = rewards.size(1)
        gae = th.zeros_like(values[:, 0])    # [B, N]
        advantages = th.zeros_like(rewards)   # [B, T, N]

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - terminated[:, t]   # [B, 1], broadcast over N
            delta = (rewards[:, t]
                     + gamma * values[:, t + 1] * next_non_terminal
                     - values[:, t])
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[:, t] = gae

        return advantages * mask

    def _mean_stat(self, values):
        return sum(values) / max(1, len(values))

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimiser.load_state_dict(
            th.load("{}/actor_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
