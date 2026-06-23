import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPOContinuousLearner:
    """Continuous FPO learner.

    The rollout policy stores fixed CFM eps/t/action points. During training we
    recompute the current CFM loss at those same points and use the loss change
    to build the FPO ratio:

        rho_s = exp(mean(clamp(L_old - L_new, -rho_clip, rho_clip)))
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

        # Minibatches are sampled over environment timesteps. Each selected
        # timestep keeps all agents together, so 2048 rollout timesteps really
        # means 2048 environment transitions rather than 2048 agent entries.
        valid_time_indices = th.nonzero(
            mask[:, :, 0].reshape(-1) > 0, as_tuple=False
        ).squeeze(1)
        minibatch_size = getattr(self.args, "fpo_minibatch_size", 256)
        actor_stats = {
            "advantage_mean": [],
            "advantage_std": [],
            "pg_loss": [],
            "cfm_loss_mean": [],
            "cfm_aux_loss": [],
            "rho_s_mean": [],
            "rho_s_std": [],
            "clip_fraction": [],
            "actor_grad_norm": [],
        }
        critic_train_stats = {
            k: [] for k in ["critic_loss", "critic_grad_norm", "td_error_abs",
                            "target_mean", "q_taken_mean"]
        }

        for _ in range(self.args.epochs):
            # The critic still uses full sequences for n-step returns. We train
            # it once per epoch, then freeze the advantages for the shuffled
            # actor minibatches in this epoch.
            advantages, epoch_critic_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            for key, values in epoch_critic_stats.items():
                critic_train_stats[key].extend(values)

            # Shuffle valid (episode, time) entries every epoch. This is
            # the PPO-style minibatch pass over the 2048-step rollout batch.
            permutation = valid_time_indices[
                th.randperm(valid_time_indices.numel(), device=valid_time_indices.device)
            ]

            for start in range(0, permutation.numel(), minibatch_size):
                mb_time_idx = permutation[start:start + minibatch_size]

                # Recompute actor hidden states after every optimizer step. If
                # we reused one full graph across minibatches, later updates
                # would backprop through stale pre-step parameters.
                h_seq = self._build_actor_hidden_sequence(batch)
                mb_cfm_loss = self._compute_cfm_loss_for_time_indices(
                    batch, h_seq, mb_time_idx
                )
                mb_initial_cfm_loss = initial_cfm_loss.reshape(
                    -1, self.n_agents, initial_cfm_loss.size(-2), initial_cfm_loss.size(-1)
                )[mb_time_idx]

                diff = th.clamp(
                    mb_initial_cfm_loss - mb_cfm_loss, -rho_clip, rho_clip
                )
                mb_rho_s = th.exp(diff.mean(dim=(-2, -1)))       # [M,N]
                advantages_by_time = advantages.reshape(-1, self.n_agents)
                mb_advantages = advantages_by_time[mb_time_idx]

                mb_rho_s = mb_rho_s.reshape(-1)
                mb_advantages = mb_advantages.reshape(-1)
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(
                    mb_rho_s, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                ) * mb_advantages
                pg_loss = -th.min(surr1, surr2).mean()

                # CFM 对冲：仅在 advantage < 0 时施加辅助损失，抵消负 advantage
                # 驱动 v_pred 偏离 target 的梯度，防止 cfm_loss 跨 rollout 漂移发散。
                # adv_scale × rho_scale 精确跟踪 PG 破坏力量级，cfm_aux_coef 控制对冲比例。
                neg_mask = (mb_advantages < 0)
                neg_adv_vals = mb_advantages[neg_mask]
                if neg_adv_vals.numel() > 0:
                    adv_scale = neg_adv_vals.abs().mean().detach()
                    rho_scale = mb_rho_s[neg_mask].mean().detach()
                else:
                    adv_scale = mb_advantages.abs().mean().detach()
                    rho_scale = th.ones(1, device=mb_advantages.device)
                cfm_aux_coef = getattr(self.args, "cfm_aux_coef", 0.5)
                cfm_aux_loss = (
                    mb_cfm_loss.mean(dim=(-2, -1)).reshape(-1) * neg_mask.float()
                ).mean()
                actor_loss = pg_loss + cfm_aux_coef * adv_scale * rho_scale * cfm_aux_loss

                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.actor_params, self.args.grad_norm_clip
                )
                self.actor_optimiser.step()

                actor_stats["advantage_mean"].append(mb_advantages.mean().item())
                actor_stats["advantage_std"].append(
                    mb_advantages.std(unbiased=False).item()
                )
                actor_stats["pg_loss"].append(pg_loss.item())
                actor_stats["cfm_loss_mean"].append(
                    mb_cfm_loss.mean(dim=(-2, -1)).mean().item()
                )
                actor_stats["cfm_aux_loss"].append(cfm_aux_loss.item())
                actor_stats["rho_s_mean"].append(mb_rho_s.mean().item())
                actor_stats["rho_s_std"].append(mb_rho_s.std(unbiased=False).item())
                actor_stats["clip_fraction"].append(
                    (
                        (mb_rho_s > 1 + self.args.eps_clip)
                        | (mb_rho_s < 1 - self.args.eps_clip)
                    ).float().mean().item()
                )
                actor_stats["actor_grad_norm"].append(grad_norm.item())

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
                        "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            self.logger.log_stat(
                "advantage_mean", self._mean_stat(actor_stats["advantage_mean"]), t_env
            )
            self.logger.log_stat(
                "advantage_std", self._mean_stat(actor_stats["advantage_std"]), t_env
            )
            self.logger.log_stat("pg_loss", self._mean_stat(actor_stats["pg_loss"]), t_env)
            self.logger.log_stat(
                "cfm_loss_mean", self._mean_stat(actor_stats["cfm_loss_mean"]), t_env
            )
            self.logger.log_stat(
                "cfm_aux_loss", self._mean_stat(actor_stats["cfm_aux_loss"]), t_env
            )
            self.logger.log_stat(
                "rho_s_mean", self._mean_stat(actor_stats["rho_s_mean"]), t_env
            )
            self.logger.log_stat(
                "rho_s_std", self._mean_stat(actor_stats["rho_s_std"]), t_env
            )
            self.logger.log_stat(
                "clip_fraction", self._mean_stat(actor_stats["clip_fraction"]), t_env
            )
            self.logger.log_stat(
                "actor_grad_norm", self._mean_stat(actor_stats["actor_grad_norm"]), t_env
            )
            self.logger.log_stat(
                "fpo_valid_transitions", valid_time_indices.numel(), t_env
            )
            self.log_stats_t = t_env

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        h_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = self.mac.forward(batch, t=t)
            h_list.append(h)
        return th.stack(h_list, dim=1)                # [B,T,N,hidden_dim]

    def _compute_cfm_loss(self, batch: EpisodeBatch, h_seq: th.Tensor) -> th.Tensor:
        action = batch["actions"][:, :-1].float()     # [B,T,N,n_actions]
        eps = batch["cfm_eps"][:, :-1]                # [B,T,N,cfm_n,n_actions]
        cfm_t = batch["cfm_t"][:, :-1]                # [B,T,N,cfm_n,1]

        act_exp = action.unsqueeze(3).expand_as(eps)
        x_t = cfm_t * eps + (1 - cfm_t) * act_exp
        h_exp = h_seq.unsqueeze(3).expand(-1, -1, -1, eps.size(3), -1)

        flat_h = h_exp.reshape(-1, h_exp.shape[-1])
        flat_x_t = x_t.reshape(-1, x_t.shape[-1])
        flat_t = cfm_t.reshape(-1, 1)

        v_pred = self.mac.agent.velocity(flat_h, flat_x_t, flat_t)
        v_pred = v_pred.reshape_as(eps)

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = eps - act_exp
        elif cfm_target_type == "eps":
            target = eps
        else:
            raise ValueError("cfm_target_type must be 'velocity' or 'eps'")

        return ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)

    def _compute_cfm_loss_for_time_indices(
        self, batch: EpisodeBatch, h_seq: th.Tensor, flat_time_indices: th.Tensor
    ) -> th.Tensor:
        # Minibatches are sampled over flattened (episode, time) entries. We
        # keep the full agent dimension for each timestep and evaluate all CFM
        # samples attached to those agents.
        action = batch["actions"][:, :-1].float().reshape(
            -1, self.n_agents, self.n_actions
        )
        eps = batch["cfm_eps"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, self.args.cfm_action_dim
        )
        cfm_t = batch["cfm_t"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, 1
        )
        h = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])

        mb_action = action[flat_time_indices]         # [M,N,A]
        mb_eps = eps[flat_time_indices]               # [M,N,cfm_n,A]
        mb_cfm_t = cfm_t[flat_time_indices]           # [M,N,cfm_n,1]
        mb_h = h[flat_time_indices]                   # [M,N,H]

        act_exp = mb_action.unsqueeze(2).expand_as(mb_eps)
        x_t = mb_cfm_t * mb_eps + (1 - mb_cfm_t) * act_exp
        h_exp = mb_h.unsqueeze(2).expand(-1, -1, mb_eps.size(2), -1)

        flat_h = h_exp.reshape(-1, h_exp.shape[-1])
        flat_x_t = x_t.reshape(-1, x_t.shape[-1])
        flat_t = mb_cfm_t.reshape(-1, 1)

        v_pred = self.mac.agent.velocity(flat_h, flat_x_t, flat_t)
        v_pred = v_pred.reshape_as(mb_eps)

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = mb_eps - act_exp
        elif cfm_target_type == "eps":
            target = mb_eps
        else:
            raise ValueError("cfm_target_type must be 'velocity' or 'eps'")

        return ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = target_critic(batch).squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {k: [] for k in ["critic_loss", "critic_grad_norm",
                                        "td_error_abs", "target_mean", "q_taken_mean"]}

        v = critic(batch)[:, :-1].squeeze(3)
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
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)
        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

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
