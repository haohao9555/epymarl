import copy

import torch as th
from torch.distributions import Beta
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry


class PPOContinuousLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
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

    def _get_log_prob_and_entropy(self, pi, actions):
        alpha, beta = pi.chunk(2, dim=-1)
        dist = Beta(alpha, beta)
        eps = getattr(self.args, "beta_action_epsilon", 1e-6)
        actions = actions.clamp(eps, 1.0 - eps)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
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

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_pi = th.stack(old_mac_out, dim=1)
        old_log_pi_taken, _ = self._get_log_prob_and_entropy(old_pi, actions)

        # Action-collapse diagnostics (same definition as fpo_continuous_learner,
        # so the two algorithms' wandb curves are directly comparable): fraction
        # of realized actions sitting within bound_eps of 0 or 1. For a Beta
        # policy the direct root-cause signal is whether alpha/beta themselves
        # collapse toward the U-shaped regime (both < 1), which makes Beta(a,b)
        # bimodal with mass piling up at both boundaries regardless of the mean.
        bound_eps = 0.02
        act_mask = mask.unsqueeze(-1).expand_as(actions)
        act_mask_sum = act_mask.sum()
        action_mean = (actions * act_mask).sum() / act_mask_sum
        action_var = (((actions - action_mean) * act_mask) ** 2).sum() / act_mask_sum
        action_std = th.sqrt(action_var)
        at_bound = ((actions < bound_eps) | (actions > 1 - bound_eps)).float()
        action_at_bound_fraction = (at_bound * act_mask).sum() / act_mask_sum

        alpha_all, beta_all = old_pi.chunk(2, dim=-1)
        alpha_mean = (alpha_all * act_mask).sum() / act_mask_sum
        beta_mean = (beta_all * act_mask).sum() / act_mask_sum
        u_shaped = ((alpha_all < 1.0) & (beta_all < 1.0)).float()
        u_shaped_fraction = (u_shaped * act_mask).sum() / act_mask_sum

        # Behavioral-pattern diagnostics ported unchanged from
        # fpo_continuous_learner.py (same thresholds/definitions), so MAFPO
        # and MAPPO's wandb curves are directly comparable on: net-force
        # magnitude (MPE opposing-pair encoding), distance-to-nearest-landmark
        # conditioned speed (far+fast / near+slow / near+still-fast), and
        # consecutive-step direction-flip oscillation. Beta samples `actions`
        # are already the final executed action (no separate raw/noise split
        # the way FPO has), so there's no phi-only counterpart here.
        if self.n_actions >= 5:
            force_x = actions[..., 2] - actions[..., 1]      # [B,T,N]
            force_y = actions[..., 4] - actions[..., 3]
            force_valid = mask.bool()
            valid_fx = force_x[force_valid]
            valid_fy = force_y[force_valid]
            if valid_fx.numel() > 0:
                force_mag = th.sqrt(valid_fx ** 2 + valid_fy ** 2)
                force_x_mean = valid_fx.mean().item()
                force_y_mean = valid_fy.mean().item()
                force_magnitude_mean = force_mag.mean().item()
                force_near_zero_fraction = (force_mag < 0.1).float().mean().item()
                force_near_max_fraction = (force_mag > 0.9).float().mean().item()
            else:
                force_x_mean = force_y_mean = force_magnitude_mean = 0.0
                force_near_zero_fraction = force_near_max_fraction = 0.0

            # Per-agent breakdown and per-timestep same-action fraction,
            # ported unchanged from fpo_continuous_learner.py -- see there
            # for the full rationale (divergence across agents vs. collapse
            # to a shared behavior, since obs_agent_id is supposed to let
            # one shared network act differently per agent).
            per_agent_fx, per_agent_fy = [], []
            for i in range(self.n_agents):
                agent_mask = mask[..., i].bool()
                fx_i = force_x[..., i][agent_mask]
                fy_i = force_y[..., i][agent_mask]
                per_agent_fx.append(fx_i.mean().item() if fx_i.numel() > 0 else 0.0)
                per_agent_fy.append(fy_i.mean().item() if fy_i.numel() > 0 else 0.0)
            force_x_agent_std = float(th.tensor(per_agent_fx).std(unbiased=False))
            force_y_agent_std = float(th.tensor(per_agent_fy).std(unbiased=False))

            still_thresh = 0.15
            is_still = (force_x.abs() < still_thresh) & (force_y.abs() < still_thresh)
            x_dominant = force_x.abs() >= force_y.abs()
            # category ids: 0=still, 1=+x, 2=-x, 3=+y, 4=-y
            category = th.zeros_like(force_x, dtype=th.long)
            category = th.where(is_still, th.zeros_like(category), category)
            moving = ~is_still
            category = th.where(moving & x_dominant & (force_x > 0), th.full_like(category, 1), category)
            category = th.where(moving & x_dominant & (force_x <= 0), th.full_like(category, 2), category)
            category = th.where(moving & (~x_dominant) & (force_y > 0), th.full_like(category, 3), category)
            category = th.where(moving & (~x_dominant) & (force_y <= 0), th.full_like(category, 4), category)

            step_valid = mask[..., 0].bool()                          # [B,T]
            cat_onehot = th.nn.functional.one_hot(category, num_classes=5)  # [B,T,N,5]
            cat_counts = cat_onehot.sum(dim=2)                        # [B,T,5]
            max_count = cat_counts.amax(dim=-1)                       # [B,T]
            valid_max_count = max_count[step_valid]
            if valid_max_count.numel() > 0:
                agents_same_action_fraction = (
                    (valid_max_count >= 2).float().mean().item()
                )
                agents_all_same_action_fraction = (
                    (valid_max_count >= self.n_agents).float().mean().item()
                )
            else:
                agents_same_action_fraction = agents_all_same_action_fraction = 0.0

            if "obs" in batch.scheme:
                n_landmarks = getattr(self.args, "n_landmarks", self.n_agents)
                obs = batch["obs"][:, :-1].float()
                landmark_end = 4 + 2 * n_landmarks
                if obs.shape[-1] >= landmark_end:
                    landmark_rel = obs[..., 4:landmark_end].reshape(
                        *obs.shape[:-1], n_landmarks, 2
                    )
                    nearest_dist = landmark_rel.norm(dim=-1).min(dim=-1)[0]

                    far_thresh = getattr(self.args, "landmark_far_thresh", 0.3)
                    near_thresh = getattr(self.args, "landmark_near_thresh", 0.15)
                    force_mag_full = th.sqrt(force_x ** 2 + force_y ** 2)
                    is_far = nearest_dist > far_thresh
                    is_near = nearest_dist < near_thresh
                    is_fast = force_mag_full > 0.7
                    is_slow = force_mag_full < 0.3
                    step_agent_valid = mask.bool()
                    denom = step_agent_valid.float().sum()

                    def _frac(cond):
                        return (
                            (cond & step_agent_valid).float().sum() / denom
                        ).item() if denom > 0 else 0.0

                    far_fast_fraction = _frac(is_far & is_fast)
                    near_slow_fraction = _frac(is_near & is_slow)
                    near_still_fast_fraction = _frac(is_near & is_fast)

                    fx_prev, fx_curr = force_x[:, :-1], force_x[:, 1:]
                    fy_prev, fy_curr = force_y[:, :-1], force_y[:, 1:]
                    mag_prev = th.sqrt(fx_prev ** 2 + fy_prev ** 2)
                    mag_curr = th.sqrt(fx_curr ** 2 + fy_curr ** 2)
                    dot = fx_prev * fx_curr + fy_prev * fy_curr
                    pair_valid = mask[:, :-1].bool() & mask[:, 1:].bool()
                    is_flip = (mag_prev > 0.5) & (mag_curr > 0.5) & (dot < 0)
                    denom_pairs = pair_valid.float().sum()
                    oscillation_fraction = (
                        (is_flip & pair_valid).float().sum() / denom_pairs
                    ).item() if denom_pairs > 0 else 0.0
                else:
                    far_fast_fraction = near_slow_fraction = 0.0
                    near_still_fast_fraction = oscillation_fraction = 0.0
            else:
                far_fast_fraction = near_slow_fraction = 0.0
                near_still_fast_fraction = oscillation_fraction = 0.0
        else:
            force_x_mean = force_y_mean = force_magnitude_mean = 0.0
            force_near_zero_fraction = force_near_max_fraction = 0.0
            force_x_agent_std = force_y_agent_std = 0.0
            agents_same_action_fraction = agents_all_same_action_fraction = 0.0
            far_fast_fraction = near_slow_fraction = 0.0
            near_still_fast_fraction = oscillation_fraction = 0.0

        for _ in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            pi = th.stack(mac_out, dim=1)

            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()

            log_pi_taken, entropy = self._get_log_prob_and_entropy(pi, actions)
            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy)
                    * mask
                ).sum()
                / mask.sum()
            )

            approx_kl = (
                (old_log_pi_taken.detach() - log_pi_taken) * mask
            ).sum() / mask.sum()
            clip_fraction = (
                (
                    (
                        (ratios > 1 + self.args.eps_clip)
                        | (ratios < 1 - self.args.eps_clip)
                    ).float()
                    * mask
                ).sum()
                / mask.sum()
            )
            entropy_mean = (entropy * mask).sum() / mask.sum()
            ratio_mean = (ratios * mask).sum() / mask.sum()
            ratio_std = th.sqrt(
                (((ratios - ratio_mean) * mask) ** 2).sum() / mask.sum()
            )
            log_pi_taken_mean = (log_pi_taken * mask).sum() / mask.sum()
            old_log_pi_taken_mean = (old_log_pi_taken * mask).sum() / mask.sum()
            advantage_mean = (advantages * mask).sum() / mask.sum()
            advantage_std = th.sqrt(
                (((advantages - advantage_mean) * mask) ** 2).sum() / mask.sum()
            )

            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat("advantage_mean", advantage_mean.item(), t_env)
            self.logger.log_stat("advantage_std", advantage_std.item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("entropy_mean", entropy_mean.item(), t_env)
            self.logger.log_stat("approx_kl", approx_kl.item(), t_env)
            self.logger.log_stat("clip_fraction", clip_fraction.item(), t_env)
            self.logger.log_stat("ratio_mean", ratio_mean.item(), t_env)
            self.logger.log_stat("ratio_std", ratio_std.item(), t_env)
            self.logger.log_stat("log_pi_taken_mean", log_pi_taken_mean.item(), t_env)
            self.logger.log_stat(
                "old_log_pi_taken_mean", old_log_pi_taken_mean.item(), t_env
            )
            self.logger.log_stat("action_mean", action_mean.item(), t_env)
            self.logger.log_stat("action_std", action_std.item(), t_env)
            self.logger.log_stat(
                "action_at_bound_fraction", action_at_bound_fraction.item(), t_env
            )
            self.logger.log_stat("alpha_mean", alpha_mean.item(), t_env)
            self.logger.log_stat("beta_mean", beta_mean.item(), t_env)
            self.logger.log_stat("u_shaped_fraction", u_shaped_fraction.item(), t_env)
            self.logger.log_stat("force_x_mean", force_x_mean, t_env)
            self.logger.log_stat("force_y_mean", force_y_mean, t_env)
            self.logger.log_stat("force_magnitude_mean", force_magnitude_mean, t_env)
            self.logger.log_stat("force_x_agent_std", force_x_agent_std, t_env)
            self.logger.log_stat("force_y_agent_std", force_y_agent_std, t_env)
            self.logger.log_stat(
                "agents_same_action_fraction", agents_same_action_fraction, t_env
            )
            self.logger.log_stat(
                "agents_all_same_action_fraction",
                agents_all_same_action_fraction,
                t_env,
            )
            self.logger.log_stat(
                "force_near_zero_fraction", force_near_zero_fraction, t_env
            )
            self.logger.log_stat(
                "force_near_max_fraction", force_near_max_fraction, t_env
            )
            self.logger.log_stat("far_fast_fraction", far_fast_fraction, t_env)
            self.logger.log_stat("near_slow_fraction", near_slow_fraction, t_env)
            self.logger.log_stat(
                "near_still_fast_fraction", near_still_fast_fraction, t_env
            )
            self.logger.log_stat("oscillation_fraction", oscillation_fraction, t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()
        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        running_log["td_error_abs"].append(
            masked_td_error.abs().sum().item() / mask_elems
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
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
                    nstep_return_t += self.args.gamma**step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma**step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma**step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
