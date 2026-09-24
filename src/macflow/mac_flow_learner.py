import copy
import csv
import os
from pathlib import Path

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry

from macflow.joint_flow_actor import JointFlowActor
from macflow.theta_metrics import ThetaMetrics


class MACFlowLearner:
    """Online, off-policy MAC-Flow: MADDPG's growing-replay-buffer /
    target-network training loop, carrying MAC-Flow's three losses
    (arXiv:2511.05005) instead of MADDPG's single deterministic
    policy-gradient loss. See macflow/README.md for the full design writeup.

    Where the offline paper trains flow-BC -> critic -> distillation
    sequentially, each stage waiting for the previous one to converge and
    then freezing it, this learner runs all three in every train() call on a
    minibatch sampled from a continuously-growing replay buffer (run.py's
    generic off-policy path -- see run.py, the same one MADDPG already uses;
    nothing there needed to change). "Frozen after convergence" is replaced
    throughout by "frozen target network, Polyak-updated" -- target_mac,
    target_critic and target_joint_flow play the exact role old_mac/
    target_critic play in fpo_continuous_learner.py / maddpg_learner.py:

      1. Flow-BC   (Eq.6):  self.joint_flow learns v_phi(t,o,x) by CFM
                            regression against whatever joint actions are
                            currently in the buffer -- no dataset-quality
                            assumption, the buffer's own composition (mix of
                            past policy versions) IS the behavior
                            distribution being cloned.
      2. Critic    (Eq.2):  per-agent Q_theta, TD target's bootstrap action
                            a' comes from target_mac (the target one-step
                            actor), not a full ODE draw from target_joint_flow
                            -- same "target actor for the next action" trick
                            TD3/DDPG/MADDPG use, avoids paying for an ODE
                            integration on every critic update.
      3. Distillation (Eq.9): self.mac (the one-step actor, also the actual
                            rollout policy) is pushed by -Q_tot (direct
                            reparameterized value gradient through the
                            critic, MADDPG-style) plus an alpha-weighted BC
                            term pulling it toward target_joint_flow's own
                            multi-step ODE endpoint for the SAME z. Using the
                            *target* flow here (not self.joint_flow, which is
                            being updated in the very same train() call) is
                            the one substitution that makes this stage-3 loss
                            well-posed online -- anchoring to a fast-moving
                            target reproduces exactly the instability this
                            repo already spent a long time fighting in
                            fpo_continuous_learner.py's old_mac/w_b machinery.

    Unlike MAFPO/FPO, there is no PPO-style importance ratio, no eps_clip, no
    trust region on the velocity field -- policy improvement is a direct
    value-gradient (DDPG-style) through the one-step actor, not an
    importance-sampled surrogate around a flow with no closed-form density.
    That sidesteps the whole ratio/ELBO-mismatch failure mode FPO's git
    history documents; the price is the usual off-policy actor-critic
    concerns instead (Q overestimation, target staleness, exploration
    scheduling), handled with the standard target-network toolbox.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.joint_action_dim = args.n_agents * args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.actor_params = list(mac.parameters())
        self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr)

        self.joint_flow = JointFlowActor(scheme, args)
        self.target_joint_flow = copy.deepcopy(self.joint_flow)
        self.flow_params = list(self.joint_flow.parameters())
        self.flow_optimiser = Adam(params=self.flow_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.theta_metrics_enabled = getattr(args, "mac_flow_theta_metrics", True)
        self.actor_theta_metrics = ThetaMetrics()
        self.flow_theta_metrics = ThetaMetrics()
        self.theta_train_updates = 0
        self._theta_trace_file = None
        self._theta_trace_writer = None

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(1,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        batch_size = batch.batch_size
        actions = batch["actions"].float()               # [B, T_full, N, A]

        rewards = batch["reward"][:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        if self.args.common_reward:
            assert rewards.size(2) == 1
            rewards = rewards.expand(-1, -1, self.n_agents).unsqueeze(-1)
        else:
            rewards = rewards.unsqueeze(-1)               # [B,T,N,1]

        filled = batch["filled"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()
        mask = filled.clone()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])   # [B,T,1]
        mask_n = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)          # [B,T,N,1]
        bootstrap_mask_n = (1 - terminated).unsqueeze(2).expand(
            -1, -1, self.n_agents, -1
        )                                                                      # [B,T,N,1]

        actions_bt = actions[:, :-1].reshape(batch_size, -1, self.joint_action_dim)

        # ==================== 1) Flow-BC (Eq.6) ====================
        flow_inputs = self.joint_flow._build_inputs(batch)          # [B,T_full,F]
        mb_flow_inputs = flow_inputs[:, :-1]                        # [B,T,F]
        h_flow = self.joint_flow.encode(mb_flow_inputs)             # [B,T,hidden]

        eps_joint = th.randn_like(actions_bt)
        t_flow = th.rand(batch_size, actions_bt.shape[1], 1, device=actions_bt.device)
        # Unbounded-logit interpolation -- see JointFlowActor's docstring for
        # why this must not run directly on the (0,1)-bounded stored actions.
        a_joint_logit = th.logit(actions_bt.clamp(1e-4, 1 - 1e-4))
        x_t = (1 - t_flow) * eps_joint + t_flow * a_joint_logit
        v_pred = self.joint_flow.velocity(h_flow, x_t, t_flow)
        flow_target = a_joint_logit - eps_joint
        per_step_flow_loss = ((v_pred - flow_target) ** 2).mean(dim=-1, keepdim=True)
        flow_loss = (per_step_flow_loss * mask).sum() / mask.sum().clamp(min=1)

        self.flow_optimiser.zero_grad()
        flow_loss.backward()
        flow_grad_norm = th.nn.utils.clip_grad_norm_(
            self.flow_params, self.args.grad_norm_clip
        )
        self.flow_optimiser.step()

        # ==================== 2) Critic (Eq.2) ====================
        central_inputs = self._build_inputs(batch)                  # [B,T_full,N,F]
        joint_actions = actions.reshape(
            batch_size, -1, 1, self.joint_action_dim
        ).expand(-1, -1, self.n_agents, -1)                         # [B,T_full,N,joint_dim]

        q_taken = self.critic(central_inputs[:, :-1], joint_actions[:, :-1].detach())
        q_taken = q_taken.view(batch_size, -1, 1)                   # [B,T*N,1]

        self.target_mac.init_hidden(batch_size)
        smoothing_std = getattr(self.args, "mac_flow_target_noise", 0.0)
        smoothing_clip = getattr(self.args, "mac_flow_target_noise_clip", 0.0)
        target_actions_list = []
        for t in range(1, batch.max_seq_length):
            target_actions_list.append(
                self.target_mac.target_actions(batch, t, smoothing_std, smoothing_clip)
            )
        target_actions = th.stack(target_actions_list, dim=1)        # [B,T,N,A]
        target_joint_actions = target_actions.reshape(
            batch_size, -1, 1, self.joint_action_dim
        ).expand(-1, -1, self.n_agents, -1)

        with th.no_grad():
            target_vals = self.target_critic(
                central_inputs[:, 1:], target_joint_actions
            )
            target_vals = target_vals.view(batch_size, -1, 1)
            if self.args.standardise_returns:
                target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            targets = (
                rewards.reshape(-1, 1)
                + self.args.gamma
                * bootstrap_mask_n.reshape(-1, 1)
                * target_vals.reshape(-1, 1)
            )
            if self.args.standardise_returns:
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = q_taken.reshape(-1, 1) - targets
        masked_td_error = td_error * mask_n.reshape(-1, 1)
        critic_loss = (masked_td_error ** 2).sum() / mask_n.reshape(-1, 1).sum().clamp(min=1)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        # ==================== 3) Q-guided distillation (Eq.9) ====================
        self.mac.init_hidden(batch_size)
        raws, zs = [], []
        for t in range(batch.max_seq_length - 1):
            inputs_t = self.mac._build_inputs(batch, t)
            h_t = self.mac.agent.encode(inputs_t, self.mac.hidden_states)
            self.mac.hidden_states = h_t
            z_t = th.randn(h_t.shape[0], self.n_actions, device=h_t.device)
            raw_t = self.mac.agent.act(h_t, z_t)
            raws.append(raw_t.view(batch_size, self.n_agents, -1))
            zs.append(z_t.view(batch_size, self.n_agents, -1))
        raw_seq = th.stack(raws, dim=1)                # [B,T,N,A], grad-tracked
        z_seq = th.stack(zs, dim=1)                     # [B,T,N,A]
        actions_pred = th.sigmoid(raw_seq)

        joint_actions_pred = actions_pred.reshape(
            batch_size, -1, 1, self.joint_action_dim
        ).expand(-1, -1, self.n_agents, -1)
        q_pi = self.critic(central_inputs[:, :-1], joint_actions_pred)
        q_pi = q_pi.view(batch_size, -1, 1)
        q_loss = -(q_pi.reshape(-1, 1) * mask_n.reshape(-1, 1)).sum() / mask_n.reshape(
            -1, 1
        ).sum().clamp(min=1)

        # BC anchor: target_joint_flow's OWN multi-step ODE endpoint for the
        # same z, NOT self.joint_flow (which this very train() call just
        # updated above) -- see class docstring.
        with th.no_grad():
            h_target_flow = self.target_joint_flow.encode(flow_inputs[:, :-1])
            z_joint = z_seq.reshape(batch_size, -1, self.joint_action_dim)
            distill_steps = getattr(
                self.args, "mac_flow_distill_steps",
                getattr(self.args, "cfm_rollout_steps", 5),
            )
            target_endpoint = self.target_joint_flow.integrate(
                h_target_flow, z_joint, distill_steps
            )                                            # unbounded latent, [B,T,joint_dim]
        target_endpoint_per_agent = target_endpoint.reshape(
            batch_size, -1, self.n_agents, self.n_actions
        )
        bc_diff = (raw_seq - target_endpoint_per_agent) ** 2
        per_step_bc_loss = bc_diff.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)  # [B,T,1]
        bc_loss = (per_step_bc_loss * mask).sum() / mask.sum().clamp(min=1)

        alpha = self._current_bc_alpha(t_env)
        reg_coef = getattr(self.args, "mac_flow_actor_reg", 0.0)
        if reg_coef > 0.0:
            reg_term = (raw_seq ** 2).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
            reg_loss = (reg_term * mask).sum() / mask.sum().clamp(min=1)
        else:
            reg_loss = th.zeros((), device=raw_seq.device)

        actor_loss = q_loss + alpha * bc_loss + reg_coef * reg_loss

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(
            self.actor_params, self.args.grad_norm_clip
        )
        self.actor_optimiser.step()

        # ==================== target networks ====================
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

        # Track EVERY train() call, even when scalar logging is less frequent.
        # This keeps the cosine tied to adjacent updates, not adjacent log rows.
        theta_stats = {}
        if self.theta_metrics_enabled:
            self.theta_train_updates += 1
            theta_stats = self.actor_theta_metrics.measure(self.actor_params)
            theta_stats.update({"flow_" + key: value for key, value in
                                self.flow_theta_metrics.measure(self.flow_params).items()})
            self._write_theta_trace(theta_stats, t_env)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_bt = mask.expand(-1, -1, self.joint_action_dim).bool()
            valid_actions = actions_bt[mask_bt]
            bound_eps = 0.02
            if valid_actions.numel() > 0:
                action_mean = valid_actions.mean().item()
                action_std = valid_actions.std(unbiased=False).item()
                at_bound = (valid_actions < bound_eps) | (valid_actions > 1 - bound_eps)
                action_at_bound_fraction = at_bound.float().mean().item()
            else:
                action_mean = action_std = action_at_bound_fraction = 0.0

            mask_elems = mask_n.sum().item()
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat(
                "td_error_abs", masked_td_error.abs().sum().item() / max(mask_elems, 1), t_env
            )
            self.logger.log_stat(
                "q_taken_mean", (q_taken.reshape(-1, 1) * mask_n.reshape(-1, 1)).sum().item()
                / max(mask_elems, 1), t_env
            )
            self.logger.log_stat("flow_bc_loss", flow_loss.item(), t_env)
            self.logger.log_stat("flow_grad_norm", flow_grad_norm.item(), t_env)
            self.logger.log_stat("distill_q_loss", q_loss.item(), t_env)
            self.logger.log_stat("distill_bc_loss", bc_loss.item(), t_env)
            self.logger.log_stat("distill_bc_alpha", alpha, t_env)
            self.logger.log_stat("actor_grad_norm", actor_grad_norm.item(), t_env)
            self.logger.log_stat("action_mean", action_mean, t_env)
            self.logger.log_stat("action_std", action_std, t_env)
            self.logger.log_stat("action_at_bound_fraction", action_at_bound_fraction, t_env)
            for key, value in theta_stats.items():
                self.logger.log_stat(key, value, t_env)
            self.log_stats_t = t_env

    def _write_theta_trace(self, metrics, t_env):
        if not getattr(self.args, "mac_flow_theta_trace", False):
            return
        if self._theta_trace_writer is None:
            token = getattr(self.args, "unique_token", f"mac_flow_seed{self.args.seed}_pid{os.getpid()}")
            path = Path(self.args.local_results_path) / "theta" / f"{token}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._theta_trace_file = path.open("x", newline="", buffering=1)
            self._theta_trace_writer = csv.DictWriter(
                self._theta_trace_file, fieldnames=["t_env", "train_updates", *metrics]
            )
            self._theta_trace_writer.writeheader()
            self.logger.console_logger.info("Writing every-update theta metrics to %s", path)
        self._theta_trace_writer.writerow(dict(t_env=t_env, train_updates=self.theta_train_updates, **metrics))

    def _current_bc_alpha(self, t_env):
        """Optional linear anneal alpha_init -> alpha_final over
        mac_flow_bc_anneal_steps env steps. Off-policy online data is
        continually refreshed by the current policy, so unlike offline RL
        there is no persistent OOD-action problem to guard against -- alpha
        can shrink from a stabilizer-strength value toward a light touch as
        training progresses, instead of staying at its offline-strength
        value for the whole run. Defaults to a constant (anneal_steps=0)."""
        alpha_init = getattr(self.args, "mac_flow_bc_alpha", 1.0)
        alpha_final = getattr(self.args, "mac_flow_bc_alpha_final", alpha_init)
        anneal_steps = getattr(self.args, "mac_flow_bc_anneal_steps", 0)
        if anneal_steps <= 0:
            return alpha_init
        frac = min(1.0, t_env / anneal_steps)
        return alpha_init + frac * (alpha_final - alpha_init)

    def _build_inputs(self, batch, t=None):
        """Central critic inputs, mirrors maddpg_learner._build_inputs (own
        copy, not an import, so nothing in maddpg_learner.py is touched)."""
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = [batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1)]
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions"][:, 0:1]), batch["actions"][:, :-1]],
                    dim=1,
                )
                inputs.append(last_actions)
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            )
        return th.cat(inputs, dim=-1)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_joint_flow.load_state_dict(self.joint_flow.state_dict())

    def _update_targets_soft(self, tau):
        for tp, p in zip(self.target_mac.parameters(), self.mac.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        for tp, p in zip(self.target_joint_flow.parameters(), self.joint_flow.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.joint_flow.cuda()
        self.target_joint_flow.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.joint_flow.state_dict(), "{}/joint_flow.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.flow_optimiser.state_dict(), "{}/flow_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        if self.theta_metrics_enabled:
            th.save({"actor": self.actor_theta_metrics.state_dict(),
                     "flow": self.flow_theta_metrics.state_dict(),
                     "train_updates": self.theta_train_updates},
                    "{}/theta_metrics.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_state(self.mac)
        self.joint_flow.load_state_dict(
            th.load("{}/joint_flow.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.target_joint_flow.load_state_dict(self.joint_flow.state_dict())
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimiser.load_state_dict(
            th.load("{}/actor_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.flow_optimiser.load_state_dict(
            th.load("{}/flow_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
        metrics_path = Path(path) / "theta_metrics.th"
        if self.theta_metrics_enabled:
            if metrics_path.exists():
                state = th.load(metrics_path, map_location="cpu", weights_only=True)
                self.actor_theta_metrics.load_state_dict(state["actor"])
                self.flow_theta_metrics.load_state_dict(state["flow"])
                self.theta_train_updates = state["train_updates"]
            else:
                self.actor_theta_metrics = ThetaMetrics()
                self.flow_theta_metrics = ThetaMetrics()
                self.theta_train_updates = 0
                self.logger.console_logger.warning(
                    "Checkpoint has no theta reference; a new reference will be captured after the next train()."
                )
