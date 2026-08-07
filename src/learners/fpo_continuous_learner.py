import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPOContinuousLearner:
    """Continuous FPO learner.

    The rollout policy stores fixed CFM eps/t/action points. By default
    (use_policyflow_ratio=False) we recompute the current CFM loss at those
    same points and use the loss change to build the FPO ratio:

        rho_s = exp(mean(clamp(L_old - L_new, -rho_clip, rho_clip)))

    Optional PolicyFlow-style additions (arXiv:2602.01156), aimed at the
    action-collapse failure mode FPO has no built-in defense against (no
    closed-form entropy, and an ELBO-style ratio the paper notes is
    asymmetrically less reliable exactly when advantage<0 wants the ratio to
    shrink):

    - use_policyflow_ratio=True replaces rho_s above with an exact Gaussian
      importance ratio. The policy is treated as a ~ N(a1(z;s), sigma^2) where
      a1(z;s) is the flow's deterministic endpoint and sigma is a learned,
      state-independent terminal noise actually injected at rollout (see
      fpo_actor.sample_action). For the realized action a and its own
      rollout-time noise n = a - a1_old (a1_old stored as batch["action_raw"]),
      the endpoint shift a1_new - a1_old is approximated (no ODE re-integration
      needed) by delta_v = mean_t[v_new(x_t,t) - v_old(x_t,t)] evaluated at the
      same CFM neighborhood points used for cfm_loss:

          log_ratio = -0.5 * sum_i[ (n_i-delta_v_i)^2/sigma_new_i^2
                                     - n_i^2/sigma_old_i^2
                                     + log(sigma_new_i^2/sigma_old_i^2) ]
          rho_s = exp(clamp(log_ratio, -rho_clip, rho_clip))

      Because sigma now appears directly in this ratio, it gets real gradient
      pressure from pg_loss in both directions (unlike w_g acting alone, which
      only ever pushes sigma up with nothing pushing back — see git history).

    - w_b: Brownian regularizer pulling the velocity field toward its
      entropy-increasing/score-corrected version (eta_t, Eq.15).
    - w_g: matching Gaussian entropy bonus that keeps sigma from being
      squeezed toward sigma_min.

    All three (use_policyflow_ratio, w_b, w_g) default off/0 via
    mafpo_continuous.yaml unless explicitly enabled.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        # Frozen velocity-field snapshot from the start of this round, used for
        # the trust-region penalty (see train()). Refreshed to match self.mac
        # once per train() call, after all epochs/minibatches for this round.
        self.old_mac = copy.deepcopy(mac)
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

        # Action-collapse diagnostic: computed once from the *actually executed*
        # rollout actions (not the K-sample entropy proxy), so it directly answers
        # "is the policy pinning actions to the [0,1] boundary" regardless of
        # whether the entropy bonus is on. bound_eps is the distance from 0/1
        # counted as "at the boundary".
        actions_taken = batch["actions"][:, :-1].float()          # [B,T,N,A]
        action_valid = mask.unsqueeze(-1).expand_as(actions_taken).bool()
        valid_actions = actions_taken[action_valid]
        bound_eps = 0.02
        if valid_actions.numel() > 0:
            action_mean = valid_actions.mean().item()
            action_std = valid_actions.std(unbiased=False).item()
            at_bound = (valid_actions < bound_eps) | (valid_actions > 1 - bound_eps)
            action_at_bound_fraction = at_bound.float().mean().item()
        else:
            action_mean = action_std = action_at_bound_fraction = 0.0

        # clamp-前诊断: action_raw 是积分终点在被 clamp(0,1) 之前的原始值。
        # overshoot = 超出 [0,1] 的距离(0 表示压根没超出，本来就在界内)。
        # overshoot_at_bound 只统计"最终落在边界上"的那些样本，看它们原始究竟
        # 冲出去多远——数值越大，说明越多是"远远越界被硬拉回来"而不是"刚好压线"。
        if "action_raw" in batch.scheme:
            action_raw = batch["action_raw"][:, :-1].float()      # [B,T,N,A]
            valid_raw = action_raw[action_valid]
            overshoot = th.clamp(-valid_raw, min=0) + th.clamp(valid_raw - 1, min=0)
            action_overshoot_mean = overshoot.mean().item() if overshoot.numel() > 0 else 0.0
            if valid_actions.numel() > 0 and at_bound.any():
                action_overshoot_at_bound_mean = overshoot[at_bound].mean().item()
            else:
                action_overshoot_at_bound_mean = 0.0
        else:
            action_overshoot_mean = action_overshoot_at_bound_mean = 0.0

        initial_cfm_loss = batch["initial_cfm_loss"][:, :-1]    # [B,T,N,cfm_n,1]
        rho_clip = getattr(self.args, "cfm_rho_clip", 3.0)
        entropy_coef = getattr(self.args, "entropy_coef", 0.0)
        entropy_n_samples = getattr(self.args, "entropy_n_samples", 4)
        lambda_trust = getattr(self.args, "lambda_trust", 0.0)
        w_b = getattr(self.args, "w_b", 0.0)
        w_g = getattr(self.args, "w_g", 0.0)
        use_pf_ratio = getattr(self.args, "use_policyflow_ratio", False)
        need_v_old = lambda_trust > 0.0 or w_b > 0.0 or use_pf_ratio

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
            "rho_s_mean": [],
            "rho_s_std": [],
            "clip_fraction": [],
            "actor_grad_norm": [],
            "entropy_mean": [],
            "trust_loss": [],
            "brownian_loss": [],
            "gaussian_entropy": [],
            "sigma_mean": [],
            "delta_v_abs_mean": [],
            "n_abs_mean": [],
        }
        critic_train_stats = {
            k: [] for k in ["critic_loss", "critic_grad_norm", "td_error_abs",
                            "target_mean", "value_mean"]
        }

        for _ in range(self.args.epochs):
            # The critic still uses full sequences for GAE. We train it once
            # per epoch, then freeze the advantages for the shuffled actor
            # minibatches in this epoch.
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
                mb_cfm_loss, mb_v_new, mb_flat_h, mb_flat_x_t, mb_flat_t = (
                    self._compute_cfm_loss_for_time_indices(batch, h_seq, mb_time_idx)
                )

                # Trust-region penalty: keep this round's velocity field close to
                # the field that generated the rollout, evaluated at the exact
                # same (h, x_t, t) points already used for the CFM loss above.
                # Unlike eps_clip (which only gates whether a sample's gradient
                # fires at all), this is a continuous, always-on pull back toward
                # last round's policy, directly limiting how far v_pred can drift
                # per round regardless of advantage sign.
                if need_v_old:
                    with th.no_grad():
                        v_old = self.old_mac.agent.velocity(
                            mb_flat_h, mb_flat_x_t, mb_flat_t
                        ).reshape_as(mb_v_new)
                if lambda_trust > 0.0:
                    trust_loss = ((mb_v_new - v_old) ** 2).mean()
                else:
                    trust_loss = th.zeros((), device=mb_cfm_loss.device)

                # Brownian regularizer (PolicyFlow, arXiv:2602.01156, Eq.15):
                # eta_t = (1-t)*v_new - (x_t - t*v_old). Under the rectified-flow
                # score-velocity relationship, (x_t - t*v_old) = -(1-t)*score_old,
                # so eta_t/(1-t) = v_new + score_old — penalizing ||eta_t||^2 pulls
                # the *current* velocity field toward the entropy-increasing
                # (score-corrected / "Brownian") version of the reference field,
                # instead of letting it collapse into a purely deterministic map
                # that concentrates mass at the action boundary. mb_flat_t is
                # flat [M*N*cfm_n, 1]; v_new/v_old are [M,N,cfm_n,A], so broadcast
                # t back against them via mb_flat_t's own leading dim.
                if w_b > 0.0:
                    t_bcast = mb_flat_t.reshape_as(mb_v_new[..., :1])
                    eta_t = (1 - t_bcast) * mb_v_new - (
                        mb_flat_x_t.reshape_as(mb_v_new) - t_bcast * v_old
                    )
                    brownian_loss = (eta_t ** 2).mean()
                else:
                    brownian_loss = th.zeros((), device=mb_cfm_loss.device)

                # Gaussian entropy bonus on the learned terminal-noise sigma
                # (PolicyFlow Eq.15's second term): sigma is a single
                # state-independent vector (fpo_actor.sigma()), so this is a
                # scalar, not batch-averaged. Maximizing it (i.e. subtracting
                # w_g * gaussian_entropy from the loss we minimize) keeps sigma
                # from being squeezed to sigma_min by the other loss terms —
                # same role as entropy_coef, but exact instead of a sampled
                # proxy, and tied to noise that's actually injected at rollout.
                if w_g > 0.0:
                    sigma = self.mac.agent.sigma()
                    gaussian_entropy = 0.5 * th.sum(
                        th.log(2 * th.pi * th.e * sigma ** 2)
                    )
                else:
                    sigma = self.mac.agent.sigma().detach()
                    gaussian_entropy = th.zeros((), device=mb_cfm_loss.device)

                advantages_by_time = advantages.reshape(-1, self.n_agents)
                mb_advantages_2d = advantages_by_time[mb_time_idx]   # [M,N]

                if use_pf_ratio:
                    # PolicyFlow-style exact Gaussian ratio (see class docstring).
                    # n = the *actually sampled* rollout noise, read straight from
                    # batch["action_noise"] -- NOT reconstructed as
                    # (action - action_raw), because once clamp() truncates a
                    # sample, that difference is no longer a real N(0,sigma^2)
                    # draw (it's the noise it *would* have taken had clamping
                    # not intervened, which is a biased quantity exactly on the
                    # boundary-heavy samples this whole investigation is about).
                    # delta_v approximates how far the flow endpoint would shift
                    # under the NEW policy, estimated from the same CFM
                    # neighborhood points as cfm_loss/eta_t above (no extra
                    # sampling, no ODE re-integration).
                    n_flat = batch["action_noise"][:, :-1].float().reshape(
                        -1, self.n_agents, self.n_actions
                    )
                    n_noise = n_flat[mb_time_idx]                   # [M,N,A]
                    delta_v = (mb_v_new - v_old).mean(dim=2)        # [M,N,cfm_n,A] -> [M,N,A]

                    sigma_new = self.mac.agent.sigma()              # [A], grad-tracked
                    with th.no_grad():
                        sigma_old = self.old_mac.agent.sigma()      # [A], frozen reference

                    log_ratio_per_dim = -0.5 * (
                        (n_noise - delta_v) ** 2 / sigma_new ** 2
                        - n_noise ** 2 / sigma_old ** 2
                        + th.log(sigma_new ** 2 / sigma_old ** 2)
                    )
                    log_ratio = log_ratio_per_dim.sum(dim=-1)       # [M,N]
                    mb_rho_s = th.exp(th.clamp(log_ratio, -rho_clip, rho_clip))

                    delta_v_abs_mean = delta_v.detach().abs().mean().item()
                    n_abs_mean = n_noise.detach().abs().mean().item()
                else:
                    mb_initial_cfm_loss = initial_cfm_loss.reshape(
                        -1, self.n_agents, initial_cfm_loss.size(-2), initial_cfm_loss.size(-1)
                    )[mb_time_idx]
                    diff = mb_initial_cfm_loss - mb_cfm_loss              # [M,N,cfm_n,1]
                    diff_mean = diff.mean(dim=(-2, -1))                   # [M,N]
                    mb_rho_s = th.exp(th.clamp(diff_mean, -rho_clip, rho_clip))  # [M,N]
                    delta_v_abs_mean = n_abs_mean = 0.0

                mb_rho_s = mb_rho_s.reshape(-1)
                mb_advantages = mb_advantages_2d.reshape(-1)
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(
                    mb_rho_s, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                ) * mb_advantages
                pg_loss = -th.min(surr1, surr2).mean()

                # Entropy bonus: FPO has no closed-form action distribution to take
                # log/entropy of, so approximate "how spread out are the actions
                # this state generates" directly — draw entropy_n_samples fresh eps
                # for each (timestep, agent) in the minibatch, integrate each
                # through the *current* velocity field, and measure the variance
                # across those samples. Collapsing to a near-deterministic map
                # (all eps -> similar action) drives this toward 0; maximizing it
                # counteracts that collapse.
                if entropy_coef > 0.0:
                    h_flat = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])
                    mb_h = h_flat[mb_time_idx]                     # [M,N,H]
                    M, N, H = mb_h.shape
                    K = entropy_n_samples
                    h_rep = mb_h.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, H)
                    ent_eps = th.rand(M * N * K, self.n_actions, device=mb_h.device)
                    n_steps = getattr(self.args, "cfm_rollout_steps", 10)
                    x1 = self.mac.agent.integrate(h_rep, ent_eps, n_steps)
                    sampled_actions = th.clamp(x1, 0.0, 1.0).reshape(M, N, K, self.n_actions)
                    entropy = sampled_actions.var(dim=2, unbiased=False).mean()
                else:
                    entropy = th.zeros((), device=mb_advantages.device)

                actor_loss = (
                    pg_loss
                    - entropy_coef * entropy
                    + lambda_trust * trust_loss
                    + w_b * brownian_loss
                    - w_g * gaussian_entropy
                )

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
                actor_stats["rho_s_mean"].append(mb_rho_s.mean().item())
                actor_stats["rho_s_std"].append(mb_rho_s.std(unbiased=False).item())
                actor_stats["clip_fraction"].append(
                    (
                        (mb_rho_s > 1 + self.args.eps_clip)
                        | (mb_rho_s < 1 - self.args.eps_clip)
                    ).float().mean().item()
                )
                actor_stats["actor_grad_norm"].append(grad_norm.item())
                actor_stats["entropy_mean"].append(entropy.item())
                actor_stats["trust_loss"].append(trust_loss.item())
                actor_stats["brownian_loss"].append(brownian_loss.item())
                actor_stats["gaussian_entropy"].append(gaussian_entropy.item())
                actor_stats["sigma_mean"].append(sigma.mean().item())
                actor_stats["delta_v_abs_mean"].append(delta_v_abs_mean)
                actor_stats["n_abs_mean"].append(n_abs_mean)

        # Refresh the trust-region reference to match the policy we just trained,
        # so next train() call's L_trust measures drift over the *next* round
        # only (not cumulative drift since the very first round).
        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps -       self.last_target_update_step)
            / self.args.target_update_interval_or_tau >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("action_mean", action_mean, t_env)
            self.logger.log_stat("action_std", action_std, t_env)
            self.logger.log_stat(
                "action_at_bound_fraction", action_at_bound_fraction, t_env
            )
            self.logger.log_stat(
                "action_overshoot_mean", action_overshoot_mean, t_env
            )
            self.logger.log_stat(
                "action_overshoot_at_bound_mean", action_overshoot_at_bound_mean, t_env
            )
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs",
                        "value_mean", "target_mean"]:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            self.logger.log_stat(
                "advantage_mean", self._mean_stat(actor_stats["advantage_mean"]), t_env
            )
            self.logger.log_stat(
                "advantage_std", self._mean_stat(actor_stats["advantage_std"]), t_env
            )
            self.logger.log_stat("pg_loss", self._mean_stat(actor_stats["pg_loss"]), t_env)
            # Only log mechanism-specific diagnostics while their coefficient
            # actually makes them nonzero -- otherwise they're flat-zero lines
            # cluttering wandb (lambda_trust/w_b/entropy_coef are all 0 while
            # use_policyflow_ratio is being tested in isolation).
            if lambda_trust > 0.0:
                self.logger.log_stat(
                    "trust_loss", self._mean_stat(actor_stats["trust_loss"]), t_env
                )
            if w_b > 0.0:
                self.logger.log_stat(
                    "brownian_loss", self._mean_stat(actor_stats["brownian_loss"]), t_env
                )
            if w_g > 0.0:
                self.logger.log_stat(
                    "gaussian_entropy", self._mean_stat(actor_stats["gaussian_entropy"]), t_env
                )
            if entropy_coef > 0.0:
                self.logger.log_stat(
                    "entropy_mean", self._mean_stat(actor_stats["entropy_mean"]), t_env
                )
            self.logger.log_stat(
                "sigma_mean", self._mean_stat(actor_stats["sigma_mean"]), t_env
            )
            if use_pf_ratio:
                self.logger.log_stat(
                    "delta_v_abs_mean", self._mean_stat(actor_stats["delta_v_abs_mean"]), t_env
                )
                self.logger.log_stat(
                    "n_abs_mean", self._mean_stat(actor_stats["n_abs_mean"]), t_env
                )
            self.logger.log_stat(
                "cfm_loss_mean", self._mean_stat(actor_stats["cfm_loss_mean"]), t_env
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
        # Unused (train() calls _compute_cfm_loss_for_time_indices instead),
        # kept in sync for consistency. See that method's comment on why this
        # interpolates toward action_raw (phi_hat), not the noisy action.
        action = batch["action_raw"][:, :-1].float()   # [B,T,N,n_actions]
        eps = batch["cfm_eps"][:, :-1]                # [B,T,N,cfm_n,n_actions]
        cfm_t = batch["cfm_t"][:, :-1]                # [B,T,N,cfm_n,1]

        act_exp = action.unsqueeze(3).expand_as(eps)
        x_t = (1 - cfm_t) * eps + cfm_t * act_exp
        h_exp = h_seq.unsqueeze(3).expand(-1, -1, -1, eps.size(3), -1)

        flat_h = h_exp.reshape(-1, h_exp.shape[-1])
        flat_x_t = x_t.reshape(-1, x_t.shape[-1])
        flat_t = cfm_t.reshape(-1, 1)

        v_pred = self.mac.agent.velocity(flat_h, flat_x_t, flat_t)
        v_pred = v_pred.reshape_as(eps)

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = act_exp - eps
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
        #
        # Interpolate toward action_raw (phi_hat, the deterministic flow
        # endpoint), NOT toward the noisy executed action -- PolicyFlow keeps
        # z, phi, and a=phi+n as three separate saved quantities and trains
        # the flow on z->phi only, so the terminal-noise sigma stays a
        # separate, orthogonal source of stochasticity instead of getting
        # baked into the flow's own regression target.
        action = batch["action_raw"][:, :-1].float().reshape(
            -1, self.n_agents, self.n_actions
        )
        eps = batch["cfm_eps"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, self.args.cfm_action_dim
        )
        cfm_t = batch["cfm_t"][:, :-1].reshape(
            -1, self.n_agents, self.args.cfm_n_samples, 1
        )
        h = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])

        mb_action = action[flat_time_indices]         # [M,N,A], = phi_hat
        mb_eps = eps[flat_time_indices]               # [M,N,cfm_n,A]
        mb_cfm_t = cfm_t[flat_time_indices]           # [M,N,cfm_n,1]
        mb_h = h[flat_time_indices]                   # [M,N,H]

        act_exp = mb_action.unsqueeze(2).expand_as(mb_eps)
        x_t = (1 - mb_cfm_t) * mb_eps + mb_cfm_t * act_exp
        h_exp = mb_h.unsqueeze(2).expand(-1, -1, mb_eps.size(2), -1)

        flat_h = h_exp.reshape(-1, h_exp.shape[-1])
        flat_x_t = x_t.reshape(-1, x_t.shape[-1])
        flat_t = mb_cfm_t.reshape(-1, 1)

        v_pred = self.mac.agent.velocity(flat_h, flat_x_t, flat_t)
        v_pred = v_pred.reshape_as(mb_eps)

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = act_exp - mb_eps
        elif cfm_target_type == "eps":
            target = mb_eps
        else:
            raise ValueError("cfm_target_type must be 'velocity' or 'eps'")

        cfm_loss = ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)
        # Also hand back v_pred and the exact (h, x_t, t) points it was evaluated
        # at, so a trust-region penalty can compare against v_old at the same
        # points without resampling.
        return cfm_loss, v_pred, flat_h, flat_x_t, flat_t

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
        self.old_mac.cuda()
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
