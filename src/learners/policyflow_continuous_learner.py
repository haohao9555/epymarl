import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class PolicyFlowContinuousLearner:
    """Continuous PolicyFlow learner (renamed from FPOContinuousLearner /
    fpo_continuous_learner.py). This is the full exact-Gaussian-ratio +
    Brownian/entropy-mechanism lineage developed in this repo; the plainer
    cfm-loss-diff-ratio + individual-actors lineage pulled in from GitHub
    lives separately as MAFPOContinuousLearner (mafpo_continuous_learner.py)
    and evolves independently -- the two are deliberately not merged.

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

        # Net-force diagnostic (pz-mpe-simple-spread specific): the 5-dim
        # continuous action isn't 5 independent controls -- MPE's own physics
        # (pettingzoo/mpe/_mpe_utils/simple_env.py) computes
        #   force_x = action[2] - action[1]
        #   force_y = action[4] - action[3]
        # (index 0 is a no-op slot). Two opposing dims pinned at the SAME
        # boundary (both 0 or both 1) cancel to zero net force -- functionally
        # identical to both sitting at 0.5 -- while one at each extreme is
        # full-throttle bang-bang control, not noise. Per-dimension
        # action_at_bound_fraction can't tell these apart; this can.
        if self.n_actions >= 5:
            force_x = actions_taken[..., 2] - actions_taken[..., 1]   # [B,T,N]
            force_y = actions_taken[..., 4] - actions_taken[..., 3]
            force_valid = mask.bool()                                # [B,T,N], same shape
            valid_fx = force_x[force_valid]
            valid_fy = force_y[force_valid]
            if valid_fx.numel() > 0:
                force_mag = th.sqrt(valid_fx ** 2 + valid_fy ** 2)
                force_x_mean = valid_fx.mean().item()
                force_y_mean = valid_fy.mean().item()
                force_magnitude_mean = force_mag.mean().item()
                # "full-throttle": net force magnitude near its max of sqrt(2)
                # (both axes maxed) or near 1 (one axis maxed) -- i.e. genuine
                # bang-bang, not two opposing dims cancelling near 0.
                force_near_zero_fraction = (force_mag < 0.1).float().mean().item()
                force_near_max_fraction = (force_mag > 0.9).float().mean().item()
            else:
                force_x_mean = force_y_mean = force_magnitude_mean = 0.0
                force_near_zero_fraction = force_near_max_fraction = 0.0

            # Per-agent breakdown: are different agents pushing in DIFFERENT
            # directions (healthy -- e.g. each heading to its own landmark),
            # or is everyone pinned to the same direction at once (a more
            # degenerate collapse, since obs_agent_id is supposed to let one
            # shared network act differently per agent)? force_x_mean/
            # force_y_mean above are pooled across agents and can't tell
            # these apart -- a high per-agent std here means agents diverge,
            # near-zero means they're all doing the same thing.
            per_agent_fx, per_agent_fy = [], []
            for i in range(self.n_agents):
                agent_mask = mask[..., i].bool()
                fx_i = force_x[..., i][agent_mask]
                fy_i = force_y[..., i][agent_mask]
                per_agent_fx.append(fx_i.mean().item() if fx_i.numel() > 0 else 0.0)
                per_agent_fy.append(fy_i.mean().item() if fy_i.numel() > 0 else 0.0)
            force_x_agent_std = float(th.tensor(per_agent_fx).std(unbiased=False))
            force_y_agent_std = float(th.tensor(per_agent_fy).std(unbiased=False))

            # Per-TIMESTEP agent-similarity, by DIRECTION CATEGORY not raw
            # distance: "agent1 moving down (or still), agent2 also moving
            # down (or still) at this same instant" should count as "same
            # action" regardless of exact force magnitude -- a continuous
            # distance threshold conflates "same direction, different speed"
            # with "different direction", which isn't what we want to catch.
            # 5 categories: still (force magnitude below still_thresh), or
            # the sign of whichever axis has bigger magnitude (+x/-x/+y/-y).
            # This directly answers "are agents collapsing to the same
            # behavior at the same time" (a shared-parameter-network failure
            # mode), as opposed to the earlier window-averaged force_x_agent*
            # stats, which can hide this behind time-varying averages that
            # happen to cancel out.
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
            # For each timestep, how many of the 5 categories are actually
            # occupied, and does the majority category cover >=2 / all N agents?
            cat_onehot = th.nn.functional.one_hot(category, num_classes=5)  # [B,T,N,5]
            cat_counts = cat_onehot.sum(dim=2)                        # [B,T,5]
            max_count = cat_counts.amax(dim=-1)                       # [B,T], size of largest same-category group
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

            # Distance-to-target-conditioned behavior (pz-mpe-simple-spread
            # specific): correlate force with how close the agent currently
            # is to its nearest landmark, to distinguish qualitatively
            # different reasons behind a high action_at_bound_fraction:
            #   far + full force        -> reasonable (racing toward target)
            #   near + low force        -> not a real collapse (settled)
            #   near + still full force -> imprecise control near the goal
            #   force flips direction every other step -> oscillation
            # obs layout (pettingzoo/mpe/simple_spread):
            #   [self_vel(2), self_pos(2), landmark_rel_pos(2*n_landmarks),
            #    other_agent_rel_pos(2*(N-1)), comm(...)]
            if "obs" in batch.scheme:
                n_landmarks = getattr(self.args, "n_landmarks", self.n_agents)
                obs = batch["obs"][:, :-1].float()                # [B,T,N,obs_dim]
                landmark_end = 4 + 2 * n_landmarks
                if obs.shape[-1] >= landmark_end:
                    landmark_rel = obs[..., 4:landmark_end].reshape(
                        *obs.shape[:-1], n_landmarks, 2
                    )
                    nearest_dist = landmark_rel.norm(dim=-1).min(dim=-1)[0]  # [B,T,N]

                    far_thresh = getattr(self.args, "landmark_far_thresh", 0.3)
                    near_thresh = getattr(self.args, "landmark_near_thresh", 0.15)
                    force_mag_full = th.sqrt(force_x ** 2 + force_y ** 2)  # [B,T,N]
                    is_far = nearest_dist > far_thresh
                    is_near = nearest_dist < near_thresh
                    is_fast = force_mag_full > 0.7
                    is_slow = force_mag_full < 0.3
                    step_agent_valid = mask.bool()                 # [B,T,N]
                    denom = step_agent_valid.float().sum()

                    def _frac(cond):
                        return (
                            (cond & step_agent_valid).float().sum() / denom
                        ).item() if denom > 0 else 0.0

                    far_fast_fraction = _frac(is_far & is_fast)
                    near_slow_fraction = _frac(is_near & is_slow)
                    near_still_fast_fraction = _frac(is_near & is_fast)

                    # Oscillation: force direction flips between consecutive
                    # timesteps while BOTH steps are still near-max magnitude
                    # -- actively slamming between extremes, not just varying
                    # speed smoothly.
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
            per_agent_fx = per_agent_fy = []
            agents_same_action_fraction = agents_all_same_action_fraction = 0.0
            far_fast_fraction = near_slow_fraction = 0.0
            near_still_fast_fraction = oscillation_fraction = 0.0

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

        # The above only checks "did the ODE (phi) leave the road" -- it says
        # nothing about whether phi was fine but got knocked out of [0,1] by
        # the added terminal noise n. What actually gets clamped is phi+n,
        # not phi alone, so measure that overshoot separately using the real
        # stored noise (batch["action_noise"]). Comparing this against the
        # phi-only overshoot above tells us how much of the boundary-pinning
        # is the flow's own doing vs. purely noise-driven.
        if "action_raw" in batch.scheme and "action_noise" in batch.scheme:
            action_noise = batch["action_noise"][:, :-1].float()      # [B,T,N,A]
            noisy_raw = action_raw + action_noise                     # phi + n, pre-clamp
            valid_noisy_raw = noisy_raw[action_valid]
            noisy_overshoot = (
                th.clamp(-valid_noisy_raw, min=0) + th.clamp(valid_noisy_raw - 1, min=0)
            )
            action_noisy_overshoot_mean = (
                noisy_overshoot.mean().item() if noisy_overshoot.numel() > 0 else 0.0
            )
            if valid_actions.numel() > 0 and at_bound.any():
                action_noisy_overshoot_at_bound_mean = noisy_overshoot[at_bound].mean().item()
            else:
                action_noisy_overshoot_at_bound_mean = 0.0
            # Of the samples that ended up at the boundary, how many had phi
            # itself already inside [0,1] -- i.e. were only pushed out by n?
            phi_valid_raw = valid_raw  # from the block above, phi alone
            phi_was_inside = (phi_valid_raw >= 0) & (phi_valid_raw <= 1)
            if valid_actions.numel() > 0 and at_bound.any():
                action_at_bound_from_noise_fraction = (
                    (phi_was_inside & at_bound).float().sum().item()
                    / at_bound.float().sum().item()
                )
            else:
                action_at_bound_from_noise_fraction = 0.0
        else:
            action_noisy_overshoot_mean = action_noisy_overshoot_at_bound_mean = 0.0
            action_at_bound_from_noise_fraction = 0.0

        # phi-only (noise-free) oscillation: identical flip definition to
        # oscillation_fraction above, but computed on action_raw (the flow's
        # own integration endpoint, pre-noise, pre-clamp) instead of the
        # executed action. Directly separates "is the learned velocity field
        # itself reversing direction" from "is this just independently
        # resampled sigma noise creating an apparent flip" -- compare this
        # against oscillation_fraction side by side rather than assuming
        # either explanation.
        if self.n_actions >= 5 and "action_raw" in batch.scheme:
            force_x_raw = action_raw[..., 2] - action_raw[..., 1]     # [B,T,N]
            force_y_raw = action_raw[..., 4] - action_raw[..., 3]
            fx_prev_r, fx_curr_r = force_x_raw[:, :-1], force_x_raw[:, 1:]
            fy_prev_r, fy_curr_r = force_y_raw[:, :-1], force_y_raw[:, 1:]
            mag_prev_r = th.sqrt(fx_prev_r ** 2 + fy_prev_r ** 2)
            mag_curr_r = th.sqrt(fx_curr_r ** 2 + fy_curr_r ** 2)
            dot_r = fx_prev_r * fx_curr_r + fy_prev_r * fy_curr_r
            pair_valid_r = mask[:, :-1].bool() & mask[:, 1:].bool()
            is_flip_r = (mag_prev_r > 0.5) & (mag_curr_r > 0.5) & (dot_r < 0)
            denom_pairs_r = pair_valid_r.float().sum()
            oscillation_fraction_phi_only = (
                (is_flip_r & pair_valid_r).float().sum() / denom_pairs_r
            ).item() if denom_pairs_r > 0 else 0.0
        else:
            oscillation_fraction_phi_only = 0.0

        rho_clip = getattr(self.args, "cfm_rho_clip", 3.0)
        entropy_coef = getattr(self.args, "entropy_coef", 0.0)
        entropy_n_samples = getattr(self.args, "entropy_n_samples", 4)
        w_b = getattr(self.args, "w_b", 0.0)
        w_g = getattr(self.args, "w_g", 0.0)
        use_pf_ratio = getattr(self.args, "use_policyflow_ratio", False)
        # initial_cfm_loss only exists in the buffer scheme when NOT using
        # the PolicyFlow ratio (see run.py's scheme setup) -- the old
        # cfm-loss-diff ratio's else-branch below is the only reader.
        initial_cfm_loss = (
            None if use_pf_ratio else batch["initial_cfm_loss"][:, :-1]
        )                                                        # [B,T,N,cfm_n,1]
        # trust_loss (lambda_trust) removed: it was permanently disabled
        # (lambda_trust=0.0) once the Brownian regularizer took over the same
        # v_new-vs-v_old role, and its computation was fully short-circuited
        # to a zero tensor -- dead code, see git history for the removed
        # mechanism (a plain ||v_new-v_old||^2 penalty, superseded by w_b).
        need_v_old = w_b > 0.0 or use_pf_ratio

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

        # Hidden states from old_mac's OWN encoder, computed once since
        # old_mac is frozen for the whole train() call. Every v_old(...) call
        # below must use this, never h_seq (which is rebuilt from self.mac's
        # encoder every minibatch as mac updates) -- the encoder (fc1/GRU)
        # does most of the representational work, so feeding old_mac's
        # velocity head the *new* encoder's hidden state made v_old and v_new
        # share almost everything except the small output-layer difference,
        # artificially shrinking delta_v more and more as mac's encoder
        # drifted from old_mac's -- this was found to be the dominant cause
        # of delta_v_abs_mean collapsing toward 0 as action collapse
        # deepened, on top of the cfm_eps-vs-real-z and pre-averaging bugs
        # fixed earlier.
        old_h_seq = None
        if need_v_old:
            with th.no_grad():
                old_h_seq = self._build_old_actor_hidden_sequence(batch)

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

                # Real-rollout-path (z, action_raw) interpolated at a FIXED,
                # deterministic t-grid -- shared by both the PolicyFlow
                # ratio's delta_v and the Brownian regularizer's eta_t below.
                # Previously these used two DIFFERENT probe paths: delta_v
                # was fixed to use the real z after finding cfm_eps collapsed
                # its corrective signal toward 0 (action_at_bound_fraction
                # climbing to 90%+ while delta_v_abs_mean shrank toward 0 in
                # a 2M-step run), but brownian_loss was never migrated and
                # kept evaluating v_new/v_old at cfm_eps-based neighbourhood
                # points -- the same "unrelated to what actually happened"
                # problem, just undiscovered in a second place. Unified here.
                #
                # t is now a fixed linspace instead of fresh random cfm_t
                # samples: for a Monte Carlo/quadrature estimate of an
                # integral over t in [0,1], an even grid has lower variance
                # than random draws of the same count, and needs no
                # rollout-time sampling or buffer storage at all (cfm_eps and
                # cfm_t are no longer read in this branch).
                if need_v_old:
                    cfm_n = getattr(self.args, "cfm_n_samples", 10)
                    z_flat = batch["z"][:, :-1].float().reshape(
                        -1, self.n_agents, self.n_actions
                    )
                    phi_flat = batch["action_raw"][:, :-1].float().reshape(
                        -1, self.n_agents, self.n_actions
                    )
                    mb_z = z_flat[mb_time_idx]                      # [M,N,A]
                    mb_phi = phi_flat[mb_time_idx]                  # [M,N,A]

                    t_vals = th.linspace(0.0, 1.0, cfm_n + 1, device=mb_z.device)[:-1]
                    t_grid = t_vals.reshape(1, 1, cfm_n, 1).expand(
                        mb_z.shape[0], mb_z.shape[1], -1, -1
                    )                                                # [M,N,cfm_n,1]

                    z_exp = mb_z.unsqueeze(2).expand(-1, -1, cfm_n, -1)
                    phi_exp = mb_phi.unsqueeze(2).expand(-1, -1, cfm_n, -1)
                    x_t_real = (1 - t_grid) * z_exp + t_grid * phi_exp   # [M,N,cfm_n,A]

                    h_flat = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])
                    mb_h = h_flat[mb_time_idx]                      # [M,N,H]
                    h_exp_real = mb_h.unsqueeze(2).expand(-1, -1, cfm_n, -1)
                    flat_h_real = h_exp_real.reshape(-1, h_exp_real.shape[-1])
                    flat_x_t_real = x_t_real.reshape(-1, x_t_real.shape[-1])
                    flat_t_real = t_grid.reshape(-1, 1)

                    v_new_real = self.mac.agent.velocity(
                        flat_h_real, flat_x_t_real, flat_t_real
                    ).reshape_as(x_t_real)

                    # v_old must go through old_mac's OWN encoder output
                    # (old_h_seq), never self.mac's h_seq -- see old_h_seq
                    # comment above the epoch loop.
                    old_h_flat = old_h_seq.reshape(-1, self.n_agents, old_h_seq.shape[-1])
                    mb_old_h = old_h_flat[mb_time_idx]              # [M,N,H]
                    old_h_exp_real = mb_old_h.unsqueeze(2).expand(-1, -1, cfm_n, -1)
                    old_flat_h_real = old_h_exp_real.reshape(-1, old_h_exp_real.shape[-1])
                    with th.no_grad():
                        v_old_real = self.old_mac.agent.velocity(
                            old_flat_h_real, flat_x_t_real, flat_t_real
                        ).reshape_as(x_t_real)

                # Brownian regularizer (PolicyFlow, arXiv:2602.01156, Eq.15):
                # eta_t = (1-t)*v_new - (x_t - t*v_old). Under the rectified-flow
                # score-velocity relationship, (x_t - t*v_old) = -(1-t)*score_old,
                # so eta_t/(1-t) = v_new + score_old — penalizing ||eta_t||^2 pulls
                # the *current* velocity field toward the entropy-increasing
                # (score-corrected / "Brownian") version of the reference field,
                # instead of letting it collapse into a purely deterministic map
                # that concentrates mass at the action boundary.
                if w_b > 0.0:
                    eta_t = (1 - t_grid) * v_new_real - (
                        x_t_real - t_grid * v_old_real
                    )
                    # ||eta_t||^2 is a squared L2 norm -- SUM over the action
                    # dimension (matching the paper's Eq.15 and the ratio's
                    # own log_ratio_per_dim.sum(dim=-1) above), then mean
                    # over the M,N,cfm_n Monte Carlo samples. Using a flat
                    # .mean() here previously averaged over the action dim
                    # too, silently shrinking brownian_loss by a constant
                    # factor of n_actions relative to the paper's formula.
                    brownian_loss = (eta_t ** 2).sum(dim=-1).mean()
                else:
                    brownian_loss = th.zeros((), device=h_seq.device)

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
                    gaussian_entropy = th.zeros((), device=h_seq.device)

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
                    n_flat = batch["action_noise"][:, :-1].float().reshape(
                        -1, self.n_agents, self.n_actions
                    )
                    n_noise = n_flat[mb_time_idx]                   # [M,N,A]

                    # Do NOT average delta_v over the cfm_n t-samples before
                    # building the ratio: log_ratio is quadratic in delta_v, so
                    # E_t[log_ratio(delta_v_t)] != log_ratio(E_t[delta_v_t]) --
                    # averaging first lets velocity shifts of opposite sign at
                    # different t cancel out, silently shrinking the ratio's
                    # corrective signal. Instead: compute a separate
                    # ratio/surrogate at each of the cfm_n t-grid points, and
                    # only average the final per-sample PPO loss -- the
                    # correct Monte Carlo estimate of E_p(t)[surrogate].
                    delta_v = v_new_real - v_old_real                # [M,N,cfm_n,A]
                    n_noise_exp = n_noise.unsqueeze(2).expand_as(delta_v)  # same n for every t

                    sigma_new = self.mac.agent.sigma()              # [A], grad-tracked
                    with th.no_grad():
                        sigma_old = self.old_mac.agent.sigma()      # [A], frozen reference

                    log_ratio_per_dim = -0.5 * (
                        (n_noise_exp - delta_v) ** 2 / sigma_new ** 2
                        - n_noise_exp ** 2 / sigma_old ** 2
                        + th.log(sigma_new ** 2 / sigma_old ** 2)
                    )
                    log_ratio = log_ratio_per_dim.sum(dim=-1)       # [M,N,cfm_n]
                    mb_rho_s_t = th.exp(th.clamp(log_ratio, -rho_clip, rho_clip))

                    mb_advantages_3d = mb_advantages_2d.unsqueeze(2).expand_as(mb_rho_s_t)
                    surr1 = mb_rho_s_t * mb_advantages_3d
                    surr2 = th.clamp(
                        mb_rho_s_t, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                    ) * mb_advantages_3d
                    pg_loss = -th.min(surr1, surr2).mean()          # mean over M,N,cfm_n

                    mb_rho_s = mb_rho_s_t.mean(dim=2)               # [M,N], for logging only
                    delta_v_abs_mean = delta_v.detach().abs().mean().item()
                    n_abs_mean = n_noise.detach().abs().mean().item()
                    # cfm_loss is the OLD ratio mechanism's own quantity and is
                    # never computed on this branch anymore (see class
                    # docstring) -- logged as 0.0, not a meaningful value here.
                    mb_cfm_loss_mean = 0.0
                else:
                    # Old cfm-loss-diff ratio: this mechanism's own design
                    # needs the cfm_eps-based neighbourhood CFM loss (a
                    # separate, independently-sampled probe path -- unrelated
                    # to the real-z path built above for brownian_loss).
                    mb_cfm_loss, _, _, _, _ = self._compute_cfm_loss_for_time_indices(
                        batch, h_seq, mb_time_idx
                    )
                    mb_initial_cfm_loss = initial_cfm_loss.reshape(
                        -1, self.n_agents, initial_cfm_loss.size(-2), initial_cfm_loss.size(-1)
                    )[mb_time_idx]
                    diff = mb_initial_cfm_loss - mb_cfm_loss              # [M,N,cfm_n,1]
                    diff_mean = diff.mean(dim=(-2, -1))                   # [M,N]
                    mb_rho_s = th.exp(th.clamp(diff_mean, -rho_clip, rho_clip))  # [M,N]
                    delta_v_abs_mean = n_abs_mean = 0.0
                    mb_cfm_loss_mean = mb_cfm_loss.mean(dim=(-2, -1)).mean().item()

                    mb_rho_s_flat = mb_rho_s.reshape(-1)
                    mb_advantages = mb_advantages_2d.reshape(-1)
                    surr1 = mb_rho_s_flat * mb_advantages
                    surr2 = th.clamp(
                        mb_rho_s_flat, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                    ) * mb_advantages
                    pg_loss = -th.min(surr1, surr2).mean()

                mb_advantages = mb_advantages_2d.reshape(-1)

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
                    ent_eps = th.randn(M * N * K, self.n_actions, device=mb_h.device)
                    n_steps = getattr(self.args, "cfm_rollout_steps", 10)
                    x1 = self.mac.agent.integrate(h_rep, ent_eps, n_steps)
                    sampled_actions = th.sigmoid(x1).reshape(M, N, K, self.n_actions)
                    entropy = sampled_actions.var(dim=2, unbiased=False).mean()
                else:
                    entropy = th.zeros((), device=mb_advantages.device)

                actor_loss = (
                    pg_loss
                    - entropy_coef * entropy
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
                actor_stats["cfm_loss_mean"].append(mb_cfm_loss_mean)
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
            self.logger.log_stat("force_x_mean", force_x_mean, t_env)
            self.logger.log_stat("force_y_mean", force_y_mean, t_env)
            self.logger.log_stat(
                "force_magnitude_mean", force_magnitude_mean, t_env
            )
            self.logger.log_stat(
                "force_near_zero_fraction", force_near_zero_fraction, t_env
            )
            self.logger.log_stat(
                "force_near_max_fraction", force_near_max_fraction, t_env
            )
            self.logger.log_stat("force_x_agent_std", force_x_agent_std, t_env)
            self.logger.log_stat("force_y_agent_std", force_y_agent_std, t_env)
            for i, (fx_i, fy_i) in enumerate(zip(per_agent_fx, per_agent_fy)):
                self.logger.log_stat(f"force_x_agent{i}_mean", fx_i, t_env)
                self.logger.log_stat(f"force_y_agent{i}_mean", fy_i, t_env)
            self.logger.log_stat(
                "agents_same_action_fraction", agents_same_action_fraction, t_env
            )
            self.logger.log_stat(
                "agents_all_same_action_fraction",
                agents_all_same_action_fraction,
                t_env,
            )
            self.logger.log_stat("far_fast_fraction", far_fast_fraction, t_env)
            self.logger.log_stat("near_slow_fraction", near_slow_fraction, t_env)
            self.logger.log_stat(
                "near_still_fast_fraction", near_still_fast_fraction, t_env
            )
            self.logger.log_stat("oscillation_fraction", oscillation_fraction, t_env)
            self.logger.log_stat(
                "oscillation_fraction_phi_only", oscillation_fraction_phi_only, t_env
            )
            self.logger.log_stat(
                "action_overshoot_mean", action_overshoot_mean, t_env
            )
            self.logger.log_stat(
                "action_overshoot_at_bound_mean", action_overshoot_at_bound_mean, t_env
            )
            self.logger.log_stat(
                "action_noisy_overshoot_mean", action_noisy_overshoot_mean, t_env
            )
            self.logger.log_stat(
                "action_noisy_overshoot_at_bound_mean",
                action_noisy_overshoot_at_bound_mean,
                t_env,
            )
            self.logger.log_stat(
                "action_at_bound_from_noise_fraction",
                action_at_bound_from_noise_fraction,
                t_env,
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
            # cluttering wandb (w_b/entropy_coef are 0 while
            # use_policyflow_ratio is being tested in isolation).
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

    def _build_old_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        """Same as _build_actor_hidden_sequence but through old_mac's own
        (frozen) encoder -- required for any v_old(...) call to be a genuine
        "what would the old policy have output" evaluation."""
        h_list = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = self.old_mac.forward(batch, t=t)
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
