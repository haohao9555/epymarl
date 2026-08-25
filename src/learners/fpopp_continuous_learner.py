import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPOPPContinuousLearner:
    """Continuous "FPO++" learner. Forked from FPOContinuousLearner
    (fpo_continuous_learner.py) -- same MAC/actor/critic
    (fpo_mac/fpo_actor/fpo_critic, unchanged, reused as-is), only the
    ratio/surrogate math in this train() differs. Two changes on top of
    plain MAFPO:

    1. No pre-averaging over the cfm_n neighbourhood samples before
       exponentiating. MAFPO computes one ratio per transition by averaging
       the cfm_n loss differences first:
           rho = exp(mean_i(L_old^(i) - L_new^(i)))
       Since exp is nonlinear, mean_i(exp(diff_i)) != exp(mean_i(diff_i)) --
       averaging the diffs first lets loss-difference swings of opposite
       sign at different sample points cancel out before the ratio ever sees
       them, discarding real dispersion information. Instead, each of the
       cfm_n sampled (tau_i, eps_i) points gets its own ratio:
           rho_i = exp(clamp(L_old^(i) - L_new^(i), -rho_clip, rho_clip)),  i=1..cfm_n
       and the PPO/SPO surrogate below is evaluated at each rho_i
       separately; only the final per-point surrogate values are averaged
       (over the minibatch AND cfm_n together), not the ratios or the raw
       loss differences.

    2. For A<0 samples, the standard PPO clip surrogate
       min(rho*A, clip(rho,1-eps,1+eps)*A) is replaced by a smooth SPO-style
       objective:
           psi_SPO(rho, A) = rho*A - (|A| / (2*eps_clip)) * (rho-1)^2
       PPO's hard clip gives exactly zero gradient once rho drops below
       1-eps_clip for A<0 -- this is the "self-extinguishing" failure mode
       MAFPO's own neg_A_active_rho_s_mean/neg_A_clip_fraction diagnostics
       were built to detect. SPO's gradient equals the unclipped policy
       gradient A at rho=1 (no discontinuity there), shrinks smoothly (not
       abruptly) as rho crosses the old clip boundary, and past that
       boundary turns into a restoring force pulling rho back toward it --
       never a flat zero-gradient region. A>=0 samples still use the
       standard PPO clip surrogate unchanged.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        # ------修复：individual actors 的独立更新，之前只是"看起来独立" ----------
        # mac.parameters() 在 individual_agents=True 时用 itertools.chain 把 3 个
        # agent 的参数拼成一个扁平列表；旧代码把这一整个列表喂给同一个 Adam +
        # 同一次 clip_grad_norm_。Adam 的一阶/二阶矩本身是逐参数的，共用一个
        # optimiser 实例不会引入耦合；但 clip_grad_norm_ 是按**全局** L2 范数联
        # 合裁剪的——某个 agent 这一步梯度偏大，会把另外两个本来正常的 agent 的
        # 梯度也按同一个缩放比例摁下去，等于通过 clipping 又把三个 agent 重新
        # 耦合在一起，跟"每个 agent 独立更新"的设计初衷矛盾。individual_agents
        # =True 时改成每个 agent 自己的 optimiser + 自己的 clip_grad_norm_，范
        # 数只在各自参数里算，互不影响；共享网络（individual_agents=False）时
        # 行为不变。
        if getattr(mac, "individual_agents", False):
            self.actor_params_per_agent = [list(agent.parameters()) for agent in mac.agents]
            self.actor_optimisers = [
                Adam(params=p, lr=args.lr) for p in self.actor_params_per_agent
            ]
            self.actor_params = None
            self.actor_optimiser = None
        else:
            self.actor_params_per_agent = None
            self.actor_optimisers = None
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

        # ------新增：noise-domination 诊断 ----------
        # individual actors (0313c24) 修复的是"共享网络导致跨智能体 Delta 混
        # 入"——症状是不同 agent 在同一时刻的动作被同一份共享参数/共享噪声主
        # 导，看起来彼此雷同而不是各自对自己的观测做出反应。这里直接量化这个
        # 症状，而不是间接猜：
        #   action_at_bound_fraction: 动作贴 [0,1] 边界的比例（噪声主导/塌缩
        #       的经典表现，hard clamp 会把越界的噪声钉在边界上）。
        #   agents_same_action_fraction / agents_all_same_action_fraction:
        #       同一时刻 >=2 / 全部 N 个 agent 的动作方向类别相同的比例（按
        #       pz-mpe-simple-spread 的 5 维动作物理意义分类，跟
        #       policyflow_continuous_learner.py 里的同名诊断同一套逻辑，直
        #       接对照两条线是否有区别）。如果 individual actors 真的解决了
        #       Delta 混入，这两个比例应该明显低于 shared-network 版本。
        actions_taken = batch["actions"][:, :-1].float()          # [B,T,N,A]
        action_valid = mask.unsqueeze(-1).expand_as(actions_taken).bool()
        valid_actions = actions_taken[action_valid]
        bound_eps = 0.02
        if valid_actions.numel() > 0:
            at_bound = (valid_actions < bound_eps) | (valid_actions > 1 - bound_eps)
            action_at_bound_fraction = at_bound.float().mean().item()
        else:
            action_at_bound_fraction = 0.0

        if self.n_actions >= 5:
            force_x = actions_taken[..., 2] - actions_taken[..., 1]   # [B,T,N]
            force_y = actions_taken[..., 4] - actions_taken[..., 3]
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
        else:
            agents_same_action_fraction = agents_all_same_action_fraction = 0.0
        # -----------------------------------------------------------------------------

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
            "rho_s_mean": [],
            "rho_s_std": [],
            "clip_fraction": [],
            "actor_grad_norm": [],
            #------新增：A<0 且未被 eps_clip 截断的样本上，诊断 rho_s * residual 是否随训练衰减----------
            # 理论上：r=exp(L_old-L_new) 随残差指数衰减，∇L_new 随残差线性增长，
            # r * ||∇L_new|| 应当 -> 0（"自我熄灭"）。若这个量不降反升，说明梯度没有
            # 随 rho_s 正常衰减，权重实际上被钉成了近似常数。
            #-----------------------------
            "neg_A_active_rho_s_mean": [],
            "neg_A_active_weighted_residual": [],
            "neg_A_clip_fraction": [],
            "jac_diag_sat_diff_abs_mean": [],
            "jac_diag_unsat_diff_abs_mean": [],
            "jac_diag_sat_fraction": [],
        }
        critic_train_stats = {
            k: [] for k in ["critic_loss", "critic_grad_norm", "td_error_abs",
                            "target_mean", "value_mean"]
        }

        # ------修复：advantage 只在 rollout 后算一次，整个 actor update phase 固定
        # 不变 ----------
        # 这里用的 target_vals 来自 self.target_critic，这整个 train() 调用期间
        # 根本不会变（soft/hard update 要等所有 epoch 跑完之后才发生），所以旧代
        # 码在每个 epoch 里重新跑一遍 GAE 算出来的 advantages 其实每次都是同一个
        # 值——不是 bug（数值没错），但白白重复计算，而且原来的注释("we train it
        # once per epoch, then freeze the advantages...")容易让人误以为 advantage
        # 会随 epoch 刷新。现在把它提到 epoch 循环外面只算一次；critic 每个 epoch
        # 依然做一次梯度更新，但用的是这同一份固定 target_returns，不再重新跑 GAE。
        advantages, target_returns = self._compute_advantages_and_targets(
            self.target_critic, batch, rewards, critic_mask
        )
        advantages = advantages.detach()
        target_returns = target_returns.detach()

        for _ in range(self.args.epochs):
            epoch_critic_stats = self._critic_gradient_step(
                self.critic, batch, target_returns, critic_mask
            )
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

                advantages_by_time = advantages.reshape(-1, self.n_agents)
                mb_advantages_2d = advantages_by_time[mb_time_idx]   # [M,N]

                # ------新增：latent-vs-action 失配诊断 + 可选雅可比修正 ----------
                # u = action_raw（无界 latent，CFM 回归的插值目标），a = sigmoid(u)
                # 才是环境真正执行的动作。diff/ratio 是在 u 空间的 CFM 回归 loss 差
                # 上算出来的代理量，从来没有对 sigmoid 的雅可比 sigma'(u)=a(1-a) 做
                # 过任何处理——如果这是一个严格的密度比，同一个 u 上算新旧密度之比
                # 时雅可比会精确抵消（policyflow_actor.py 的 exact ratio 就是这么论
                # 证的），但 cfm-loss-diff 根本不是一个显式密度比，没有这层保护。
                # sigmoid 饱和区（a 接近 0/1）里 u 的一次大幅移动只对应 a 的一点点
                # 移动——ratio 却完全按 u 空间的尺度在报告"策略变了多少"，把这个被
                # 夸大的信号乘进 A<0 时的 SPO 惩罚 / A>=0 时的 PPO clip，可能是残余
                # 不稳定的一个额外来源，跟"共享网络梯度冲突"那条线正交。
                #
                # w = 4*a*(1-a) ∈ [0,1]：u=0（未饱和，a=0.5）时 w=1，不打折；越往
                # 饱和区走 w→0。mb_action 用已经存好的执行动作，不需要重新算 sigmoid。
                mb_action = actions_taken.reshape(-1, self.n_agents, self.n_actions)[
                    mb_time_idx
                ]                                                    # [M,N,A]
                jac_w = (4.0 * mb_action * (1.0 - mb_action)).mean(dim=-1)  # [M,N], in [0,1]

                # ------FPO++ 改动 1：不对 cfm_n 个采样点先取平均，直接逐点算 ratio ----------
                # 见类 docstring。diff/rho 都保留 cfm_n 维，不在这里 reduce。
                diff = (mb_initial_cfm_loss - mb_cfm_loss).squeeze(-1)   # [M,N,cfm_n]

                # 诊断：饱和区 (jac_w 小) vs 非饱和区 (jac_w 大) 样本各自的 |diff| 均值，
                # 无条件记录，不受下面的开关影响。
                with th.no_grad():
                    sat_mask = jac_w < 0.5          # a 落在 [0.146, 0.854] 之外算"饱和"
                    diff_flat_by_transition = diff.abs().mean(dim=-1)   # [M,N]
                    if sat_mask.any():
                        jac_diag_sat_diff_abs_mean = diff_flat_by_transition[sat_mask].mean().item()
                    else:
                        jac_diag_sat_diff_abs_mean = 0.0
                    if (~sat_mask).any():
                        jac_diag_unsat_diff_abs_mean = diff_flat_by_transition[~sat_mask].mean().item()
                    else:
                        jac_diag_unsat_diff_abs_mean = 0.0
                    jac_diag_sat_fraction = sat_mask.float().mean().item()

                if getattr(self.args, "jacobian_ratio_correction", False):
                    diff = diff * jac_w.unsqueeze(-1)   # [M,N,cfm_n], broadcast over cfm_n
                mb_rho_s_pt = th.exp(th.clamp(diff, -rho_clip, rho_clip))  # [M,N,cfm_n]
                # ------残差幅度代理，用于诊断 exp(-x)*x -> 0 是否真的在发生（同样逐点，
                # 不做 cfm_n 平均）----------
                # sqrt(L_new) 正比于 CFM 回归残差 ||v_pred - target|| 的均方根，
                # 是 ||∇_v_pred L_new|| 的廉价代理（无需对每个样本单独反传求参数梯度）。
                # -----------------------------------------------------------------------------
                cfm_residual_rms_pt = mb_cfm_loss.squeeze(-1).sqrt()     # [M,N,cfm_n]

                cfm_n = mb_rho_s_pt.shape[-1]
                mb_advantages_pt = mb_advantages_2d.unsqueeze(-1).expand(-1, -1, cfm_n)  # [M,N,cfm_n]

                mb_rho_s = mb_rho_s_pt.reshape(-1)
                mb_advantages = mb_advantages_pt.reshape(-1)
                cfm_residual_rms = cfm_residual_rms_pt.reshape(-1)

                # ------FPO++ 改动 2：A<0 时用 SPO 式平滑目标替换 PPO 硬 clip ----------
                # psi_SPO(rho, A) = rho*A - (|A| / (2*eps_clip)) * (rho-1)^2
                # 见类 docstring。A>=0 仍用标准 PPO clip surrogate。
                neg_mask = mb_advantages < 0
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(
                    mb_rho_s, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                ) * mb_advantages
                ppo_surr = th.min(surr1, surr2)
                spo_surr = (
                    mb_rho_s * mb_advantages
                    - (mb_advantages.abs() / (2 * self.args.eps_clip))
                    * (mb_rho_s - 1) ** 2
                )
                surr = th.where(neg_mask, spo_surr, ppo_surr)
                # 最终只在这一步对 (minibatch, cfm_n) 一起取平均 -- ratio 和逐点 surrogate
                # 本身全程没有被平均过。
                pg_loss = -surr.mean()
                actor_loss = pg_loss

                if self.actor_optimisers is not None:
                    for opt in self.actor_optimisers:
                        opt.zero_grad()
                    actor_loss.backward()
                    grad_norm = th.stack([
                        th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
                        for params in self.actor_params_per_agent
                    ]).mean()   # logged as the mean of 3 independently-computed norms
                    for opt in self.actor_optimisers:
                        opt.step()
                else:
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
                actor_stats["cfm_loss_mean"].append(mb_cfm_loss.mean().item())
                actor_stats["rho_s_mean"].append(mb_rho_s.mean().item())
                actor_stats["rho_s_std"].append(mb_rho_s.std(unbiased=False).item())
                actor_stats["clip_fraction"].append(
                    (
                        (mb_rho_s > 1 + self.args.eps_clip)
                        | (mb_rho_s < 1 - self.args.eps_clip)
                    ).float().mean().item()
                )
                actor_stats["actor_grad_norm"].append(grad_norm.item())

                # ------A<0 诊断，检验"残差变大->有效权重是否随之衰减"（沿用 MAFPO 的
                # 诊断定义，现在作用在逐点 ratio 上，样本量是 M*N*cfm_n 而不是 M*N，
                # 粒度更细；unclipped_mask 只是拿旧的 clip 边界当参照系诊断用，SPO
                # 分支本身并不真的做这个 clip）----------
                with th.no_grad():
                    unclipped_mask = (
                        (mb_rho_s >= 1 - self.args.eps_clip)
                        & (mb_rho_s <= 1 + self.args.eps_clip)
                    )
                    active_neg = neg_mask & unclipped_mask   # A<0 且梯度未被 clip 置零的样本
                    if active_neg.any():
                        actor_stats["neg_A_active_rho_s_mean"].append(
                            mb_rho_s[active_neg].mean().item()
                        )
                        actor_stats["neg_A_active_weighted_residual"].append(
                            (mb_rho_s[active_neg] * cfm_residual_rms[active_neg])
                            .mean()
                            .item()
                        )
                    if neg_mask.any():
                        actor_stats["neg_A_clip_fraction"].append(
                            (neg_mask & ~unclipped_mask).float().sum().item()
                            / neg_mask.float().sum().item()
                        )
                    actor_stats["jac_diag_sat_diff_abs_mean"].append(jac_diag_sat_diff_abs_mean)
                    actor_stats["jac_diag_unsat_diff_abs_mean"].append(jac_diag_unsat_diff_abs_mean)
                    actor_stats["jac_diag_sat_fraction"].append(jac_diag_sat_fraction)
                # -----------------------------------------------------------------------------

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
            self.logger.log_stat("action_at_bound_fraction", action_at_bound_fraction, t_env)
            self.logger.log_stat(
                "agents_same_action_fraction", agents_same_action_fraction, t_env
            )
            self.logger.log_stat(
                "agents_all_same_action_fraction", agents_all_same_action_fraction, t_env
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
            # ------新增：A<0 侧的"权重是否随残差衰减"诊断 ----------
            self.logger.log_stat(
                "neg_A_active_rho_s_mean",
                self._mean_stat(actor_stats["neg_A_active_rho_s_mean"]),
                t_env,
            )
            self.logger.log_stat(
                "neg_A_active_weighted_residual",
                self._mean_stat(actor_stats["neg_A_active_weighted_residual"]),
                t_env,
            )
            self.logger.log_stat(
                "neg_A_clip_fraction",
                self._mean_stat(actor_stats["neg_A_clip_fraction"]),
                t_env,
            )
            # ------新增：u(latent)-vs-a(action) 失配诊断，见上面 jac_w 的注释 ----------
            self.logger.log_stat(
                "jac_diag_sat_diff_abs_mean",
                self._mean_stat(actor_stats["jac_diag_sat_diff_abs_mean"]),
                t_env,
            )
            self.logger.log_stat(
                "jac_diag_unsat_diff_abs_mean",
                self._mean_stat(actor_stats["jac_diag_unsat_diff_abs_mean"]),
                t_env,
            )
            self.logger.log_stat(
                "jac_diag_sat_fraction",
                self._mean_stat(actor_stats["jac_diag_sat_fraction"]),
                t_env,
            )
            # -----------------------------------------------------------------------------
            self.log_stats_t = t_env

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        h_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = self.mac.forward(batch, t=t)
            h_list.append(h)
        return th.stack(h_list, dim=1)                # [B,T,N,hidden_dim]

    def _compute_cfm_loss(self, batch: EpisodeBatch, h_seq: th.Tensor) -> th.Tensor:
        # action_raw (sigmoid 之前的无界积分终点), 不是执行动作 -- 见
        # fpo_actor.py/fpo_mac.py 顶部注释。
        action = batch["action_raw"][:, :-1].float()   # [B,T,N,n_actions]
        eps = batch["cfm_eps"][:, :-1]                # [B,T,N,cfm_n,n_actions]
        cfm_t = batch["cfm_t"][:, :-1]                # [B,T,N,cfm_n,1]

        act_exp = action.unsqueeze(3).expand_as(eps)
        x_t = (1 - cfm_t) * eps + cfm_t * act_exp
        h_exp = h_seq.unsqueeze(3).expand(-1, -1, -1, eps.size(3), -1)

        v_pred = self.mac.velocity(h_exp, x_t, cfm_t)

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
        # action_raw (sigmoid 之前的无界积分终点), 不是执行动作 -- interpolating
        # toward the post-sigmoid action would reintroduce the exact boundary-
        # squash regression problem sigmoid was meant to fix (see
        # fpo_actor.py's top-of-file comment).
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

        mb_action = action[flat_time_indices]         # [M,N,A]
        mb_eps = eps[flat_time_indices]               # [M,N,cfm_n,A]
        mb_cfm_t = cfm_t[flat_time_indices]           # [M,N,cfm_n,1]
        mb_h = h[flat_time_indices]                   # [M,N,H]

        act_exp = mb_action.unsqueeze(2).expand_as(mb_eps)
        x_t = (1 - mb_cfm_t) * mb_eps + mb_cfm_t * act_exp
        h_exp = mb_h.unsqueeze(2).expand(-1, -1, mb_eps.size(2), -1)

        v_pred = self.mac.velocity(h_exp, x_t, mb_cfm_t)

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = act_exp - mb_eps
        elif cfm_target_type == "eps":
            target = mb_eps
        else:
            raise ValueError("cfm_target_type must be 'velocity' or 'eps'")

        return ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)

    def _compute_advantages_and_targets(self, target_critic, batch, rewards, mask):
        """GAE + lambda-returns, computed once per train() call (not once per
        epoch -- see the comment at the call site)."""
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

        return advantages, target_returns

    def _critic_gradient_step(self, critic, batch, target_returns, mask):
        """One critic gradient step against a FIXED target_returns (computed
        once outside the epoch loop) -- called once per epoch, same as
        before, just no longer recomputing GAE each time."""
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
        return running_log

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
        if self.actor_optimisers is not None:
            th.save(
                [opt.state_dict() for opt in self.actor_optimisers],
                "{}/actor_opt.th".format(path),
            )
        else:
            th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path),
                    map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.actor_optimisers is not None:
            opt_states = th.load(
                "{}/actor_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
            for opt, state in zip(self.actor_optimisers, opt_states):
                opt.load_state_dict(state)
        else:
            self.actor_optimiser.load_state_dict(
                th.load("{}/actor_opt.th".format(path),
                        map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path),
                    map_location=lambda storage, loc: storage))
