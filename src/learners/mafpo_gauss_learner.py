import copy
import math

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class MAFPOGaussLearner:
    """MAFPO-Gauss：flow 均值 + 显式高斯 + 精确 PPO ratio + ADER 熵分配。

    跟 PolicyFlow / FPO++ 的区别（2026-09-20 设计）：

    1. **obs/state 归一化**（components/obs_normalizer.py）：MAC 和 critic 共
       用一个运行均值-方差；本批数据训练完之后才 update，训练内统计量与
       rollout 时一致。

    2. **ratio 直接来自 (mu, sigma)，不用 FPO 的 CFM-loss/ELBO 代理**。给定
       rollout 存的 eps，策略是精确高斯 N(u; mu_theta(s,eps), sigma_i^2)，
       u = mu_old + n（n 是 rollout 真实采的噪声）：
           log rho = sum_d [ -(u-mu_new)^2/(2 sigma_new^2) + (u-mu_old)^2/(2 sigma_old^2)
                             - log(sigma_new/sigma_old) ]
       mu_new 是用当前 theta 在同一个 eps 上**重新积分** flow（K 步 Euler，
       不是 PolicyFlow 那个 delta_v 一阶近似），mu_old 直接读 buffer。theta=
       theta_old 时 mu_new==mu_old、rho==1，跟标准 PPO 完全一致，然后
       min(rho A, clip(rho) A)。

    3. **ADER 加在总熵上**（gauss_sigma_mode="ader"）：sigma_i 不进 PPO loss
       （训练内冻结，ratio 里 sigma_new==sigma_old 项抵消），每次 train()
       末尾用 theta 固定时的 dJ/dlog sigma_i 更新一次：
           g_{i,d} = E_valid[ A_i * ((n_{i,d}/sigma_{i,d})^2 - 1) ]
       这是高斯策略对 log sigma 的精确 score-function 梯度（policy 现在是显
       式高斯，pathwise 和 score 给的是同一个量），也就是"V 对 sigma 的导
       数"。更新：
           g_ema <- (1-a) g_ema + a g
           step  = clamp(ader_lr * g_ema, ±ader_max_log_k_step)
           step  = step - mean(step)            # ader_entropy_budget=True：总熵守恒
           log sigma <- clamp(log sigma + step, [log sigma_min, log sigma_max])
       所有 agent 从同一个 sigma_init 出发（初始熵相同），之后总熵
       sum_{i,d} log sigma_{i,d} 不变，只在 agent（和动作维）之间重新分配探
       索强度。ader_entropy_budget=False 则退化成各自独立的梯度上升。
       gauss_sigma_mode="ppo" 时 sigma 是普通参数、由 PPO loss + entropy_coef
       训练（标准连续 PPO）；"fixed" 时恒为 sigma_init。

    4. **额外的 Q_psi(s, a) 只管 sigma**（ader_estimator="q_pathwise"，默认）：
       原 V critic 原样不动、照常给 advantage；另建一个联合 Q_psi(s, a_1..a_N)
       （modules/critics/mafpo_gauss_q_critic.py），回归 V 那份 lambda-return
       （对 agent 取平均），每个 epoch 跟 V 一起训一次。sigma 的梯度走
       reparameterization 路径（DDPG/SAC 式，方差远低于 score-function）：
           a_i = sigmoid(mu_old_i + sigma_i * z_i),  z_i = n_i / sigma_old_i
           g_{i,d} = E[ dQ_psi(s, a) / dlog sigma_{i,d} ]     （autograd，theta 固定）
       ader_estimator="score" 则退回第 3 条的 A*((n/sigma)^2-1)。两个估计每次
       都算、都记（ader_score_* / ader_score_sf_*），只有选中的那个进更新。

    critic / GAE / return 标准化沿用 policyflow_continuous_learner（target
    critic 软更新、每个 epoch 训一次 critic 然后冻结 advantage）。
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.mac = mac

        self.sigma_mode = str(getattr(args, "gauss_sigma_mode", "ader")).lower()
        assert self.sigma_mode in ("ader", "ppo", "fixed"), self.sigma_mode
        # sigma 只在 "ppo" 模式下进优化器，其余模式由 learner 手写更新/冻结。
        sigma_params = [mac.agent.raw_sigma]
        other_params = [p for p in mac.parameters() if p is not mac.agent.raw_sigma]
        self.actor_params = other_params + (sigma_params if self.sigma_mode == "ppo" else [])
        self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.critic.normalizer = mac.obs_normalizer
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.normalizer = mac.obs_normalizer
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        # 只服务 sigma 的联合 Q_psi(s,a)，见类 docstring 第 4 条。
        self.ader_estimator = str(getattr(args, "ader_estimator", "q_pathwise")).lower()
        assert self.ader_estimator in ("q_pathwise", "score"), self.ader_estimator
        self.q_critic = None
        if self.sigma_mode == "ader" and self.ader_estimator == "q_pathwise":
            self.q_critic = critic_registry[getattr(args, "q_critic_type", "mafpo_gauss_q_critic")](scheme, args)
            self.q_critic.normalizer = mac.obs_normalizer
            self.q_params = list(self.q_critic.parameters())
            self.q_optimiser = Adam(params=self.q_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

        # 标量 sigma（gauss_sigma_per_dim=False）时 ADER 的自由度是 [N,1]
        self.sigma_shape = tuple(mac.agent.raw_sigma.shape)
        # delta_v ratio (PolicyFlow-style): the endpoint shift mu_new - mu_old is
        # estimated as mean_t[v_new(x_t,t) - v_old(x_t,t)] along the flow-matching
        # interpolation path x_t = (1-t)*eps + t*mu_old, instead of re-integrating
        # the ODE. Needs a frozen copy of the actor for v_old; the copy SHARES the
        # observation normaliser (which is only updated at the very end of train(),
        # so both see identical statistics and the difference comes from the actor
        # weights alone).
        self.use_delta_v_ratio = bool(getattr(args, "use_delta_v_ratio", False))
        self.delta_v_n_points = int(getattr(args, "delta_v_n_points", 8))
        self.old_mac = None
        if self.use_delta_v_ratio:
            self.old_mac = copy.deepcopy(mac)
            self.old_mac.obs_normalizer = mac.obs_normalizer
            for p_ in self.old_mac.agent.parameters():
                p_.requires_grad_(False)

        self.ader_g_ema = th.zeros(*self.sigma_shape, device=device)
        self.ader_g_sq_ema = th.zeros(*self.sigma_shape, device=device)
        self.ader_update_count = 0

    # ── train ────────────────────────────────────────────────────────────────

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_bt = mask[:, :, 0].clone()                          # [B,T]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        if self.args.common_reward:
            assert rewards.size(2) == 1
            rewards = rewards.expand(-1, -1, self.n_agents)
        mask = mask.repeat(1, 1, self.n_agents)                  # [B,T,N]
        critic_mask = mask.clone()

        # rollout 存下来的高斯量，平铺成 [B*T, N, A]
        mu_old_all = batch["action_raw"][:, :-1].float().reshape(-1, self.n_agents, self.n_actions)
        noise_all = batch["action_noise"][:, :-1].float().reshape(-1, self.n_agents, self.n_actions)
        eps_all = batch["z"][:, :-1].float().reshape(-1, self.n_agents, self.n_actions)
        u_all = mu_old_all + noise_all
        with th.no_grad():
            sigma_old = self.mac.agent.sigma().detach().clone()  # [N,A]，整个 train() 内冻结

        # old_mac is frozen for this whole train() call, so its hidden sequence is
        # computed once. It must come from old_mac's OWN encoder: feeding the new
        # encoder's h to the old velocity head would leave v_old and v_new sharing
        # everything but the last layer and shrink delta_v artificially (the failure
        # policyflow_continuous_learner.py documents at its _build_old_actor_hidden_sequence).
        old_h_seq = None
        if self.use_delta_v_ratio:
            with th.no_grad():
                old_h_seq = self.old_mac.agent.encode_sequence(
                    self.old_mac._build_inputs_all(batch)
                ).reshape(-1, self.n_agents, self.args.hidden_dim)

        valid_time_indices = th.nonzero(mask[:, :, 0].reshape(-1) > 0, as_tuple=False).squeeze(1)
        minibatch_size = getattr(self.args, "fpo_minibatch_size", 512)
        rho_clip = float(getattr(self.args, "cfm_rho_clip", 3.0))
        eps_clip = float(self.args.eps_clip)
        entropy_coef = float(getattr(self.args, "entropy_coef", 0.0))
        normalise_adv = bool(getattr(self.args, "normalise_advantages", True))

        actor_stats = {k: [] for k in [
            "pg_loss", "ppo_clip_fraction", "actor_grad_norm", "ratio_mean", "approx_kl",
            "mu_shift_abs_mean", "entropy_total",
        ]}
        critic_train_stats = {k: [] for k in [
            "critic_loss", "critic_grad_norm", "td_error_abs", "target_mean", "value_mean",
            "q_loss", "q_mean",
        ]}

        first_advantages = None
        for _ in range(self.args.epochs):
            advantages, target_returns, epoch_critic_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()                     # [B,T,N]
            for key, values in epoch_critic_stats.items():
                critic_train_stats[key].extend(values)
            if self.q_critic is not None:
                q_loss, q_mean = self._q_gradient_step(batch, target_returns.detach(), mask_bt)
                critic_train_stats["q_loss"].append(q_loss)
                critic_train_stats["q_mean"].append(q_mean)
            if normalise_adv:
                valid = mask.bool()
                if valid.any():
                    a = advantages[valid]
                    advantages = (advantages - a.mean()) / (a.std(unbiased=False) + 1e-8) * mask
            if first_advantages is None:
                first_advantages = advantages.clone()
            adv_flat = advantages.reshape(-1, self.n_agents)     # [B*T, N]

            permutation = valid_time_indices[
                th.randperm(valid_time_indices.numel(), device=valid_time_indices.device)
            ]
            for start in range(0, permutation.numel(), minibatch_size):
                mb = permutation[start:start + minibatch_size]
                h_seq = self._build_actor_hidden_sequence(batch)          # [B,T,N,H]
                h_mb = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])[mb]   # [M,N,H]

                sigma_new = self.mac.agent.sigma()                        # [N,A]
                if self.sigma_mode != "ppo":
                    sigma_new = sigma_new.detach()
                u = u_all[mb]
                mu_old = mu_old_all[mb]
                if self.use_delta_v_ratio:
                    # mu_new - mu_old, estimated on the interpolation path rather
                    # than by re-integrating: mu = eps + int_0^1 v dt, so the shift
                    # is mean_t[v_new - v_old] evaluated at the same x_t. At
                    # theta = theta_old, v_new == v_old exactly, so the ratio is
                    # still exactly 1.
                    delta_mu = self._delta_v(h_mb, old_h_seq[mb], eps_all[mb], mu_old)
                    mu_new = mu_old + delta_mu
                else:
                    mu_new = self.mac.agent.mean(h_mb, eps_all[mb])       # [M,N,A]
                log_ratio = (
                    -0.5 * ((u - mu_new) / sigma_new) ** 2
                    + 0.5 * ((u - mu_old) / sigma_old) ** 2
                    - th.log(sigma_new / sigma_old)
                ).sum(dim=-1)                                              # [M,N]
                ratio = th.exp(th.clamp(log_ratio, -rho_clip, rho_clip))
                adv = adv_flat[mb]                                         # [M,N]
                surr1 = ratio * adv
                surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                pg_loss = -th.min(surr1, surr2).mean()

                entropy_total = self.mac.agent.entropy_per_agent().sum()
                actor_loss = pg_loss
                if self.sigma_mode == "ppo" and entropy_coef > 0:
                    actor_loss = actor_loss - entropy_coef * entropy_total

                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
                self.actor_optimiser.step()

                with th.no_grad():
                    clipped = ((adv > 0) & (ratio > 1 + eps_clip)) | ((adv < 0) & (ratio < 1 - eps_clip))
                    actor_stats["pg_loss"].append(pg_loss.item())
                    actor_stats["ppo_clip_fraction"].append(clipped.float().mean().item())
                    actor_stats["actor_grad_norm"].append(grad_norm.item())
                    actor_stats["ratio_mean"].append(ratio.mean().item())
                    actor_stats["approx_kl"].append((ratio - 1 - log_ratio).mean().item())
                    actor_stats["mu_shift_abs_mean"].append((mu_new - mu_old).abs().mean().item())
                    actor_stats["entropy_total"].append(entropy_total.item())

        # ── ADER：theta 固定（用第一个 epoch、任何 actor 更新之前的 advantage
        # 快照）时 dJ/dlog sigma_{i,d} 的 score-function 估计，train() 末尾更新
        # 一次 sigma，下一批 rollout 才用上。
        ader_diag = None
        if self.sigma_mode == "ader":
            ader_diag = self._ader_update(
                batch, first_advantages.reshape(-1, self.n_agents), mu_old_all, noise_all,
                sigma_old, valid_time_indices,
            )

        if self.use_delta_v_ratio:
            self.old_mac.load_state(self.mac)          # theta_old <- theta for the next batch

        # 本批训练完了才更新归一化统计量（理由见 ObsNormalizer 的 docstring）。
        self.mac.obs_normalizer.update(batch, mask_bt)

        self.critic_training_steps += 1
        tau = self.args.target_update_interval_or_tau
        if tau > 1 and (self.critic_training_steps - self.last_target_update_step) / tau >= 1.0:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.last_target_update_step = self.critic_training_steps
        elif tau <= 1.0:
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for key in critic_train_stats:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            for key in actor_stats:
                self.logger.log_stat(key, self._mean_stat(actor_stats[key]), t_env)
            with th.no_grad():
                actions_taken = batch["actions"][:, :-1].float()
                valid_a = actions_taken[mask.unsqueeze(-1).expand_as(actions_taken).bool()]
                if valid_a.numel() > 0:
                    self.logger.log_stat("action_std", valid_a.std(unbiased=False).item(), t_env)
                    self.logger.log_stat(
                        "action_at_bound_fraction",
                        ((valid_a < 0.02) | (valid_a > 0.98)).float().mean().item(), t_env,
                    )
                self.logger.log_stat("mu_abs_max", mu_old_all.abs().max().item(), t_env)
                sigma = self.mac.agent.sigma()
                ent = self.mac.agent.entropy_per_agent()
                self.logger.log_stat("sigma_mean", sigma.mean().item(), t_env)
                for i in range(self.n_agents):
                    self.logger.log_stat(f"sigma_agent_{i}", sigma[i].mean().item(), t_env)
                    self.logger.log_stat(f"entropy_agent_{i}", ent[i].item(), t_env)
                    if self.mac.agent.sigma_per_dim:
                        for d in range(self.n_actions):
                            self.logger.log_stat(f"sigma_agent_{i}_dim_{d}", sigma[i, d].item(), t_env)
                if ader_diag is not None:
                    for i in range(self.n_agents):
                        self.logger.log_stat(f"ader_score_agent_{i}", ader_diag["g"][i].mean().item(), t_env)
                        self.logger.log_stat(f"ader_score_ema_agent_{i}", ader_diag["g_ema"][i].mean().item(), t_env)
                        self.logger.log_stat(f"ader_score_sf_agent_{i}", ader_diag["g_sf"][i].mean().item(), t_env)
                        if ader_diag.get("g_q") is not None:
                            self.logger.log_stat(f"ader_score_q_agent_{i}", ader_diag["g_q"][i].mean().item(), t_env)
                    self.logger.log_stat("ader_update_count", self.ader_update_count, t_env)
            self.log_stats_t = t_env

    # ── ADER ─────────────────────────────────────────────────────────────────

    def _q_gradient_step(self, batch, target_returns, mask_bt):
        """Q_psi(s, a_taken) 朝 V 的 lambda-return（agent 均值）回归一次。"""
        state = batch["state"][:, :-1]                                 # [B,T,S]
        actions = batch["actions"][:, :-1].float()                     # [B,T,N,A]
        q = self.q_critic(state, actions).squeeze(-1)                  # [B,T]
        target = target_returns.mean(dim=2)                            # [B,T]
        m = mask_bt
        loss = (((q - target) * m) ** 2).sum() / m.sum()
        self.q_optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()
        return loss.item(), ((q * m).sum() / m.sum()).item()

    def _ader_update(self, batch, adv_flat, mu_old_all, noise_all, sigma_old, valid_idx):
        """adv_flat [B*T,N]（已 normalize、已 mask），mu_old_all/noise_all
        [B*T,N,A]，sigma_old [N,A]。返回诊断 dict。"""
        n = noise_all[valid_idx]                                       # [M,N,A]
        per_dim = self.sigma_shape[-1] == self.n_actions
        with th.no_grad():
            a = adv_flat[valid_idx].unsqueeze(-1)                      # [M,N,1]
            g_sf = (a * ((n / sigma_old) ** 2 - 1.0)).mean(dim=0)      # [N,A]，score-function
            if not per_dim:
                g_sf = g_sf.sum(dim=-1, keepdim=True)                  # 标量 sigma：各维梯度相加 [N,1]

        g_q = None
        if self.q_critic is not None:
            # reparam：a = sigmoid(mu_old + sigma * z)，z = n / sigma_old；对 log sigma 求
            # autograd，Q 的参数不动（只取 grad 到 log_sigma 叶子）。叶子形状跟
            # raw_sigma 一致（[N,A] 或 [N,1]），标量模式下靠广播自动把各维梯度相加。
            log_sigma = th.log(sigma_old if per_dim else sigma_old[:, :1]).clone().requires_grad_(True)
            z = (n / sigma_old).detach()
            u = mu_old_all[valid_idx] + th.exp(log_sigma) * z          # [M,N,A]
            act = th.sigmoid(u)
            state = batch["state"][:, :-1].reshape(-1, batch["state"].shape[-1])[valid_idx]
            q = self.q_critic(state, act).squeeze(-1)                  # [M]
            (g_q,) = th.autograd.grad(q.mean(), log_sigma)
            g_q = g_q.detach()

        g = g_q if self.ader_estimator == "q_pathwise" else g_sf

        with th.no_grad():
            ema_alpha = float(getattr(self.args, "ader_ema_alpha", 0.05))
            self.ader_g_ema = self.ader_g_ema.to(g.device)
            self.ader_g_ema = (1 - ema_alpha) * self.ader_g_ema + ema_alpha * g
            self.ader_g_sq_ema = self.ader_g_sq_ema.to(g.device)
            self.ader_g_sq_ema = (1 - ema_alpha) * self.ader_g_sq_ema + ema_alpha * g ** 2
            self.ader_update_count += 1

            warmup = int(getattr(self.args, "ader_warmup_updates", 10))
            interval = max(1, int(getattr(self.args, "ader_update_interval", 1)))
            if self.ader_update_count > warmup and self.ader_update_count % interval == 0:
                ader_lr = float(getattr(self.args, "ader_lr", 1.0))
                max_step = float(getattr(self.args, "ader_max_log_k_step", 0.05))
                direction = self.ader_g_ema
                if bool(getattr(self.args, "ader_step_normalise", True)):
                    # Adam 式归一：EMA(g)/sqrt(EMA(g^2)) 落在 [-1,1]，ader_lr 直接是
                    # "每次更新最多走多少 log sigma"，跟 Q 的数值尺度（standardise
                    # _returns 之后 ~N(0,1)、dQ/da 早期 1e-5 量级）脱钩。
                    direction = direction / th.sqrt(self.ader_g_sq_ema + 1e-12)
                step = th.clamp(ader_lr * direction, -max_step, max_step)
                log_s_old = th.log(sigma_old if per_dim else sigma_old[:, :1])
                log_s = log_s_old + step
                if bool(getattr(self.args, "ader_entropy_budget", True)):
                    log_s = self._project_entropy_budget(log_s, log_s_old.sum())
                self.mac.agent.set_log_sigma(log_s)
        return {"g": g, "g_ema": self.ader_g_ema.clone(), "g_sf": g_sf, "g_q": g_q}

    def _project_entropy_budget(self, log_s, target_sum, iters=8):
        """把 log_s 投影到 {sum = target_sum} ∩ [log sigma_min, log sigma_max]。
        只减均值不够：碰到上下界的维度被 clamp 掉之后总和就漏了（2026-09-20
        第一条 MPE run 里 entropy_total 从 3.22 漏到 0.80，就是 sigma_max 那头
        被截、sigma_min 那头继续往下走）。这里迭代：clamp → 把残差平均分给
        还没贴边的维度 → 再 clamp，几轮就收敛；全部贴边时放弃守恒。"""
        lo = math.log(self.mac.agent.sigma_min) + 1e-6
        hi = math.log(self.mac.agent.sigma_max) - 1e-6
        for _ in range(iters):
            log_s = th.clamp(log_s, lo, hi)
            free = (log_s > lo + 1e-5) & (log_s < hi - 1e-5)
            resid = target_sum - log_s.sum()
            if resid.abs() < 1e-6 or free.sum() == 0:
                break
            log_s = log_s + resid / free.sum() * free.float()
        return th.clamp(log_s, lo, hi)

    # ── critic / GAE（同 policyflow_continuous_learner）───────────────────────

    def _delta_v(self, h_new, h_old, eps, mu_old):
        """mean_t[ v_theta(h_new, x_t, t) - v_theta_old(h_old, x_t, t) ] on
        x_t = (1-t)*eps + t*mu_old, t a fixed linspace of delta_v_n_points.
        Shapes: h_* [M,N,H], eps/mu_old [M,N,A] -> [M,N,A]."""
        K = self.delta_v_n_points
        M, N, A = eps.shape
        t = th.linspace(0.0, 1.0, K, device=eps.device).view(1, 1, K, 1)
        x_t = (1.0 - t) * eps.unsqueeze(2) + t * mu_old.unsqueeze(2)       # [M,N,K,A]
        t_b = t.expand(M, N, K, 1).reshape(-1, 1)
        x_b = x_t.reshape(-1, A)
        v_new = self.mac.agent.velocity(h_new.unsqueeze(2).expand(M, N, K, -1).reshape(-1, h_new.shape[-1]), x_b, t_b)
        with th.no_grad():
            v_old = self.old_mac.agent.velocity(
                h_old.unsqueeze(2).expand(M, N, K, -1).reshape(-1, h_old.shape[-1]), x_b, t_b)
        return (v_new - v_old).view(M, N, K, A).mean(dim=2)

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        """[B, T, N, H]，带梯度。整段一次算（GRU 走 cuDNN、MLP 批量前向），
        替代原来 T 步 Python 循环调 mac.forward——结果逐位一致，只是快。"""
        return self.mac.agent.encode_sequence(self.mac._build_inputs_all(batch))

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = target_critic(batch).squeeze(3)          # [B,T+1,N]
        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
        terminated = batch["terminated"][:, :-1].float()
        gae_lambda = getattr(self.args, "gae_lambda", 0.95)
        advantages = self.compute_gae(rewards, mask, target_vals, terminated, self.args.gamma, gae_lambda)
        target_returns = advantages + target_vals[:, :-1]
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {k: [] for k in ["critic_loss", "critic_grad_norm", "td_error_abs", "target_mean", "value_mean"]}
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
        return advantages, target_returns, running_log

    def compute_gae(self, rewards, mask, values, terminated, gamma, gae_lambda):
        T = rewards.size(1)
        gae = th.zeros_like(values[:, 0])
        advantages = th.zeros_like(rewards)
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - terminated[:, t]
            delta = rewards[:, t] + gamma * values[:, t + 1] * next_non_terminal - values[:, t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[:, t] = gae
        return advantages * mask

    def _mean_stat(self, values):
        return sum(values) / max(1, len(values))

    def cuda(self):
        self.mac.cuda()
        if self.old_mac is not None:
            self.old_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        if self.q_critic is not None:
            self.q_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        if self.q_critic is not None:
            th.save(self.q_critic.state_dict(), "{}/q_critic.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda s, l: s))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimiser.load_state_dict(th.load("{}/actor_opt.th".format(path), map_location=lambda s, l: s))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda s, l: s))
