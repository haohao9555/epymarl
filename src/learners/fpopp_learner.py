import copy
import os

import torch as th
from torch.optim import Adam, AdamW

from components.adams import AdamS

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPOPPLearner:
    """连续动作（Box(0,1)）多智能体 flow 策略的 FPO++ learner。

    2026-09-05 起 ratio/surrogate 逐条对齐官方实现 amazon-far/fpo-control
    （isaaclab_fpo/algorithms/fpo.py 的 FPO.update()，以及 manipulation 版
    finetune_online_rl.py / src/flow_model.py）：

    CFM 误差：逐探测点的回归误差走 mac.cfm_error()——官方 FPO++ fine-tuning
    版的 modified Huber（|e|<=delta 时是 e^2，之外线性，对 v_pred 的梯度上界
    2*delta；cfm_loss_huber_delta，默认 1.0）+ 可配置的 action 维 reduction
    （cfm_loss_reduction："mean"/"sum"/"sqrt"）。rollout 存的 L_old 和这里重算
    的 L_new 必须是同一个函数，所以它只在 mafpo_mac.py 里定义一次。

    Ratio：没有闭式的 log-prob，用 CFM 回归 loss 当 ELBO 的代理。对一笔
    transition 上采样的 cfm_n 个 (tau_i, eps_i) 探测点，每个点各自算一个 ratio
    （官方 "Per-sample log ratios (no averaging before exp)"）：
        rho_i = exp(ste_clamp(C(L_old^(i)) - C'(L_new^(i)), max=rho_clip))
    C 是 old/new 共用的普通 clamp(max=cfm_loss_clip_max)（官方 cfm_loss_clamp），
    C' 在 A<0 的样本上再额外夹 cfm_loss_clip_max_neg_adv（官方
    cfm_loss_clamp_negative_advantages），ste_clamp 是官方 clamp_ste（前向截断、
    反向恒等）。逐点算，不先对 cfm_n 取平均（exp 是非线性的，先平均会让符号
    相反的点互相抵消，在 ratio 看到它们之前就丢掉了信息）。只在最后对
    (minibatch, N, cfm_n) 一起取平均得到 surrogate。

    Surrogate（fpo_trust_region）：默认 "ppo" = 原版 FPO 的标准 PPO clip
    min(rho*A, clip(rho,1-eps,1+eps)*A)，对所有 A 一视同仁。"aspo" = 官方
    FPO++ 的 trust_region_mode="aspo"：A>0 用标准 PPO clip；A<=0 用
    平滑的 SPO 目标替代 PPO 硬 clip：
        psi_SPO(rho, A) = rho*A - (|A| / (2*eps_clip)) * (rho-1)^2
    PPO 的 clip 在 rho 越出边界后梯度恒为 0；SPO 在 rho=1 处梯度跟未裁剪的策略
    梯度一致，越出边界后梯度平滑衰减而不是直接归零，所以一个已经偏得很远的样
    本仍然会被往回拉，而不是彻底失活。

    Ratio 的 log-diff（算完 diff 之后）用 straight-through clamp：前向数值被
    截断到 rho_clip，反向梯度按恒等函数穿透，一个已经越界的点仍然带着把它往
    回拉的梯度，不会彻底失活（见 _ste_clamp()）。

    但给 diff 喂数据的 old/new CFM loss 各自的上界 clamp（cfm_loss_clip_max）
    刻意用的是普通 clamp，不是 STE（官方也是 torch.clamp）：回归误差对 v_pred
    的原始梯度没有归一化约束。如果这里也用 STE，一旦 old/new 同时超过上限，
    diff=C_L-C_L=0，rho=1，ratio 本身看起来毫无异常，但未截断的原始回归梯度
    会绕过这层"看起来正常"的伪装直接传回 v_pred，等于让一个已经烂掉的点在
    诊断量完全看不出来的情况下主导更新。这里就是要让梯度真的断掉：C_L 是
    "离谱到不该再提供学习信号"的安全阀，跟 log-diff 那层 STE clamp（刻意保留
    梯度）解决的是两个不同的问题，不能合并成一步。

    优化器：fpo_optimizer 选 "adams"（默认，Adam + Stable Weight Decay，
    components/adams.py）/ "adamw"（官方）/ "adam"，decay 系数 fpo_weight_decay
    （默认 1e-4），见 _make_optimiser()。每次 log 顺带记 actor_param_norm
    （actor 全部参数的 L2 范数）和 swd_v_bar_sqrt（SWD 的 decay 尺度）。执行动作的 clip 映射和 rollout 扰动在 mafpo_mac.py。

    Config 里几个互相独立的轴（都在 __init__ 里生效）：
      - fpo_individual_agents（从 mac 上读）：每个 agent 一套独立参数，还是所
        有 agent 共用一套（靠 obs_agent_id one-hot 区分）。
      - fpo_actor_optim_per_agent：每个 agent 独立 Adam + 独立
        clip_grad_norm_，还是一个联合优化器/联合裁剪。只有 actor 本来就是独
        立参数时才有意义——共享一套参数的话不管这个开关怎么设都只会有一个
        Adam/一次裁剪。
      - fpo_anchor_per_agent：anchor/reflow 的 hinge-squared 惩罚按 agent 拆开
        算再求和，还是对整个 batch 的均值算一个全局惩罚。
      - critic_type：mafpo_critic（agent-id-conditioned，N 个独立的 GAE
        advantage）还是 mafpo_shared_critic（单一联合 V(s)，一个 advantage 广
        播给所有 agent）。从 critic 实例自己的 `per_agent_values` 属性读取。
        单一共享 advantage 要求 common_reward=True。

    wandb 只留最关键的几个：critic 侧 value_mean/critic_loss，actor 侧
    cfm_loss_mean/ppo_clip_fraction/actor_grad_norm。
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        # surrogate 的 trust region 形式，见 train() 里的 surrogate 段。
        self.trust_region = str(getattr(args, "fpo_trust_region", "ppo")).lower()
        assert self.trust_region in ("ppo", "aspo"), f"未知 fpo_trust_region={self.trust_region!r}"
        self.logger = logger
        self.mac = mac

        individual_agents = getattr(mac, "individual_agents", False)
        self.individual_agents = individual_agents

        # actor 优化器/梯度裁剪的粒度。按 agent 拆开：每个 agent 自己的 Adam +
        # 自己的 clip_grad_norm_，这样某个 agent 这一步梯度偏大，不会通过共享
        # 的全局范数裁剪把其它 agent 本来正常的更新也一起摁下去。只有 actor
        # 本身就是独立参数时才生效。
        self.optim_per_agent = individual_agents and getattr(
            args, "fpo_actor_optim_per_agent", individual_agents
        )
        if self.optim_per_agent:
            self.actor_params_per_agent = [list(agent.parameters()) for agent in mac.agents]
            self.actor_optimisers = [
                self._make_optimiser(p) for p in self.actor_params_per_agent
            ]
            self.actor_params = None
            self.actor_optimiser = None
        else:
            self.actor_params_per_agent = None
            self.actor_optimisers = None
            self.actor_params = list(mac.parameters())
            self.actor_optimiser = self._make_optimiser(self.actor_params)

        # anchor/reflow 聚合方式：按 agent 拆开求和，还是对全局均值算一个。跟
        # optim_per_agent 相互独立。
        self.anchor_per_agent = getattr(args, "fpo_anchor_per_agent", individual_agents)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = self._make_optimiser(self.critic_params)

        # advantage 的宽度：N 个按 agent 拆开的 GAE advantage，还是一个共享
        # advantage 广播给所有 agent——从 critic 自己声明的 per_agent_values
        # 类属性读取（见 mafpo_critic.py / mafpo_shared_critic.py），不额外设
        # 一个可能跟 critic_type 脱节的旗标。
        self.per_agent_values = getattr(self.critic, "per_agent_values", True)
        self.advantage_width = self.n_agents if self.per_agent_values else 1
        if not self.per_agent_values:
            assert self.args.common_reward, (
                f"critic_type={args.critic_type!r} 只产生单一联合 V(s) 的共享 "
                "advantage——需要 common_reward=True（按 agent 拆开的 reward "
                "配一个共享 advantage 没有良好定义）。"
            )

        self.log_stats_t = -self.args.learner_log_interval - 1

        # ── PFO（Proximal Feature Optimization，Moalla et al. 2024，NeurIPS）：
        # L_total = L_FPO + lambda * ||phi_theta(s) - phi_theta_old(s)||^2，phi 取
        # 速度网络倒数第二层的预激活（见 mafpo_mac.velocity_features）。theta_old
        # 是这次 train() 开始时的参数，也就是产生这批 rollout 的策略。原文用它
        # 抑制 PPO 的表征崩塌，这里当作"抑制中心漂移"的一条已有解法来对照。
        # fpo_pfo_coef=0（默认）时整条路径不建立，old_mac 也不创建。
        self.pfo_coef = float(getattr(args, "fpo_pfo_coef", 0.0))
        self.old_mac = copy.deepcopy(mac) if self.pfo_coef > 0 else None

        # ── theta 轨迹诊断：每次 train() 调用记一次，用来区分参数是在扩散
        # 还是在系统性漂移。三个量：单次更新的位移 step_norm、相对第一次调用
        # 的累计位移 disp_norm、以及相邻两次更新方向的夹角余弦 step_cos。
        # 随机游走时 disp ~ sqrt(K)*step 且 cos≈0；系统漂移时 disp ~ K*step
        # 且 cos>0。开销是每次调用拍平一次参数，可忽略。
        self._theta_ref = None
        self._theta_prev = None
        self._prev_step = None

        # ── adaptive lr（官方 FPO.update() 的 schedule="adaptive"）：用新旧速度
        # 场在同一批探测点上的 x1_pred 均方差当 KL 代理，超过 2*desired_kl 就把
        # lr 除以 1.5，低于 desired_kl/2 就乘 1.5，夹在 [1e-5, 1e-2]。
        # fpo_lr_schedule="fixed"（默认）时完全不动。
        self.current_lr = float(args.lr)

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.advantage_width,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

        # ── ADER：agent-wise adaptive base-noise scale（见 mafpo_mac.py
        # 顶部关于 ader_k 的说明）。学习率/更新算法全在这个 learner 里，
        # mac 只负责保存/暴露当前 k。ader_enabled=False 时下面这些状态都不
        # 会被用到——mac 自己已经强制 ader_k=ones，这里只是保持 beta_ema/
        # update_count 初始化好，方便 checkpoint 逻辑不用特判。
        self.ader_enabled = getattr(mac, "ader_enabled", False)
        self.ader_mode = getattr(mac, "ader_mode", "adaptive")
        self.beta_ema = th.ones(self.n_agents, device=device) / self.n_agents
        # gradient 模式用的 g 的 EMA（g 本身是 ~2k 条 transition 上的蒙特卡洛
        # 均值，单批噪声 std≈0.02，直接当梯度步太抖）。
        self.ader_g_ema = th.zeros(self.n_agents, device=device)
        self.ader_update_count = 0
        self._ader_log_cache = {}

    def _flat_actor_params(self):
        if self.actor_params_per_agent is not None:
            ps = [p for group in self.actor_params_per_agent for p in group]
        else:
            ps = self.actor_params
        return th.cat([p.detach().reshape(-1) for p in ps])

    def _track_theta(self):
        """在一次 train() 的全部梯度步之后调用一次。返回
        (step_norm, disp_norm, step_cos)。"""
        with th.no_grad():
            flat = self._flat_actor_params()
            if self._theta_ref is None:
                self._theta_ref = flat.clone()
                self._theta_prev = flat.clone()
            step = flat - self._theta_prev
            sn = step.norm()
            dn = (flat - self._theta_ref).norm()
            cos = th.zeros((), device=flat.device)
            if self._prev_step is not None:
                pn = self._prev_step.norm()
                if sn > 0 and pn > 0:
                    cos = th.dot(step, self._prev_step) / (sn * pn)
            self._prev_step = step.clone()
            self._theta_prev = flat.clone()
        return sn.item(), dn.item(), cos.item()

    def _make_optimiser(self, params):
        """fpo_optimizer 选优化器（actor 逐 agent/联合 和 critic 都走这一个工厂）：
          - "adams"（默认）：Adam + Stable Weight Decay（Xie et al. 2023），见
            components/adams.py。decay 项除以 sqrt(v_bar)（v_hat 的全局均值），
            跟 Adam 的自适应梯度步同尺度。这里选它而不是 AdamW 的原因：RL 里
            actor 梯度经常是 1e-3 量级，AdamW 的 (1 - lr*wd) 收缩相对
            lr*m/sqrt(v) 的梯度步小了几个数量级，decay 实际上没有约束住 W 的
            增长（theta_disp_norm 一直涨）；SWD 把两者拉到同一尺度，wd 才真
            的在管权重范数。
          - "adamw"：官方 FPO++ 的选择（FpoRslRlPpoAlgorithmCfg.weight_decay=1e-4）。
          - "adam"：无 decay。
        fpo_weight_decay<=0 时不管选什么都退回 Adam。
        weight decay 是官方实现里唯一持续把速度网络权重（因而把积分终点 x1）
        往回拉的力：执行动作的 clip 之外 reward 是平的、advantage 的噪声让 x1
        做随机游走，没有这一项就没有任何东西阻止 x1 漂出 clip 区（见
        mafpo_mac.py 文件头）。官方是 actor+critic 一个优化器一起 decay，这里
        actor（逐 agent 或联合）和 critic 各自的优化器都用同一个工厂。"""
        weight_decay = float(getattr(self.args, "fpo_weight_decay", 1e-4))
        kind = str(getattr(self.args, "fpo_optimizer", "adams")).lower()
        assert kind in ("adams", "adamw", "adam"), f"未知 fpo_optimizer={kind!r}"
        if weight_decay <= 0 or kind == "adam":
            return Adam(params=params, lr=self.args.lr)
        if kind == "adamw":
            return AdamW(params=params, lr=self.args.lr, weight_decay=weight_decay)
        return AdamS(params=params, lr=self.args.lr, weight_decay=weight_decay)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # rewards：common_reward 时 [B,T,1]，否则 [B,T,N]。mask/terminated：[B,T,1]。
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        base_mask = batch["filled"][:, :-1].float()
        base_mask[:, 1:] = base_mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # critic_mask 跟着 advantage 的宽度走（N 还是 1）；mask 永远是
        # [B,T,N]——下面逐 agent 的诊断和 ratio/surrogate 机制不管用哪种
        # critic 都按 agent 展开算。
        if self.per_agent_values:
            if self.args.common_reward:
                assert rewards.size(2) == 1
                rewards = rewards.expand(-1, -1, self.n_agents)
            mask = base_mask.repeat(1, 1, self.n_agents)
            critic_mask = mask.clone()
        else:
            mask = base_mask.repeat(1, 1, self.n_agents)
            critic_mask = base_mask.clone()

        initial_cfm_loss = batch["initial_cfm_loss"][:, :-1]    # [B,T,N,cfm_n,1]
        rho_clip = getattr(self.args, "cfm_rho_clip", 3.0)

        # ── x1 诊断：直接量 flow 的无界积分终点 x1=action_raw 的幅度分布，用来
        # 对照 sigmoid / clip 两种执行映射。只读 buffer，不进任何 loss。
        with th.no_grad():
            x1 = batch["action_raw"][:, :-1].float()            # [B,T,N,A]
            acts = batch["actions"][:, :-1].float()             # [B,T,N,A]
            m_a = mask.unsqueeze(-1).expand_as(x1)
            denom_a = m_a.sum().clamp(min=1.0)
            clip_c = float(getattr(self.args, "fpo_action_clip", 2.0))
            x1_stats = {
                "x1_abs_mean": (x1.abs() * m_a).sum() / denom_a,
                "x1_abs_max": (x1.abs() * m_a).max(),
                "x1_beyond_clip_fraction": ((x1.abs() > clip_c).float() * m_a).sum() / denom_a,
                "action_at_bound_fraction": (
                    ((acts < 0.02) | (acts > 0.98)).float() * m_a
                ).sum() / denom_a,
            }

        # minibatch 是在环境 timestep 上抽样的；每个被选中的 timestep 把所有
        # agent 一起带上。
        valid_time_indices = th.nonzero(
            mask[:, :, 0].reshape(-1) > 0, as_tuple=False
        ).squeeze(1)
        minibatch_size = getattr(self.args, "fpo_minibatch_size", 256)
        actor_stats = {
            "cfm_loss_mean": [],
            "ppo_clip_fraction": [],
            "actor_grad_norm": [],
            "cfm_loss_clip_fraction": [],
            "actor_lr": [],
            "x1_pred_kl": [],
            "pfo_penalty": [],
            "pg_loss": [],
            "theta_step_norm": [],
            "theta_step_cos": [],
        }
        theta_disp = None

        # ── adaptive lr 用的"旧"速度场：train() 开始时参数就是 rollout 参数
        # （on-policy，buffer 每次 train 后清空），所以在任何梯度步之前对全部
        # 探测点算一遍 v_old 存下来，之后每个 minibatch 拿新 v 和它比。官方是
        # rollout 时把 x1_pred 存进 storage，这里等价，但不用改 buffer scheme。
        lr_schedule = getattr(self.args, "fpo_lr_schedule", "fixed")
        old_v_all = None
        if lr_schedule == "adaptive":
            desired_kl = float(getattr(self.args, "fpo_desired_kl", 0.01))
            with th.no_grad():
                h_old = self._build_actor_hidden_sequence(batch)
                all_time_idx = th.arange(
                    h_old.shape[0] * h_old.shape[1], device=h_old.device
                )
                old_v_all, _, _ = self._velocity_for_time_indices(
                    batch, h_old, all_time_idx
                )                                                   # [B*T,N,K,A]
        critic_train_stats = {k: [] for k in ["critic_loss", "value_mean"]}

        # GAE + lambda-return：整个 train() 调用只算一次，后面所有 epoch/
        # minibatch 都固定用这一份（critic 每个 epoch 仍然做一次梯度更新，但
        # 用的都是这同一份固定 target_returns，不重新算 GAE）。ader_advantage
        # 是给 ADER score 用的另一份快照：normalize 之后、adv_clip 之前、已
        # 乘 mask、且 detach——跟 actor 实际用来算 surrogate 的 advantages
        # 是两份数据，不能混用（actor 的那份还要过 adv_clip）。
        advantages, target_returns, v_old, ader_advantage = self._compute_advantages_and_targets(
            batch, rewards, critic_mask
        )
        advantages = advantages.detach()
        target_returns = target_returns.detach()
        v_old = v_old.detach()

        # ── ADER score g_i：score-function estimator，dJ/dlog(k_i) ≈
        # E[A * (||z_i||^2 - d)]，必须在 critic/actor 参数更新之前、用
        # rollout policy 对应的固定 advantage 算好并 detach；真正拿 g 去改
        # k 要等本次 train() 的所有 actor epoch 都跑完（见 train() 末尾对
        # _update_ader_k 的调用）。score-function 版全程 no_grad；pathwise
        # 版只对 log s 这个叶子做 autograd.grad（不进 theta 的 .grad）——两者
        # 都不是 loss，绝不能 backward()。
        ader_g = None
        if self.ader_enabled:
            with th.no_grad():
                if self.advantage_width == 1:
                    a_team = ader_advantage           # [B,T,1]
                else:
                    a_team = ader_advantage.mean(dim=2, keepdim=True)  # [B,T,1]
                a_team = a_team.expand(-1, -1, self.n_agents)          # [B,T,N]

                z = batch["flow_z"][:, :-1].float()                    # [B,T,N,A]
                a_dim = z.shape[-1]
                score = (z.pow(2).sum(dim=-1) - a_dim) / (2 * a_dim) ** 0.5  # [B,T,N]

                weighted = mask * a_team * score
                denom = mask.sum(dim=(0, 1)).clamp(min=1e-8)           # [N]
                ader_g_sf = weighted.sum(dim=(0, 1)) / denom           # [N]
                ader_z_sq_mean = (mask * z.pow(2).sum(dim=-1)).sum(dim=(0, 1)) / denom  # [N]

            self._ader_log_cache["ader_z_sq_mean"] = ader_z_sq_mean
            self._ader_log_cache["ader_g_sf"] = ader_g_sf
            estimator = getattr(self.args, "ader_estimator", "pathwise")
            if estimator == "pathwise":
                # theta 还是 rollout policy（epoch 循环之前），走 reparam 路径
                # 对 log s 求 autograd，见 _ader_pathwise_score()。
                ader_g = self._ader_pathwise_score(
                    batch, a_team, mask, valid_time_indices, minibatch_size,
                    initial_cfm_loss, rho_clip,
                ).detach()
            elif estimator == "score":
                ader_g = ader_g_sf
            else:
                raise ValueError(f"ader_estimator must be 'pathwise' or 'score', got {estimator!r}")

        # 诊断：执行动作 / 无界积分终点 x1 按 agent 拆开的标准差，也就是动作
        # 分布的「宽度」。跟 ADER 无关，无条件算——2026-09-10 之前它跟
        # cfm_loss_clip_fraction 一起被写在 _log_ader_stats() 里，而那个函数只
        # 在 ader_enabled=True 时调用，于是所有 baseline run 都缺这两条。缺了
        # 就只能看到分布的中心（msd_probe 的 d_mu_act 量的就是中心），看不到
        # 宽度——而 x1 的尾部爆炸（x1_abs_max 12 倍增长而 x1_abs_mean 只有
        # 2.5 倍）恰恰是宽度问题，不是中心问题。
        with th.no_grad():
            acts = batch["actions"][:, :-1].float()                # [B,T,N,A]
            acts_raw = batch["action_raw"][:, :-1].float()         # [B,T,N,A]
            mask_a = critic_mask.expand(-1, -1, self.n_agents).unsqueeze(-1).float()
            denom_a = mask_a.sum(dim=(0, 1)).clamp(min=1e-8)       # [N,1]
            mean_a = (mask_a * acts).sum(dim=(0, 1)) / denom_a
            var_a = (mask_a * (acts - mean_a) ** 2).sum(dim=(0, 1)) / denom_a
            action_std = var_a.clamp(min=0).sqrt().mean(dim=-1)    # [N]
            mean_ar = (mask_a * acts_raw).sum(dim=(0, 1)) / denom_a
            var_ar = (mask_a * (acts_raw - mean_ar) ** 2).sum(dim=(0, 1)) / denom_a
            action_raw_std = var_ar.clamp(min=0).sqrt().mean(dim=-1)  # [N]
        self._ader_log_cache["action_std"] = action_std
        self._ader_log_cache["action_raw_std"] = action_raw_std

        # PFO：theta_old = 这次 train() 开始时的参数（= 产生这批 rollout 的策略），
        # 整个 train() 调用内冻结；它的 hidden 序列也只滚一次。
        h_old_seq = None
        if self.old_mac is not None:
            self.old_mac.load_state(self.mac)
            with th.no_grad():
                h_old_seq = self._build_actor_hidden_sequence(batch, mac=self.old_mac)

        for _ in range(self.args.epochs):
            epoch_critic_stats = self._critic_gradient_step(
                self.critic, batch, target_returns, critic_mask, v_old
            )
            for key, values in epoch_critic_stats.items():
                critic_train_stats[key].extend(values)

            # 每个 epoch 打乱有效的 (episode, time) 下标，按 minibatch 扫一遍
            # （标准 PPO 式对同一批 rollout 数据的多次 minibatch 遍历）。
            permutation = valid_time_indices[
                th.randperm(valid_time_indices.numel(), device=valid_time_indices.device)
            ]

            for start in range(0, permutation.numel(), minibatch_size):
                mb_time_idx = permutation[start:start + minibatch_size]

                # 每个 minibatch 都重新算一遍 actor hidden state 和 CFM
                # loss——上一步已经更新过参数，不能复用旧图。
                h_seq = self._build_actor_hidden_sequence(batch)
                mb_cfm_loss, mb_v_pred, mb_cfm_t_pt = self._compute_cfm_loss_for_time_indices(
                    batch, h_seq, mb_time_idx
                )
                if old_v_all is not None:
                    # 官方 kl_mean = mean((x1_pred_new - x1_pred_old)^2)；x1_pred 是
                    # 探测点上的去噪预测，新旧之差 = t * (v_new - v_old)（官方 t 约
                    # 定是噪声端为 1，换算到这里的 t 就是这个因子）。先调 lr 再做
                    # 这个 minibatch 的梯度步，顺序同官方。
                    with th.no_grad():
                        kl_proxy = (
                            (mb_cfm_t_pt * (mb_v_pred - old_v_all[mb_time_idx])) ** 2
                        ).mean().item()
                    self._adapt_lr(kl_proxy, desired_kl)
                    actor_stats["x1_pred_kl"].append(kl_proxy)
                actor_stats["actor_lr"].append(self.current_lr)
                mb_initial_cfm_loss = initial_cfm_loss.reshape(
                    -1, self.n_agents, initial_cfm_loss.size(-2), initial_cfm_loss.size(-1)
                )[mb_time_idx]

                # advantage 广播：advantages_by_time 最后一维是 advantage_width
                # （N 或 1）；.expand(-1, n_agents) 在已经是 N 时是个 no-op，是
                # 1 时自动广播——两种 critic 用同一行代码处理。
                advantages_by_time = advantages.reshape(-1, self.advantage_width)
                mb_advantages_2d = advantages_by_time[mb_time_idx].expand(
                    -1, self.n_agents
                )   # [M,N]

                # ---- 张量规范：从这里到 pg_loss 全程保持显式的 [M,N,K]，K=cfm_n ----
                # 不靠隐式广播——advantage 先显式 expand 到 [M,N,K]，ratio 也是
                # [M,N,K]，做逐元素乘之前 assert 两边形状完全一致。n=1 时 N=1，
                # 走的是同一条路径，没有任何 [M,1,1] 对 [M,1,K] 的隐式广播
                # （官方 fpo.py 在同一位置也是一组 shape assert）。
                M = mb_time_idx.numel()
                N = self.n_agents
                K = self.args.cfm_n_samples
                assert mb_cfm_loss.shape == (M, N, K, 1), (mb_cfm_loss.shape, (M, N, K, 1))
                assert mb_initial_cfm_loss.shape == (M, N, K, 1), (
                    mb_initial_cfm_loss.shape, (M, N, K, 1)
                )
                assert mb_advantages_2d.shape == (M, N), (mb_advantages_2d.shape, (M, N))
                mb_advantages_pt = mb_advantages_2d.unsqueeze(-1).expand(M, N, K)  # [M,N,K]

                # ---- ratio：old/new CFM loss 各自先夹一个上界 cfm_loss_clip_max ----
                # （官方 cfm_loss_clamp："Applied symmetrically to both old and
                # current CFM losses"）两边都用普通 clamp，不是 STE。old_cfm_loss_c
                # 是 rollout 时存的无梯度 buffer 数据，STE 与否对它无所谓。
                # new_cfm_loss_c 不能用 STE：一旦 old/new 同时超过上限，
                # diff = C_L - C_L = 0，rho=exp(0)=1，ratio 和 ppo_clip_fraction 这
                # 些诊断量看起来完全正常，但如果这里用 STE，未截断的原始回归梯
                # 度会绕过这层伪装直接传回 v_pred，等于让一个已经烂掉的点在"看
                # 起来人畜无害"的掩护下主导更新，恰好是这个 clamp 本该防住的问
                # 题。普通 clamp 在这里就是要让梯度真的断掉：C_L 是"离谱到不该
                # 再提供学习信号"的安全阀，这跟下面对 log-ratio 的 STE clamp（那
                # 里刻意保留梯度）解决的是两个不同的问题，不要合并。
                cfm_loss_clip_max = getattr(self.args, "cfm_loss_clip_max", 20.0)
                old_cfm_loss_c = th.clamp(mb_initial_cfm_loss, max=cfm_loss_clip_max)
                new_cfm_loss_c = th.clamp(mb_cfm_loss, max=cfm_loss_clip_max)

                # 诊断：new CFM loss 有多大比例撞到了这个上限——k 太大会让
                # 大量探测点的回归误差本身就离谱，一旦触顶，diff 里 new 那
                # 一侧被钉死，ratio 的梯度信号跟着失活（见类 docstring 里
                # cfm_loss_clip_max 那段）。按 agent 拆开看，因为不同 agent
                # 的 k 可能已经分化开了。
                with th.no_grad():
                    actor_stats["cfm_loss_clip_fraction"].append(
                        (mb_cfm_loss.squeeze(-1) >= cfm_loss_clip_max).float().mean(dim=(0, 2))
                    )

                # 官方 cfm_loss_clamp_negative_advantages：A<0 的样本对 new loss
                # 再单独夹一个（通常更低的）上界 cfm_loss_clip_max_neg_adv。A<0
                # 时 SPO 是在把这些探测点的回归误差往上推（"避开坏动作"）——纯
                # 回归误差没有归一化约束，可以被无限推高而不需要在别处补偿，
                # 这个上界让"推开"到此为止、之后梯度断掉（官方："Prevents the
                # policy from being destabilized by extreme ratios when
                # aggressively avoiding bad actions"）。默认等于 cfm_loss_clip_max
                # （官方默认两者都是 20，同样是 no-op），要单独收紧时再设。
                cfm_loss_clip_max_neg = getattr(
                    self.args, "cfm_loss_clip_max_neg_adv", cfm_loss_clip_max
                )
                if self.trust_region == "aspo" and cfm_loss_clip_max_neg < cfm_loss_clip_max:
                    new_cfm_loss_c = th.where(
                        mb_advantages_pt.unsqueeze(-1) < 0,
                        th.clamp(new_cfm_loss_c, max=cfm_loss_clip_max_neg),
                        new_cfm_loss_c,
                    )

                # 逐点 log-ratio，保留 cfm_n 维不 reduce（原因见类 docstring）。
                diff = (old_cfm_loss_c - new_cfm_loss_c).squeeze(-1)   # [M,N,K]

                # 对 log-ratio 做 STE clamp（官方 clamp_ste），只截上界：diff 很负
                # 时 rho 本来就会平滑趋近 0（没有爆炸风险），只有 diff 很正的一
                # 侧需要在 exp() 之前截住。
                diff = self._ste_clamp(diff, -float("inf"), rho_clip)
                mb_rho_s_pt = th.exp(diff)                              # [M,N,K]
                assert mb_rho_s_pt.shape == mb_advantages_pt.shape == (M, N, K), (
                    mb_rho_s_pt.shape, mb_advantages_pt.shape, (M, N, K)
                )

                mb_rho_s = mb_rho_s_pt.reshape(-1)
                mb_advantages = mb_advantages_pt.reshape(-1)

                # ---- surrogate。fpo_trust_region:
                #   "ppo"（默认）：原版 FPO（McAllister et al. 2025）的标准 PPO
                #     clip，对所有 A 一视同仁：min(rho*A, clip(rho)*A)。
                #   "aspo"：官方 FPO++ 的 trust_region_mode="aspo"：A>0 用 PPO
                #     clip，A<=0 用平滑 SPO（官方 torch.where(advantages > 0,
                #     ppo, spo)），配套 cfm_loss_clip_max_neg_adv 一起生效。
                # clip 边界用论文标准的线性 [1-eps, 1+eps]，不是 log 空间对称。
                clip_lo = 1.0 - self.args.eps_clip
                clip_hi = 1.0 + self.args.eps_clip
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(mb_rho_s, clip_lo, clip_hi) * mb_advantages
                ppo_surr = th.min(surr1, surr2)
                if self.trust_region == "aspo":
                    pos_mask = mb_advantages > 0
                    spo_surr = (
                        mb_rho_s * mb_advantages
                        - (mb_advantages.abs() / (2 * self.args.eps_clip))
                        * (mb_rho_s - 1) ** 2
                    )
                    surr = th.where(pos_mask, ppo_surr, spo_surr)
                else:
                    surr = ppo_surr
                # 只在这一步对 (minibatch, N, cfm_n) 一起取平均——ratio 和逐点
                # surrogate 本身全程没有被平均过。
                pg_loss = -surr.mean()

                # ---- anchor：CFM 回归误差的 hinge-squared 惩罚，只在
                # batch/agent 均值超过 cfm_anchor_threshold 时启动。
                # anchor_per_agent=True 时按 agent 拆开算再求和（每一项只依赖
                # 对应 agent 自己的参数，求和不会产生跨 agent 的参数级耦合）；
                # False 时对整体取一个全局均值。coef<=0 时整段短路成零张量
                # （不短路的话，即便系数是 0，也会用完全没被 cfm_loss_clip_max
                # 保护过的原始 mb_cfm_loss 算 excess——一旦它冲到 inf，
                # `0 * inf = NaN` 会混进 actor_loss）。
                cfm_anchor_coef = getattr(self.args, "cfm_anchor_coef", 0.0)
                if cfm_anchor_coef > 0:
                    if self.anchor_per_agent:
                        cfm_loss_per_agent = mb_cfm_loss.mean(dim=(0, 2, 3))  # [N]
                        anchor_excess_per_agent = th.clamp(
                            cfm_loss_per_agent - self.args.cfm_anchor_threshold, min=0.0
                        )
                        anchor_loss = (
                            cfm_anchor_coef * anchor_excess_per_agent ** 2
                        ).sum()
                    else:
                        cfm_loss_batch_mean = mb_cfm_loss.mean()
                        anchor_excess = th.clamp(
                            cfm_loss_batch_mean - self.args.cfm_anchor_threshold, min=0.0
                        )
                        anchor_loss = cfm_anchor_coef * anchor_excess ** 2
                else:
                    anchor_loss = th.zeros((), device=mb_cfm_loss.device)

                # ---- reflow：flow 速度场的自蒸馏直线度正则（Rectified Flow
                # reflow）——用模型自己生成的 (x0, x1_hat) 配对重新回归，而不
                # 依赖 rollout 时真实执行的动作。跟 anchor 是两个独立开关，同
                # 一套 hinge-squared 结构和 per-agent/全局聚合方式。
                reflow_coef = getattr(self.args, "cfm_reflow_coef", 0.0)
                if reflow_coef > 0:
                    mb_h = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])[mb_time_idx]  # [M,N,H]
                    with th.no_grad():
                        reflow_eps = th.randn(
                            mb_h.shape[0], mb_h.shape[1], self.n_actions, device=mb_h.device
                        )
                        if self.ader_enabled:
                            # 用这个 train() 调用期间冻结的同一个 k（rollout
                            # 采样时用的那个），不是重新采一份独立的 N(0,I)
                            # ——reflow 的 base 分布跟当前 batch 的 flow base
                            # 分布是同一个，见类顶部的说明。
                            k_reflow = self.mac.get_ader_k(mb_h.device).view(1, self.n_agents, 1)
                            reflow_eps = reflow_eps * k_reflow
                        x1_hat = self.mac.integrate(
                            mb_h, reflow_eps, getattr(self.args, "cfm_rollout_steps", 10)
                        )
                    reflow_t = th.rand(mb_h.shape[0], mb_h.shape[1], 1, device=mb_h.device)
                    x_t_reflow = (1 - reflow_t) * reflow_eps + reflow_t * x1_hat
                    v_pred_reflow = self.mac.velocity(mb_h, x_t_reflow, reflow_t)
                    target_reflow = x1_hat - reflow_eps
                    reflow_sqerr_per_agent = (
                        (v_pred_reflow - target_reflow) ** 2
                    ).mean(dim=-1).mean(dim=0)  # [N]
                    reflow_loss_mean = reflow_sqerr_per_agent.mean()
                    reflow_threshold = getattr(self.args, "cfm_reflow_threshold", 1.5)
                    if self.anchor_per_agent:
                        reflow_excess_per_agent = th.clamp(
                            reflow_sqerr_per_agent - reflow_threshold, min=0.0
                        )
                        reflow_loss = (
                            reflow_coef * reflow_excess_per_agent ** 2
                        ).sum()
                    else:
                        reflow_excess = th.clamp(reflow_loss_mean - reflow_threshold, min=0.0)
                        reflow_loss = reflow_coef * reflow_excess ** 2
                else:
                    reflow_loss_mean = th.zeros((), device=mb_cfm_loss.device)
                    reflow_loss = th.zeros((), device=mb_cfm_loss.device)

                if self.old_mac is not None:
                    pfo_penalty = self._pfo_penalty(batch, h_seq, h_old_seq, mb_time_idx)
                    pfo_loss = self.pfo_coef * pfo_penalty
                else:
                    pfo_penalty = th.zeros((), device=mb_cfm_loss.device)
                    pfo_loss = pfo_penalty

                actor_loss = pg_loss + anchor_loss + reflow_loss + pfo_loss

                # ---- 优化器步骤 ----
                if self.actor_optimisers is not None:
                    for opt in self.actor_optimisers:
                        opt.zero_grad()
                    actor_loss.backward()
                    grad_norm = th.stack([
                        th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
                        for params in self.actor_params_per_agent
                    ]).mean()   # 记录成 N 个独立算出来的范数的均值
                    for opt in self.actor_optimisers:
                        opt.step()
                else:
                    self.actor_optimiser.zero_grad()
                    actor_loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self.actor_params, self.args.grad_norm_clip
                    )
                    self.actor_optimiser.step()

                actor_stats["cfm_loss_mean"].append(mb_cfm_loss.mean().item())
                # 只统计 PPO clip 真正改变了梯度的样本，不是整批落在区间外的
                # 比例：A>0 时只有 rho>clip_hi 生效（rho<clip_lo 那侧
                # min(surr1,surr2) 恒选未截断的 surr1）；A<0 时只有 rho<clip_lo
                # 生效。aspo 下 A<=0 走 SPO，跟 [clip_lo,clip_hi] 无关，所以
                # 只算 A>0 那一半。
                with th.no_grad():
                    clipped = (mb_advantages > 0) & (mb_rho_s > clip_hi)
                    if self.trust_region == "ppo":
                        clipped = clipped | ((mb_advantages < 0) & (mb_rho_s < clip_lo))
                actor_stats["ppo_clip_fraction"].append(clipped.float().mean().item())
                actor_stats["actor_grad_norm"].append(grad_norm.item())
                actor_stats["pfo_penalty"].append(pfo_penalty.item())
                actor_stats["pg_loss"].append(pg_loss.item())

        sn, theta_disp, cos = self._track_theta()
        actor_stats["theta_step_norm"].append(sn)
        actor_stats["theta_step_cos"].append(cos)

        # ── ADER：本次 train() 调用的所有 actor epoch/minibatch 到这里才算
        # 真正结束——k 全程冻结到此为止，g 转成新 k 的更新只在这一处发生一
        # 次，下一批 rollout 才会用上新值。fixed 模式下 _update_ader_k 只
        # 更新 beta_ema 这个统计量，不碰 mac.ader_k。
        ader_diag = None
        if self.ader_enabled:
            ader_diag = self._update_ader_k(ader_g)

        # ---- 日志：只留最关键的几个 ----
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for key in ["critic_loss", "value_mean"]:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            for key in ["cfm_loss_mean", "ppo_clip_fraction", "actor_grad_norm", "pg_loss"]:
                self.logger.log_stat(key, self._mean_stat(actor_stats[key]), t_env)
            for key, value in x1_stats.items():
                self.logger.log_stat(key, value.item(), t_env)
            for key in ["theta_step_norm", "theta_step_cos"]:
                self.logger.log_stat(key, self._mean_stat(actor_stats[key]), t_env)

            # cfm_loss_clip_fraction / action_std / action_raw_std 以前只在
            # _log_ader_stats() 里记，而那个函数只在 ader_enabled=True 时调用，
            # 于是所有 baseline run 都没有这三条。它们跟 ADER 没有任何关系：
            #   cfm_loss_clip_fraction —— 多少比例的探测点 CFM loss 已经撞到
            #     cfm_loss_clip_max。那是个普通 clamp（梯度真的断掉），所以撞
            #     顶的点既不再被推开也不再被拉回，是优化器的盲区；这个比例是
            #     判断"尾部是否已经脱离控制"的直接指标。
            #   action_std / action_raw_std —— 动作分布的宽度。中心（均值）漂
            #     没漂 msd_probe 能看，宽度只有这里能看。
            # 无条件记录，跟 ader_enabled 解耦。
            clip_frac_list = actor_stats.get("cfm_loss_clip_fraction", [])
            if clip_frac_list:
                clip_frac_mean = th.stack(clip_frac_list).mean(dim=0)   # [N]
                self.logger.log_stat(
                    "cfm_loss_clip_fraction", clip_frac_mean.mean().item(), t_env
                )
            for _name in ("action_std", "action_raw_std"):
                _v = self._ader_log_cache.get(_name)
                if _v is not None:
                    self.logger.log_stat(_name, _v.mean().item(), t_env)
            if theta_disp is not None:
                self.logger.log_stat("theta_disp_norm", theta_disp, t_env)
            # W 有没有被 decay 管住：actor 全部参数的 L2 范数（绝对量，跟
            # theta_disp_norm 那种相对初始点的位移互补），以及 SWD 实际用的
            # decay 尺度 sqrt(v_bar)（越小说明 AdamW 式的 decay 在这里越无效）。
            with th.no_grad():
                self.logger.log_stat(
                    "actor_param_norm",
                    th.sqrt(sum((p.detach().float() ** 2).sum() for p in self.mac.parameters())).item(),
                    t_env,
                )
            _actor_opts = self.actor_optimisers or [self.actor_optimiser]
            _v_bars = [o.last_v_bar_sqrt for o in _actor_opts if hasattr(o, "last_v_bar_sqrt")]
            if _v_bars:
                self.logger.log_stat("swd_v_bar_sqrt", sum(_v_bars) / len(_v_bars), t_env)
            if self.old_mac is not None:
                self.logger.log_stat("pfo_penalty", self._mean_stat(actor_stats["pfo_penalty"]), t_env)
            if actor_stats["x1_pred_kl"]:
                self.logger.log_stat("x1_pred_kl", self._mean_stat(actor_stats["x1_pred_kl"]), t_env)
                self.logger.log_stat("actor_lr", self._mean_stat(actor_stats["actor_lr"]), t_env)
            if self.ader_enabled:
                self._log_ader_stats(t_env, actor_stats, ader_diag)
            self.log_stats_t = t_env

    def _ader_pathwise_score(
        self, batch, a_team, mask, valid_time_indices, chunk_size,
        initial_cfm_loss, rho_clip,
    ):
        """ADER 的 pathwise（reparameterization）估计：
            g_i = -dL_MAFPO / dlog s_i = dJ / dlog s_i，
            J(s) = mean[ rho(s) * A ]，rho(s) = exp(C(L_old) - C(L_new(s)))，
        theta 固定在 rollout policy（本函数在任何 actor 更新之前调用），动作
        a 固定，只有 CFM 探测噪声写成 s 的函数：eps(s) = z_cfm * s，
        z_cfm = 存下来的 cfm_eps / flow_k。于是
            x_t = (1-t) eps(s) + t a,   u = a - eps(s),
            dl/dlog s = s * 2 e^T [ (1-t) J_v z - z ]   （由 autograd 完成）
        在 s = k_old 处 L_new == L_old、rho == 1，PPO clip 不生效，所以
        dJ/dlog s_i = mean_{agent i 的条目}[ A * (-dl/dlog s_i) ]：
        优势为正的样本若噪声放大后回归误差反而变小（更"像"当前策略）就推
        高 s_i，反之压低。跟 score-function 版（||z||^2 - d）相比，这是
        利用了 CFM 结构的低方差路径梯度，代价是继承了 FPO 用 CFM loss 代
        替 log-prob 的近似（少了 base 分布归一化项 -d*log s）。

        梯度只对 log_s 这个叶子求（autograd.grad），不会往 theta 的 .grad
        里累积。按 chunk_size 分块以控制显存。返回 [N]，已 detach。"""
        n = self.n_agents
        k_old = self.mac.get_ader_k(a_team.device).detach()
        log_s = th.log(k_old).clone().requires_grad_(True)         # [N]

        with th.no_grad():
            h_seq = self._build_actor_hidden_sequence(batch)
        eps_all = batch["cfm_eps"][:, :-1].reshape(
            -1, n, self.args.cfm_n_samples, self.args.cfm_action_dim
        )
        k_all = batch["flow_k"][:, :-1].float().reshape(-1, n, 1, 1)   # rollout 时的 k
        z_all = eps_all / k_all.clamp(min=1e-6)                        # 还原标准高斯探测点
        a_flat = a_team.reshape(-1, n)                                  # [B*T, N]
        old_flat = initial_cfm_loss.reshape(-1, n, self.args.cfm_n_samples, 1)
        cfm_loss_clip_max = getattr(self.args, "cfm_loss_clip_max", 20.0)

        total = th.zeros((), device=a_team.device)
        count = 0
        for start in range(0, valid_time_indices.numel(), chunk_size):
            idx = valid_time_indices[start:start + chunk_size]
            s = th.exp(log_s).view(1, n, 1, 1)
            eps_s = z_all[idx] * s                                      # [M,N,K,A]，带 log_s 的图
            v_pred, target, _ = self._velocity_for_time_indices(
                batch, h_seq, idx, eps_override=eps_s
            )
            new_loss = self.mac.cfm_error(v_pred, target)               # [M,N,K,1]
            diff = (
                th.clamp(old_flat[idx], max=cfm_loss_clip_max)
                - th.clamp(new_loss, max=cfm_loss_clip_max)
            ).squeeze(-1)                                               # [M,N,K]
            rho = th.exp(self._ste_clamp(diff, -float("inf"), rho_clip))
            adv = a_flat[idx].unsqueeze(-1).expand_as(rho)
            total = total + (rho * adv).sum()
            count += rho.numel()
        j = total / max(count, 1) * n      # 每个 agent 自己那 1/N 的条目，均值按 agent 归一
        (g,) = th.autograd.grad(j, log_s)
        return g

    def _update_ader_k(self, g: th.Tensor) -> dict:
        """g: [N]，score-function 估计 dJ/dlog(k_i)（已经 detach）。整段
        no_grad——k 从 g 到新值全是手写的标量公式，不是任何 optimizer 的
        step，也不建立/消费 autograd 图（禁止事项：不要用 SGD/Adam 更新
        k，不要对 eps=k*z / Euler integration 建立 k 的 autograd 路径）。

        ader_mode:
          - "adaptive"：g 在 agent 维标准化 → softmax → EMA 得 beta，再按固定
            variance budget 分配 k（只用 g 的相对排序，总方差守恒，N=1 时不动）。
          - "gradient"：把 g 真正当 dJ/dlog k_i 用，theta 固定（g 是在本次
            train() 改 theta 之前、用 rollout policy 的数据算的）：
                g_ema <- (1-a) g_ema + a g
                log k <- log k + clamp(ader_lr * g_ema, ±ader_max_log_k_step)
            每个 agent 独立升降，不做跨 agent 归一化，N=1 也能用。
          - "fixed"：k 不动。
        warmup（ader_update_count <= ader_warmup_updates）和 "fixed" 都只更新
        beta_ema / g_ema 这些统计量，不改 mac 里的 k；update_interval 控制"每
        隔几次完整 train() 调用才真正生效一次"，但从不在 minibatch/epoch 内部
        触发。"""
        with th.no_grad():
            device = g.device
            n = self.n_agents
            ema_alpha = getattr(self.args, "ader_ema_alpha", 0.05)
            self.ader_g_ema = self.ader_g_ema.to(device)
            self.ader_g_ema = (1 - ema_alpha) * self.ader_g_ema + ema_alpha * g

            if n > 1 and g.std(unbiased=False) > 1e-6:
                g_norm = (g - g.mean()) / (g.std(unbiased=False) + 1e-8)
            else:
                g_norm = th.zeros_like(g)
            g_norm = th.clamp(g_norm, -5.0, 5.0)

            temperature = getattr(self.args, "ader_temperature", 1.0)
            beta = th.softmax(g_norm / temperature, dim=0)

            self.beta_ema = self.beta_ema.to(device)
            self.beta_ema = (1 - ema_alpha) * self.beta_ema + ema_alpha * beta

            self.ader_update_count += 1

            k_old = self.mac.get_ader_k(device).clone()

            warmup_updates = getattr(self.args, "ader_warmup_updates", 10)
            update_interval = max(1, int(getattr(self.args, "ader_update_interval", 1)))
            do_update = (
                self.ader_mode in ("adaptive", "gradient")
                and self.ader_update_count > warmup_updates
                and self.ader_update_count % update_interval == 0
            )
            k_min = getattr(self.args, "ader_k_min", 0.5)
            k_max = getattr(self.args, "ader_k_max", 1.5)
            max_log_step = getattr(self.args, "ader_max_log_k_step", 0.05)

            if do_update and self.ader_mode == "gradient":
                ader_lr = float(getattr(self.args, "ader_lr", 1.0))
                delta_log_k = th.clamp(ader_lr * self.ader_g_ema, -max_log_step, max_log_step)
                k_new = th.clamp(th.exp(th.log(k_old) + delta_log_k), k_min, k_max)
            elif do_update:
                ader_k_init = float(getattr(self.args, "ader_k_init", 1.0))
                alloc_strength = getattr(self.args, "ader_allocation_strength", 0.25)

                # 固定 variance budget（clamp 之前）：
                # sum_i k_candidate_i^2 = N * ader_k_init^2。
                k_candidate = ader_k_init * th.sqrt(
                    (1 - alloc_strength) + alloc_strength * n * self.beta_ema
                )
                k_candidate = th.clamp(k_candidate, k_min, k_max)

                # 单次更新的 log-k 步长上限，防止一次更新跳太远。
                delta_log_k = th.clamp(
                    th.log(k_candidate) - th.log(k_old), -max_log_step, max_log_step
                )
                k_new = th.exp(th.log(k_old) + delta_log_k)
                k_new = th.clamp(k_new, k_min, k_max)
            else:
                k_new = k_old

            self.mac.set_ader_k(k_new)

            # 诊断：k_old -> k_new 这一步的 Gaussian base KL，只用来衡量更
            # 新幅度，不能跟 FPO CFM ratio 简单相加当成完整 action-policy
            # KL（theta 和 k 同时变化时两者一般不可加，见类外的说明）。
            action_dim = self.n_actions
            kl = action_dim / 2.0 * (
                (k_old ** 2) / (k_new ** 2) - 1 - th.log((k_old ** 2) / (k_new ** 2))
            )

            return {
                "g": g,
                "g_ema": self.ader_g_ema.clone(),
                "beta_ema": self.beta_ema.clone(),
                "k_new": k_new,
                "kl": kl,
            }

    def _log_ader_stats(self, t_env, actor_stats, ader_diag):
        g = ader_diag["g"]
        beta_ema = ader_diag["beta_ema"]
        k = ader_diag["k_new"]
        kl = ader_diag["kl"]
        z_sq_mean = self._ader_log_cache.get("ader_z_sq_mean")
        action_std = self._ader_log_cache.get("action_std")
        action_raw_std = self._ader_log_cache.get("action_raw_std")

        clip_frac_list = actor_stats.get("cfm_loss_clip_fraction", [])
        clip_frac_mean = (
            th.stack(clip_frac_list).mean(dim=0) if clip_frac_list
            else th.zeros(self.n_agents, device=k.device)
        )

        for i in range(self.n_agents):
            self.logger.log_stat(f"ader_k_agent_{i}", k[i].item(), t_env)
            self.logger.log_stat(f"ader_score_agent_{i}", g[i].item(), t_env)
            self.logger.log_stat(f"ader_score_ema_agent_{i}", ader_diag["g_ema"][i].item(), t_env)
            g_sf = self._ader_log_cache.get("ader_g_sf")
            if g_sf is not None:
                self.logger.log_stat(f"ader_score_sf_agent_{i}", g_sf[i].item(), t_env)
            self.logger.log_stat(f"ader_beta_agent_{i}", beta_ema[i].item(), t_env)
            if z_sq_mean is not None:
                self.logger.log_stat(f"ader_z_sq_mean_agent_{i}", z_sq_mean[i].item(), t_env)
            if action_std is not None:
                self.logger.log_stat(f"action_std_agent_{i}", action_std[i].item(), t_env)
            if action_raw_std is not None:
                self.logger.log_stat(f"action_raw_std_agent_{i}", action_raw_std[i].item(), t_env)
            self.logger.log_stat(
                f"cfm_loss_clip_fraction_agent_{i}", clip_frac_mean[i].item(), t_env
            )

        self.logger.log_stat("ader_k_mean", k.mean().item(), t_env)
        self.logger.log_stat("ader_k_min", k.min().item(), t_env)
        self.logger.log_stat("ader_k_max", k.max().item(), t_env)
        self.logger.log_stat("ader_base_kl_mean", kl.mean().item(), t_env)
        self.logger.log_stat("ader_base_kl_max", kl.max().item(), t_env)
        self.logger.log_stat("ader_update_count", self.ader_update_count, t_env)

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch, mac=None) -> th.Tensor:
        """把 actor 的编码器沿整条 episode 滚一遍。返回 [B,T,N,hidden_dim]。
        mac 默认是当前策略；PFO 需要用老策略再滚一遍，所以可以显式传入。"""
        mac = self.mac if mac is None else mac
        h_list = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = mac.forward(batch, t=t)
            h_list.append(h)
        return th.stack(h_list, dim=1)

    def _compute_cfm_loss(self, batch: EpisodeBatch, h_seq: th.Tensor) -> th.Tensor:
        """整条 episode 上的 CFM 回归 loss（train() 实际调用的是
        _compute_cfm_loss_for_time_indices，这个方法没被用到，留着对照参考）。

        插值目标用 action_raw（sigmoid 之前的无界积分终点），不是执行动作——
        见 mafpo_actor.py/mafpo_mac.py。返回 [B,T,N,cfm_n,1]。"""
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

        return self.mac.cfm_error(v_pred, target)

    def _adapt_lr(self, kl: float, desired_kl: float):
        """官方 FPO.update() 的 adaptive 分支，逐字对应：
        kl > 2*desired -> lr/1.5（下限 1e-5）；0 < kl < desired/2 -> lr*1.5
        （上限 1e-2）。官方 actor+critic 共用一个优化器，这里 actor（逐 agent
        或联合）和 critic 的优化器一起改。"""
        if kl > desired_kl * 2.0:
            self.current_lr = max(1e-5, self.current_lr / 1.5)
        elif 0.0 < kl < desired_kl / 2.0:
            self.current_lr = min(1e-2, self.current_lr * 1.5)
        optimisers = list(self.actor_optimisers or []) + (
            [self.actor_optimiser] if self.actor_optimiser is not None else []
        ) + [self.critic_optimiser]
        for opt in optimisers:
            for group in opt.param_groups:
                group["lr"] = self.current_lr

    def _velocity_for_time_indices(
        self, batch: EpisodeBatch, h_seq: th.Tensor, flat_time_indices: th.Tensor,
        eps_override: th.Tensor = None,
    ):
        """给平铺 (episode,time) 下标算探测点上的速度预测和回归目标。返回
        (v_pred [M,N,cfm_n,A], target [M,N,cfm_n,A], cfm_t [M,N,cfm_n,1])。
        eps_override（[M,N,cfm_n,A]，可带 autograd 图）替换 buffer 里存的
        cfm_eps——ADER pathwise 估计用它把探测噪声写成 s 的函数。"""
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
        mb_eps = eps[flat_time_indices] if eps_override is None else eps_override  # [M,N,cfm_n,A]
        mb_cfm_t = cfm_t[flat_time_indices]           # [M,N,cfm_n,1]
        mb_h = h[flat_time_indices]                   # [M,N,H]

        act_exp = mb_action.unsqueeze(2).expand_as(mb_eps)
        x_t = (1 - mb_cfm_t) * mb_eps + mb_cfm_t * act_exp
        h_exp = mb_h.unsqueeze(2).expand(-1, -1, mb_eps.size(2), -1)

        v_pred = self.mac.velocity(h_exp, x_t, mb_cfm_t)
        self._last_probe = (x_t, mb_cfm_t)            # 供 PFO 复用同一组探测点

        cfm_target_type = getattr(self.args, "cfm_target_type", "velocity")
        if cfm_target_type == "velocity":
            target = act_exp - mb_eps
        elif cfm_target_type == "eps":
            target = mb_eps
        else:
            raise ValueError("cfm_target_type must be 'velocity' or 'eps'")
        return v_pred, target, mb_cfm_t

    def _compute_cfm_loss_for_time_indices(
        self, batch: EpisodeBatch, h_seq: th.Tensor, flat_time_indices: th.Tensor
    ):
        """给一个 minibatch 的平铺 (episode,time) 下标算 CFM 回归 loss，保留
        完整的 agent 维。跟 _compute_cfm_loss 一样用 action_raw 插值。误差函数
        是 mac.cfm_error()——跟 rollout 时算 initial_cfm_loss 的是同一个（Huber
        delta / reduction 配置只在那一处读），否则 old/new 不可比。返回
        (loss [M,N,cfm_n,1], v_pred [M,N,cfm_n,A], cfm_t [M,N,cfm_n,1])。"""
        v_pred, target, mb_cfm_t = self._velocity_for_time_indices(
            batch, h_seq, flat_time_indices
        )
        return self.mac.cfm_error(v_pred, target), v_pred, mb_cfm_t

    def _pfo_penalty(self, batch, h_seq, h_old_seq, flat_time_indices):
        """||phi_theta - phi_theta_old||^2，在这个 minibatch 的 CFM 探测点上算。

        探测点 (x_t, t) 直接复用 _velocity_for_time_indices 刚构造的那一组
        （self._last_probe），保证 PFO 惩罚和策略损失看的是同一批位置。两边的
        h 各自用自己的参数前滚——这正是原文 phi_theta(S_t) vs phi_theta_old(S_t)
        的含义：同样的状态，不同的参数。沿特征维求平方和（原文的 L2 范数平方），
        再对 (minibatch, agent, 探测点) 取均值。
        """
        x_t, mb_cfm_t = self._last_probe
        n_probe = x_t.shape[2]
        h_new = h_seq.reshape(-1, self.n_agents, h_seq.shape[-1])[flat_time_indices]
        h_new = h_new.unsqueeze(2).expand(-1, -1, n_probe, -1)
        f_new = self.mac.velocity_features(h_new, x_t, mb_cfm_t)
        with th.no_grad():
            h_old = h_old_seq.reshape(-1, self.n_agents, h_old_seq.shape[-1])[flat_time_indices]
            h_old = h_old.unsqueeze(2).expand(-1, -1, n_probe, -1)
            f_old = self.old_mac.velocity_features(h_old, x_t, mb_cfm_t)
        return ((f_new - f_old) ** 2).sum(dim=-1).mean()

    def _compute_advantages_and_targets(self, batch, rewards, mask):
        """GAE + lambda-return，整个 train() 调用只算一次，后面所有 epoch/
        minibatch 复用（不是每个 epoch 重算）。

        value baseline 用当前（rollout 结束时那份）critic，没有 target
        网络——critic 本身直接朝 lambda-return 回归。advantage 在整个 rollout
        的有效 mask 内做 normalize（均值 0、标准差 1）再 clamp 到
        [-adv_clip, adv_clip]，然后才交给 actor。

        self.critic(batch) 对 agent-id-conditioned 的 critic 是
        [B,T+1,N,1]（squeeze(3) 变成 [B,T+1,N]），对单一联合 V(s) 已经是
        [B,T+1,1]。后面的 GAE/normalize/clamp 都是对最后一维通用的，两种情况
        代码不用分叉。

        返回 (advantages, target_returns, v_old, ader_advantage)：v_old 是
        destandardise 之前的原始 critic 输出（[B,T,W]，T 维已经切掉
        bootstrap 那一步），供 value clip 当锚点用，见
        _critic_gradient_step；ader_advantage 是专门留给 ADER score 用的另
        一份快照——normalize 之后、adv_clip 之前、已经乘过 mask、且
        detach，跟下面 actor 真正用的 advantages（还要再过 adv_clip）是两
        份数据。"""
        with th.no_grad():
            values = self.critic(batch)
            if self.per_agent_values:
                values = values.squeeze(3)   # [B,T+1,N,1] -> [B,T+1,N]
            # v_old：这个 train() 调用一开始、epoch 循环之前那份 critic 原始
            # 输出（还没做下面的 destandardise），跟后面 epoch 里
            # _critic_gradient_step 算出来的 v 是同一个 scale——留给 value
            # clip 当"更新前"的锚点用，见 _critic_gradient_step。
            v_old = values[:, :-1].clone()

        if self.args.standardise_returns:
            values = values * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        terminated = batch["terminated"][:, :-1].float()   # [B, T, 1]
        gae_lambda = getattr(self.args, "gae_lambda", 0.95)
        advantages = self.compute_gae(
            rewards, mask, values, terminated, self.args.gamma, gae_lambda
        )                                                   # [B, T, advantage_width], masked
        target_returns = advantages + values[:, :-1]        # critic 要回归的 lambda-return

        valid = mask.bool()
        if valid.any():
            adv_valid = advantages[valid]
            adv_mean = adv_valid.mean()
            adv_std = adv_valid.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        ader_advantage = (advantages * mask).detach()
        adv_clip = getattr(self.args, "adv_clip", 5.0)
        advantages = th.clamp(advantages, -adv_clip, adv_clip) * mask

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        return advantages, target_returns, v_old, ader_advantage

    def _critic_gradient_step(self, critic, batch, target_returns, mask, v_old):
        """对固定的 target_returns（整个 train() 调用只算一次）做一次 critic
        梯度更新——每个 epoch 调一次。

        value_clip_eps > 0 时做 PPO2 式 value clip：v 相对 v_old（这个
        train() 调用开始时的快照，epoch 间不变）的位移被夹到
        [-value_clip_eps, value_clip_eps]，loss 取"未裁剪"和"裁剪后"两个
        squared error 里更大的那个——本轮里 v 想往哪个方向跑多远都不受
        target_returns 直接限制，唯一的约束就是这个位移上限，跟 actor 侧
        eps_clip 的 trust-region 思路是一回事，只是作用对象从 ratio 换成
        v 本身。value_clip_eps<=0 时完全退化成原来的纯 MSE（不引入任何行
        为变化）。"""
        running_log = {k: [] for k in ["critic_loss", "value_mean"]}

        v = critic(batch)[:, :-1]
        if self.per_agent_values:
            v = v.squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask

        value_clip_eps = getattr(self.args, "value_clip_eps", 0.0)
        if value_clip_eps > 0:
            v_clipped = v_old + th.clamp(v - v_old, -value_clip_eps, value_clip_eps)
            loss_unclipped = (v - target_returns.detach()) ** 2
            loss_clipped = (v_clipped - target_returns.detach()) ** 2
            loss_per_elem = th.max(loss_unclipped, loss_clipped)
            loss = (loss_per_elem * mask).sum() / mask.sum()
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()
        running_log["critic_loss"].append(loss.item())
        running_log["value_mean"].append((v * mask).sum().item() / mask_elems)
        return running_log

    def compute_gae(self, rewards, mask, values, terminated, gamma, gae_lambda):
        """GAE advantage 估计（反向递推）。对最后一维通用：不管这一维是 N
        （按 agent 拆开的 advantage）还是 1（单一共享 advantage）都不用改代码。

        rewards:    [B, T, W]   （W = advantage_width，N 或 1）
        mask:       [B, T, W]
        values:     [B, T+1, W]  rollout 时的 critic 输出（含 T 处的 bootstrap）
        terminated: [B, T, 1]    该 step episode 是否结束
        返回:       advantages [B, T, W]，无效 step 处为 0
        """
        T = rewards.size(1)
        gae = th.zeros_like(values[:, 0])    # [B, W]
        advantages = th.zeros_like(rewards)   # [B, T, W]

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - terminated[:, t]   # [B, 1]，广播到 W
            delta = (rewards[:, t]
                     + gamma * values[:, t + 1] * next_non_terminal
                     - values[:, t])
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[:, t] = gae

        return advantages * mask

    @staticmethod
    def _ste_clamp(x: th.Tensor, min_val: float, max_val: float) -> th.Tensor:
        """Straight-through clamp。前向数值就是
        th.clamp(x, min_val, max_val)；反向梯度按恒等函数穿透（处处
        d(output)/d(x) = 1，包括 [min_val, max_val] 之外的部分），而不是
        th.clamp 自己的反向（区间内是 1，区间外是 0）。
        x + (clamp(x) - x).detach() 就能做到这一点：detach 掉的那部分不贡献
        梯度，autograd 只看得到 `x` 这一项；前向数值不受影响，因为
        x + (clamp(x) - x) 恒等于 clamp(x)。"""
        clamped = th.clamp(x, min_val, max_val)
        return x + (clamped - x).detach()

    def _mean_stat(self, values):
        return sum(values) / max(1, len(values))

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        if self.old_mac is not None:
            self.old_mac.cuda()

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
        if self.ader_enabled:
            th.save(
                {
                    "k": self.mac.get_ader_k().detach().cpu(),
                    "beta_ema": self.beta_ema.detach().cpu(),
                    "update_count": self.ader_update_count,
                },
                "{}/ader_state.th".format(path),
            )

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path),
                    map_location=lambda storage, loc: storage))
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
        self.current_lr = float(self.critic_optimiser.param_groups[0]["lr"])
        # ADER 状态是单独的文件——旧 checkpoint（在这个功能加进来之前存的）
        # 没有它是预期情况，不报错，直接保持配置初始化出来的值，向后兼容。
        ader_path = "{}/ader_state.th".format(path)
        if self.ader_enabled and os.path.exists(ader_path):
            state = th.load(ader_path, map_location=lambda storage, loc: storage)
            self.mac.set_ader_k(state["k"])
            self.beta_ema = state["beta_ema"].to(self.beta_ema.device)
            self.ader_update_count = state["update_count"]
