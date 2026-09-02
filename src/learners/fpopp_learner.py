import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class FPOPPLearner:
    """连续动作（Box(0,1)）多智能体 flow 策略的 FPO++ learner。

    Ratio：没有闭式的 log-prob，用 CFM 回归 loss 当 ELBO 的代理。对一笔
    transition 上采样的 cfm_n 个 (tau_i, eps_i) 探测点，每个点各自算一个 ratio：
        rho_i = exp(clamp(L_old^(i) - L_new^(i), -inf, rho_clip))
    逐点算，不先对 cfm_n 取平均（exp 是非线性的，先平均会让符号相反的点互相抵
    消，在 ratio 看到它们之前就丢掉了信息）。只在最后对 (minibatch, cfm_n) 一
    起取平均得到 surrogate。

    Surrogate：A>=0 用标准 PPO clip；A<0 用平滑的 SPO 目标替代 PPO 硬 clip：
        psi_SPO(rho, A) = rho*A - (|A| / (2*eps_clip)) * (rho-1)^2
    PPO 的 clip 在 rho 越出边界后梯度恒为 0；SPO 在 rho=1 处梯度跟未裁剪的策略
    梯度一致，越出边界后梯度平滑衰减而不是直接归零，所以一个已经偏得很远的样
    本仍然会被往回拉，而不是彻底失活。

    Ratio 的 log-diff（算完 diff 之后）用 straight-through clamp：前向数值被
    截断到 rho_clip，反向梯度按恒等函数穿透，一个已经越界的点仍然带着把它往
    回拉的梯度，不会彻底失活（见 _ste_clamp()）。

    但给 diff 喂数据的 old/new CFM loss 各自的上界 clamp（cfm_loss_clip_max）
    刻意用的是普通 clamp，不是 STE：mb_cfm_loss 是 MSE，对 v_pred 的原始梯度
    正比于残差、没有上界。如果这里也用 STE，一旦 old/new 同时超过上限，
    diff=C_L-C_L=0，rho=1，ratio 本身看起来毫无异常，但未截断的原始 MSE 梯度
    会绕过这层"看起来正常"的伪装直接传回 v_pred——残差越离谱梯度越大，等于
    让一个已经烂掉的点在诊断量完全看不出来的情况下主导更新。这里就是要让梯
    度真的断掉：C_L 是"离谱到不该再提供学习信号"的安全阀，跟 log-diff 那层
    STE clamp（刻意保留梯度）解决的是两个不同的问题，不能合并成一步。

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
                Adam(params=p, lr=args.lr) for p in self.actor_params_per_agent
            ]
            self.actor_params = None
            self.actor_optimiser = None
        else:
            self.actor_params_per_agent = None
            self.actor_optimisers = None
            self.actor_params = list(mac.parameters())
            self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr)

        # anchor/reflow 聚合方式：按 agent 拆开求和，还是对全局均值算一个。跟
        # optim_per_agent 相互独立。
        self.anchor_per_agent = getattr(args, "fpo_anchor_per_agent", individual_agents)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

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

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.advantage_width,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

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
        }
        critic_train_stats = {k: [] for k in ["critic_loss", "value_mean"]}

        # GAE + lambda-return：整个 train() 调用只算一次，后面所有 epoch/
        # minibatch 都固定用这一份（critic 每个 epoch 仍然做一次梯度更新，但
        # 用的都是这同一份固定 target_returns，不重新算 GAE）。
        advantages, target_returns, v_old = self._compute_advantages_and_targets(
            batch, rewards, critic_mask
        )
        advantages = advantages.detach()
        target_returns = target_returns.detach()
        v_old = v_old.detach()

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
                mb_cfm_loss = self._compute_cfm_loss_for_time_indices(
                    batch, h_seq, mb_time_idx
                )
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

                # ---- ratio：old/new CFM loss 各自先夹一个上界 cfm_loss_clip_max ----
                # 两边都用普通 clamp，不是 STE。old_cfm_loss_c 是 rollout 时存
                # 的无梯度 buffer 数据，STE 与否对它无所谓。new_cfm_loss_c 不能
                # 用 STE：mb_cfm_loss 是 MSE，其对 v_pred 的原始梯度正比于残
                # 差、没有上界；一旦 old/new 同时超过上限，diff = C_L - C_L = 0，
                # rho=exp(0)=1，ratio 和 ppo_clip_fraction 这些诊断量看起来完全正
                # 常，但如果这里用 STE，未截断的原始 MSE 梯度会绕过这层伪装直
                # 接传回 v_pred——残差越离谱这个梯度越大，等于让一个已经烂掉的
                # 点在"看起来人畜无害"的掩护下主导更新，恰好是这个 clamp 本该
                # 防住的问题。普通 clamp 在这里就是要让梯度真的断掉：C_L 是"离
                # 谱到不该再提供学习信号"的安全阀，这跟下面改动 4 的 STE clamp
                # （那里刻意保留梯度）解决的是两个不同的问题，不要合并。
                cfm_loss_clip_max = getattr(self.args, "cfm_loss_clip_max", 20.0)
                old_cfm_loss_c = th.clamp(mb_initial_cfm_loss, max=cfm_loss_clip_max)
                new_cfm_loss_c = th.clamp(mb_cfm_loss, max=cfm_loss_clip_max)

                # 逐点 log-ratio，保留 cfm_n 维不 reduce（原因见类 docstring）。
                diff = (old_cfm_loss_c - new_cfm_loss_c).squeeze(-1)   # [M,N,cfm_n]

                # 对 ratio 本身再做一次 STE clamp，只截上界：diff 很负时 rho
                # 本来就会平滑趋近 0（没有爆炸风险），只有 diff 很正的一侧需要
                # 在 exp() 之前截住。
                diff = self._ste_clamp(diff, -float("inf"), rho_clip)
                mb_rho_s_pt = th.exp(diff)  # [M,N,cfm_n]

                cfm_n = mb_rho_s_pt.shape[-1]
                mb_advantages_pt = mb_advantages_2d.unsqueeze(-1).expand(-1, -1, cfm_n)  # [M,N,cfm_n]

                mb_rho_s = mb_rho_s_pt.reshape(-1)
                mb_advantages = mb_advantages_pt.reshape(-1)

                # ---- PPO/SPO surrogate：A>=0 用 PPO clip，A<0 用平滑 SPO ----
                # clip 边界用论文标准的线性 [1-eps, 1+eps]，不是 log 空间对称。
                clip_lo = 1.0 - self.args.eps_clip
                clip_hi = 1.0 + self.args.eps_clip
                neg_mask = mb_advantages < 0
                surr1 = mb_rho_s * mb_advantages
                surr2 = th.clamp(mb_rho_s, clip_lo, clip_hi) * mb_advantages
                ppo_surr = th.min(surr1, surr2)
                spo_surr = (
                    mb_rho_s * mb_advantages
                    - (mb_advantages.abs() / (2 * self.args.eps_clip))
                    * (mb_rho_s - 1) ** 2
                )
                surr = th.where(neg_mask, spo_surr, ppo_surr)
                # 只在这一步对 (minibatch, cfm_n) 一起取平均——ratio 和逐点
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

                actor_loss = pg_loss + anchor_loss + reflow_loss

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
                # ASPO 下只有 A>=0 且 rho>clip_hi 真正被 PPO 硬 clip 改变了梯度
                # （A>=0 时 rho<clip_lo 那侧 min(surr1,surr2) 恒选未截断的
                # surr1，clip 不生效；A<0 走的是 SPO，跟这个 [clip_lo,clip_hi]
                # 区间无关，越界不代表梯度被截断）——所以只统计这一个子集，不
                # 是整批样本落在区间外的比例。
                actor_stats["ppo_clip_fraction"].append(
                    (~neg_mask & (mb_rho_s > clip_hi)).float().mean().item()
                )
                actor_stats["actor_grad_norm"].append(grad_norm.item())

        # ---- 日志：只留最关键的几个 ----
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for key in ["critic_loss", "value_mean"]:
                self.logger.log_stat(key, self._mean_stat(critic_train_stats[key]), t_env)
            for key in ["cfm_loss_mean", "ppo_clip_fraction", "actor_grad_norm"]:
                self.logger.log_stat(key, self._mean_stat(actor_stats[key]), t_env)
            self.log_stats_t = t_env

    def _build_actor_hidden_sequence(self, batch: EpisodeBatch) -> th.Tensor:
        """把 actor 的编码器沿整条 episode 滚一遍。返回 [B,T,N,hidden_dim]。"""
        h_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            h = self.mac.forward(batch, t=t)
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

        return ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)

    def _compute_cfm_loss_for_time_indices(
        self, batch: EpisodeBatch, h_seq: th.Tensor, flat_time_indices: th.Tensor
    ) -> th.Tensor:
        """给一个 minibatch 的平铺 (episode,time) 下标算 CFM 回归 loss，保留
        完整的 agent 维。跟 _compute_cfm_loss 一样用 action_raw 插值。返回
        [M,N,cfm_n,1]。"""
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

        返回 (advantages, target_returns, v_old)：v_old 是 destandardise 之
        前的原始 critic 输出（[B,T,W]，T 维已经切掉 bootstrap 那一步），供
        value clip 当锚点用，见 _critic_gradient_step。"""
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
        adv_clip = getattr(self.args, "adv_clip", 5.0)
        advantages = th.clamp(advantages, -adv_clip, adv_clip) * mask

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        return advantages, target_returns, v_old

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
