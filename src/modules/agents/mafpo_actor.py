import torch as th
import torch.nn as nn
import torch.nn.functional as F

#------MAFPO 独立 Actor 网络（速度场）----------
# 从 GitHub origin/current-mafpo @ 0313c24 拉取而来，是这条独立演化线自己的
# actor——跟本仓库本地开发的 PolicyFlow 线（policyflow_actor.py）刻意分开维
# 护，互不影响。
#
# 2026-08-24 本地追加修改：hard clamp(x1,0,1) 换成 sigmoid(x1)。5M 步实测
# hard clamp 版本 1M 步时 action_at_bound_fraction=0.93、
# agents_all_same_action_fraction=0.77——individual actors 独立开了参数，但
# 硬边界本身仍然是噪声/未训练输出被大量钉死在 0/1 的机制（clamp 的导数在边
# 界外恒为 0，越界多远都一样，等价于把"离谱的输出"和"刚好压线的输出"混为一
# 谈）。换成 sigmoid 后不再有硬边界，越界越远梯度只是变小不是消失，同时把
# CFM 回归的插值目标从"sigmoid 之后的、被压缩过的 action"改回"sigmoid 之前
# 的无界 latent 终点 x1"（即下面 sample_action 新增返回的第 4 个值），避免
# 流本身的训练信号又在压缩边界上重新踩坑——这正是 policyflow_actor.py 当初
# 从 hard clamp 换成 sigmoid+无界 latent 时踩过的同一个坑，这里直接照搬结
# 论，不重新踩一遍。
#-----------------------------


class MAFPOActor(nn.Module):
    """独立 Actor 网络，输出速度场 v(h, x_t, t)，用于 CFM。

    结构:
        obs → fc1 → ReLU → GRU → h
        [h, x_t, embed_t(t)] → vel_fc1 → ReLU → vel_fc2 → velocity

    rollout 采样（K 步 Euler，从 t=0 积分到 t=1，K = args.cfm_rollout_steps）：
        x_0 = eps ~ N(0,I)
        x_{t+dt} = x_t + dt * v(h, x_t, t)
        action = sigmoid(x_1)      （全程无界 latent，sigmoid 是唯一的映射步）
    训练 CFM   : cfm_loss = ||v(h, x_t, t) - (x_1 - eps)||²   （插值目标是无界的
                 x_1，不是 sigmoid 之后的 action——见 mafpo_mac.py 里
                 _last_x1_raw 的存储/使用）
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        n_actions = args.n_actions

        # obs 时序编码器
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)

        # 时间嵌入维度。官方 isaaclab_fpo 的 timestep_embed_dim 默认 8，
        # manipulation 版的 SinusoidalPosEmb 是 32——两套实现没有任何一份把 t
        # 当裸标量喂进去。0 = 退回旧的裸标量行为（只为 A/B 对照保留）。
        self.t_embed_dim = int(getattr(args, "fpo_timestep_embed_dim", 8))
        assert self.t_embed_dim >= 0 and (self.t_embed_dim == 0 or self.t_embed_dim % 2 == 0), \
            f"fpo_timestep_embed_dim 必须是 0 或正偶数，得到 {self.t_embed_dim}"
        t_in = self.t_embed_dim if self.t_embed_dim > 0 else 1

        # 速度场 MLP: 输入 = [h, x_t, embed_t(t)]
        self.vel_fc1 = nn.Linear(hidden_dim + n_actions + t_in, hidden_dim)
        self.vel_fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    # ── obs 编码 ──────────────────────────────────────────────────────────────

    def encode(self, inputs, hidden_state):
        """obs → GRU → h，返回新 hidden state。"""
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        return h

    # ── 速度场预测 ────────────────────────────────────────────────────────────

    def embed_t(self, t):
        """Fourier 时间嵌入，等价于官方 isaaclab_fpo/modules/actor_critic.py 的
        _embed_timestep：freqs = 2^[0 .. d/2-1]，输出 [cos(t*f), sin(t*f)]。

        为什么必须要：速度场 v(h, x_t, t) 对 t 的依赖是 flow matching 的全部内
        容。裸标量 t 只有 1 维，过一层 ReLU 之后网络基本只能表达 t 的分段线性
        调制，K 步 Euler 积分就退化成近似"常速度场"的一次性映射，而
        ratio = exp(L_old - L_new) 整个建在这个残差上。官方两套实现、六份任务
        配置无一例外都用嵌入（isaaclab 8 维 Fourier，manipulation 32 维
        SinusoidalPosEmb）。

        注意 t 的约定：官方 t=1 是噪声端，我们 t=1 是动作端（见
        fpopp_learner.py 的 x_t 构造）。这只是 t <-> 1-t 的重标定，嵌入本身
        对两种约定同样有效，网络自己学得出来。

        t: [..., 1] -> [..., t_embed_dim]
        """
        if self.t_embed_dim <= 0:
            return t
        freqs = 2.0 ** th.arange(
            self.t_embed_dim // 2, device=t.device, dtype=t.dtype
        )
        scaled = t * freqs
        return th.cat([th.cos(scaled), th.sin(scaled)], dim=-1)

    def velocity(self, h, x_t, t):
        """速度场: [h, x_t, t] → v。

        h:   [..., hidden_dim]
        x_t: [..., n_actions]   插值点
        t:   [..., 1]           时间标量 0~1

        输出用 tanh 限幅到 [-cfm_velocity_bound, cfm_velocity_bound]——原样搬自
        policyflow_actor.py（见其 velocity() 的详细注释）。2026-08-24 教训：把
        CFM 回归目标从 hard-clamp 后的 action 换成无界的 x1_raw 之后，5M 步实
        测里 cfm_loss_mean 从第一个 checkpoint 的 ~2.5e6 一路飙到 2M 步时的
        ~2e21，训练整体崩掉——回归目标一旦无界，没有这层限幅速度场权重就会被
        推向发散，这不是可选项，是无界 latent 方案的必需搭档。
        """
        inp = th.cat([h, x_t, self.embed_t(t)], dim=-1)
        raw = self.vel_fc2(F.relu(self.vel_fc1(inp)))
        bound = getattr(self.args, "cfm_velocity_bound", 8.0)
        if bound <= 0:
            # <=0 关闭限幅：速度场直接用网络原始输出（官方 isaaclab 版没有任何限幅）
            return raw
        return bound * th.tanh(raw / bound)

    # ── MAC 兼容接口 ──────────────────────────────────────────────────────────

    def forward(self, inputs, hidden_state):
        """MAC 调用接口：返回 (h, h)，h 供 learner 计算 CFM loss。"""
        h = self.encode(inputs, hidden_state)
        return h, h

    # ── K 步 Euler 积分（t=0 → t=1）───────────────────────────────────────────

    def integrate(self, h, eps, n_steps):
        """K 步 Euler 积分，从 x_0=eps（t=0）走到 x_1（t=1）。

        h:       [..., hidden_dim]
        eps:     [..., n_actions]  x_0
        n_steps: int，积分步数（K=1 退化为原来的一步 Euler）

        返回 x_1 估计（未裁剪到 [0,1]）: [..., n_actions]
        """
        x = eps
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            x = x + dt * self.velocity(h, x, t)
        return x

    # ── rollout 动作采样（K 步 flow）────────────────────────────────────────

    def sample_action(self, inputs, hidden_state):
        """K 步 flow 采样。

        x_0 = eps ~ N(0,I)
        action = sigmoid(integrate(h, eps, K))

        返回: (action, h, eps, x1)
            action: [..., n_actions]  实际执行的动作，sigmoid(x1) ∈ (0,1)
            h:      [..., hidden_dim] 更新后的 hidden state
            eps:    [..., n_actions]  本次采样的噪声（供 initial_cfm_loss 用）
            x1:     [..., n_actions]  sigmoid 之前的无界积分终点，CFM 回归的
                    插值目标用它而不是 action 本身（见类 docstring）
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        eps = th.randn(*h.shape[:-1], n_act, device=h.device)
        n_steps = getattr(self.args, "cfm_rollout_steps", 1)
        x1 = self.integrate(h, eps, n_steps)
        action = th.sigmoid(x1)
        return action, h, eps, x1

#-----------------------------
