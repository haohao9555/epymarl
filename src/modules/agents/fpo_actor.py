import torch as th
import torch.nn as nn
import torch.nn.functional as F

#------新增：FPO 独立 Actor 网络（速度场）----------
#-----------------------------


class FPOActor(nn.Module):
    """独立 Actor 网络，输出速度场 v(h, x_t, t)，用于 CFM。

    结构:
        obs → fc1 → ReLU → GRU → h
        [h, x_t, t] → vel_fc1 → ReLU → vel_fc2 → velocity

    rollout 采样（K 步 Euler，从 t=0 积分到 t=1，K = args.cfm_rollout_steps）：
        x_0 = eps ~ N(0,I)
        x_{t+dt} = x_t + dt * v(h, x_t, t)
        action = sigmoid(x_1 + n)，n ~ N(0, sigma^2)（PolicyFlow 风格的可学习末端
                 噪声，见 sigma()；仅在非 test_mode 下叠加。全程无界 latent 空间，
                 sigmoid 是唯一映射进 (0,1) 的一步，取代原来的 hard clamp）
    训练 CFM   : cfm_loss = ||v(h, x_t, t) - (action_raw - eps)||²
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

        # 速度场 MLP: 输入 = [h, x_t, t]
        self.vel_fc1 = nn.Linear(hidden_dim + n_actions + 1, hidden_dim)
        self.vel_fc2 = nn.Linear(hidden_dim, n_actions)

        # PolicyFlow 风格的可学习末端噪声标准差(state-independent，仿照经典
        # 连续 PPO 的 log_std 做法)。sigma() 同时用于: (a) rollout 时在积分
        # 终点叠加真正的探索噪声，(b) Brownian regularizer 里的高斯熵项
        # w_g * H[N(0,sigma^2)]——sigma 必须真正参与采样，这个熵项才不是摆设。
        #
        # 参数化用 sigma = sigma_min + (sigma_max-sigma_min)*sigmoid(raw_sigma)，
        # 不用 exp(raw).clamp(min=sigma_min)：后者只有下限、没有上限，实测长跑
        # (20M 步)里 sigma_mean 从 0.27 一路单调涨到 0.9 都没有平台迹象——超过
        # 动作范围[0,1]宽度的一半后，噪声本身就足以主导贴边行为(clamp 人为制造
        # 贴边，跟训练早期 sigma_init 选太大是同一种机制，只是这次是训练过程中
        # 自己爬上去的，不是初始化的问题)。sigmoid 参数化天然把 sigma 约束在
        # [sigma_min, sigma_max]，两端梯度平滑趋于 0(饱和特性类似 tanh 限幅
        # velocity 那次的思路)，不会无限爬升。
        sigma_init = getattr(args, "sigma_init", 0.1)
        self.sigma_min = getattr(args, "sigma_min", 0.01)
        self.sigma_max = getattr(args, "sigma_max", 1.0)
        # 反解初始化: sigmoid(raw_init) = (sigma_init-sigma_min)/(sigma_max-sigma_min)
        p_init = (sigma_init - self.sigma_min) / (self.sigma_max - self.sigma_min)
        p_init = min(max(p_init, 1e-4), 1 - 1e-4)  # 避免 logit 在 0/1 处发散
        raw_init = th.log(th.tensor(p_init) / (1 - th.tensor(p_init))).item()
        self.raw_sigma = nn.Parameter(th.full((n_actions,), raw_init))

    def sigma(self):
        # 诊断用逃生舱: sigma_fixed_value 设置后直接返回常数，完全不碰
        # self.raw_sigma——不进计算图，梯度出不去、也进不来。用来隔离"σ数值
        # 大小"和"σ在训练中漂移"这两件事：固定在不同常数下对比
        # oscillation_fraction，纯粹看数值大小的影响，不涉及学习动态。
        fixed = getattr(self.args, "sigma_fixed_value", None)
        if fixed is not None:
            return th.full(
                (self.args.n_actions,), float(fixed), device=self.raw_sigma.device
            )
        return self.sigma_min + (self.sigma_max - self.sigma_min) * th.sigmoid(
            self.raw_sigma
        )

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

    def velocity(self, h, x_t, t):
        """速度场: [h, x_t, t] → v。

        h:   [..., hidden_dim]
        x_t: [..., n_actions]   插值点
        t:   [..., 1]           时间标量 0~1

        输出用 tanh 限幅到 [-cfm_velocity_bound, cfm_velocity_bound]：CFM loss
        对 advantage<0 的样本会主动推高 loss（让 v_pred 远离 target），这个方向
        本身没有上界，长期训练会把 vel_fc2 的权重推到发散。限幅之后 v_pred 出不
        了这个范围，cfm_loss 天然有上限，同时饱和区梯度趋于 0，权重也不会被继
        续无限推大。

        用 bound*tanh(raw/bound) 而不是 bound*tanh(raw)：tanh 本身的饱和阈值固
        定在 raw≈±2~3，跟 bound 无关——如果直接 bound*tanh(raw)，调小 bound(比
        如从 8.0 调到 1.5)并不会让"需要多大的 raw 才饱和"跟着变，饱和依然在
        raw≈3 附近发生，只是乘出来的动态范围变窄了。这样 raw 更容易被推过这个
        固定阈值进入饱和区，一旦 v_new 和 v_old 都饱和，输出几乎相等（tanh 在
        那里梯度趋于 0），δv=v_new-v_old 会被人为压得很小——这是 PolicyFlow
        ratio 的 delta_v 持续萎缩、修正信号消失的第三个来源（另外两个是 v_old
        错用了新网络的 hidden state、以及对多个 t 采样点先平均再算 ratio）。
        先除以 bound 再过 tanh，饱和阈值会跟着 bound 等比例缩放，raw 对 bound
        的相对灵敏度不再因为调小 bound 而意外改变。
        """
        inp = th.cat([h, x_t, t], dim=-1)
        raw = self.vel_fc2(F.relu(self.vel_fc1(inp)))
        if not getattr(self.args, "use_velocity_bound", True):
            # Diagnostic-only escape hatch to test whether the tanh bound is
            # actually needed once use_policyflow_ratio replaces the old
            # cfm-loss-diff ratio (the specific pathway the bound was added
            # for). Expect this to reintroduce unbounded divergence via
            # delta_v in the new ratio instead -- see conversation notes.
            return raw
        bound = getattr(self.args, "cfm_velocity_bound", 8.0)
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

        x_0 = eps ~ N(0,I)  (基分布，跟 action 本身的 [0,1] 区间无关——z 的分布
                 形状本身对 ratio 的可算性没有影响，因为 z 不依赖网络参数 θ，
                 会在 old/new policy 的 ratio 里精确抵消掉；见 2026-08-13 对话
                 记录的推导。换成高斯是更贴近 rectified flow / stochastic
                 interpolant 文献里的标准约定)
        u = integrate(h, eps, K) + n,  n ~ N(0, sigma^2)   (全程无界 latent 空间)
        action = sigmoid(u)                                (唯一一步映射进 (0,1))

        用 sigmoid 代替 hard clamp：之前 action = clamp(x1+n, 0, 1) 有个隐藏的
        模型失配——ratio 把 n 当成未截断的高斯来算似然，但真实执行的动作是截断
        过的，两者在贴边样本上对不上。换成 sigmoid 后，x1、n、u = x1+n 全程都在
        无界实数空间，n 就是货真价实、从未被截断过的高斯噪声，ratio 对 u 算出
        来的似然是精确的，不再有这个失配。sigmoid 是固定、不含参数的变换，同一
        个 u 在 new/old policy 下经过的是同一个 sigmoid，其雅可比在 ratio 里精确
        抵消，所以现有的 ratio 公式（对 n 和 Δv 算）不需要因为这个改动而改变。

        返回: (action, h, eps, x1_raw, noise)
            action: [..., n_actions]  实际执行的动作（sigmoid(x1+noise)，∈(0,1)，
                    永远不会真正等于 0/1，只会渐近逼近）
            h:      [..., hidden_dim] 更新后的 hidden state
            eps:    [..., n_actions]  本次采样的基分布噪声（供 initial_cfm_loss 用）
            x1_raw: [..., n_actions]  sigmoid 之前的积分终点 phi_hat（无界 latent，
                    不再是"该落在[0,1]、有时候越界"的量，是训练插值的目标端点）
            noise:  [..., n_actions]  真实采样的高斯噪声 n ~ N(0,sigma^2)，PolicyFlow
                    ratio 需要的就是这个未经任何变换的 n 本身。
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        eps = th.randn(*h.shape[:-1], n_act, device=h.device)
        n_steps = getattr(self.args, "cfm_rollout_steps", 1)
        x1 = self.integrate(h, eps, n_steps)
        # PolicyFlow 风格：flow 积分终点叠加可学习的高斯噪声 u = x1 + n，
        # n ~ N(0, sigma^2)，让 sigma 真正参与探索，而不只是个装饰参数。
        noise = th.randn_like(x1) * self.sigma()
        action = th.sigmoid(x1 + noise)
        return action, h, eps, x1, noise

#-----------------------------
