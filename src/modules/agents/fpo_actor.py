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
        x_0 = eps ~ Uniform(0,1)
        x_{t+dt} = x_t + dt * v(h, x_t, t)
        action = clamp(x_1 + n, 0, 1)，n ~ N(0, sigma^2)（PolicyFlow 风格的
                 可学习末端噪声，见 sigma()；仅在非 test_mode 下叠加）
    训练 CFM   : cfm_loss = ||v(h, x_t, t) - (action - eps)||²
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
        # 初始化到 sigma≈sigma_init(默认 0.1)而不是 exp(0)=1.0：动作范围只有
        # [0,1] 宽，sigma=1.0 这么大的噪声叠加 clamp 会人为制造出贴边质量(等
        # 价于 truncated Gaussian 在窄区间上的两端堆积)，训练早期
        # action_at_bound_fraction 虚高就是这个人为假象，不是 velocity 场本身
        # 塌缩——sigma_init 给一个和动作范围匹配的起点，避免这段虚高。
        sigma_init = getattr(args, "sigma_init", 0.1)
        self.log_sigma = nn.Parameter(
            th.full((n_actions,), th.log(th.tensor(sigma_init)).item())
        )

    def sigma(self):
        sigma_min = getattr(self.args, "sigma_min", 0.01)
        return th.exp(self.log_sigma).clamp(min=sigma_min)

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

        x_0 = eps ~ Uniform(0,1)  (基分布改成 Uniform，跟 action 本身同一个区间)
        action = clamp(integrate(h, eps, K), 0, 1)

        返回: (action, h, eps, x1_raw, noise)
            action: [..., n_actions]  实际执行的动作（已 clamp(x1+noise, 0, 1)）
            h:      [..., hidden_dim] 更新后的 hidden state
            eps:    [..., n_actions]  本次采样的噪声（供 initial_cfm_loss 用）
            x1_raw: [..., n_actions]  clamp 之前的积分终点 phi_hat，用于诊断"贴边是
                    刚好压线还是冲出界很远被硬拉回来"，也是训练插值的目标端点
            noise:  [..., n_actions]  真实采样的高斯噪声 n ~ N(0,sigma^2)，必须单独
                    存下来——clamp 之后的 action 一旦真的被截断过，就没法用
                    action-x1_raw 反推出这个 n 了（截断后的差值已经不是真正的高
                    斯样本），PolicyFlow ratio 需要的是这个未经改动的 n 本身。
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        eps = th.rand(*h.shape[:-1], n_act, device=h.device)
        n_steps = getattr(self.args, "cfm_rollout_steps", 1)
        x1 = self.integrate(h, eps, n_steps)
        # PolicyFlow 风格：flow 积分终点叠加可学习的高斯噪声 a = x1 + n，
        # n ~ N(0, sigma^2)，让 sigma 真正参与探索，而不只是个装饰参数。
        noise = th.randn_like(x1) * self.sigma()
        action = th.clamp(x1 + noise, 0.0, 1.0)
        return action, h, eps, x1, noise

#-----------------------------
