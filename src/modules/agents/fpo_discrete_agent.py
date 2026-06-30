import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FPODiscreteAgent(nn.Module):
    """离散 FPO 智能体：RNN 编码器 + 速度场网络。

    CFM 在 n_actions 维 one-hot 空间中操作：
        x_0 = eps ~ N(0,I),  x_1 = one_hot(a)
        x_t = (1-t)*eps + t*one_hot(a)
        velocity target = one_hot(a) - eps

    rollout 采样（K 步 Euler，从 t=0 积分到 t=1，K = args.cfm_rollout_steps）：
        x_{t+dt} = x_t + dt * v(h, x_t, t)
        a = argmax(x_1)

    注：一步 Euler 仅在给定 h 时目标动作分布退化为单点质量（确定性策略）时才
    是精确解——此时最优 v(h, x_0, 0) = E[x_1|h] - x_0，噪声被精确抵消。当策略
    仍有熵（探索阶段）时，边际速度场在 t→1 时才逐渐依赖 x_t 收窄向具体模态，
    单步会把不同 eps 的输出都拉向同一个"平均方向"，丧失探索多样性。多步积分
    让网络在中间 t 重新评估 x_t，更贴近标准 flow matching 的采样过程。
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        n_actions = args.n_actions

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)

        # 速度场 MLP: 输入 = [h, x_t, t]
        self.vel_fc1 = nn.Linear(hidden_dim + n_actions + 1, hidden_dim)
        self.vel_fc2 = nn.Linear(hidden_dim, n_actions)
        # 最后一层小权重初始化：初始 velocity ≈ 0，使 L_old ≈ L_new，rho ≈ 1，避免早期崩溃
        nn.init.orthogonal_(self.vel_fc2.weight, gain=0.01)
        nn.init.constant_(self.vel_fc2.bias, 0.0)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def encode(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        return h

    def velocity(self, h, x_t, t):
        """速度场: [h, x_t, t] → v。

        h:   [..., hidden_dim]
        x_t: [..., n_actions]
        t:   [..., 1]
        """
        return self.vel_fc2(F.relu(self.vel_fc1(th.cat([h, x_t, t], dim=-1))))

    def integrate(self, h, eps, n_steps):
        """K 步 Euler 积分，从 x_0=eps（t=0）走到 x_1（t=1）。

        h:       [..., hidden_dim]
        eps:     [..., n_actions]  x_0
        n_steps: int，积分步数（K=1 退化为原来的一步 Euler）

        返回 logits（x_1 估计）: [..., n_actions]
        """
        x = eps
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            x = x + dt * self.velocity(h, x, t)
        return x

    def forward(self, inputs, hidden_state):
        """返回 (h, h)，h 供 learner 计算 CFM loss。"""
        h = self.encode(inputs, hidden_state)
        return h, h

    def sample_action(self, inputs, hidden_state):
        """K 步 flow 采样，返回 (action, h, eps)。

        action: [..., 1]  long integer（环境接收的离散动作）
        h:      [..., hidden_dim]
        eps:    [..., n_actions]  本次噪声（供 initial_cfm_loss 用）
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        eps = th.randn(*h.shape[:-1], n_act, device=h.device)
        n_steps = getattr(self.args, "cfm_rollout_steps", 1)
        logits = self.integrate(h, eps, n_steps)
        action = logits.argmax(dim=-1, keepdim=True)
        return action, h, eps
