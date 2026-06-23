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

    rollout 采样: action = clamp(eps - v(h, eps, t=1), 0, 1)
    训练 CFM   : cfm_loss = ||v(h, x_t, t) - (eps - action)||²
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
        """
        inp = th.cat([h, x_t, t], dim=-1)
        return self.vel_fc2(F.relu(self.vel_fc1(inp)))

    # ── MAC 兼容接口 ──────────────────────────────────────────────────────────

    def forward(self, inputs, hidden_state):
        """MAC 调用接口：返回 (h, h)，h 供 learner 计算 CFM loss。"""
        h = self.encode(inputs, hidden_state)
        return h, h

    # ── rollout 动作采样（一步 flow）────────────────────────────────────────

    def sample_action(self, inputs, hidden_state):
        """One-step flow 采样。

        x_1 = eps ~ N(0,I)
        action = clamp(eps - v(h, eps, t=1), 0, 1)

        返回: (action, h, eps)
            action: [..., n_actions]  实际执行的动作
            h:      [..., hidden_dim] 更新后的 hidden state
            eps:    [..., n_actions]  本次采样的噪声（供 initial_cfm_loss 用）
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        eps = th.randn(*h.shape[:-1], n_act, device=h.device)
        t = th.ones(*h.shape[:-1], 1, device=h.device)
        v = self.velocity(h, eps, t)
        action = th.clamp(eps - v, 0.0, 1.0)
        return action, h, eps

#-----------------------------
