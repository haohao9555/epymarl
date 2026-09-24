import torch as th
import torch.nn as nn
import torch.nn.functional as F


class JointQCritic(nn.Module):
    """联合动作价值 Q_psi(s, a_1..a_N)，MAFPO-Gauss 专用。

    只用来对 sigma 求导控制探索强度（learner 的 ader_estimator="q_pathwise"），
    不参与 advantage——advantage 仍由原来的 V critic 给。输入 = 归一化 state
    + 所有 agent 的执行动作拼接（[0,1] 区间），输出一个标量。回归目标是 V
    critic 同一份 lambda-return（对 agent 取平均），每个 epoch 训一次。"""

    normalizer = None

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        input_shape = scheme["state"]["vshape"] + self.n_agents * self.n_actions
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, state, actions):
        """state [..., S]，actions [..., N, A] -> Q [..., 1]。"""
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
        a = actions.reshape(*actions.shape[:-2], self.n_agents * self.n_actions)
        x = F.relu(self.fc1(th.cat([state, a], dim=-1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
