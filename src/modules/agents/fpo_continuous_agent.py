import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FlowPolicyMLP(nn.Module):
    """CFM velocity predictor: (obs, x_t, t) -> predicted velocity."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, x_t, t):
        return self.net(th.cat([obs, x_t, t], dim=-1))


class FPOContinuousAgent(nn.Module):
    """FPO policy for continuous Box(0,1) actions.

    Policy head: GRU + Beta distribution parameters [alpha_1..n, beta_1..n].
    flow_policy: MLP (obs, x_t, t) -> velocity, called by FPOContinuousLearner.compute_cfm_loss.
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        # ------ 新增：FPO 连续动作 Agent（Beta 策略头 + CFM 流网络）----------
        # -----------------------------------------------------------------------------
        # Policy head — same as RNNContinuousAgent
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions * 2)

        # Flow policy uses raw env obs (obs_shape), not the expanded input_shape
        # args.obs_shape is set in run.py from env_info["obs_shape"]
        obs_dim = getattr(args, "obs_shape", input_shape)
        self.flow_policy = FlowPolicyMLP(obs_dim, args.n_actions, args.hidden_dim)
        # -----------------------------------------------------------------------------

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in) if self.args.use_rnn else F.relu(self.rnn(x))
        raw_alpha, raw_beta = self.fc2(h).chunk(2, dim=-1)
        eps = getattr(self.args, "beta_param_epsilon", 1e-4)
        alpha = F.softplus(raw_alpha) + eps
        beta = F.softplus(raw_beta) + eps
        return th.cat([alpha, beta], dim=-1), h
