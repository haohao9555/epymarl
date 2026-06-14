import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNContinuousAgent(nn.Module):
    """RNN policy head for continuous Box(0, 1) actions.

    The output is concatenated Beta distribution parameters:
    [alpha_1..alpha_n, beta_1..beta_n].
    """

    def __init__(self, input_shape, args):
        super(RNNContinuousAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions * 2)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        raw_alpha, raw_beta = self.fc2(h).chunk(2, dim=-1)
        eps = getattr(self.args, "beta_param_epsilon", 1e-4)
        alpha = F.softplus(raw_alpha) + eps
        beta = F.softplus(raw_beta) + eps

        return th.cat([alpha, beta], dim=-1), h
