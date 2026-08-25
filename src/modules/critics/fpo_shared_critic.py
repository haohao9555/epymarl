# code adapted from https://github.com/AnujMahajanOxf/MAVEN
#
# Single joint state-value V(s), shared across all agents -- no per-agent
# one-hot conditioning, no per-agent output slot. Used by
# fpopp_shared_continuous_learner.py so every agent is scaled by the SAME
# single advantage at each timestep, unlike fpo_critic.py (a
# shared-parameter but agent-id-conditioned critic that still produces N
# distinct output slots via a one-hot input).

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SharedVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SharedVCritic, self).__init__()

        self.args = args
        input_shape = scheme["state"]["vshape"]
        self.output_type = "v"

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t + 1)
        x = F.relu(self.fc1(batch["state"][:, ts]))
        x = F.relu(self.fc2(x))
        return self.fc3(x)          # [B, max_t, 1]
