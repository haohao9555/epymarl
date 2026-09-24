import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OneStepActor(nn.Module):
    """Decentralized one-step policy mu_w(o_i, z_i) -> a_i (MAC-Flow's stage-3
    distillation target). Unlike fpo_actor.FPOActor this does NOT integrate an
    ODE at rollout time -- it is a single forward pass, which is the whole
    point of MAC-Flow's distillation (fast decentralized execution). z is a
    fresh N(0,I) noise vector concatenated with the encoded observation; it is
    both the policy's reparameterization input (mirrors the joint flow's own
    z, so the distillation BC term in mac_flow_learner compares like-for-like)
    and, together with the learned sigma below, its exploration source at
    rollout.

    Structure:
        obs -> fc1 -> ReLU -> GRU -> h
        [h, z] -> fc2 -> ReLU -> fc3 -> raw   (unbounded logit space)
        action = sigmoid(raw + n),  n ~ N(0, sigma^2)  (rollout only)

    The unbounded-latent-then-final-sigmoid design and the sigmoid-parametrized
    learnable sigma are carried over unchanged from fpo_actor.py, where this
    exact combination was arrived at after several failed alternatives (hard
    clamp -> truncated-Gaussian boundary pileup; unbounded exp(raw_sigma) ->
    sigma_mean drifting past 0.9 over long runs with no plateau). Re-deriving
    that here would just reproduce the same failure modes from scratch.
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

        self.fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        sigma_init = getattr(args, "mac_flow_sigma_init", 0.1)
        self.sigma_min = getattr(args, "mac_flow_sigma_min", 0.01)
        self.sigma_max = getattr(args, "mac_flow_sigma_max", 1.0)
        p_init = (sigma_init - self.sigma_min) / (self.sigma_max - self.sigma_min)
        p_init = min(max(p_init, 1e-4), 1 - 1e-4)
        raw_init = th.log(th.tensor(p_init) / (1 - th.tensor(p_init))).item()
        self.raw_sigma = nn.Parameter(th.full((n_actions,), raw_init))

    def sigma(self):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * th.sigmoid(
            self.raw_sigma
        )

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

    def act(self, h, z):
        """[h, z] -> raw (unbounded pre-sigmoid action). Kept separate from
        encode() so the learner's distillation loss can re-run this with a
        freshly-sampled z at every training step without re-running the GRU
        (h is cheap to keep around / recompute once per minibatch)."""
        inp = th.cat([h, z], dim=-1)
        return self.fc3(F.relu(self.fc2(inp)))

    def forward(self, inputs, hidden_state):
        """MAC-compatible interface: returns (h, h)."""
        h = self.encode(inputs, hidden_state)
        return h, h

    def sample_action(self, inputs, hidden_state):
        """Rollout sampling: fresh z ~ N(0,I) plus learned terminal noise.

        Returns: (action, h, z, raw, noise) -- raw and noise mirror
        fpo_actor.sample_action's (x1_raw, noise) naming so the parallel with
        the existing, already-debugged rollout path is explicit.
        """
        h = self.encode(inputs, hidden_state)
        n_act = self.args.n_actions
        z = th.randn(*h.shape[:-1], n_act, device=h.device)
        raw = self.act(h, z)
        noise = th.randn_like(raw) * self.sigma()
        action = th.sigmoid(raw + noise)
        return action, h, z, raw, noise
