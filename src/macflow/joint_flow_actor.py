import torch as th
import torch.nn as nn
import torch.nn.functional as F


class JointFlowActor(nn.Module):
    """MAC-Flow stage-1: a single centralized velocity field
    v_phi(t, o_joint, x_t) over the FULL joint action (dim = n_agents *
    n_actions), trained with a flow-matching / behavior-cloning loss against
    whatever joint actions are sitting in the replay buffer (arXiv:2511.05005
    Eq.6). Non-recurrent by design: it is conditioned on the global env
    state (Markovian in the MPE tasks this is aimed at), the same choice
    fpo_critic.py's CentralVCritic already makes for its centrally-conditioned
    network -- only the decentralized actors (fpo_actor.py, one_step_actor.py)
    need the GRU for partial observability.

    Flow-matching is run in an UNBOUNDED logit space, not directly on the
    stored (0,1) actions: fpo_actor.py's design notes (see its docstring)
    found that letting a bounded-support CFM target interact with hard
    boundaries manufactures truncated-Gaussian-style boundary pileup. Since
    this network doesn't control how the buffer's actions were produced (they
    may come from many past policy versions, not just this run's own one-step
    actor), there is no "raw pre-sigmoid" value to reuse the way fpo_actor
    does at rollout time -- instead we invert the stored (0,1) action with
    logit() before interpolating, and the ODE integrator's own output is
    mapped back through sigmoid only by the caller (mac_flow_learner), never
    inside this module. That keeps this module symmetric with one_step_actor
    (both operate in the same unbounded latent space; sigmoid is applied by
    whoever needs an executable action).

    velocity() output is tanh-bounded to +-cfm_velocity_bound for the same
    reason fpo_actor.velocity() is: an uncapped CFM loss on advantage-shaped
    (here: replay-buffer-shaped) targets has no ceiling and drives weights to
    diverge over long runs.
    """

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.joint_action_dim = args.n_agents * args.n_actions
        hidden_dim = args.hidden_dim

        input_shape = self._get_input_shape(scheme)
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.vel_fc1 = nn.Linear(hidden_dim + self.joint_action_dim + 1, hidden_dim)
        self.vel_fc2 = nn.Linear(hidden_dim, self.joint_action_dim)

    def encode(self, state_inputs):
        x = F.relu(self.fc1(state_inputs))
        h = F.relu(self.fc2(x))
        return h

    def velocity(self, h, x_t, t):
        inp = th.cat([h, x_t, t], dim=-1)
        raw = self.vel_fc2(F.relu(self.vel_fc1(inp)))
        bound = getattr(self.args, "cfm_velocity_bound", 8.0)
        return bound * th.tanh(raw / bound)

    def integrate(self, h, eps, n_steps):
        """K-step Euler, t=0 -> t=1. Returns the unbounded-latent endpoint
        (caller applies sigmoid if an executable/comparable action is
        needed)."""
        x = eps
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            x = x + dt * self.velocity(h, x, t)
        return x

    def _build_inputs(self, batch, t=None):
        """Returns [B, T, F] (t=None, full sequence) or [B, 1, F] (single t) --
        caller reshapes as needed. Never flattens the batch dim itself, unlike
        the decentralized MACs' _build_inputs, since this network is called
        once per timestep for the whole joint batch, not once per agent."""
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = [batch["state"][:, ts]]
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].reshape(bs, max_t, -1))
        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        return input_shape
