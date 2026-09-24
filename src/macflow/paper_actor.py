"""MAC-Flow actor (faithful port): the BC flow plus the distilled one-step policy.

Both networks are per-agent and conditioned on the agent's own observation with
its one-hot id appended -- the official implementation never builds a network
over the joint action, and never uses the global state (the state field is
read from the dataset but commented out in agents/macflow.py::total_loss).

Actions live in [-1, 1] internally, as in the official code (`jnp.clip(...,-1,1)`
after every flow/one-step evaluation). This repo's environments expose Box(0,1),
so the MAC converts with a = (u + 1) / 2 at the boundary; keeping the internal
convention at [-1,1] matters because the flow's base distribution is N(0, I) and
its scale is only sensible against a [-1,1] action range.
"""

import torch as th
import torch.nn as nn

from macflow.paper_nets import ActorVectorField


class MACFlowPaperActor(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        hid = tuple(getattr(args, "macflow_actor_hidden_dims", (512, 512, 512, 512)))
        ln = bool(getattr(args, "macflow_actor_layer_norm", False))
        n_actions = args.n_actions
        self.flow_steps = int(getattr(args, "macflow_flow_steps", 10))

        self.bc_flow = ActorVectorField(input_shape, n_actions, hid, ln, with_time=True)
        self.onestep = ActorVectorField(input_shape, n_actions, hid, ln, with_time=False)

    # ── the distilled policy (what acts in the environment) ──────────────────
    def act(self, obs, z):
        """mu_w(o, z) -> action in [-1, 1]. One forward pass, no ODE."""
        return self.onestep(obs, z).clamp(-1.0, 1.0)

    # ── the BC flow (training only) ──────────────────────────────────────────
    def velocity(self, obs, x, t):
        return self.bc_flow(obs, x, t)

    @th.no_grad()
    def flow_action(self, obs, z):
        """Euler integration of the BC flow from the SAME z, i.e. the
        distillation target mu_phi(o, z). Matches compute_flow_actions()."""
        x = z
        for i in range(self.flow_steps):
            t = x.new_full(x.shape[:-1] + (1,), i / self.flow_steps)
            x = x + self.bc_flow(obs, x, t) / self.flow_steps
        return x.clamp(-1.0, 1.0)

    # ── epymarl MAC plumbing (no recurrence: hidden state is unused) ─────────
    def init_hidden(self):
        return next(self.parameters()).new_zeros(1, 1)

    def forward(self, inputs, hidden_state=None):
        return self.act(inputs, th.randn(*inputs.shape[:-1], self.args.n_actions,
                                         device=inputs.device)), hidden_state
