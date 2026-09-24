"""Controller for the faithful MAC-Flow port.

Owns the actor (BC flow + one-step policy); the critics live in the learner,
mirroring MADDPGMAC/MACFlowMAC in this repo.

Action convention: the actor works in [-1, 1] (official), the environments here
expose Box(0, 1), so the conversion a_env = (u + 1) / 2 happens here and only
here -- the buffer stores the env-space action, and the learner converts back
with u = 2 * a_env - 1 before touching the networks.

Evaluation samples a fresh z ~ N(0, I) just like training. The official eval
loop passes a *fixed* PRNGKey every step (continuous_main.py::_evaluate), which
reuses the same noise vector at every timestep -- almost certainly unintended,
so it is not reproduced here; set macflow_eval_zero_z=True to evaluate at z = 0
instead, which is the deterministic analogue used by the other controllers in
this repo.
"""

import torch as th

from modules.agents import REGISTRY as agent_REGISTRY


class MACFlowPaperMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.hidden_states = None
        self.eval_zero_z = bool(getattr(args, "macflow_eval_zero_z", False))

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)          # [B*N, F]
        with th.no_grad():
            z = th.zeros(inputs.shape[0], self.args.n_actions, device=inputs.device) \
                if (test_mode and self.eval_zero_z) else \
                th.randn(inputs.shape[0], self.args.n_actions, device=inputs.device)
            u = self.agent.act(inputs, z)                     # [-1, 1]
        actions = (u + 1.0) / 2.0                             # -> Box(0, 1)
        return actions.view(ep_batch.batch_size, self.n_agents, -1)[bs]

    def target_actions_seq(self, ep_batch, obs_all):
        """a'_i for every transition, from the one-step policy on o_{t+1}.
        obs_all is [B, Tf, N, F] as returned by build_inputs_all."""
        nxt = obs_all[:, 1:]
        z = th.randn(*nxt.shape[:-1], self.args.n_actions, device=nxt.device)
        return self.agent.act(nxt, z)

    def target_actions(self, ep_batch, t):
        """a' for the TD bootstrap, drawn from the one-step policy (official
        critic_loss samples next_actions from the *current* policy, not a target
        actor -- there is no target actor in MAC-Flow)."""
        inputs = self._build_inputs(ep_batch, t)
        with th.no_grad():
            z = th.randn(inputs.shape[0], self.args.n_actions, device=inputs.device)
            u = self.agent.act(inputs, z)
        return u.view(ep_batch.batch_size, self.n_agents, -1)

    def build_inputs_all(self, batch, upto=None):
        """[B, T, N, F] for every timestep -- the networks are feed-forward, so
        the whole batch goes through in one call."""
        bs = batch.batch_size
        T = batch.max_seq_length if upto is None else upto
        inputs = [batch["obs"][:, :T]]
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device)
                          .view(1, 1, self.n_agents, self.n_agents).expand(bs, T, -1, -1))
        return th.cat(inputs, dim=-1)

    def init_hidden(self, batch_size):
        self.hidden_states = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
