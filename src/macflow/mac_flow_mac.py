import torch as th

from modules.agents import REGISTRY as agent_REGISTRY


class MACFlowMAC:
    """Controller for online MAC-Flow. Owns the decentralized one-step actor
    (OneStepActor) -- the thing that actually acts in the environment, same
    role MADDPGMAC's agent plays for MADDPG. The joint flow and the critics
    live in mac_flow_learner.py instead (mirrors how MADDPGLearner owns its
    critic directly rather than routing it through this controller).

    select_actions() is deliberately a single forward pass (no ODE
    integration) -- that is the entire point of the distilled one-step
    policy: it has to be cheap enough to run every environment step, unlike
    the joint flow which is only ever evaluated inside train().
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        B = ep_batch.batch_size

        if test_mode:
            # Deterministic eval: z=0 (the base distribution's mean), no
            # terminal noise -- same convention as FPOMAC's test_mode.
            h = self.agent.encode(inputs, self.hidden_states)
            self.hidden_states = h
            n_act = self.args.n_actions
            z = th.zeros((*h.shape[:-1], n_act), device=h.device)
            raw = self.agent.act(h, z)
            action = th.sigmoid(raw)
        else:
            action, self.hidden_states, _, _, _ = self.agent.sample_action(
                inputs, self.hidden_states
            )

        action = action.view(B, self.n_agents, -1)
        return action[bs]

    def target_actions(self, ep_batch, t, smoothing_std=0.0, smoothing_clip=0.0):
        """Used by the learner for the critic's TD bootstrap target (via
        target_mac). Fresh z each call, like a real rollout draw -- the
        policy IS a distribution over z, so this is the natural analogue of
        sampling a'~pi in a stochastic-policy actor-critic, not a
        deterministic point estimate. Optional clipped Gaussian smoothing on
        top (TD3-style) guards the critic against exploiting sharp errors at
        a single deterministic action.
        """
        inputs = self._build_inputs(ep_batch, t)
        h, self.hidden_states = self.agent(inputs, self.hidden_states)
        n_act = self.args.n_actions
        z = th.randn(*h.shape[:-1], n_act, device=h.device)
        raw = self.agent.act(h, z)
        if smoothing_std > 0.0:
            noise = (th.randn_like(raw) * smoothing_std).clamp(
                -smoothing_clip, smoothing_clip
            )
            raw = raw + noise
        action = th.sigmoid(raw)
        return action.view(ep_batch.batch_size, self.n_agents, -1)

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        h, self.hidden_states = self.agent(inputs, self.hidden_states)
        return h.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
        )

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
            th.load("{}/agent.th".format(path),
                    map_location=lambda storage, loc: storage)
        )

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
