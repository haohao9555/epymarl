import itertools

import torch as th
import torch.nn as nn

from modules.agents import REGISTRY as agent_REGISTRY

#------新增：FPO 专用 MAC，调用 FPOActor.sample_action() 采样，并计算 initial_cfm_loss----------
#-----------------------------


class FPOMAC:
    """FPO 多智能体控制器。

    与 ContinuousMAC 的区别:
      - select_actions(): 调用 agent.sample_action()，通过 K 步 Euler flow 产生动作
      - forward():        返回 h（hidden state），供 learner 计算 CFM loss
      - compute_initial_cfm_loss(): 在 rollout 时用当前策略计算初始 CFM loss
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.individual_agents = getattr(args, "fpo_individual_agents", False)
        if self.individual_agents:
            self.agents = nn.ModuleList(
                [agent_REGISTRY[args.agent](input_shape, args) for _ in range(self.n_agents)]
            )
            self.agent = None
        else:
            self.agent = agent_REGISTRY[args.agent](input_shape, args)
            self.agents = None
        self.hidden_states = None

    # ── rollout 动作采样 ──────────────────────────────────────────────────────

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        B = ep_batch.batch_size

        if test_mode:
            # 确定性评估：用 eps=0（N(0,I) 的众数）而不是重新随机采样，
            # 与 BetaActionSelector 在 test_mode 下返回分布均值而非 sample() 的约定一致
            # （见 components/action_selectors.py 的 BetaActionSelector）。
            h = self._encode(inputs, self.hidden_states, ep_batch.batch_size)
            self.hidden_states = h
            n_act = self.args.n_actions
            eps = th.zeros(*h.shape[:-1], n_act, device=h.device)
            n_steps = getattr(self.args, "cfm_rollout_steps", 1)
            x1 = self.integrate(h, eps, n_steps)
            action = th.clamp(x1, 0.0, 1.0)
            self._last_eps = eps
        else:
            if self.individual_agents:
                h = self._encode(inputs, self.hidden_states, B)
                n_act = self.args.n_actions
                eps = th.randn(*h.shape[:-1], n_act, device=h.device)
                n_steps = getattr(self.args, "cfm_rollout_steps", 1)
                action = th.clamp(self.integrate(h, eps, n_steps), 0.0, 1.0)
                self.hidden_states = h
                self._last_eps = eps
            else:
                action, self.hidden_states, self._last_eps = self.agent.sample_action(
                    inputs, self.hidden_states
                )

        # action: [B*N, n_actions] → [B, N, n_actions]
        action = action.view(B, self.n_agents, -1)
        return action[bs]

    # ── learner forward（返回 h 供 CFM loss 计算）────────────────────────────

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        if self.individual_agents:
            h = self._encode(inputs, self.hidden_states, ep_batch.batch_size)
            self.hidden_states = h
        else:
            h, self.hidden_states = self.agent(inputs, self.hidden_states)
        return h.view(ep_batch.batch_size, self.n_agents, -1)   # [B, N, hidden_dim]

    # ── rollout 时计算 initial_cfm_loss ──────────────────────────────────────

    def compute_initial_cfm_loss(
        self, cfm_eps, cfm_t, actions, bs=slice(None)
    ):
        """用当前 hidden state 和流网络计算 rollout 时的 CFM loss。

        cfm_eps:  [B, N, cfm_n, n_actions]
        cfm_t:    [B, N, cfm_n, 1]
        actions:  [B, N, n_actions]

        返回: initial_cfm_loss [B, N, cfm_n, 1]，存入 buffer，训练时当参考基线。
        """
        with th.no_grad():
            full_batch_size = self.hidden_states.shape[0]
            if self.hidden_states.dim() == 2:
                full_batch_size //= self.n_agents
            h = self.hidden_states.reshape(
                full_batch_size, self.n_agents, -1
            )[bs]
            B, N = cfm_eps.shape[0], cfm_eps.shape[1]
            cfm_n = cfm_eps.shape[2]

            # 扩维与 cfm_n 对齐
            act_exp = actions.unsqueeze(2).expand_as(cfm_eps)         # [B,N,cfm_n,n_act]
            x_t = (1 - cfm_t) * cfm_eps + cfm_t * act_exp            # 插值点

            h_exp = h.reshape(B, N, 1, -1).expand(-1, -1, cfm_n, -1)

            v_pred = self.velocity(h_exp, x_t, cfm_t)

            target = act_exp - cfm_eps                                 # velocity target
            cfm_loss = ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)  # [B,N,cfm_n,1]
        return cfm_loss

    # ── 通用接口 ──────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self._init_agent_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
        )

    def parameters(self):
        if self.individual_agents:
            return itertools.chain(*(agent.parameters() for agent in self.agents))
        return self.agent.parameters()

    def load_state(self, other_mac):
        if self.individual_agents:
            for agent, other_agent in zip(self.agents, other_mac.agents):
                agent.load_state_dict(other_agent.state_dict())
        else:
            self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        if self.individual_agents:
            self.agents.cuda()
        else:
            self.agent.cuda()

    def save_models(self, path):
        if self.individual_agents:
            th.save([agent.state_dict() for agent in self.agents], "{}/agent.th".format(path))
        else:
            th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        state = th.load("{}/agent.th".format(path),
                        map_location=lambda storage, loc: storage)
        if self.individual_agents:
            for agent, agent_state in zip(self.agents, state):
                agent.load_state_dict(agent_state)
        else:
            self.agent.load_state_dict(state)

    def velocity(self, h, x_t, t):
        if not self.individual_agents:
            v = self.agent.velocity(
                h.reshape(-1, h.shape[-1]),
                x_t.reshape(-1, x_t.shape[-1]),
                t.reshape(-1, t.shape[-1]),
            )
            return v.reshape_as(x_t)

        outputs = []
        for agent_id, agent in enumerate(self.agents):
            agent_h = h[:, agent_id]
            agent_x = x_t[:, agent_id]
            agent_t = t[:, agent_id]
            agent_v = agent.velocity(
                agent_h.reshape(-1, agent_h.shape[-1]),
                agent_x.reshape(-1, agent_x.shape[-1]),
                agent_t.reshape(-1, agent_t.shape[-1]),
            )
            outputs.append(agent_v.reshape_as(agent_x))
        return th.stack(outputs, dim=1)

    def integrate(self, h, eps, n_steps):
        if not self.individual_agents:
            x = self.agent.integrate(
                h.reshape(-1, h.shape[-1]),
                eps.reshape(-1, eps.shape[-1]),
                n_steps,
            )
            return x.reshape_as(eps)

        outputs = []
        for agent_id, agent in enumerate(self.agents):
            agent_x = agent.integrate(h[:, agent_id], eps[:, agent_id], n_steps)
            outputs.append(agent_x)
        return th.stack(outputs, dim=1)

    def _encode(self, inputs, hidden_states, batch_size):
        if not self.individual_agents:
            return self.agent.encode(inputs, hidden_states)

        inputs = inputs.reshape(batch_size, self.n_agents, -1)
        hidden_states = hidden_states.reshape(batch_size, self.n_agents, -1)
        outputs = []
        for agent_id, agent in enumerate(self.agents):
            outputs.append(agent.encode(inputs[:, agent_id], hidden_states[:, agent_id]))
        return th.stack(outputs, dim=1)

    def _init_agent_hidden(self):
        if self.individual_agents:
            return th.cat([agent.init_hidden() for agent in self.agents], dim=0)
        return self.agent.init_hidden()

    # ── 输入构建（与 ContinuousMAC 相同）─────────────────────────────────────

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

#-----------------------------
