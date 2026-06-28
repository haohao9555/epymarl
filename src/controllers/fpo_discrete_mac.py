import torch as th
import torch.nn.functional as F

from modules.agents import REGISTRY as agent_REGISTRY


class FPODiscreteMAC:
    """离散 FPO 多智能体控制器。

    - select_actions(): 调用 agent.sample_action()，一步 flow → argmax 得离散动作
    - forward():        返回 h（hidden state），供 learner 计算 CFM loss
    - compute_initial_cfm_loss(): rollout 时用当前策略计算 CFM loss（以 one-hot 为目标）
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        action, self.hidden_states, self._last_eps = self.agent.sample_action(
            inputs, self.hidden_states
        )
        B = ep_batch.batch_size
        action = action.view(B, self.n_agents, 1)
        return action[bs]

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        h, self.hidden_states = self.agent(inputs, self.hidden_states)
        return h.view(ep_batch.batch_size, self.n_agents, -1)   # [B, N, hidden_dim]

    def compute_initial_cfm_loss(self, cfm_eps, cfm_t, actions, bs=slice(None)):
        """rollout 时计算 CFM loss，以 one_hot(action) 为流目标。

        cfm_eps:  [B, N, cfm_n, n_actions]
        cfm_t:    [B, N, cfm_n, 1]
        actions:  [B, N, 1]  long integer

        返回: [B, N, cfm_n, 1]
        """
        with th.no_grad():
            full_batch_size = self.hidden_states.shape[0]
            if self.hidden_states.dim() == 2:
                full_batch_size //= self.n_agents
            h = self.hidden_states.reshape(full_batch_size, self.n_agents, -1)[bs]
            B, N = cfm_eps.shape[0], cfm_eps.shape[1]
            cfm_n = cfm_eps.shape[2]

            one_hot_a = F.one_hot(
                actions.squeeze(-1), self.args.n_actions
            ).float()                                                    # [B, N, n_actions]
            act_exp = one_hot_a.unsqueeze(2).expand_as(cfm_eps)          # [B, N, cfm_n, n_actions]
            x_t = (1 - cfm_t) * cfm_eps + cfm_t * act_exp               # 插值点

            h_exp = h.reshape(B, N, 1, -1).expand(-1, -1, cfm_n, -1)
            flat_h   = h_exp.reshape(-1, h_exp.shape[-1])
            flat_x_t = x_t.reshape(-1, x_t.shape[-1])
            flat_t   = cfm_t.reshape(-1, 1)

            v_pred = self.agent.velocity(flat_h, flat_x_t, flat_t)
            v_pred = v_pred.reshape(B, N, cfm_n, -1)

            target = act_exp - cfm_eps                                   # velocity target
            cfm_loss = ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)
        return cfm_loss

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
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
