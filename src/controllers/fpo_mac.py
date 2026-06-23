import torch as th

from modules.agents import REGISTRY as agent_REGISTRY

#------新增：FPO 专用 MAC，调用 FPOActor.sample_action() 采样，并计算 initial_cfm_loss----------
#-----------------------------


class FPOMAC:
    """FPO 多智能体控制器。

    与 ContinuousMAC 的区别:
      - select_actions(): 调用 agent.sample_action()，通过一步 flow 产生动作
      - forward():        返回 h（hidden state），供 learner 计算 CFM loss
      - compute_initial_cfm_loss(): 在 rollout 时用当前策略计算初始 CFM loss
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.hidden_states = None

    # ── rollout 动作采样 ──────────────────────────────────────────────────────

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        action, self.hidden_states, self._last_eps = self.agent.sample_action(
            inputs, self.hidden_states
        )
        # action: [B*N, n_actions] → [B, N, n_actions]
        B = ep_batch.batch_size
        action = action.view(B, self.n_agents, -1)
        if test_mode:
            # 测试时直接用均值（无随机性）
            action = th.clamp(
                self._last_eps.view(B, self.n_agents, -1)
                - self.agent.velocity(
                    self.hidden_states.view(B * self.n_agents, -1),
                    self._last_eps,
                    th.ones(B * self.n_agents, 1, device=action.device),
                ).view(B, self.n_agents, -1),
                0.0, 1.0,
            )
        return action[bs]

    # ── learner forward（返回 h 供 CFM loss 计算）────────────────────────────

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        h, self.hidden_states = self.agent(inputs, self.hidden_states)
        return h.view(ep_batch.batch_size, self.n_agents, -1)   # [B, N, hidden_dim]

    # ── rollout 时计算 initial_cfm_loss ──────────────────────────────────────

    def compute_initial_cfm_loss(self, cfm_eps, cfm_t, actions):
        """用当前 hidden state 和流网络计算 rollout 时的 CFM loss。

        cfm_eps:  [B, N, cfm_n, n_actions]
        cfm_t:    [B, N, cfm_n, 1]
        actions:  [B, N, n_actions]

        返回: initial_cfm_loss [B, N, cfm_n, 1]，存入 buffer，训练时当参考基线。
        """
        with th.no_grad():
            h = self.hidden_states                              # [B*N, hidden_dim]
            B, N = cfm_eps.shape[0], cfm_eps.shape[1]
            cfm_n = cfm_eps.shape[2]

            # 扩维与 cfm_n 对齐
            act_exp = actions.unsqueeze(2).expand_as(cfm_eps)         # [B,N,cfm_n,n_act]
            x_t = cfm_t * cfm_eps + (1 - cfm_t) * act_exp            # 插值点

            h_exp = h.view(B, N, 1, -1).expand(-1, -1, cfm_n, -1)    # [B,N,cfm_n,hidden]

            flat_h    = h_exp.reshape(-1, h_exp.shape[-1])
            flat_x_t  = x_t.reshape(-1, x_t.shape[-1])
            flat_t    = cfm_t.reshape(-1, 1)

            v_pred = self.agent.velocity(flat_h, flat_x_t, flat_t)
            v_pred = v_pred.reshape(B, N, cfm_n, -1)

            target = cfm_eps - act_exp                                 # velocity target
            cfm_loss = ((v_pred - target) ** 2).mean(dim=-1, keepdim=True)  # [B,N,cfm_n,1]
        return cfm_loss

    # ── 通用接口 ──────────────────────────────────────────────────────────────

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
