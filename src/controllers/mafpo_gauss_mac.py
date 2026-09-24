import math

import torch as th

from components.obs_normalizer import ObsNormalizer
from modules.agents import REGISTRY as agent_REGISTRY


class MAFPOGaussMAC:
    """MAFPO-Gauss 控制器：共享 actor（obs_agent_id one-hot 区分 agent），
    每个 agent 自己的 sigma_i 在 actor 里按 agent 下标取。

    rollout 存三样东西给 learner 算精确高斯 ratio（scheme 键沿用 PolicyFlow
    的名字，parallel_runner 不用改）：
        action_raw   = mu_old        高斯均值（flow 端点 + 均值头）
        action_noise = n             真实采样的 N(0, sigma_i^2) 噪声
        z            = eps           flow 起点 x_0
    执行动作 = sigmoid(mu_old + n)。test_mode 下 eps=0、n=0（确定性评估）。

    obs 归一化：self.obs_normalizer 在这里创建，learner 把同一个实例挂到
    critic 上；更新时机见 components/obs_normalizer.py。
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # eps_rho generalises the two extremes into one timescale knob. eps is
        # carried through time as an AR(1),
        #     eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * xi_t,   xi ~ N(0, I),
        # whose marginal is EXACTLY N(0, I) at every step, so pi(a|h,eps) and the
        # PPO ratio are untouched -- only the temporal correlation of the eps
        # sequence changes. Correlation time is tau ~ 1/(1-rho).
        #   rho = 0    every step independent (the original behaviour)
        #   rho = 1    one draw per episode   (eps_per_episode=True)
        # The tradeoff the knob resolves: at rho = 0 a role agreed at t is thrown
        # away at t+1, and since eps is i.i.d. in time the only way to keep a role
        # stable is to stop depending on eps at all -- the objective rewards the
        # degeneracy we measure (the flow carries 0.4-0.7% of action variance). At
        # rho = 1 the role persists, but every timestep of an episode shares one
        # draw, so on HalfCheetah an update sees 8 independent eps instead of 8000
        # and the eps-dependent gradient's standard error grows ~32x.
        self.eps_rho = float(getattr(args, "eps_rho", 0.0))
        if bool(getattr(args, "eps_per_episode", False)):
            self.eps_rho = 1.0
        assert 0.0 <= self.eps_rho <= 1.0, self.eps_rho
        self.episode_eps = None
        # test_eps_mode: "zero" evaluates mu(h, 0), an arbitrary point of the flow
        # -- neither its mean nor a sample. While the flow is degenerate that is
        # harmless, but for a mu(h, .) with two good modes, mu(h, 0) is the
        # mode-averaged midpoint, i.e. the worst action: the evaluation would hide
        # the method exactly when it starts working. "sample" draws eps ~ N(0, I)
        # and keeps only the terminal Gaussian noise at zero.
        self.test_eps_mode = str(getattr(args, "test_eps_mode", "zero")).lower()
        assert self.test_eps_mode in ("zero", "sample"), self.test_eps_mode
        device = "cuda" if args.use_cuda else "cpu"
        self.obs_normalizer = ObsNormalizer(scheme, args, device)
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        B = ep_batch.batch_size
        n_act = self.args.n_actions
        with th.no_grad():
            h = self.agent.encode(inputs, self.hidden_states)
            self.hidden_states = h
            if test_mode and self.test_eps_mode == "zero":
                eps = th.zeros(B * self.n_agents, n_act, device=h.device)
            elif self.eps_rho > 0.0:
                xi = th.randn(B * self.n_agents, n_act, device=h.device)
                if self.episode_eps is None or self.episode_eps.shape != xi.shape:
                    self.episode_eps = xi                       # episode start
                else:
                    r = self.eps_rho
                    self.episode_eps = r * self.episode_eps + math.sqrt(1.0 - r * r) * xi
                eps = self.episode_eps
            else:
                eps = th.randn(B * self.n_agents, n_act, device=h.device)
            # mean() gets [B, N, *]: the flow's inter-agent attention (when on)
            # needs the agent axis, and the shape is inert without it.
            mu = self.agent.mean(
                h.view(B, self.n_agents, -1), eps.view(B, self.n_agents, n_act)
            ).view(B, self.n_agents, n_act)
            sigma = self.agent.sigma().unsqueeze(0).expand(B, -1, -1)      # [B,N,A]
            noise = th.zeros_like(mu) if test_mode else th.randn_like(mu) * sigma
            action = th.sigmoid(mu + noise)

        self._last_x1_raw = mu
        self._last_noise = noise
        self._last_eps = eps.view(B, self.n_agents, n_act)
        return action[bs]

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        h, self.hidden_states = self.agent(inputs, self.hidden_states)
        return h.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )
        # a new episode means a new role draw
        self.episode_eps = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.obs_normalizer.state_dict(), "{}/obs_norm.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.obs_normalizer.load_state_dict(
            th.load("{}/obs_norm.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [self.obs_normalizer.normalize_obs(batch["obs"][:, t])]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _build_inputs_all(self, batch):
        """一次构造训练要用的全部时刻的 actor 输入：[B, T, N, F]，
        T = max_seq_length - 1，与逐步 _build_inputs(batch, t) 逐位一致。"""
        bs = batch.batch_size
        T = batch.max_seq_length - 1
        inputs = [self.obs_normalizer.normalize_obs(batch["obs"][:, :T])]          # [B,T,N,O]
        if self.args.obs_last_action:
            prev = th.cat([th.zeros_like(batch["actions"][:, :1]), batch["actions"][:, :T - 1]], dim=1)
            inputs.append(prev)
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents)
                .expand(bs, T, -1, -1)
            )
        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
