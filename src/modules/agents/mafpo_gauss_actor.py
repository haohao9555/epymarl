import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAFPOGaussActor(nn.Module):
    """MAFPO-Gauss actor：flow 给均值，外面套一个显式高斯。

        obs(归一化) → fc1 → ReLU → GRU → h
        mu  = Euler_K( v_theta(h, x_t, t) ; x_0 = eps ~ N(0,I) )       flow 端点即高斯均值
        u   = mu + n,   n ~ N(0, sigma_i^2)                            sigma_i: 每个 agent 独立、
                                                                        state-independent、逐动作维
        action = sigmoid(u)                                             唯一一步映射进 (0,1)

    策略在给定 eps 下是精确高斯 N(u; mu_theta(s, eps), sigma_i^2)，所以 ratio
    直接从 (mu, sigma) 里算（见 mafpo_gauss_learner），不再走 FPO 的 CFM-loss
    /ELBO 代理。eps 的分布不依赖参数，在新旧策略的联合密度比里精确约掉；
    E_{eps|s,u}[∇log N(u;mu(s,eps))] = ∇log pi(u|s)（Fisher 恒等式），所以条件
    在 eps 上的 PPO 是边缘策略梯度的无偏估计。

    sigma 参数化沿用 policyflow_actor：sigma = sigma_min + (sigma_max-sigma_min)
    * sigmoid(raw)，形状 [n_agents, n_actions]，所有 agent 从同一个 sigma_init
    出发（初始熵相同）。谁来改 sigma 由 learner 的 gauss_sigma_mode 决定：
    "ader"（learner 按 dJ/dlog sigma_i 在 agent 间分配，见 set_log_sigma）、
    "ppo"（作为普通参数进 PPO loss）、"fixed"。
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        n_actions = args.n_actions
        n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        # use_rnn=True 用 nn.GRU（单层）而不是 GRUCell：数学一样，但训练时
        # encode_sequence 可以把整段 T 步一次交给 cuDNN 融合 kernel 做（正反传
        # 都不走 Python 循环，T=1000 时快 ~100 倍）；rollout 时按 seq_len=1 逐步调。
        if args.use_rnn:
            self.rnn = nn.GRU(hidden_dim, hidden_dim)
            # cuDNN 对 RNN 默认开 TF32（10 位尾数），1000 步累积下来和逐步 GRUCell
            # 差 ~5e-4；关掉后逐位一致（6e-6）而且这么小的 GRU 反而更快。
            th.backends.cudnn.allow_tf32 = False
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)

        # gauss_mu_source="flow"（默认）：mu = K 步 Euler 积分的 flow 端点。
        # gauss_mu_source="mlp"：mu = 两层 MLP(h)，不积分、不看 eps —— 这就是
        # 把 flow 换成普通高斯策略头的消融（= squashed-Gaussian MAPPO），其余
        # （obs 归一化、GAE、PPO-clip、minibatch、critic、ratio 公式）完全不变。
        # mu 不依赖 eps 时，learner 的 ratio 自动退化成标准高斯 PPO ratio。
        self.mu_source = str(getattr(args, "gauss_mu_source", "flow")).lower()
        assert self.mu_source in ("flow", "mlp"), self.mu_source
        self.t_embed_dim = int(getattr(args, "fpo_timestep_embed_dim", 8))
        # flow_param="velocity": the head outputs v directly (original).
        # flow_param="endpoint": the head outputs g, its guess of where the
        #   trajectory ends, and the velocity is *defined* as the average speed
        #   needed to cover the rest of the path,
        #       v(h, x_t, t) = (g(h, x_t, t) - x_t) / (1 - t),
        #   which turns the Euler update into x <- x + (g - x)/(K - k). Two
        #   consequences, both structural rather than penalised:
        #     * a g that is consistent along the trajectory *is* a straight
        #       path, so self-consistency needs no extra loss term;
        #     * the last step has coefficient 1/(K-(K-1)) = 1, so the endpoint
        #       is exactly the final guess, mu = g_{K-1}. With the head
        #       zero-initialised the telescoping product ends in 0/1, so mu = 0
        #       at init (zero torque after the sigmoid) instead of mu ~ eps,
        #       which is what made the velocity parameterisation waste its first
        #       ~1M steps contracting the base noise.
        self.flow_param = str(getattr(args, "flow_param", "velocity")).lower()
        assert self.flow_param in ("velocity", "endpoint"), self.flow_param
        self.flow_attention = False
        if self.mu_source == "flow":
            # 速度场，时间用 Fourier 嵌入（同 mafpo_actor.embed_t，官方 isaaclab 默认 8 维）。
            t_in = self.t_embed_dim if self.t_embed_dim > 0 else 1
            self.vel_fc1 = nn.Linear(hidden_dim + n_actions + t_in, hidden_dim)
            self.vel_fc2 = nn.Linear(hidden_dim, n_actions)
            if self.flow_param == "endpoint" and bool(getattr(args, "endpoint_zero_init", True)):
                nn.init.zeros_(self.vel_fc2.weight)
                nn.init.zeros_(self.vel_fc2.bias)
            # Inter-agent attention INSIDE the ODE: every Euler step is one round
            # of negotiation. Each agent turns (h_i, x_k^i, t) into a token,
            # attends over the other agents' tokens -- i.e. over where everyone
            # currently thinks they are heading -- and its endpoint guess g_i is
            # formed after seeing them. So the attention weights decide whom I
            # listen to and g_i decides where I go; the velocity follows from g_i
            # by the endpoint parameterisation. K integration steps = K rounds.
            #
            # This makes execution require communication (each round needs the
            # other agents' current x_k), so it is a communicating-agents method,
            # not strict CTDE. The channel is small: A floats per agent per round.
            #
            # out_proj is zero-initialised so the attention contributes nothing at
            # init -- the run starts exactly as the attention-free version and
            # grows coupling only if it pays off.
            self.flow_attention = bool(getattr(args, "flow_attention", False))
            if self.flow_attention:
                heads = int(getattr(args, "flow_attention_heads", 4))
                self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
                nn.init.zeros_(self.attn.out_proj.weight)
                nn.init.zeros_(self.attn.out_proj.bias)
        else:
            self.mu_fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.mu_fc2 = nn.Linear(hidden_dim, n_actions)
            # mu_attention: one round of inter-agent attention over the MLP's
            # hidden features -- a Gaussian policy that COMMUNICATES, i.e. the
            # baseline a reviewer will reach for ("isn't this just CommNet /
            # TarMAC?"). It is the control that isolates our actual claim: not
            # that agents talk, but that they talk over a latent that carries
            # each agent's own sampling noise. Here the noise is added after the
            # mean, where attention cannot see it, so two agents with identical
            # observations still get identical means.
            #
            # out_proj is zero-initialised, so at init this is exactly the MLP
            # baseline. Unlike the flow head's attention it gets a non-zero
            # gradient from step one, because mu_fc2 is not zero-initialised.
            self.mu_attention = bool(getattr(args, "mu_attention", False))
            if self.mu_attention:
                heads = int(getattr(args, "flow_attention_heads", 4))
                self.mu_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
                nn.init.zeros_(self.mu_attn.out_proj.weight)
                nn.init.zeros_(self.mu_attn.out_proj.bias)

        # sigma_param:
        #   "sigmoid" (default, this repo's line) -- sigma is bounded to
        #       [sigma_min, sigma_max] by a sigmoid. Chosen after an unbounded
        #       exp() parameterisation was observed to drift past 0.9 with no
        #       plateau (see macflow/one_step_actor.py's note).
        #   "exp" -- the textbook continuous-control PPO/MAPPO parameterisation,
        #       sigma = exp(log_std) with a free log_std and sigma_init = 1.0 by
        #       convention. Needed to check that the MAPPO baseline is not being
        #       weakened by this repo's bounded variant.
        self.sigma_param = str(getattr(args, "sigma_param", "sigmoid")).lower()
        assert self.sigma_param in ("sigmoid", "exp"), self.sigma_param
        self.sigma_min = float(getattr(args, "sigma_min", 0.01))
        self.sigma_max = float(getattr(args, "sigma_max", 1.0))
        sigma_init = float(getattr(args, "sigma_init", 0.3))
        if self.sigma_param == "exp":
            raw_init = math.log(sigma_init)
        else:
            p_init = (sigma_init - self.sigma_min) / (self.sigma_max - self.sigma_min)
            p_init = min(max(p_init, 1e-4), 1 - 1e-4)
            raw_init = math.log(p_init / (1 - p_init))
        # gauss_sigma_per_dim=False：每个 agent 只有一个标量 sigma，广播到所有动作维
        # （MPE 这种"5 维其实只是 2 维移动 + 1 个 no-op"的环境没必要逐维区分，
        # 逐维时 no-op 维会把熵预算吸走）。sigma() 对外始终返回 [n_agents, n_actions]。
        self.sigma_per_dim = bool(getattr(args, "gauss_sigma_per_dim", True))
        sigma_shape = (n_agents, n_actions) if self.sigma_per_dim else (n_agents, 1)
        self.raw_sigma = nn.Parameter(th.full(sigma_shape, raw_init))

    # ── sigma ────────────────────────────────────────────────────────────────

    def sigma(self):
        """[n_agents, n_actions]（标量模式下由 [n_agents, 1] 广播而来）。"""
        if self.sigma_param == "exp":
            s = th.exp(self.raw_sigma)                     # unbounded, textbook
        else:
            s = self.sigma_min + (self.sigma_max - self.sigma_min) * th.sigmoid(self.raw_sigma)
        return s.expand(-1, self.args.n_actions)

    @th.no_grad()
    def set_log_sigma(self, log_sigma):
        """learner（ADER 模式）直接写 sigma：log_sigma [n_agents, n_actions] 或
        [n_agents, 1]，先 clamp 到 [sigma_min, sigma_max] 再反解 raw。标量模式下
        传进来 [n_agents, n_actions] 时对动作维取均值。"""
        if log_sigma.shape != self.raw_sigma.shape:
            log_sigma = log_sigma.mean(dim=-1, keepdim=True)
        if self.sigma_param == "exp":
            self.raw_sigma.copy_(log_sigma)
            return
        s = th.clamp(th.exp(log_sigma), self.sigma_min + 1e-6, self.sigma_max - 1e-6)
        p = (s - self.sigma_min) / (self.sigma_max - self.sigma_min)
        self.raw_sigma.copy_(th.log(p / (1 - p)))

    def entropy_per_agent(self):
        """高斯熵 [n_agents]：sum_d (0.5*log(2*pi*e) + log sigma_{i,d})。"""
        return (0.5 * math.log(2 * math.pi * math.e) + th.log(self.sigma())).sum(dim=-1)

    # ── encoder / flow ───────────────────────────────────────────────────────

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def encode(self, inputs, hidden_state):
        """单步：inputs [M, F]，hidden_state [..., H] -> h [M, H]（rollout 用）。"""
        x = F.relu(self.fc1(inputs))
        if self.args.use_rnn:
            h_in = hidden_state.reshape(1, -1, self.args.hidden_dim).contiguous()
            _, h = self.rnn(x.unsqueeze(0), h_in)
            return h.squeeze(0)
        return F.relu(self.rnn(x))

    def encode_sequence(self, inputs_seq):
        """整段：inputs_seq [B, T, N, F] -> h [B, T, N, H]（learner 训练用），
        初始 hidden 为零，和 init_hidden() + 逐步 encode() 的结果逐位一致。
        GRU 走 cuDNN 的整段调用；MLP 直接批量前向。"""
        B, T, N, _ = inputs_seq.shape
        x = F.relu(self.fc1(inputs_seq))                                  # [B,T,N,H]
        if self.args.use_rnn:
            x = x.permute(1, 0, 2, 3).reshape(T, B * N, self.args.hidden_dim)
            out, _ = self.rnn(x)                                          # h0 = 0
            return out.reshape(T, B, N, self.args.hidden_dim).permute(1, 0, 2, 3)
        return F.relu(self.rnn(x))

    def forward(self, inputs, hidden_state):
        h = self.encode(inputs, hidden_state)
        return h, h

    def embed_t(self, t):
        if self.t_embed_dim <= 0:
            return t
        freqs = 2.0 ** th.arange(self.t_embed_dim // 2, device=t.device, dtype=t.dtype)
        scaled = t * freqs
        return th.cat([th.cos(scaled), th.sin(scaled)], dim=-1)

    def _head(self, h, x_t, t):
        """Raw head output: v under flow_param="velocity", g under "endpoint".
        With flow_attention the agent axis must be the second-to-last one
        ([..., N, *]) because one round of inter-agent attention runs here."""
        inp = th.cat([h, x_t, self.embed_t(t)], dim=-1)
        z = F.relu(self.vel_fc1(inp))                              # [..., N, H]
        if self.flow_attention:
            lead, N, H = z.shape[:-2], z.shape[-2], z.shape[-1]
            assert N == self.args.n_agents, (
                f"flow_attention needs the agent axis at dim -2, got {z.shape}")
            zf = z.reshape(-1, N, H)
            zf = zf + self.attn(zf, zf, zf, need_weights=False)[0]  # residual
            z = zf.reshape(*lead, N, H)
        raw = self.vel_fc2(z)
        bound = float(getattr(self.args, "cfm_velocity_bound", 0.0))
        if bound > 0:
            return bound * th.tanh(raw / bound)
        return raw

    def velocity(self, h, x_t, t):
        """The ODE's velocity field. Under the endpoint parameterisation it is
        derived from the endpoint guess; t < 1 always (the integrator never
        evaluates t = 1), so 1 - t >= 1/n_steps and the division is bounded."""
        out = self._head(h, x_t, t)
        if self.flow_param == "endpoint":
            return (out - x_t) / (1.0 - t).clamp(min=1e-6)
        return out

    def integrate(self, h, eps, n_steps):
        x = eps
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            if self.flow_param == "endpoint":
                # x + dt*(g - x)/(1 - t) with t = i/n_steps is exactly
                # x + (g - x)/(n_steps - i); the integer form avoids the float
                # division and makes the final step land exactly on g.
                x = x + (self._head(h, x, t) - x) / (n_steps - i)
            else:
                x = x + dt * self.velocity(h, x, t)
        return x

    def mean(self, h, eps):
        """高斯均值：flow 端点（默认），或 MLP(h)（gauss_mu_source="mlp"，忽略
        eps）。h/eps 的前导维任意。"""
        if self.mu_source == "mlp":
            z = F.relu(self.mu_fc1(h))
            if getattr(self, "mu_attention", False):
                lead, N, H = z.shape[:-2], z.shape[-2], z.shape[-1]
                assert N == self.args.n_agents, (
                    f"mu_attention needs the agent axis at dim -2, got {z.shape}")
                zf = z.reshape(-1, N, H)
                zf = zf + self.mu_attn(zf, zf, zf, need_weights=False)[0]
                z = zf.reshape(*lead, N, H)
            return self.mu_fc2(z)
        n_steps = getattr(self.args, "cfm_rollout_steps", 10)
        return self.integrate(h, eps, n_steps)
