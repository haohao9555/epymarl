import itertools

import torch as th
import torch.nn as nn

from modules.agents import REGISTRY as agent_REGISTRY

#------MAFPO 专用 MAC，调用 MAFPOActor.sample_action() 采样，并计算 initial_cfm_loss----------
# 从 GitHub origin/current-mafpo @ 0313c24 拉取而来（"Use individual actors
# for continuous MAFPO"）：fpo_individual_agents=True 时，N 个智能体各用一份
# 独立参数的 actor（nn.ModuleList），修复了共享网络导致的跨智能体 Δ 混入问题
# ——跟本仓库本地开发的 PolicyFlow 线（policyflow_mac.py）刻意分开维护。
#
# 2026-08-24 本地追加修改：hard clamp 换成 sigmoid（见 mafpo_actor.py 顶部注
# 释）。self._last_x1_raw 存的是压缩/裁剪之前的无界积分终点，供
# compute_initial_cfm_loss / parallel_runner.py 的 collect_action_raw 读取，
# 让 CFM 回归的插值目标是无界 latent，不是被压缩过的 action。
#
# 2026-09-05 对齐 FPO++ 官方实现（amazon-far/fpo-control）：
#   1. 执行动作默认改回官方的"线性映射 + 硬 clip"（fpo_action_map="clip"，
#      x1 in [-fpo_action_clip, fpo_action_clip] <-> action in [0,1]，对应官方
#      isaaclab_fpo 的 clip_actions=2.0 / actor_scale），存进 buffer、喂给 CFM
#      的仍然是未裁剪的 x1——clip 只发生在"执行"这一步，flow 本身永远看不到
#      边界（官方 FPO::act() 存的是 policy.act() 的原始输出，wrapper 才 clip）。
#      sigmoid 保留为 fpo_action_map="sigmoid" 可选项。为什么要换：sigmoid 永
#      远不到 1，advantage normalize 之后"再往外一点"总是正 advantage，边界最
#      优的动作维（MPE 里满力推进）上 x1 会被系统性地推向饱和区，饱和之后
#      reward 对 x1 不再敏感、CFM 目标 |x1-eps| 超过速度网络的限幅、回归残差
#      单调上升——这正是所有长跑 log 里 action_at_bound_fraction 单调上升然后
#      cfm_loss/grad_norm 爆炸的前奏。硬 clip 之后边界外 reward 是平的，没有
#      系统性外推力，只剩 weight decay（learner 里的 AdamS/AdamW，fpo_optimizer）把 x1 拉回来。
#   2. rollout 时给 x1 加官方 action_perturb_std 的高斯扰动（训练模式，
#      fpo_action_perturb_std，官方默认 0.02）：执行和存 buffer 的都是扰动后
#      的 x1，CFM 目标永远不是 h 的确定性函数，flow 的推前分布有一个最小宽度。
#   3. CFM 回归误差统一走 cfm_regression_error()（官方 FPO++ fine-tuning 版的
#      modified Huber + 可配置的 action 维 reduction），rollout 的 initial loss
#      和 learner 重算的 new loss 必须用同一个函数，见该函数 docstring。
#-----------------------------


def cfm_regression_error(pred, target, huber_delta=0.0, reduction="mean"):
    """FPO++ 官方的逐探测点 CFM 回归误差。

    对应 fpo-control/manipulation_experiments/src/flow_model.py 的
    _compute_squared_error（modified Huber）和 isaaclab_fpo/modules/
    actor_critic.py 的 _compute_squared_error（action 维 reduction）。rollout 时
    存进 buffer 的 initial_cfm_loss 和训练时重算的 new loss 必须走同一个函数，
    否则 exp(L_old - L_new) 这个 ratio 本身就没有意义。

    huber_delta > 0：|e| <= delta 时是 e^2（跟 MSE 完全一致，不带 0.5 因子），
    |e| > delta 时是 2*delta*|e| - delta^2（一阶连续）。对 v_pred 的梯度上界是
    2*delta，不再随残差无限增长——纯 MSE 下一个离群探测点的梯度正比于残差，
    残差越大梯度越大，会在 exp(ratio) 之前就主导整个 minibatch 的更新方向
    （log 里 actor_grad_norm 从 ~1 冲到几百上千就是这个机制）。
    huber_delta <= 0：纯平方误差（官方 isaaclab locomotion 版本）。
    reduction：沿 action 维 "mean"（/D，官方 tracking 任务）、"sum"、"sqrt"
    （/sqrt(D)，官方 locomotion 默认，variance-preserving）。返回 [..., 1]。
    """
    diff = pred - target
    if huber_delta > 0:
        abs_diff = diff.abs()
        err = th.where(
            abs_diff <= huber_delta,
            diff ** 2,
            2.0 * huber_delta * abs_diff - huber_delta ** 2,
        )
    else:
        err = diff ** 2
    if reduction == "mean":
        return err.mean(dim=-1, keepdim=True)
    if reduction == "sum":
        return err.sum(dim=-1, keepdim=True)
    if reduction == "sqrt":
        return err.sum(dim=-1, keepdim=True) / (err.shape[-1] ** 0.5)
    raise ValueError(f"cfm_loss_reduction must be 'mean', 'sum' or 'sqrt', got {reduction!r}")


class MAFPOMAC:
    """MAFPO 多智能体控制器。

    与 ContinuousMAC 的区别:
      - select_actions(): 调用 agent.sample_action()，通过 K 步 Euler flow 产生动作
      - forward():        返回 h（hidden state），供 learner 计算 CFM loss
      - compute_initial_cfm_loss(): 在 rollout 时用当前策略计算初始 CFM loss

    fpo_individual_agents=True 时，self.agents 是 N 个独立参数的 actor
    （nn.ModuleList），self.agent 为 None；velocity()/integrate()/_encode()
    等方法据此在"共享单个 agent"和"逐 agent 分别调用"之间切换。
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

        # ── ADER：agent-wise 自适应 base-noise scale k_i（见类底部 ADER 相关方法
        # 的说明）。关闭时强制全 1，等价于原来的 N(0,I) 行为。
        self._init_ader_k(args)

    # ── ADER：agent-wise adaptive base-noise scale ──────────────────────────
    #
    # eps_i = k_i * z_i，z_i ~ N(0,I)。k_i 只是 batch-level 的 exploration
    # controller：不参与 actor 的 Adam 优化，也不通过 Euler integration 对 k
    # 建立 autograd 路径（那条路径专属于 theta）。k 在一整个 on-policy rollout
    # batch 收集期间、以及该 batch 对应的 FPOPPLearner.train() 里所有
    # epoch/minibatch 之间保持冻结，只在 train() 结束后由 learner 调用
    # set_ader_k() 更新一次，下一批 rollout 才会用上新值——具体的更新算法（g
    # score、EMA、variance-budget 分配、log-k 步长裁剪）都在
    # fpopp_learner.py，这里只保存/暴露当前值。
    def _init_ader_k(self, args):
        n = self.n_agents
        self.ader_enabled = getattr(args, "ader_enabled", False)
        self.ader_mode = getattr(args, "ader_mode", "adaptive")
        assert self.ader_mode in ("fixed", "adaptive", "gradient"), f"未知 ader_mode={self.ader_mode!r}"
        ader_k_init = float(getattr(args, "ader_k_init", 1.0))

        if not self.ader_enabled:
            init_k = th.ones(n)
        elif self.ader_mode == "fixed":
            ader_k_values = getattr(args, "ader_k_values", []) or []
            if len(ader_k_values) > 0:
                assert len(ader_k_values) == n, (
                    f"ader_k_values 长度 {len(ader_k_values)} 必须等于 n_agents={n}"
                )
                init_k = th.tensor(ader_k_values, dtype=th.float32)
            else:
                init_k = th.full((n,), ader_k_init)
        else:  # "adaptive" / "gradient"
            init_k = th.full((n,), ader_k_init)

        self.ader_k = init_k.float()

    def get_ader_k(self, device=None):
        """返回当前 agent-wise k，形状 [n_agents]。总是安全地移到 `device`
        （不传则原样返回），这样即便忘了显式调用 cuda() 也不会因为 device
        不匹配而崩。"""
        if device is not None:
            device = th.device(device)
            if self.ader_k.device != device:
                self.ader_k = self.ader_k.to(device)
        return self.ader_k

    def set_ader_k(self, new_k):
        """由 learner 在 train() 结束后调用一次。new_k 可以是 Tensor/list/
        ndarray，非 Tensor 会被转换；结果永远 detach（k 不是可训练参数，不该
        带着计算图）。"""
        if not th.is_tensor(new_k):
            new_k = th.as_tensor(new_k, dtype=th.float32)
        self.ader_k = new_k.detach().to(self.ader_k.device, dtype=th.float32).reshape(self.n_agents)

    # ── rollout 动作采样 ──────────────────────────────────────────────────────

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """统一采样路径：不管 individual_agents 还是共享 actor，都在这里由
        MAC 自己采 z/eps、算 k、调用 self.integrate()——共享 actor 不再走
        MAFPOActor.sample_action() 自己内部采样，这样 ADER 的 agent-wise k
        才能对两条路径都生效（sample_action() 方法仍然留在 mafpo_actor.py
        里，只是 MAC 不再调用它）。

        test_mode 下 z 和 eps 都是 0（N(0,I) 的众数，确定性评估，跟
        BetaActionSelector 的约定一致），k 因而不影响 test action，也不加
        action_perturb 扰动（官方 act() 只在 self.training 下扰动）；但仍然
        照常写 _last_z/_last_eps/_last_k，避免 runner 那边字段缺失。

        x1 是无界的积分终点：它（含扰动）原样存成 action_raw / 用来算
        initial_cfm_loss；执行动作 = _x1_to_action(x1)，见文件头注释第 1 条。
        """
        inputs = self._build_inputs(ep_batch, t_ep)
        B = ep_batch.batch_size
        n_act = self.args.n_actions
        n_steps = getattr(self.args, "cfm_rollout_steps", 1)

        h = self._encode(inputs, self.hidden_states, B)
        self.hidden_states = h
        h_bn = h.reshape(B, self.n_agents, -1)
        device = h_bn.device

        k = self.get_ader_k(device).view(1, self.n_agents, 1)

        if test_mode:
            z = th.zeros(B, self.n_agents, n_act, device=device)
            eps = th.zeros_like(z)
        else:
            z = th.randn(B, self.n_agents, n_act, device=device)
            # eps = k*z 只是采样时的一个逐元素缩放，不建立对 k 的 autograd
            # 路径（k 从不参与 actor 的 backward，见类顶部 ADER 说明）；这里
            # 不需要额外 no_grad，因为 k 本身就不是 requires_grad 的参数。
            eps = k * z

        x1 = self.integrate(h_bn, eps, n_steps)
        if not test_mode:
            # 官方 ActorCritic.act()：训练模式下给 flow 输出加 N(0, std^2) 扰动，
            # "can be interpreted as an entropy regularizer"；扰动后的值既执行也
            # 存进 buffer（官方 transition.actions 就是扰动后的），所以下面的
            # initial_cfm_loss / action_raw 看到的都是同一个 x1。
            perturb_std = float(getattr(self.args, "fpo_action_perturb_std", 0.02))
            if perturb_std > 0:
                x1 = x1 + perturb_std * th.randn_like(x1)
        action = self._x1_to_action(x1)

        self._last_z = z
        self._last_eps = eps
        self._last_k = k.expand(B, self.n_agents, 1).clone()
        self._last_x1_raw = x1

        action = action.view(B, self.n_agents, -1)
        return action[bs]

    def _x1_to_action(self, x1):
        """无界积分终点 x1 -> 环境执行的 Box(0,1) 动作。逐元素，对 [B,N,A] 或
        任何前导维都一样，n=1 和 n=3 走完全相同的代码。

        "clip"（默认，官方语义）：action = clamp(0.5 + 0.5 * x1 / c, 0, 1)，
        c = fpo_action_clip（默认 2.0 = 官方 isaaclab clip_actions）。x1 in
        [-c, c] 线性映射到 [0, 1]，c 之外硬 clip；c=2 时 x1=0 附近的斜率 0.25
        跟 sigmoid'(0) 一致，N(0,I) 的初始 x1 只有 ~4.6% 落在 clip 区。clip
        只作用在执行动作上，buffer 里的 action_raw 是未裁剪的 x1，CFM 回归
        目标从不经过这个 clip（官方：存 policy.act() 原始输出，wrapper 才
        clamp），所以速度场不会在边界上学到奇异点。
        "sigmoid"：2026-08-24 到 2026-09-05 之间的旧行为，保留作对照。"""
        action_map = getattr(self.args, "fpo_action_map", "clip")
        if action_map == "clip":
            clip = float(getattr(self.args, "fpo_action_clip", 2.0))
            return th.clamp(0.5 + 0.5 * x1 / clip, 0.0, 1.0)
        if action_map == "sigmoid":
            return th.sigmoid(x1)
        raise ValueError(f"fpo_action_map must be 'clip' or 'sigmoid', got {action_map!r}")

    def cfm_error(self, v_pred, target):
        """CFM 回归误差的唯一入口（rollout 的 initial loss 和 learner 的 new loss
        都从这里走），配置见文件头 cfm_regression_error() 的说明。默认
        cfm_loss_huber_delta=1.0（官方 FPO++ fine-tuning 各任务用 0.1~1.0），
        cfm_loss_reduction="mean"（跟现有 cfm_loss_clip_max / cfm_rho_clip 阈值
        的量纲一致；官方 locomotion 默认是 "sqrt"）。返回 [..., 1]。"""
        return cfm_regression_error(
            v_pred,
            target,
            huber_delta=float(getattr(self.args, "cfm_loss_huber_delta", 1.0)),
            reduction=getattr(self.args, "cfm_loss_reduction", "mean"),
        )

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
        actions:  [B, N, n_actions]   无界的 x1（含扰动），不是执行动作

        返回: initial_cfm_loss [B, N, cfm_n, 1]，存入 buffer，训练时当参考基线。
        误差函数 = self.cfm_error()，跟 learner 重算 new loss 用的是同一个。
        """
        with th.no_grad():
            full_batch_size = self.hidden_states.shape[0]
            if self.hidden_states.dim() == 2:
                full_batch_size //= self.n_agents
            h = self.hidden_states.reshape(
                full_batch_size, self.n_agents, -1
            )[bs]
            B, N, cfm_n, n_act = cfm_eps.shape
            assert N == self.n_agents, (N, self.n_agents)
            assert actions.shape == (B, N, n_act), (actions.shape, (B, N, n_act))
            assert cfm_t.shape == (B, N, cfm_n, 1), (cfm_t.shape, (B, N, cfm_n, 1))
            assert h.shape[:2] == (B, N), (h.shape, (B, N))

            # 扩维与 cfm_n 对齐
            act_exp = actions.unsqueeze(2).expand_as(cfm_eps)         # [B,N,cfm_n,n_act]
            x_t = (1 - cfm_t) * cfm_eps + cfm_t * act_exp            # 插值点

            h_exp = h.reshape(B, N, 1, -1).expand(-1, -1, cfm_n, -1)

            v_pred = self.velocity(h_exp, x_t, cfm_t)

            target = act_exp - cfm_eps                                 # velocity target
            cfm_loss = self.cfm_error(v_pred, target)                 # [B,N,cfm_n,1]
            assert cfm_loss.shape == (B, N, cfm_n, 1), cfm_loss.shape
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
        self.ader_k = self.ader_k.cuda()

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

    def velocity_features(self, h, x_t, t):
        """速度网络倒数第二层的预激活（vel_fc1 的输出，ReLU 之前）。

        PFO（Proximal Feature Optimization，Moalla et al. 2024）惩罚的就是这个
        量在新旧策略之间的差。取预激活而不是激活值，是因为死掉的 ReLU 神经元
        传不回梯度。h 也喂进这一层，所以编码器的漂移一并被覆盖。
        形状与 velocity() 一致：输入 [..., N, ...]，输出 [..., N, hidden_dim]。
        """
        # 时间嵌入必须和 velocity() 用同一个（embed_t 无参数，各 agent 一致，
        # 所以取哪一个 agent 的都行）——两边不一致的话 PFO 惩罚的就不是速度网
        # 络真正的倒数第二层了。
        any_agent = self.agent if self.agent is not None else self.agents[0]
        inp = th.cat([h, x_t, any_agent.embed_t(t)], dim=-1)
        if not self.individual_agents:
            f = self.agent.vel_fc1(inp.reshape(-1, inp.shape[-1]))
            return f.reshape(*inp.shape[:-1], -1)

        outputs = []
        for agent_id, agent in enumerate(self.agents):
            xi = inp[:, agent_id]
            f = agent.vel_fc1(xi.reshape(-1, xi.shape[-1]))
            outputs.append(f.reshape(*xi.shape[:-1], -1))
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
