# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    # Agent-id-conditioned: forward() returns N distinct output slots -> N
    # distinct GAE advantages ([B,T,N]). Read by FPOPPLearner
    # (fpopp_learner.py) to decide the advantage width/shape --
    # see mafpo_shared_critic.py's SharedVCritic for the other case.
    per_agent_values = True

    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # 全局 state，每个 agent 一份一样的（靠后面的 agent-id one-hot 区分
        # 输出）
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # 可选：每个 agent 自己的局部观测也拼进去（state 已经覆盖不到的信息）
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # 可选：上一步所有 agent 执行的动作也拼进去。连续动作直接用原始动作
        # 向量 batch["actions"]，不是 one-hot——one-hot 编码只对离散动作（类
        # 别下标）有意义，连续动作本来就是浮点向量，没有"类别"可以 one-hot。
        if self.args.obs_last_action:
            if t == 0:
                last_actions = th.zeros_like(batch["actions"][:, 0:1])
            elif isinstance(t, int):
                last_actions = batch["actions"][:, slice(t - 1, t)]
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions"][:, 0:1]), batch["actions"][:, :-1]], dim=1
                )
            # 所有 agent 的上一步动作拼成一条向量，每一行（每个 agent 的
            # value）都看到同一份
            last_actions = last_actions.reshape(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        # agent-id one-hot：这个 critic 的 fc1/fc2/fc3 是所有 agent 共用同一
        # 套参数，全靠这个 one-hot 让 fc3 对不同 agent 输出不同的 value（不
        # 是"离散动作"的 one-hot，是"第几个 agent"的 one-hot，两者是完全不
        # 同的东西）
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents   # agent-id one-hot
        return input_shape
