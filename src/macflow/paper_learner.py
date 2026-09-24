"""Faithful PyTorch port of MAC-Flow's learner (arXiv:2511.05005).

Ported from the official JAX `agents/macflow.py` (critic_loss / actor_loss /
update). **This is a different algorithm from `mac_flow_learner.py`**, which was
an earlier online re-interpretation written in this repo; the differences that
matter, all resolved here in favour of the official code:

  |                     | mac_flow_learner.py (old)      | this file (official)        |
  | flow                | ONE centralized flow over the  | per-agent flow over the     |
  |                     | joint action, conditioned on   | agent's own action,         |
  |                     | the global state               | conditioned on its own obs  |
  | critic              | MADDPGCritic(state, a_joint),  | per-agent Q_k(o_i, a_i),    |
  |                     | per-agent TD                   | 2-ensemble, TD on the       |
  |                     |                                | agent-MEAN Q_tot (IGM)      |
  | distillation target | TARGET copy of the joint flow  | the current BC flow         |
  | Q term              | plain -Q                       | -Q normalised by 1/|Q|      |
  | BC weight alpha     | 1.0 annealed to 0.1            | constant 3.0                |
  | exploration         | extra learned Gaussian sigma   | z ~ N(0,I) only             |

The losses, verbatim from the paper's Eq. 2/6/9 as the official code implements
them:

  critic   MSE( mean_i Q(o_i,a_i) ,  mean_i [ r + gamma (1-d) mean_k Q_target(o'_i, a'_i) ] )
           with a' ~ the one-step policy (there is no target actor)
  actor    ||v_phi(o, x_t, t) - (x_1 - x_0)||^2                      flow-matching BC
           + alpha * ||mu_w(o,z) - Euler_K(v_phi; z)||^2             distillation
           - lambda * mean_i Q(o_i, mu_w(o_i,z_i)),  lambda = 1/|Q|  value guidance
  Polyak   critic only, tau = 0.005

Both losses are optimised by a single Adam over all three networks, as in the
official `total_loss` (one optimiser, `loss = critic_loss + actor_loss`).

Runs through run.py's ordinary off-policy path (growing replay buffer, sample a
minibatch of episodes per train() call) -- the learner name is deliberately not
in run.py's on-policy list.
"""

import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from macflow.paper_nets import TwinValue


class MACFlowPaperLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.mac = mac

        obs_dim = scheme["obs"]["vshape"] + (self.n_agents if args.obs_agent_id else 0)
        hid = tuple(getattr(args, "macflow_value_hidden_dims", (512, 512, 512, 512)))
        self.critic = TwinValue(obs_dim, self.n_actions, hid,
                                layer_norm=bool(getattr(args, "macflow_layer_norm", True)))
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.params = list(mac.parameters()) + list(self.critic.parameters())
        self.optimiser = Adam(params=self.params, lr=args.lr)

        self.alpha = float(getattr(args, "macflow_alpha", 3.0))
        self.tau = float(getattr(args, "macflow_tau", 0.005))
        self.gamma = float(getattr(args, "macflow_discount", args.gamma))
        self.q_agg = str(getattr(args, "macflow_q_agg", "mean"))
        self.normalize_q_loss = bool(getattr(args, "macflow_normalize_q_loss", True))

        self.log_stats_t = -self.args.learner_log_interval - 1
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    @staticmethod
    def _crop(x, rows, idx):
        """x[b, idx[b]] for a [B, T, ...] tensor -- picks one random window per
        episode."""
        return x[rows, idx]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        B = batch.batch_size
        # Official sampling is `batch_size` subsequences of `sequence_length`
        # (32 x 20), not whole episodes: epymarl's replay buffer returns full
        # episodes, so crop one random window per episode. Without this the
        # 512x4 MLPs see 50x more rows per update than the official code and
        # the update does not fit in memory on MuJoCo-length episodes.
        Tf = min(int(getattr(self.args, "macflow_seq_len", 20)) + 1, batch.max_seq_length)
        T = Tf - 1                                              # transitions
        rows = th.arange(B, device=batch.device).unsqueeze(1)
        if batch.max_seq_length > Tf:
            start = th.randint(0, batch.max_seq_length - Tf + 1, (B, 1), device=batch.device)
        else:
            start = th.zeros(B, 1, dtype=th.long, device=batch.device)
        idx = start + th.arange(Tf, device=batch.device).unsqueeze(0)   # [B,Tf]

        obs = self._crop(self.mac.build_inputs_all(batch), rows, idx)   # [B,Tf,N,F]
        # env actions are Box(0,1); the networks work in [-1,1] (official convention)
        act = self._crop(batch["actions"].float(), rows, idx)[:, :T] * 2.0 - 1.0

        rewards = self._crop(batch["reward"], rows, idx)[:, :T]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        if self.args.common_reward:
            rewards = rewards.expand(-1, -1, self.n_agents)     # [B,T,N]
        term_w = self._crop(batch["terminated"].float(), rows, idx)      # [B,Tf,1]
        terminated = term_w[:, :T].expand(-1, -1, self.n_agents)
        mask = self._crop(batch["filled"].float(), rows, idx)[:, :T]
        mask[:, 1:] = mask[:, 1:] * (1 - term_w[:, :T - 1])
        mask_bt = mask[:, :, 0]                                  # [B,T]
        denom = mask_bt.sum().clamp(min=1)

        # ── 1) critic: TD on the agent-mean Q (IGM average mixer) ────────────
        with th.no_grad():
            next_u = self.mac.target_actions_seq(batch, obs)     # [B,T,N,A] in [-1,1]
            next_qs = self.target_critic(obs[:, 1:Tf], next_u)   # [2,B,T,N]
            next_q = next_qs.min(dim=0)[0] if self.q_agg == "min" else next_qs.mean(dim=0)
            target = rewards + self.gamma * (1.0 - terminated) * next_q
            mixed_target = target.mean(dim=-1)                   # [B,T]

        qs = self.critic(obs[:, :T], act)                        # [2,B,T,N]
        mixed_q = qs.mean(dim=-1)                                # [2,B,T]
        critic_loss = (((mixed_q - mixed_target.unsqueeze(0)) ** 2).mean(dim=0)
                       * mask_bt).sum() / denom

        # ── 2) actor: flow-matching BC + distillation + value guidance ───────
        z0 = th.randn_like(act)
        t = th.rand(*act.shape[:-1], 1, device=act.device)
        x_t = (1 - t) * z0 + t * act
        pred_v = self.mac.agent.velocity(obs[:, :T], x_t, t)
        bc_flow_loss = ((((pred_v - (act - z0)) ** 2).mean(dim=(-1, -2))) * mask_bt).sum() / denom

        z = th.randn_like(act)
        flow_target = self.mac.agent.flow_action(obs[:, :T], z)  # no grad (Euler under no_grad)
        pi_u = self.mac.agent.act(obs[:, :T], z)
        distill_loss = ((((pi_u - flow_target) ** 2).mean(dim=(-1, -2))) * mask_bt).sum() / denom

        pi_qs = self.critic(obs[:, :T], pi_u)                    # [2,B,T,N]
        pi_q = pi_qs.mean(dim=0).mean(dim=-1)                    # [B,T]
        q_term = -(pi_q * mask_bt).sum() / denom
        if self.normalize_q_loss:
            lam = (1.0 / (pi_q.detach().abs() * mask_bt).sum().clamp(min=1e-8) * denom)
            q_term = lam * q_term
        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_term

        loss = critic_loss + actor_loss
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        with th.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                mse = ((((pi_u - act) ** 2).mean(dim=(-1, -2))) * mask_bt).sum() / denom
                acts_env = (act + 1.0) / 2.0
                valid = acts_env[mask.unsqueeze(-1).expand_as(act).bool()]
                for k, v in [("critic_loss", critic_loss), ("flow_bc_loss", bc_flow_loss),
                             ("distill_loss", distill_loss), ("q_loss", q_term),
                             ("actor_loss", actor_loss), ("grad_norm", grad_norm),
                             ("q_mean", pi_q.mean()), ("target_mean", mixed_target.mean()),
                             ("bc_mse_vs_data", mse)]:
                    self.logger.log_stat(k, float(v), t_env)
                if valid.numel() > 0:
                    self.logger.log_stat("action_std", valid.std(unbiased=False).item(), t_env)
                    self.logger.log_stat("action_at_bound_fraction",
                                         ((valid < 0.02) | (valid > 0.98)).float().mean().item(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path),
                                            map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path),
                                               map_location=lambda storage, loc: storage))
