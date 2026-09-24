# MAC-Flow (online, off-policy) — design notes

This folder implements an online, off-policy version of MAC-Flow
([arXiv:2511.05005](https://arxiv.org/abs/2511.05005), "Multi-agent
Coordination via Flow Matching"), following the MADDPG pattern already used
elsewhere in this repo: a growing replay buffer, the agent acting in the env
with its current policy, and continuous training on that buffer.

**The original MAFPO/FPO code (`controllers/fpo_mac.py`,
`modules/agents/fpo_actor.py`, `modules/critics/fpo_critic.py`,
`learners/fpo_continuous_learner.py`, `config/algs/mafpo*.yaml`) is untouched.**
Everything new lives in this folder; the only edits to shared framework files
are three additive registry lines (one import + one dict entry each) in
`controllers/__init__.py`, `learners/__init__.py`, and
`modules/agents/__init__.py`. The critic is reused as-is
(`critic_type: "maddpg_critic"`), so `modules/critics/__init__.py` and
`modules/critics/maddpg.py` are untouched too.

## What changed vs. offline MAC-Flow

Offline MAC-Flow trains three stages **sequentially**, each waiting for the
previous one to converge and then freezing it:

1. **Flow-BC** (Eq.6) — a single joint velocity field `v_phi(t,o,x)` over the
   full joint action, CFM-regressed against a fixed offline dataset.
2. **Critic** (Eq.2) — per-agent `Q_theta` (IGM), TD-trained with the
   bootstrap action drawn from the now-frozen joint flow.
3. **Q-guided distillation** (Eq.9) — decentralized one-step policies
   `mu_w_i(o_i, z_i)` trained to maximize `Q_tot` plus a BC term anchored to
   the frozen joint flow's own ODE endpoint for the same `z`.

This package runs all three **in every `train()` call**, on a minibatch
sampled from a buffer that keeps growing as the current policy collects new
data (`run.py`'s existing generic off-policy path — the same one MADDPG
already uses; nothing there needed to change). "Frozen after convergence" is
replaced by "frozen *target network*, Polyak-updated", exactly like
`target_mac`/`target_critic` already work in `maddpg_learner.py`:

| Offline stage | Online replacement |
|---|---|
| `pi_phi` trained to convergence, then used for ②'s bootstrap | `target_mac` (target **one-step actor**, cheap 1-step forward — not a full ODE draw from the flow, avoids paying for integration on every critic update) |
| `mu_phi(o,z)` trained to convergence, then frozen as ③'s BC anchor | `target_joint_flow`'s own multi-step ODE endpoint, Polyak-updated — anchoring to `self.joint_flow` (updated in the very same `train()` call) would reproduce exactly the "chasing a moving target" instability `fpo_continuous_learner.py`'s `old_mac`/`w_b` machinery was built to fight |
| Dataset assumed high-enough quality, so `alpha` must be strong enough to suppress OOD-action exploitation | Buffer is continually refreshed by the *current* policy, so there's no persistent OOD problem — `alpha` anneals down over training (`mac_flow_bc_alpha` → `mac_flow_bc_alpha_final`) instead of staying at offline strength |

## Files

- `one_step_actor.py` — `OneStepActor`: the decentralized one-step policy
  `mu_w_i(o_i,z_i)`, registered as agent `mac_flow_one_step_actor`. This is
  the network that actually acts in the environment.
- `mac_flow_mac.py` — `MACFlowMAC` controller (registered as
  `mac_flow_mac`), owns the `OneStepActor` for rollout, mirrors
  `MADDPGMAC`/`FPOMAC`'s split of responsibilities.
- `joint_flow_actor.py` — `JointFlowActor`: the single, non-recurrent,
  centrally-conditioned joint velocity field `v_phi(t,o,x)` (dim =
  `n_agents * n_actions`). Lives only inside the learner (like MADDPG's
  critic), not part of the agent registry.
- `mac_flow_learner.py` — `MACFlowLearner` (registered as
  `mac_flow_learner`): runs all three losses per `train()` call, owns
  `target_mac` / `target_critic` / `target_joint_flow`.

## Key design choices worth knowing before tuning

- **Unbounded-latent flow matching.** Both `OneStepActor` and
  `JointFlowActor` operate in an unbounded pre-sigmoid latent space; `sigmoid`
  is the only step that ever maps into the env's `(0,1)` action range. This is
  carried over unchanged from `fpo_actor.py`, where it replaced an earlier
  hard-clamp design after that was found to manufacture truncated-Gaussian-style
  boundary pileup. `JointFlowActor` additionally has to invert stored `(0,1)`
  buffer actions via `logit()` before interpolating, since (unlike
  `fpo_actor.py`'s rollout path) it has no raw pre-sigmoid value to reuse —
  the buffer's actions may come from many past policy versions.
- **No PPO ratio, no trust region.** Policy improvement is a direct
  reparameterized value gradient through the one-step actor (`-Q_tot`,
  MADDPG-style), not an importance-sampled surrogate around a flow with no
  closed-form density. This sidesteps the ratio/ELBO-mismatch failure mode
  `fpo_continuous_learner.py`'s git history documents at length; the tradeoff
  is picking up the usual off-policy actor-critic concerns instead (Q
  overestimation, target staleness) — the standard target-network toolbox
  handles these, not flow-specific machinery.
- **`mac_flow_distill_steps` is independent of the critic's bootstrap.** The
  critic's target action always comes from `target_mac`'s single forward
  pass (cheap); only the distillation BC anchor pays for a multi-step ODE
  integration, and only against the *target* flow.

## Suggested next experiment

### Parameter trajectory diagnostics

`mac_flow_theta_metrics: True` records `theta_step_norm`, `theta_step_cos`,
and `theta_disp_norm` for the executed one-step actor. The joint flow has
independent metrics named `flow_theta_*`; neither group includes critic or
target-network parameters. The fixed reference is captured after the first
complete `train()` call. Adjacent update vectors are measured after every
`train()`, including calls between scalar log entries. `*_step_cos_valid`
distinguishes a real cosine from the zero placeholder when either update
vector is unavailable or has zero length.

`mac_flow_theta_trace: True` additionally writes every update to
`<local_results_path>/theta/<unique_token>.csv`. Scalar logs retain the usual
`learner_log_interval`. Model checkpoints include `theta_metrics.th` so the
reference and previous update are preserved when those checkpoints are loaded;
older checkpoints without that file start a new reference after the next
training call. This diagnostic state does not add a loss or change the optimizer.

Run this against `mafpo_continuous` on the same env
(`pz-mpe-simple-spread-v3`) and compare `action_at_bound_fraction` and
`return_mean` stability — the core hypothesis this design is testing is that
removing the ratio machinery entirely (not just regularizing it harder, as
MAFPO's `w_b`/`use_policyflow_ratio` do) produces a materially more stable
training curve.
