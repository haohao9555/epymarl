#!/bin/bash
# After the textbook-MAPPO baseline finishes: MAFPO endpoint K5 + inter-agent
# attention, with sigma LEARNED by PPO -- no ADER, no frozen sigma.
#
# Freezing sigma was only ever an experimental control; as a method it has no
# justification (a policy should be able to anneal its own exploration), and on
# Hopper it actively hurt. ADER is dropped here so the run has one sigma
# mechanism, the same one the MAPPO baselines use.
#
# sigma uses the TEXTBOOK parameterisation, sigma = exp(log_std), init 1.0.
# The bounded sigmoid variant this repo had been using was carried over from
# macflow/one_step_actor.py -- another algorithm's tuning, adopted without
# justification -- and it hurts at both ends: dsigma/draw is a bell curve, flat
# near 1.0 and flat near 0, so the policy starts under-explored (0.3) and later
# stalls near 0.06 instead of annealing. Measured at 1.18M on this task: the
# textbook MAPPO is at 4,355 with sigma still gliding 1.00 -> 0.63, against
# 3,363 for the bounded variant.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_attn_ppo.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
alive(){ ps -eo args --no-headers | grep -v grep | grep -q "name=mappo_textbook_hc6x1_10M"; }

say "armed: waiting for the textbook MAPPO baseline"
while alive; do sleep 60; done
say "-> MAFPO endpoint+attention K5, sigma learned by PPO (no ADER), 10M"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  flow_param=endpoint endpoint_zero_init=True cfm_rollout_steps=5 \
  flow_attention=True flow_attention_heads=4 \
  gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
  t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
  name=mafpo_v0_attn_ppoSigma_hc6x1_10M wandb_project=MAFPO_V0 \
  wandb_run_name=MAFPO-V0-attn-K5-ppoSigma-HalfCheetah6x1-10M seed=0 \
  > logs/mafpo_v0_attn_ppoSigma_hc6x1_10M.log 2>&1 < /dev/null &
say "launched"
