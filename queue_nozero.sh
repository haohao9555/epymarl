#!/bin/bash
# One-variable ablation off mafpo_v0_attn_ppoSigma_hc6x1_10M (tail 6781 /6):
# only endpoint_zero_init flips True -> False.
#
# Why: with endpoint_zero_init=True, vel_fc2.weight = 0, so the backward pass
# multiplies by that zero and dL/d(vel_fc1) is IDENTICALLY zero at init -- as is
# dL/d(attention), whose own out_proj is zero too. Measured on the real actor:
#   MLP (MAPPO)      total grad 2.25e-01   mu_fc1 1.05e-01   mu_fc2 1.90e-01
#   K5 + zero_init   total grad 1.11e-01   vel_fc1 0.00e+00  vel_fc2 1.08e-01  attn 0.00e+00
#   K5 no zero_init  total grad 2.48e-01   vel_fc1 1.14e-01  vel_fc2 2.08e-01
# So the flow head starts as a single linear layer on FROZEN random features while
# MAPPO trains both of its layers from step one, and attention cannot move at all.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_nozero.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

say "armed: waiting for a free slot (max 2 concurrent)"
while [ "$(njobs)" -ge 2 ]; do sleep 60; done
say "-> mafpo_v0_attn_nozero_hc6x1_10M (endpoint_zero_init=False, everything else identical)"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  flow_param=endpoint endpoint_zero_init=False cfm_rollout_steps=5 \
  flow_attention=True flow_attention_heads=4 \
  gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
  t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
  name=mafpo_v0_attn_nozero_hc6x1_10M wandb_project=MAFPO_V0 \
  wandb_run_name=MAFPO-V0-attn-K5-NOzeroinit-HalfCheetah6x1-10M seed=0 \
  > logs/mafpo_v0_attn_nozero_hc6x1_10M.log 2>&1 < /dev/null &
say "launched"
