#!/bin/bash
# When the 10M endpoint run finishes, launch a 3M probe of the same configuration
# plus inter-agent attention inside the ODE (K=5 integration steps = 5 rounds of
# negotiation, one attention pass per Euler step).
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_attn3m.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
alive(){ ps -eo args --no-headers | grep -v grep | grep -q "name=mafpo_v0_endpoint_K5_hc6x1_10M"; }

say "armed: waiting for the 10M endpoint run"
while alive; do sleep 60; done
say "10M done -> launch the 3M attention probe (K=5, 5 negotiation rounds)"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  flow_param=endpoint endpoint_zero_init=True cfm_rollout_steps=5 \
  flow_attention=True flow_attention_heads=4 \
  t_max=3050000 lr=0.0003 save_model=True save_model_interval=1500000 \
  name=mafpo_v0_endpoint_attn_K5_hc6x1_3M wandb_project=MAFPO_V0 \
  wandb_run_name=MAFPO-V0-endpoint-attn-K5-HalfCheetah6x1-3M seed=0 \
  > logs/mafpo_v0_endpoint_attn_K5_hc6x1_3M.log 2>&1 < /dev/null &
say "launched"
