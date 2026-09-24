#!/bin/bash
# Order as requested:
#   1. MAFPO no-zero-init on Hopper-3x1        (vs mafpo_v0_attn_ppoSigma_hopper_10M, 606)
#   2. MAPPO textbook on Humanoid-9|8          (the missing baseline there)
#   3. MAFPO no-zero-init on Humanoid-9|8      (vs the zero-init Humanoid now running)
#
# The MAFPO runs flip ONE flag off their zero-init counterparts:
# endpoint_zero_init True -> False.  Everything else is identical, sigma stays
# LEARNED (gauss_sigma_mode=ppo, sigma_param=exp, sigma_init=1.0).
#
# Why: endpoint_zero_init zeroes vel_fc2.weight, so the backward pass multiplies
# by that zero and dL/d(vel_fc1) -- and dL/d(attention) -- are IDENTICALLY zero at
# init. Measured per-layer on the real actor:
#   MLP (MAPPO)       total 2.25e-01   mu_fc1  1.05e-01   mu_fc2  1.90e-01
#   K5 + zero_init    total 1.11e-01   vel_fc1 0.00e+00   vel_fc2 1.08e-01   attn 0.00e+00
#   K5 no zero_init   total 2.48e-01   vel_fc1 1.14e-01   vel_fc2 2.08e-01
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_next3.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
wait_slot(){ while [ "$(njobs)" -ge 2 ]; do sleep 60; done; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
wandb_project=MAFPO_V0 seed=0"
NOZERO="flow_param=endpoint endpoint_zero_init=False cfm_rollout_steps=5 \
flow_attention=True flow_attention_heads=4"
MLP="gauss_mu_source=mlp"

launch(){   # launch <name> <wandb_name> <env_key> <extra...>
  local name=$1 wname=$2 key=$3; shift 3
  wait_slot
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key=$key $COMMON "$@" name=$name wandb_run_name=$wname \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 90
}

say "=== armed: 3 runs in the requested order, max 2 concurrent ==="
launch mafpo_v0_attn_nozero_hopper_10M   MAFPO-V0-attn-K5-NOzeroinit-Hopper3x1-10M  mamujoco-Hopper-3x1   $NOZERO
launch mappo_textbook_humanoid_10M       MAPPO-textbook-expSigma-Humanoid9p8-10M    mamujoco-Humanoid-9p8 $MLP
launch mafpo_v0_attn_nozero_humanoid_10M MAFPO-V0-attn-K5-NOzeroinit-Humanoid9p8-10M mamujoco-Humanoid-9p8 $NOZERO
say "=== all three launched ==="
