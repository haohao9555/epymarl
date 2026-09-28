#!/bin/bash
# 4th Ant-4x2 lr 1e-3 run, added 2026-09-26: CommFlow with attention + attn_layernorm=True and
# NO gate (the user: "LN 不和门控一起"). out_proj starts at its default init x0.01 (small, not
# zero), the same way run 14 did. Otherwise identical to the other flow runs of
# queue_ant_lr1e3.sh. Waits until that queue has launched all three of its runs, then for a
# free slot (max 2 concurrent).
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_ant_lr1e3.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

say "=== re-armed: #4 is now CommFlow attn + LN WITHOUT gate (out_proj x0.01), after the first three ==="
until grep -q "all three launched" $LOG; do sleep 60; done
sleep 120
while [ "$(njobs)" -ge 2 ]; do sleep 60; done
name=commflow_LN_ant4x2_lr1e3_10M
say "-> $name"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-Ant-4x2 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
  lr=0.001 t_max=10050000 save_model=True save_model_interval=2500000 wandb_project=MAFPO_V0 seed=0 \
  gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
  endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero \
  flow_attention=True flow_attention_heads=4 attn_layernorm=True attn_out_init_scale=0.01 \
  name=$name wandb_run_name=CommFlow-LN-nogate-small-epsStep-Ant4x2-lr1e-3-10M \
  > logs/$name.log 2>&1 < /dev/null &
say "=== #4 launched ==="
