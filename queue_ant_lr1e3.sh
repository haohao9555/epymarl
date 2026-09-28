#!/bin/bash
# Ant-4x2, lr 1e-3, seed 0, 10M -- in the order asked for on 2026-09-26:
#   1. MAPPO (textbook: exp sigma, init 1.0, learned by PPO)
#   2. MAFPO without attention
#   3. CommFlow = MAFPO + gated attention (z + a*m, a per channel, init 0.01; out_proj at
#      its default init; no normalisation anywhere in the attention)
# Both flow runs: endpoint head, K=5, eps redrawn every step, output layer = default init x0.01.
# Everything else (sigma, lr, seed, critic, PPO) identical across the three.
# At most 2 main.py jobs at once; each run starts as soon as a slot frees, in order.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_ant_lr1e3.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="env_args.key=mamujoco-Ant-4x2 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
lr=0.001 t_max=10050000 save_model=True save_model_interval=2500000 wandb_project=MAFPO_V0 seed=0"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with $COMMON "$@" name=$name wandb_run_name=$wname \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: 3 Ant-4x2 lr1e-3 runs (MAPPO -> MAFPO noattn -> CommFlow gate), max 2 concurrent ==="
launch mappo_ant4x2_lr1e3_10M MAPPO-textbook-Ant4x2-lr1e-3-10M \
  gauss_mu_source=mlp
launch mafpo_noattn_ant4x2_lr1e3_10M MAFPO-noattn-small-epsStep-Ant4x2-lr1e-3-10M \
  $FLOW flow_attention=False
launch commflow_gate_ant4x2_lr1e3_10M CommFlow-gate0.01-small-epsStep-Ant4x2-lr1e-3-10M \
  $FLOW flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0
say "=== all three launched ==="
