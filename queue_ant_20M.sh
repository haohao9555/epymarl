#!/bin/bash
# Ant-4x2, lr 1e-3, 20M -- asked for on 2026-09-27:
#   1. MAPPO lr 1e-3 (seed 0) resumed from its 10M checkpoint to 20M, the same way the
#      CommFlow gate run was continued (commflow_gate_ant4x2_lr1e3_20M): identical config,
#      checkpoint_path + t_max 20.05M. Neither 10M checkpoint has reward_norm.th, so both
#      resumes restart rew_ms/ret_ms alike.
#   2. CommFlow gate, seed 42, from scratch to 15M
#   3. MAPPO, seed 42, from scratch to 15M
#   (seed 42 cut from 20M to 15M on 2026-09-27: past ~16M the seed-0 gate run gets
#    faster but falls more, and test returns swing by +-500 between evaluations)
# Configs are those of queue_ant_lr1e3.sh (MAPPO = run 1, CommFlow gate = run 3).
# At most 2 main.py jobs at once; each starts as soon as a slot frees, in order.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_ant_20M.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="env_args.key=mamujoco-Ant-4x2 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
lr=0.001 save_model=True save_model_interval=2500000 wandb_project=MAFPO_V0"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero"
GATE="flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"
MAPPO_CKPT="results/models/mappo_ant4x2_lr1e3_10M_seed0_mamujoco-Ant-4x2_2026-09-26 12:30:22.775455"

launch(){
  name=$1; shift
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with $COMMON "$@" name=$name > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== re-armed: MAPPO 10->20M resume, then seed 42 CommFlow gate + MAPPO (15M, lr 1e-3), max 2 concurrent ==="
launch mappo_ant4x2_lr1e3_20M gauss_mu_source=mlp seed=0 t_max=20050000 "checkpoint_path=$MAPPO_CKPT" \
  wandb_run_name=MAPPO-Ant4x2-lr1e-3-10Mto20M
launch commflow_gate_ant4x2_lr1e3_15M_s42 $FLOW $GATE seed=42 t_max=15050000 \
  wandb_run_name=CommFlow-gate-Ant4x2-lr1e-3-15M-seed42
launch mappo_ant4x2_lr1e3_15M_s42 gauss_mu_source=mlp seed=42 t_max=15050000 \
  wandb_run_name=MAPPO-Ant4x2-lr1e-3-15M-seed42
say "=== all three launched ==="
