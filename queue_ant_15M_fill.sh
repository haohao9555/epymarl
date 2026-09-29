#!/bin/bash
# Ant-4x2, lr 1e-3, 15M: fill CommFlow / MAFPO / MAPPO up to seeds {0, 1, 42} (2026-09-28).
# Already done on the old machine (wandb MAFPO_V0), skipped here:
#   CommFlow s0 (10M->20M), CommFlow s42 (15M), MAPPO s0 (10M->20M), MAPPO s42 (15M).
# MAFPO s0 only reached 10M there and its checkpoint is not on this machine, so it reruns
# from scratch. Settings are those of queue_ant_lr1e3.sh (configs of s0 and s42 verified
# identical apart from the seed). MAPPO+attn is out of scope for now.
# Runs start one by one whenever fewer than MAXJ main.py jobs are running AND at least
# MIN_FREE_MIB of VRAM is free: a MuJoCo run takes ~1.74 GB, so 10 of them would not fit
# in the 5080's 16.3 GB (dispersion runs take ~0.6 GB).
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_ant_15M_fill.log
MAXJ=${MAXJ:-10}
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

COMMON="env_args.key=mamujoco-Ant-4x2 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
lr=0.001 t_max=15050000 save_model=True save_model_interval=2500000 wandb_project=commflow"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero"
GATE="flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ] || [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 120; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: Ant-4x2 lr1e-3 15M fill (5 runs), max $MAXJ jobs, >= $MIN_FREE_MIB MiB free VRAM ==="
launch commflow_gate_ant4x2_lr1e3_15M_s1  CommFlow-gate-Ant4x2-lr1e-3-15M-seed1  $FLOW $GATE seed=1
launch mappo_ant4x2_lr1e3_15M_s1          MAPPO-Ant4x2-lr1e-3-15M-seed1          gauss_mu_source=mlp seed=1
launch mafpo_noattn_ant4x2_lr1e3_15M_s42  MAFPO-noattn-Ant4x2-lr1e-3-15M-seed42  $FLOW flow_attention=False seed=42
launch mafpo_noattn_ant4x2_lr1e3_15M_s1   MAFPO-noattn-Ant4x2-lr1e-3-15M-seed1   $FLOW flow_attention=False seed=1
launch mafpo_noattn_ant4x2_lr1e3_15M_s0   MAFPO-noattn-Ant4x2-lr1e-3-15M-seed0   $FLOW flow_attention=False seed=0
say "=== all five launched ==="
