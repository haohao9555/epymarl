#!/bin/bash
# HalfCheetah-6x1, lr 1e-3, 15M: CommFlow / MAFPO / MAPPO x seeds {8, 42, 10} = 9 runs (2026-09-28).
# Same recipe as the Ant-4x2 mainline (queue_ant_15M_fill.sh): sigma as MAPPO (exp, init 1,
# PPO-trained, no entropy bonus), flow = endpoint K=5, small output init, eps redrawn each step,
# gate 0.01, test_eps_mode=zero. wandb project "commflow".
# Waits until queue_ant_15M_fill.sh has dispatched all its runs, then uses the same gate:
# fewer than MAXJ main.py jobs AND at least MIN_FREE_MIB of free VRAM.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_hc6x1_15M.log
MAXJ=${MAXJ:-10}
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

COMMON="env_args.key=mamujoco-HalfCheetah-6x1 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp \
sigma_init=1.0 lr=0.001 t_max=15050000 save_model=True save_model_interval=2500000 wandb_project=commflow"
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

say "=== armed: HalfCheetah-6x1 lr1e-3 15M, 9 runs; waiting for the Ant fill queue to dispatch ==="
until grep -q "all five launched" logs/queue_ant_15M_fill.log 2>/dev/null; do sleep 120; done
for s in 8 42 10; do
  launch commflow_gate_hc6x1_lr1e3_15M_s$s CommFlow-gate-HC6x1-lr1e-3-15M-seed$s $FLOW $GATE seed=$s
  launch mappo_hc6x1_lr1e3_15M_s$s         MAPPO-HC6x1-lr1e-3-15M-seed$s         gauss_mu_source=mlp seed=$s
  launch mafpo_noattn_hc6x1_lr1e3_15M_s$s  MAFPO-noattn-HC6x1-lr1e-3-15M-seed$s  $FLOW flow_attention=False seed=$s
done
say "=== all nine launched ==="
