#!/bin/bash
# The 5 HalfCheetah-6x1 lr 1e-3 15M runs still waiting (2026-09-29): MAPPO s42, MAFPO s42,
# CommFlow/MAPPO/MAFPO s10. Same arguments as queue_hc6x1_15M.sh. Swimmer is on hold
# (user), so the pending Swimmer MAFPO lr3e-4 run from queue_swim_lr3e4_then_hc.sh is dropped.
# Gate: fewer than MAXJ main.py jobs AND at least MIN_FREE_MIB of free VRAM.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_hc_rest.log
MAXJ=${MAXJ:-10}
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero"
GATE="flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"
HC="env_args.key=mamujoco-HalfCheetah-6x1 gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp \
sigma_init=1.0 lr=0.001 t_max=15050000 save_model=True save_model_interval=2500000 wandb_project=commflow"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ] || [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 120; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with $HC "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: HalfCheetah lr1e-3 15M x5 (MAPPO s42, MAFPO s42, s10 x3); max $MAXJ jobs, >= $MIN_FREE_MIB MiB free ==="
launch mappo_hc6x1_lr1e3_15M_s42          MAPPO-HC6x1-lr1e-3-15M-seed42          gauss_mu_source=mlp seed=42
launch mafpo_noattn_hc6x1_lr1e3_15M_s42   MAFPO-noattn-HC6x1-lr1e-3-15M-seed42   $FLOW flow_attention=False seed=42
launch commflow_gate_hc6x1_lr1e3_15M_s10  CommFlow-gate-HC6x1-lr1e-3-15M-seed10  $FLOW $GATE seed=10
launch mappo_hc6x1_lr1e3_15M_s10          MAPPO-HC6x1-lr1e-3-15M-seed10          gauss_mu_source=mlp seed=10
launch mafpo_noattn_hc6x1_lr1e3_15M_s10   MAFPO-noattn-HC6x1-lr1e-3-15M-seed10   $FLOW flow_attention=False seed=10
say "=== all five launched ==="
