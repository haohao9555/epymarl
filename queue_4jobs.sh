#!/bin/bash
# Four runs, at most two at a time (more OOMs). Ordered as requested:
#   1. MAFPO endpoint+attention K5, HalfCheetah-6x1, 10M
#   2. MAFPO endpoint+attention K5, Hopper-3x1,      10M
#   3. MAPPO textbook (mlp mu, exp sigma),  Hopper-3x1, 10M
#   4. MAFPO endpoint+attention K5, Humanoid-9|8,   10M
# sigma is learned by PPO everywhere (sigma_param=exp, init 1.0), ADER off, so
# every run has exactly one exploration mechanism and it is the same one.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_4jobs.log
MAXJOBS=2
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
# count only session leaders: each run also spawns 8 env workers
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
wait_slot(){ while [ "$(njobs)" -ge $MAXJOBS ]; do sleep 60; done; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
wandb_project=MAFPO_V0 seed=0"
FLOW="flow_param=endpoint endpoint_zero_init=True cfm_rollout_steps=5 \
flow_attention=True flow_attention_heads=4"
MLP="gauss_mu_source=mlp"

launch(){   # launch <name> <wandb_run_name> <env_key> <extra args...>
  local name=$1 wname=$2 key=$3; shift 3
  wait_slot
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key=$key $COMMON "$@" name=$name wandb_run_name=$wname \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 90               # let it grab its GPU memory before the next slot check
}

say "=== queue armed (max $MAXJOBS concurrent) ==="
launch mafpo_v0_attn_ppoSigma_hc6x1_10M   MAFPO-V0-attn-K5-ppoSigma-HalfCheetah6x1-10M mamujoco-HalfCheetah-6x1 $FLOW
launch mafpo_v0_attn_ppoSigma_hopper_10M  MAFPO-V0-attn-K5-ppoSigma-Hopper3x1-10M      mamujoco-Hopper-3x1      $FLOW
launch mappo_textbook_hopper_10M          MAPPO-textbook-expSigma-Hopper3x1-10M        mamujoco-Hopper-3x1      $MLP
launch mafpo_v0_attn_ppoSigma_humanoid_10M MAFPO-V0-attn-K5-ppoSigma-Humanoid9p8-10M   mamujoco-Humanoid-9p8    $FLOW
say "=== all four launched; waiting for them to finish ==="
while [ "$(njobs)" -gt 0 ]; do sleep 120; done
say "=== all four finished ==="
