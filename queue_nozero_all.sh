#!/bin/bash
# Is endpoint_zero_init the cause?  Three one-variable ablations: each is its
# zero_init=True counterpart with that single flag flipped, nothing else touched.
# sigma stays LEARNED everywhere (gauss_sigma_mode=ppo, sigma_param=exp, init 1.0).
#
#   counterpart runs to compare against (all seed 0, same code):
#     HalfCheetah-6x1  mafpo_v0_attn_ppoSigma_hc6x1_10M      tail/6 = 6781
#     Hopper-3x1       mafpo_v0_attn_ppoSigma_hopper_10M     tail/3 =  606
#     Humanoid-9|8     mafpo_v0_attn_ppoSigma_humanoid_10M   (running)
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_nozero_all.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
wait_slot(){ while [ "$(njobs)" -ge 2 ]; do sleep 60; done; }

launch(){   # launch <env_key> <short>
  local key=$1 short=$2 name=mafpo_v0_attn_nozero_${short}_10M
  wait_slot
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key=$key \
    flow_param=endpoint endpoint_zero_init=False cfm_rollout_steps=5 \
    flow_attention=True flow_attention_heads=4 \
    gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
    t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
    name=$name wandb_project=MAFPO_V0 \
    wandb_run_name=MAFPO-V0-attn-K5-NOzeroinit-${short}-10M seed=0 \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 90
}

say "=== armed: 3 no-zero-init ablations, max 2 concurrent ==="
launch mamujoco-HalfCheetah-6x1 hc6x1
launch mamujoco-Hopper-3x1      hopper
launch mamujoco-Humanoid-9p8    humanoid
say "=== all three launched ==="
