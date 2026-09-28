#!/bin/bash
# VMAS dispersion pilot (2026-09-27): N=4 agents spawn at the origin with identical
# observations and must split over 4 food particles; parameter sharing, NO agent id.
# time_limit 40 (see config/envs/vmas.yaml): a perfect split eats all 4 in ~26 steps,
# herding eats ~2.35 on average, so the return separates the two.
# One seed each, lr 3e-4 (the setting that was stable for both MAPPO and the gated
# CommFlow on Ant), sigma as MAPPO, 3M steps:
#   1. CommFlow           flow + gated attention in every Euler step, eps redrawn each step
#   2. MAPPO
#   3. MAPPO + attention  one gated attention round over h (communication without the noise)
#   4. MAFPO              flow, no attention
#   5. CommFlow epsEp     as 1 but one eps per episode (role persistence)
# Starts only after queue_ant_20M.sh has launched all its runs, then fills free slots,
# at most 2 main.py jobs at once.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_disp.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 lr=0.0003 \
obs_agent_id=False t_max=3050000 save_model=True save_model_interval=1000000 \
wandb_project=MAFPO_V0 seed=0"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 test_eps_mode=zero"
GATE="flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; shift
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=vmas \
    with $COMMON "$@" name=$name wandb_run_name=DISP-$name > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: dispersion pilot (5 runs, N=4, seed 0, 3M), waits for the Ant 20M queue ==="
until grep -q "all three launched" logs/queue_ant_20M.log; do sleep 60; done
sleep 120
launch disp_commflow_n4 $FLOW flow_attention=True $GATE eps_per_episode=False eps_rho=0.0
launch disp_mappo_n4 gauss_mu_source=mlp
launch disp_mappo_attn_n4 gauss_mu_source=mlp mu_attention=True $GATE
launch disp_mafpo_n4 $FLOW flow_attention=False eps_per_episode=False eps_rho=0.0
launch disp_commflow_epsEp_n4 $FLOW flow_attention=True $GATE eps_per_episode=True
say "=== all five launched ==="
