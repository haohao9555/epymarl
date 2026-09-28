#!/bin/bash
# 2026-09-27: dispersion jumps the queue (user). Replaces queue_disp.sh and the not-yet-
# launched tail of queue_ant_20M.sh. Runs already training are left alone; as slots free
# (max 2 main.py jobs), start:
#   1-5. the dispersion pilot (see queue_disp.sh for the design): N=4, seed 0, lr 3e-4,
#        3M, no agent id -- CommFlow, MAPPO, MAPPO+attention, MAFPO, CommFlow epsEp
#   6.   mappo_ant4x2_lr1e3_15M_s42 (the seed-42 MAPPO partner of the running
#        commflow_gate_ant4x2_lr1e3_15M_s42), config unchanged from queue_ant_20M.sh
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_front_disp.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

SIGMA="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 wandb_project=MAFPO_V0"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 test_eps_mode=zero"
GATE="flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"
DISP="$SIGMA lr=0.0003 obs_agent_id=False t_max=3050000 save_model=True save_model_interval=1000000 seed=0"

launch(){
  envcfg=$1; name=$2; shift 2
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=$envcfg \
    with "$@" name=$name > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: 5 dispersion runs first, then the seed-42 Ant MAPPO; max 2 concurrent ==="
launch vmas disp_commflow_n4 $DISP $FLOW flow_attention=True $GATE eps_per_episode=False eps_rho=0.0 \
  wandb_run_name=DISP-disp_commflow_n4
launch vmas disp_mappo_n4 $DISP gauss_mu_source=mlp wandb_run_name=DISP-disp_mappo_n4
launch vmas disp_mappo_attn_n4 $DISP gauss_mu_source=mlp mu_attention=True $GATE \
  wandb_run_name=DISP-disp_mappo_attn_n4
launch vmas disp_mafpo_n4 $DISP $FLOW flow_attention=False eps_per_episode=False eps_rho=0.0 \
  wandb_run_name=DISP-disp_mafpo_n4
launch vmas disp_commflow_epsEp_n4 $DISP $FLOW flow_attention=True $GATE eps_per_episode=True \
  wandb_run_name=DISP-disp_commflow_epsEp_n4
launch mamujoco mappo_ant4x2_lr1e3_15M_s42 env_args.key=mamujoco-Ant-4x2 $SIGMA lr=0.001 \
  save_model=True save_model_interval=2500000 gauss_mu_source=mlp seed=42 t_max=15050000 \
  wandb_run_name=MAPPO-Ant4x2-lr1e-3-15M-seed42
say "=== all six launched ==="
