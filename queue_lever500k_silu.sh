#!/bin/bash
# Lever 500k: rerun the failed CommFlow runs with a SiLU token activation in the flow head
# (flow_head_act=silu), everything else identical to queue_lever500k_{seeds,ablations}.sh (2026-09-29):
#   CommFlow K=5 s0     (reached 0.996, collapsed at ~320k)
#   CommFlow K=5 s8     (never took off; last-round ReLUs died, flow switched off)
#   CommFlow K=1 s0     (peaked 0.88, collapsed)
#   CommFlow noGate s42 (peaked 0.985, collapsed)
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_lever500k_silu.log
MAXJ=${MAXJ:-10}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="use_cuda=False obs_agent_id=False gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=sigmoid \
sigma_min=0.1 sigma_max=0.5 sigma_init=0.45 lr=0.001 fpo_rollout_timesteps=256 t_max=500000 \
save_model=True save_model_interval=100000 use_wandb=True wandb_project=commflow \
test_eps_mode=sample test_sample_noise=True flow_head_act=silu"
FLOW="gauss_mu_source=flow flow_param=endpoint endpoint_zero_init=False endpoint_init_scale=1.0 flow_attention=True flow_attention_heads=4 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ]; do sleep 20; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=lever \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 10
}

say "=== lever 500k SiLU reruns ==="
launch lever500k_cf_silu_s0     LEVER500k-CommFlow-SiLU-s0     $FLOW cfm_rollout_steps=5 attn_gate_init=0.01 seed=0
launch lever500k_cf_silu_s8     LEVER500k-CommFlow-SiLU-s8     $FLOW cfm_rollout_steps=5 attn_gate_init=0.01 seed=8
launch lever500k_cfk1_silu_s0   LEVER500k-CommFlow-K1-SiLU-s0  $FLOW cfm_rollout_steps=1 attn_gate_init=0.01 seed=0
launch lever500k_cfnogate_silu_s42 LEVER500k-CommFlow-noGate-SiLU-s42 $FLOW cfm_rollout_steps=5 attn_gate_init=0 seed=42
say "=== all four launched ==="
