#!/bin/bash
# Lever 500k: can the collapse of CommFlow K=5 (SiLU) be stopped by constraining the per-round
# amplification c = dg/dx of the flow head, without killing the coordination? (2026-09-29)
# Same config as queue_lever500k_silu.sh (sigma in [0.1,0.5], SiLU, K=5, gate 0.01), seeds 0 and 8
# (both collapsed at ~180k / ~225k), one run per fix:
#   step   : fuse -- penalise |g_{k+1}-g_k| beyond margin 1.0, coef 1.0 (flow_cons_mode=step)
#   anchor : every round's guess towards sg(mu), coef 0.1 (flow_cons_mode=anchor)
#   bound  : g = 5*tanh(g/5) (cfm_velocity_bound=5), the GRU-style bounded candidate
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_lever500k_cons.log
MAXJ=${MAXJ:-10}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
COMMON="use_cuda=False obs_agent_id=False gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=sigmoid \
sigma_min=0.1 sigma_max=0.5 sigma_init=0.45 lr=0.001 fpo_rollout_timesteps=256 t_max=500000 \
save_model=True save_model_interval=100000 use_wandb=True wandb_project=commflow \
test_eps_mode=sample test_sample_noise=True flow_head_act=silu \
gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False endpoint_init_scale=1.0 \
flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"
launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ]; do sleep 20; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=lever \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 10
}
say "=== lever 500k: flow_cons fixes on SiLU K=5, seeds 0/8 ==="
for s in 0 8; do
  launch lever500k_cf_silu_step_s$s   LEVER500k-CommFlow-SiLU-step-s$s   flow_cons_mode=step flow_cons_coef=1.0 flow_cons_margin=1.0 seed=$s
  launch lever500k_cf_silu_anchor_s$s LEVER500k-CommFlow-SiLU-anchor-s$s flow_cons_mode=anchor flow_cons_coef=0.1 seed=$s
  launch lever500k_cf_silu_bound_s$s  LEVER500k-CommFlow-SiLU-bound5-s$s cfm_velocity_bound=5.0 seed=$s
done
say "=== all six launched ==="
