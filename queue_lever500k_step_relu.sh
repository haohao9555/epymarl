#!/bin/bash
# Lever 500k: rerun the ReLU CommFlow runs that collapsed, each with its original configuration
# plus the step fuse A (flow_cons_mode=step, coef 1.0, margin 1.0) (2026-09-29).
#   CommFlow K=5 s0       (0.996 -> 0.200)      same args as lever500k_cf_s0
#   CommFlow noGate s42   (0.985 -> 0.36)       same args as lever500k_cfnogate_s42
#   CommFlow noGate s0    (0.995 -> 0.79)       same args as lever500k_cfnogate_s0
# (K=1 s0 also collapsed, but with a single round there is no pair of guesses to constrain.)
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_lever500k_step_relu.log
MAXJ=${MAXJ:-10}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
COMMON="use_cuda=False obs_agent_id=False gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=sigmoid \
sigma_min=0.1 sigma_max=0.5 sigma_init=0.45 lr=0.001 fpo_rollout_timesteps=256 t_max=500000 \
save_model=True save_model_interval=100000 use_wandb=True wandb_project=commflow \
test_eps_mode=sample test_sample_noise=True \
gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False endpoint_init_scale=1.0 \
flow_attention=True flow_attention_heads=4 attn_out_init_scale=1.0 \
flow_cons_mode=step flow_cons_coef=1.0 flow_cons_margin=1.0"
launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ]; do sleep 20; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=lever \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 10
}
say "=== lever 500k: ReLU collapsed runs + step fuse ==="
launch lever500k_cf_step_s0        LEVER500k-CommFlow-step-s0        attn_gate_init=0.01 seed=0
launch lever500k_cfnogate_step_s42 LEVER500k-CommFlow-noGate-step-s42 attn_gate_init=0 seed=42
launch lever500k_cfnogate_step_s0  LEVER500k-CommFlow-noGate-step-s0  attn_gate_init=0 seed=0
say "=== all three launched ==="
