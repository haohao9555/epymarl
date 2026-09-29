#!/bin/bash
# VMAS dispersion, seed 1, 5M, wandb project "commflow" (2026-09-28).
# CommFlow with the noise as a per-episode random identity, vs the running
# disp5m_commflow_n4_s1 (eps redrawn every step, no last action) as control:
#   A. CommFlow epsEp       eps drawn once per episode (eps_rho = 1)
#   B. CommFlow epsEp + LA  as A, plus the previous action fed to the actor's GRU
#                           (actor_last_action; critic input unchanged)
# Everything else as queue_disp5m_s1.sh. WAIT_FOR_S1=1 (default) starts only after
# the four disp5m_*_s1 runs have finished, so the saturated GPU is not shared by 9 runs.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_disp5m_epsep_s1.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 lr=0.0003 \
obs_agent_id=False t_max=5050000 save_model=True save_model_interval=1000000 \
wandb_project=commflow seed=1"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 test_eps_mode=sample"
GATE="flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=vmas \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 20
}

if [ "${WAIT_FOR_S1:-1}" = "1" ]; then
  say "waiting for the disp5m_*_s1 runs to finish"
  while pgrep -f "name=disp5m_[a-z]*_n4_s1 " > /dev/null; do sleep 60; done
fi
say "=== dispersion 5M seed 1: CommFlow epsEp / epsEp+LA ==="
launch disp5m_commflow_epsep_n4_s1   DISP5M-commflow-epsEp-n4-s1   $FLOW $GATE eps_per_episode=True
launch disp5m_commflow_epsepla_n4_s1 DISP5M-commflow-epsEp-LA-n4-s1 $FLOW $GATE eps_per_episode=True actor_last_action=True
say "=== both launched ==="
