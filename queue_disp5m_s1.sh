#!/bin/bash
# VMAS dispersion, seed 1, 5M, wandb project "commflow" (new machine, 2026-09-28).
# Same setting as the old machine's DISP5M runs: N=4, time_limit 40, shared reward,
# no agent id, lr 3e-4, sigma as MAPPO; flow runs evaluate with eps sampled (test_eps_mode=sample).
#   1. CommFlow           flow + gated attention in every Euler step, eps redrawn each step
#   2. MAPPO
#   3. MAPPO + attention  one gated attention round over h (communication without the noise)
#   4. MAFPO              flow, no attention
# All four start at once (well under the machine's ~10 concurrent runs).
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_disp5m_s1.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 lr=0.0003 \
obs_agent_id=False t_max=5050000 save_model=True save_model_interval=1000000 \
wandb_project=commflow seed=1"
FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=sample"
GATE="flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=vmas \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 20
}

say "=== dispersion 5M seed 1: CommFlow / MAPPO / MAPPO+attn / MAFPO ==="
launch disp5m_commflow_n4_s1  DISP5M-commflow-n4-s1  $FLOW flow_attention=True $GATE
launch disp5m_mappo_n4_s1     DISP5M-mappo-n4-s1     gauss_mu_source=mlp
launch disp5m_mappoattn_n4_s1 DISP5M-mappoattn-n4-s1 gauss_mu_source=mlp mu_attention=True $GATE
launch disp5m_mafpo_n4_s1     DISP5M-mafpo-n4-s1     $FLOW flow_attention=False
say "=== all four launched ==="
