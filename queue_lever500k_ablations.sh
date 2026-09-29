#!/bin/bash
# Lever pulling (m = N = 5), 500k episodes, seeds {0, 42, 8}: the comparison set around CommFlow
# (2026-09-29). Same configuration as queue_lever500k_seeds.sh (CommFlow + MAPPO):
# sigma learnable in [0.1, 0.5] (sigmoid, init 0.45), updates every 256 episodes, CPU only.
# In priority order:
#   1. MAFPO          flow, no attention                  -> flow alone stays under the 0.672 cap
#   2. MAPPO+attn     Gaussian mean head + attention on h -> talking about state is not enough
#   3. CommFlow h_only  messages carry h but not x_k/eps  -> it is the exchanged noise that matters
#   4. CommFlow K=1   a single round of communication
#   5. CommFlow no gate (attn_gate_init=0)
# Waits until queue_lever500k_seeds.sh has dispatched all six runs, then starts a run whenever
# fewer than MAXJ main.py jobs are running.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_lever500k_ablations.log
MAXJ=${MAXJ:-10}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="use_cuda=False obs_agent_id=False gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=sigmoid \
sigma_min=0.1 sigma_max=0.5 sigma_init=0.45 lr=0.001 fpo_rollout_timesteps=256 t_max=500000 \
save_model=True save_model_interval=100000 use_wandb=True wandb_project=commflow \
test_eps_mode=sample test_sample_noise=True"
FLOW="gauss_mu_source=flow flow_param=endpoint endpoint_zero_init=False endpoint_init_scale=1.0"
ATT="flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ]; do sleep 20; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=lever \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 10
}

say "=== armed: lever 500k ablations x3 seeds; waiting for queue_lever500k_seeds.sh to dispatch ==="
until grep -q "all six launched" logs/queue_lever500k_seeds.log 2>/dev/null; do sleep 30; done
for s in 0 42 8; do launch lever500k_mafpo_s$s    LEVER500k-MAFPO-sig0.1to0.5-s$s    $FLOW cfm_rollout_steps=5 flow_attention=False seed=$s; done
for s in 0 42 8; do launch lever500k_mappoatt_s$s LEVER500k-MAPPOattn-sig0.1to0.5-s$s gauss_mu_source=mlp mu_attention=True $ATT seed=$s; done
for s in 0 42 8; do launch lever500k_cfhonly_s$s  LEVER500k-CommFlow-hOnly-sig0.1to0.5-s$s $FLOW cfm_rollout_steps=5 flow_attention=True $ATT flow_attention_kv=h_only seed=$s; done
for s in 0 42 8; do launch lever500k_cfk1_s$s     LEVER500k-CommFlow-K1-sig0.1to0.5-s$s $FLOW flow_attention=True $ATT cfm_rollout_steps=1 seed=$s; done
for s in 0 42 8; do launch lever500k_cfnogate_s$s LEVER500k-CommFlow-noGate-sig0.1to0.5-s$s $FLOW cfm_rollout_steps=5 flow_attention=True flow_attention_heads=4 attn_gate_init=0 attn_out_init_scale=1.0 seed=$s; done
say "=== all fifteen launched ==="
