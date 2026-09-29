#!/bin/bash
# Lever pulling (m = N = 5), 50k episodes, seeds {0, 42, 8}: CommFlow vs MAPPO (2026-09-29).
# Both with the configuration that first worked: sigma learnable in [0.1, 0.5]
# (sigma_param=sigmoid, init 0.45), updates every 256 episodes, test with n sampled.
# CommFlow adds endpoint_init_scale 1.0 (flow output layer at default init), gate 0.01, K=5.
# CPU only (use_cuda=False); starts a run whenever fewer than MAXJ main.py jobs are running.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_lever_seeds.log
MAXJ=${MAXJ:-10}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="use_cuda=False obs_agent_id=False gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=sigmoid \
sigma_min=0.1 sigma_max=0.5 sigma_init=0.45 lr=0.001 fpo_rollout_timesteps=256 t_max=50000 \
save_model=True save_model_interval=25000 use_wandb=True wandb_project=commflow \
test_eps_mode=sample test_sample_noise=True"
CF="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=1.0 flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ]; do sleep 20; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=lever \
    with $COMMON "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 10
}

say "=== lever 50k, seeds 0/42/8, CommFlow + MAPPO, sigma in [0.1,0.5] ==="
for s in 0 42 8; do
  launch lever50k_cf_s$s    LEVER50k-CommFlow-sig0.1to0.5-s$s $CF seed=$s
  launch lever50k_mappo_s$s LEVER50k-MAPPO-sig0.1to0.5-s$s    gauss_mu_source=mlp seed=$s
done
say "=== all six launched ==="
