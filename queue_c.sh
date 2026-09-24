#!/bin/bash
# Queue C (2026-09-23 Sydney). Completes the 2x2 that separates "flow" from
# "entropy control" on HalfCheetah-6x1. Keeps at most 2 GPU jobs alive.
#
#              sigma learned by PPO      sigma fixed 0.3
#  mu = flow   <- #1, launched here      already have (6x1 ADER run)
#  mu = MLP    already have (MAPPO)      <- #2, launched here
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python; LOG=logs/queue_c.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
alive(){ ps -eo cmd | grep -v grep | grep -q "name=$1"; }

say "queue C armed: waiting for MAPPO-Gauss 6x1"
while alive mappo_gauss_hc6x1_10M; do sleep 60; done
say "MAPPO-Gauss 6x1 done -> launch #1 flow + PPO-learned sigma"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 t_max=10050000 lr=0.0003 \
  gauss_sigma_mode=ppo entropy_coef=0.0 save_model=True save_model_interval=2000000 \
  name=mafpo_gauss_hc6x1_flow_ppoSigma_10M wandb_project=MAFPO-entropy \
  wandb_run_name=MAFPO-Gauss-HC6x1-flow-PPOsigma-10M seed=0 \
  > logs/mafpo_gauss_hc6x1_flow_ppoSigma_10M.log 2>&1 < /dev/null &

say "waiting for Hopper-3x1"
while alive mafpo_gauss_hopper3x1_gru_fixed03; do sleep 60; done
say "Hopper done -> launch #2 MLP head + fixed sigma 0.3"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 t_max=10050000 lr=0.0003 \
  gauss_mu_source=mlp gauss_sigma_mode=fixed sigma_init=0.3 entropy_coef=0.0 \
  save_model=True save_model_interval=2000000 \
  name=mappo_gauss_hc6x1_mlp_fixedSigma_10M wandb_project=MAFPO-entropy \
  wandb_run_name=MAPPO-Gauss-HC6x1-mlp-fixedSigma0.3-10M seed=0 \
  > logs/mappo_gauss_hc6x1_mlp_fixedSigma_10M.log 2>&1 < /dev/null &

say "both launched; redrawing figures"
$PY scripts/fig_hc6x1_vs_macflow.py >> $LOG 2>&1 || say "fig1 failed"
$PY scripts/fig_hc6x1_gradsteps.py  >> $LOG 2>&1 || say "fig2 failed"
say "queue C done"
