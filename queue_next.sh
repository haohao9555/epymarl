#!/bin/bash
# Order: running 3M attention probe -> 10M MAFPO-attention (NO ADER) -> textbook MAPPO.
# One job at a time.
#
#  #1 MAFPO endpoint K5 + inter-agent attention, 10M, sigma FIXED at 0.3.
#     ADER is dropped here: it is an orthogonal sigma-allocation mechanism, and
#     on the 3M probe it pushed the mean sigma to 0.54 (vs 0.38 without
#     attention) while the total entropy budget stayed conserved -- the training
#     return fell to 6.5k against 17.7k for the non-attention run, i.e. the
#     policy was fine but its exploration noise was too large. Freezing sigma
#     isolates what the attention itself does.
#  #2 MAPPO with the TEXTBOOK sigma parameterisation: sigma = exp(log_std),
#     unbounded, init 1.0, entropy_coef 0. Checks whether the 4,671 measured for
#     MAPPO was depressed by this repo's bounded sigmoid sigma (init 0.3).
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_next.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
alive(){ ps -eo args --no-headers | grep -v grep | grep -q "name=$1"; }
wait_for(){ while alive "$1"; do sleep 60; done; }

say "re-armed: attention 10M will run WITHOUT ADER (sigma fixed 0.3)"
wait_for mafpo_v0_endpoint_attn_K5_hc6x1_3M

say "-> #1 MAFPO endpoint+attention K5, sigma fixed 0.3, 10M"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  flow_param=endpoint endpoint_zero_init=True cfm_rollout_steps=5 \
  flow_attention=True flow_attention_heads=4 \
  gauss_sigma_mode=fixed sigma_init=0.3 entropy_coef=0.0 \
  t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
  name=mafpo_v0_attn_fixedsig_hc6x1_10M wandb_project=MAFPO_V0 \
  wandb_run_name=MAFPO-V0-endpoint-attn-K5-fixedSigma-HalfCheetah6x1-10M seed=0 \
  > logs/mafpo_v0_attn_fixedsig_hc6x1_10M.log 2>&1 < /dev/null &
sleep 120
wait_for mafpo_v0_attn_fixedsig_hc6x1_10M

say "-> #2 MAPPO, textbook sigma = exp(log_std), init 1.0, 10M"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  gauss_mu_source=mlp gauss_sigma_mode=ppo entropy_coef=0.0 \
  sigma_param=exp sigma_init=1.0 \
  t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
  name=mappo_textbook_hc6x1_10M wandb_project=MAFPO_V0 \
  wandb_run_name=MAPPO-textbook-expSigma-HalfCheetah6x1-10M seed=0 \
  > logs/mappo_textbook_hc6x1_10M.log 2>&1 < /dev/null &
say "both launched"
