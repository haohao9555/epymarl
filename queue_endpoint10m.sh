#!/bin/bash
# When the 3M endpoint-parameterisation probe finishes, launch the full 10M run
# of the same configuration (flow_param=endpoint, K=5, exact re-integration).
# The 3M run stays on disk as its own record; the 10M one is a fresh run so the
# curve is continuous rather than stitched.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_endpoint10m.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
alive(){ ps -eo args --no-headers | grep -v grep | grep -q "name=mafpo_v0_endpoint_K5_hc6x1_3M"; }

say "armed: waiting for the 3M probe to finish"
while alive; do sleep 60; done
say "3M probe done -> launch the 10M run"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-HalfCheetah-6x1 \
  flow_param=endpoint endpoint_zero_init=True cfm_rollout_steps=5 \
  t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
  name=mafpo_v0_endpoint_K5_hc6x1_10M wandb_project=MAFPO_V0 \
  wandb_run_name=MAFPO-V0-endpoint-K5-HalfCheetah6x1-10M seed=0 \
  > logs/mafpo_v0_endpoint_K5_hc6x1_10M.log 2>&1 < /dev/null &
say "launched"
