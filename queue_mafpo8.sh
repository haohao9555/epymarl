#!/bin/bash
# MAFPO with the delta_v ratio (option A) on the eight standard MaMuJoCo tasks.
# Two jobs at a time; wandb project "MAFPO" (fresh).
#
# Option A = the PPO ratio's endpoint shift is estimated as
#   mean_t[ v_new(x_t,t) - v_old(x_t,t) ],  x_t = (1-t)*eps + t*mu_old
# so the velocity field receives gradient along the flow-matching interpolation
# path instead of only at the ODE endpoint. Everything else is the config default
# (flow mu, ADER entropy budget, lr 3e-4, GRU).
#
# Idempotent: a task whose run is already alive, or whose log already says
# "Finished Training", is skipped, so this script can be re-armed after a restart
# without duplicating work.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_mafpo8.log
PROJ=MAFPO
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }

# Count PARENT runs only. Each parallel_runner run is 1 parent + batch_size_run
# worker processes sharing the same cmdline, so `grep -c` counts 9 per run. The
# parent is the session leader (setsid), i.e. PID == SID; its PPID stays this
# script for as long as this script lives, so PPID is not a usable discriminator.
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2 && /src\/main\.py/' | wc -l; }
wait_slot(){ while [ "$(njobs)" -ge 2 ]; do sleep 120; done; }
running(){ ps -eo args --no-headers | grep -v grep | grep -q "name=$1"; }

TASKS=(
  "mamujoco-HalfCheetah-2x3"
  "mamujoco-HalfCheetah-6x1"
  "mamujoco-Hopper-3x1"
  "mamujoco-Walker2d-2x3"
  "mamujoco-Ant-2x4"
  "mamujoco-Ant-4x2"
  "mamujoco-Humanoid-9p8"
  "mamujoco-HumanoidStandup-9p8"
)

say "queue armed: 8 tasks, delta_v ratio, project=$PROJ"
for KEY in "${TASKS[@]}"; do
  SHORT=$(echo "$KEY" | sed 's/mamujoco-//')
  NAME="mafpo_dv_${SHORT}_10M"
  if running "$NAME"; then say "skip $SHORT (already running)"; continue; fi
  if grep -q "Finished Training" "logs/${NAME}.log" 2>/dev/null; then
    say "skip $SHORT (already finished)"; continue
  fi
  wait_slot
  say "-> $SHORT"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key="$KEY" use_delta_v_ratio=True t_max=10050000 lr=0.0003 \
    save_model=True save_model_interval=2500000 \
    name="$NAME" wandb_project=$PROJ \
    wandb_run_name="MAFPO-deltaV-${SHORT}-10M" seed=0 \
    > "logs/${NAME}.log" 2>&1 < /dev/null &
  sleep 120
done
say "all 8 launched or accounted for"
