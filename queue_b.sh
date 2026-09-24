#!/bin/bash
# Stage B (2026-09-22 21:45 Sydney). Max 2 GPU jobs. Currently: MAC-Flow + MAPPO-6x1.
# When MAC-Flow ends -> Hopper-3x1 (ours, fixed sigma 0.3) + redraw figure.
# When MAPPO-6x1 or Hopper ends -> MAPPO-2x3.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python; LOG=logs/queue_b.log
say(){ echo "[$(TZ=Australia/Sydney date +%H:%M) Syd] $*" >> $LOG; }
MACFLOW_PID=43383
say "stage B armed: waiting for MAC-Flow (pid $MACFLOW_PID)"
while kill -0 $MACFLOW_PID 2>/dev/null; do sleep 60; done
say "MAC-Flow done -> launch Hopper-3x1 (fixed sigma 0.3, 10M, save_model)"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco with env_args.key=mamujoco-Hopper-3x1 t_max=10050000 lr=0.0003 gauss_sigma_mode=fixed sigma_init=0.3 save_model=True save_model_interval=2000000 name=mafpo_gauss_hopper3x1_gru_fixed03_lr3e4_10M wandb_project=MAFPO-entropy wandb_run_name=MAFPO-Gauss-Hopper3x1-GRU-fixedSigma0.3-lr3e-4-10M seed=0 > logs/mafpo_gauss_hopper3x1_gru_fixed03_lr3e4_10M.log 2>&1 < /dev/null &
sleep 30
say "redraw MAC-Flow comparison figure"
$PY /tmp/claude-0/-root--cache-conda-epymarl-main/bf975d84-e78f-4c3a-9a68-1324167abe18/scratchpad/compare_fig.py >> $LOG 2>&1 || say "figure script failed"
say "waiting for a free slot (MAPPO-6x1 or Hopper)"
while pgrep -f "name=mappo_hc6x1_10M" >/dev/null && pgrep -f "name=mafpo_gauss_hopper3x1_gru_fixed03" >/dev/null; do sleep 60; done
say "slot free -> launch MAPPO HalfCheetah-2x3 (10M, save_model)"
setsid nohup $PY src/main.py --config=mappo_continuous --env-config=mamujoco with env_args.key=mamujoco-HalfCheetah-2x3 t_max=10050000 record_mov=False save_model=True save_model_interval=2000000 name=mappo_hc2x3_10M wandb_project=MAFPO-entropy wandb_run_name=MAPPO-HalfCheetah2x3-10M seed=0 > logs/mappo_hc2x3_10M.log 2>&1 < /dev/null &
say "stage B done"
