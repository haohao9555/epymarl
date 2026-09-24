#!/bin/bash
# Queue D (2026-09-23 Sydney), picks up after queue C's two 2x2 cells.
#   #3 MAC-Flow PyTorch port, OFFLINE on the OMIGA 6halfcheetah Expert vault
#      -> validates the port against the official JAX run (target ~3,900)
#   #4 Hopper-3x1 with PPO-learned sigma
#      -> tests whether the frozen entropy is why the fixed-sigma Hopper run
#         stalls at ~800 while its episodes still end at ~200/1000 steps
#   #5 MAFPO-Gauss first attempt on MaMuJoCo Humanoid-9|8
# Never more than 2 GPU jobs alive at once.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_d.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo args --no-headers | grep -v grep | grep -cE "src/main\.py|macflow/train_offline\.py"; }
wait_slot(){ while [ "$(njobs)" -ge 2 ]; do sleep 120; done; }

say "queue D armed: MAC-Flow offline -> Hopper sigma=ppo -> Humanoid"
sleep 300                       # let queue C's launches register first
wait_slot

say "-> #3 MAC-Flow PyTorch OFFLINE, OMIGA 6halfcheetah/Expert, 500k grad steps"
setsid nohup $PY src/macflow/train_offline.py \
  --data /data/omiga_6halfcheetah_Expert.npz --scenario HalfCheetah --agent-conf 6x1 \
  --steps 500000 --eval-interval 50000 --eval-episodes 10 --alpha 3.0 --seed 0 \
  --out results/macflow_offline_6hc_expert \
  --wandb-project MAFPO-entropy --wandb-name MACFlow-pytorch-offline-6hc-Expert \
  > logs/macflow_pytorch_offline_6hc_expert.log 2>&1 < /dev/null &
sleep 90

wait_slot
say "-> #4 Hopper-3x1, flow, sigma learned by PPO (10M)"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-Hopper-3x1 t_max=10050000 lr=0.0003 \
  gauss_sigma_mode=ppo entropy_coef=0.0 save_model=True save_model_interval=2000000 \
  name=mafpo_gauss_hopper3x1_ppoSigma_10M wandb_project=MAFPO-entropy \
  wandb_run_name=MAFPO-Gauss-Hopper3x1-flow-PPOsigma-10M seed=0 \
  > logs/mafpo_gauss_hopper3x1_ppoSigma_10M.log 2>&1 < /dev/null &
sleep 90

wait_slot
say "-> #5 MAFPO-Gauss on Humanoid-9|8 (first attempt, 10M)"
setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
  with env_args.key=mamujoco-Humanoid-9p8 t_max=10050000 lr=0.0003 \
  save_model=True save_model_interval=2000000 \
  name=mafpo_gauss_humanoid9p8_10M wandb_project=MAFPO-entropy \
  wandb_run_name=MAFPO-Gauss-Humanoid9p8-ADERq-lr3e-4-10M seed=0 \
  > logs/mafpo_gauss_humanoid9p8_10M.log 2>&1 < /dev/null &
say "queue D done (all launched)"
