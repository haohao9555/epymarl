#!/bin/bash
# MAPPO baselines first, so every later MAFPO run has something to be compared
# against on the same env, same code, same seed.
#   done      HalfCheetah-6x1   tail/6 = 7220
#   running   Hopper-3x1
#   -> 1      Humanoid-9|8      (hardest; the generalisation claim rests on it)
#   -> 2      Ant-4x2           (4 symmetric legs)
#   -> 3      ManySegmentSwimmer-10x2   (10 identical segments)
# Then the MAFPO side with all three algorithm fixes on:
#   endpoint_zero_init=False  eps_per_episode=True  test_eps_mode=sample
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_baselines.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

COMMON="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 \
t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
wandb_project=MAFPO_V0 seed=0"

launch(){   # launch <name> <wandb> <key> <extra...>
  local name=$1 wname=$2 key=$3; shift 3
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key=$key $COMMON "$@" name=$name wandb_run_name=$wname \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 90
}

say "=== armed: MAPPO baselines first ==="
launch mappo_textbook_humanoid_10M MAPPO-textbook-expSigma-Humanoid9p8-10M mamujoco-Humanoid-9p8 gauss_mu_source=mlp
launch mappo_textbook_ant4x2_10M   MAPPO-textbook-expSigma-Ant4x2-10M      mamujoco-Ant-4x2      gauss_mu_source=mlp
launch mappo_textbook_swim10x2_10M MAPPO-textbook-expSigma-Swimmer10x2-10M mamujoco-ManySegmentSwimmer-10x2 gauss_mu_source=mlp
say "=== 5 MAPPO baselines dispatched ==="
