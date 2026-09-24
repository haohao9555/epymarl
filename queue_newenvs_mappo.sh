#!/bin/bash
# MAPPO baselines on the two symmetric factorisations. These are needed for the
# paper whatever the zero-init ablation says, so they fill slots productively
# while the MAFPO side waits for that answer.
#   Ant-4x2                 4 morphologically symmetric legs
#   ManySegmentSwimmer-10x2 10 identical segments; a travelling-wave gait is a
#                           phase assignment among interchangeable agents
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
LOG=logs/queue_newenvs_mappo.log
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }

# do not interleave with the earlier queue: wait for it to finish dispatching
say "waiting for queue_next3 to finish dispatching"
while pgrep -f 'queue_next3[.]sh' > /dev/null; do sleep 120; done
say "queue_next3 done dispatching"

COMMON="gauss_mu_source=mlp gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp \
sigma_init=1.0 t_max=10050000 lr=0.0003 save_model=True save_model_interval=2500000 \
wandb_project=MAFPO_V0 seed=0"

launch(){   # launch <short> <env_key>
  local short=$1 key=$2 name=mappo_textbook_${short}_10M
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with env_args.key=$key $COMMON name=$name \
    wandb_run_name=MAPPO-textbook-expSigma-${short}-10M \
    > logs/$name.log 2>&1 < /dev/null &
  sleep 90
}
launch ant4x2   mamujoco-Ant-4x2
launch swim10x2 mamujoco-ManySegmentSwimmer-10x2
say "=== both launched ==="
