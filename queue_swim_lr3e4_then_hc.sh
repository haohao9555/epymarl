#!/bin/bash
# One dispatcher for everything still waiting (2026-09-29), so two queues never race for a slot.
#  1) ManySegmentSwimmer-10x2, lr 3e-4, 5M, seed 0: CommFlow / MAFPO / MAPPO.
#     Single-variable test against the lr 1e-3 runs that all collapsed (sigma ~0.03, KL blow-ups,
#     negative returns); every other setting is the same as that first Swimmer batch.
#  2) The 5 HalfCheetah-6x1 lr 1e-3 15M runs left from queue_hc6x1_15M.sh
#     (MAPPO s42, MAFPO s42, CommFlow/MAPPO/MAFPO s10).
# Gate: fewer than MAXJ main.py jobs AND at least MIN_FREE_MIB of free VRAM.
cd /root/.cache/conda/epymarl-main
PY=/venv/MPE/bin/python
mkdir -p logs
LOG=logs/queue_swim_lr3e4_then_hc.log
MAXJ=${MAXJ:-10}
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}
say(){ echo "[$(TZ=Australia/Sydney date '+%m-%d %H:%M') Syd] $*" >> $LOG; }
njobs(){ ps -eo pid,sid,args --no-headers | awk '$1==$2' | grep -c "[s]rc/main.py"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

FLOW="gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False \
endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0 test_eps_mode=zero"
GATE="flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0"
SIG="gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 wandb_project=commflow save_model=True"
SWIM="env_args.key=mamujoco-ManySegmentSwimmer-10x2 lr=0.0003 t_max=5050000 save_model_interval=2500000 seed=0 $SIG"
HC="env_args.key=mamujoco-HalfCheetah-6x1 lr=0.001 t_max=15050000 save_model_interval=2500000 $SIG"

launch(){
  name=$1; wname=$2; shift 2
  while [ "$(njobs)" -ge "$MAXJ" ] || [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 120; done
  say "-> $name"
  setsid nohup $PY src/main.py --config=mafpo_gauss --env-config=mamujoco \
    with "$@" name=$name wandb_run_name=$wname > logs/$name.log 2>&1 < /dev/null &
  sleep 120
}

say "=== armed: Swimmer lr3e-4 5M x3, then HalfCheetah lr1e-3 15M x5; max $MAXJ jobs, >= $MIN_FREE_MIB MiB free ==="
launch commflow_gate_swim10x2_lr3e4_5M_s0 CommFlow-gate-Swim10x2-lr3e-4-5M-s0 $SWIM $FLOW $GATE
launch mappo_swim10x2_lr3e4_5M_s0         MAPPO-Swim10x2-lr3e-4-5M-s0         $SWIM gauss_mu_source=mlp
launch mafpo_noattn_swim10x2_lr3e4_5M_s0  MAFPO-noattn-Swim10x2-lr3e-4-5M-s0  $SWIM $FLOW flow_attention=False
say "=== Swimmer dispatched; HalfCheetah remainder next ==="
launch mappo_hc6x1_lr1e3_15M_s42          MAPPO-HC6x1-lr1e-3-15M-seed42          $HC gauss_mu_source=mlp seed=42
launch mafpo_noattn_hc6x1_lr1e3_15M_s42   MAFPO-noattn-HC6x1-lr1e-3-15M-seed42   $HC $FLOW flow_attention=False seed=42
for s in 10; do
  launch commflow_gate_hc6x1_lr1e3_15M_s$s CommFlow-gate-HC6x1-lr1e-3-15M-seed$s $HC $FLOW $GATE seed=$s
  launch mappo_hc6x1_lr1e3_15M_s$s         MAPPO-HC6x1-lr1e-3-15M-seed$s         $HC gauss_mu_source=mlp seed=$s
  launch mafpo_noattn_hc6x1_lr1e3_15M_s$s  MAFPO-noattn-HC6x1-lr1e-3-15M-seed$s  $HC $FLOW flow_attention=False seed=$s
done
say "=== all eight launched ==="
