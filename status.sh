#!/bin/bash
# Status of the 4-run queue. Reads results/sacred/*/metrics.json, not logs/: the
# runs' stdout is block-buffered into the log files, so the logs lag by minutes.
# Returns are divided by n_agents for the single-agent (literature) scale, since
# this repo sums the common reward.
cd /root/.cache/conda/epymarl-main
echo "=== $(TZ=Australia/Sydney date '+%m-%d %H:%M') Sydney ==="
echo "--- queue ---"; for q in logs/queue_baselines.log logs/queue_next3.log; do [ -f "$q" ] && { echo "  [$q]"; tail -4 "$q"; }; done
echo "--- running ---"
ps -eo sid,pid,etime,args --no-headers | awk '$1==$2' | grep "[s]rc/main.py" | while read -r sid pid et rest; do
  key=$(echo "$rest" | grep -o "key=[^ ]*"); nm=$(echo "$rest" | grep -oP "(?<= )name=\S+")
  printf "  %-40s %-28s up %s\n" "${nm#name=}" "${key#key=}" "$et"
done
echo "--- progress (from sacred) ---"
/venv/MPE/bin/python - <<'PY'
import glob, json, os, time
# MAPPO baselines first, then the MAFPO runs on the same env
want = ["mappo_textbook_hc6x1_10M",     "mappo_textbook_hopper_10M",
        "mappo_textbook_humanoid_10M",  "mappo_textbook_ant4x2_10M",
        "mappo_textbook_swim10x2_10M",
        "mafpo_v0_attn_ppoSigma_hc6x1_10M", "mafpo_v0_attn_ppoSigma_hopper_10M",
        "mafpo_v0_attn_ppoSigma_humanoid_10M",
        "mafpo_v0_fixed_hc6x1_10M", "mafpo_v0_fixed_hopper_10M",
        "mafpo_v0_fixed_humanoid_10M", "mafpo_v0_fixed_ant4x2_10M",
        "mafpo_v0_fixed_swim10x2_10M"]
N = {"hc6x1": 6, "hopper": 3, "humanoid": 2, "ant4x2": 4, "swim10x2": 10}
rows = {}
for d in glob.glob("results/sacred/*/*/[0-9]*"):
    try:
        cfg = json.load(open(f"{d}/config.json")); nm = cfg.get("name")
        if nm not in want: continue
        m = json.load(open(f"{d}/metrics.json"))
        k = "test_return_mean"
        if k not in m or not m[k]["steps"]: continue
        age = time.time() - os.path.getmtime(f"{d}/metrics.json")
        prev = rows.get(nm)
        if prev is None or m[k]["steps"][-1] > prev[0]:
            rows[nm] = (m[k]["steps"][-1], m[k]["values"][-1], age, cfg.get("t_max", 0))
    except Exception:
        pass
for nm in want:
    if nm not in rows:
        continue
    t, r, age, tmax = rows[nm]
    n = next(v for k, v in N.items() if k in nm)
    live = "LIVE" if age < 300 else f"idle {age/3600:.1f}h"
    print(f"  {nm:38s} {t:>9}/{tmax:<9} return {r:9.0f}  (/{n} = {r/n:6.0f})  {live}")
PY
