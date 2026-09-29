"""Summary table for the lever-pulling runs (src/envs/lever_env.py).

Groups every sacred run whose name starts with PREFIX (default "lever500k_") by method
(the name without the trailing _s<seed>) and prints, per seed and averaged over seeds:
distinct-lever fraction at a few checkpoints, the step where it first exceeds the
independent cap 1-(1-1/m)^m, the final train/test values, the final sigma, and -- with
--diag -- M1 (mu-only / executed), M4 (cross-agent |dmu_i/deps_j|) and M5 (attention off)
from scripts/lever_diag.py on the last checkpoint.

Usage: python scripts/lever_summary.py [--prefix lever500k_] [--diag]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(d):
    c = json.load(open(os.path.join(d, "config.json")))
    r = json.load(open(os.path.join(d, "run.json")))
    m = json.load(open(os.path.join(d, "metrics.json")))
    return c, r, m


def at(steps, vals, T):
    i = min(range(len(steps)), key=lambda k: abs(steps[k] - T))
    return vals[i]


def diag(d):
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts/lever_diag.py"), d, "--n", "4096"],
                         capture_output=True, text=True, cwd=REPO).stdout
    grab = lambda pat: (re.search(pat, out) or [None, None])[1]
    return {"mu_only": grab(r"mu-only ([\d.]+) \| executed"), "cross": grab(r"\(j!=i\) ([\d.]+)"),
            "attn_off": grab(r"attention off \(gate=0\): mu-only [\d.]+ \| executed ([\d.]+)")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="lever500k_")
    p.add_argument("--diag", action="store_true")
    a = p.parse_args()
    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(REPO, "results/sacred/mafpo_gauss/lever/[0-9]*"))):
        try:
            c, r, m = load(d)
        except Exception:
            continue
        if not c["name"].startswith(a.prefix):
            continue
        method = re.sub(r"_s\d+$", "", c["name"][len(a.prefix):])
        groups[method].append((c, r, m, d))

    Ts = [50e3, 100e3, 200e3, 300e3]
    head = f"{'method':12s} {'seed':>4s} " + " ".join(f"{int(T/1000):>5d}k" for T in Ts) + \
        f" {'final':>6s} {'test':>6s} {'sigma':>6s} {'>cap@':>6s} {'status':>9s}"
    if a.diag:
        head += f" {'muOnly':>6s} {'cross':>6s} {'attOff':>6s}"
    print(head)
    for method, runs in groups.items():
        finals = []
        for c, r, m, d in sorted(runs, key=lambda x: x[0]["seed"]):
            s, v = m["distinct_frac_mean"]["steps"], m["distinct_frac_mean"]["values"]
            n = int(c["env_args"]["n_agents"]); mm = int(c["env_args"].get("n_levers") or n)
            cap = 1 - (1 - 1 / mm) ** n
            cross = next((int(s[i] / 1000) for i in range(len(s)) if v[i] > cap), None)
            fin = sum(v[-3:]) / len(v[-3:]); finals.append(fin)
            te = m["test_distinct_frac_mean"]["values"][-1]; sg = m["sigma_mean"]["values"][-1]
            row = f"{method:12s} {c['seed']:>4d} " + " ".join(
                f"{at(s, v, T):6.3f}" if T <= s[-1] + 1 else "     -" for T in Ts) + \
                f" {fin:6.3f} {te:6.3f} {sg:6.3f} {str(cross) + 'k' if cross is not None else '-':>6s} {r['status']:>9s}"
            if a.diag:
                dg = diag(d)
                row += f" {dg['mu_only'] or '-':>6s} {dg['cross'] or '-':>6s} {dg['attn_off'] or '-':>6s}"
            print(row)
        print(f"{method:12s} {'mean':>4s} " + " " * (7 * len(Ts)) + f"{sum(finals) / len(finals):6.3f}")


if __name__ == "__main__":
    main()
