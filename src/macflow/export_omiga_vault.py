"""Export an OMIGA / OG-MARL flashbax vault to a plain .npz the PyTorch side can read.

The vaults are orbax stores and need flashbax + jax, which only the
`/workspace/macflow-venv` environment has; the PyTorch trainer
(`macflow/train_offline.py`) runs in `/venv/MPE`. This script is the bridge and
is meant to be run once per dataset:

    /workspace/macflow-venv/bin/python src/macflow/export_omiga_vault.py \
        --vault omiga/mamujoco/6halfcheetah.vlt --uid Expert \
        --rel-dir /data --out /data/omiga_6halfcheetah_Expert.npz

What the OMIGA MaMuJoCo vaults contain (verified on 6halfcheetah/Expert):
  observations (T, N, 23)  the FULL 17-dim single-agent state, one-hot agent id
                           appended, then each vector normalised to zero mean /
                           unit std -- so every agent sees the same 17 numbers
                           and the id is already inside the observation. Do NOT
                           append an agent id again downstream.
  actions      (T, N, 1)   NOT clipped to the action box: range is about
                           [-3.3, 3.2]. MuJoCo clamps ctrl to the actuator
                           range internally, so these execute as [-1, 1], but
                           the stored values are what MAC-Flow's BC flow
                           regresses onto (the official code does not clip the
                           flow-matching target either).
  rewards      (T, N)      identical across agents (shared team reward)
  terminals / truncations  (T, N), both set on the final step of each episode
  infos/state  (T, 23)     unused by the official algorithm
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="omiga/mamujoco/6halfcheetah.vlt")
    ap.add_argument("--uid", default="Expert")
    ap.add_argument("--rel-dir", default="/data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from flashbax.vault import Vault

    exp = Vault(vault_name=args.vault, vault_uid=args.uid, rel_dir=args.rel_dir).read().experience
    get = lambda k: np.asarray(exp[k][0], dtype=np.float32)
    obs, act = get("observations"), get("actions")
    rew, term, trunc = get("rewards"), get("terminals"), get("truncations")

    # episode boundaries: an episode ends where either flag fires (agent 0 is
    # representative -- the flags are identical across agents)
    ends = np.where((term[:, 0] > 0) | (trunc[:, 0] > 0))[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts + 1
    rets = np.array([rew[s:e + 1, 0].sum() for s, e in zip(starts, ends)])

    np.savez(args.out, observations=obs, actions=act, rewards=rew,
             terminals=term, truncations=trunc,
             ep_starts=starts.astype(np.int64), ep_lengths=lengths.astype(np.int64))
    print(f"{args.out}: T={len(obs)} N={obs.shape[1]} obs_dim={obs.shape[2]} act_dim={act.shape[2]}")
    print(f"  episodes={len(rets)} len={lengths.min()}..{lengths.max()} "
          f"return mean={rets.mean():.1f} std={rets.std():.1f} max={rets.max():.1f}")
    print(f"  action range [{act.min():.3f}, {act.max():.3f}]")


if __name__ == "__main__":
    main()
