"""collect_occupancy.py — aggregate normalised space-usage maps for the
hybrid agent (ppo_2M_v2).

For each episode the room is resampled onto a fixed NB_X x NB_Y grid in
*relative* coordinates, so rooms of different sizes can be averaged. We
accumulate, per normalised cell, how often it ends up occupied by
furniture. The complement (1 - occupancy) is the "kept-clear" map — it
reveals the circulation space and door clearance the agent learns to
leave open (the visual counterpart of the efficiency / compactness reward).

Outputs plots/occupancy_data.npz:
    checkpoints     (K,)              step counts of the sampled checkpoints
    occ_rate        (K, NB_Y, NB_X)   occupancy rate per checkpoint (evolution)
    occ_final       (NB_Y, NB_X)      occupancy rate of the final agent
    occ_bed/.../    (NB_Y, NB_X)      per-category occupancy of the final agent

Usage:
    python collect_occupancy.py --n-eps 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom, CATALOG
from train import FactoredMaskablePolicy

RUN = "ppo_2M_v2"
NB_X, NB_Y = 26, 22          # normalised bins (matches obs-grid aspect)
CATS = ["bed", "wardrobe", "desk", "cabinet", "nightstand"]
N_EVO = 6                     # checkpoints sampled for the evolution map


def load_model(path: str) -> MaskablePPO:
    return MaskablePPO.load(path, custom_objects={
        "policy_class":  FactoredMaskablePolicy,
        "learning_rate": 0.0,
        "lr_schedule":   lambda _: 0.0,
        "clip_range":    lambda _: 0.0,
    })


def accumulate(model, n_eps: int, seed0: int, per_cat: bool = False):
    """Return (occ_rate, per_cat_rate dict|None) on the NB_Y x NB_X grid."""
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    occ = np.zeros((NB_Y, NB_X))
    tot = np.zeros((NB_Y, NB_X))
    cat_occ = {c: np.zeros((NB_Y, NB_X)) for c in CATS} if per_cat else None

    # precompute normalised bin index for a room cell
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            m = env.action_masks()
            a, _ = model.predict(obs, action_masks=m, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                break
        rw, rh = env.room_w, env.room_h
        # category id per room cell (0 = empty)
        cat_grid = {}
        if per_cat:
            for p in env.placed:
                c = CATALOG[p.fid].cat
                for dy in range(p.fh):
                    for dx in range(p.fw):
                        cat_grid[(p.x + dx, p.y + dy)] = c
        for cy in range(rh):
            by = min(NB_Y - 1, int(cy / rh * NB_Y))
            for cx in range(rw):
                bx = min(NB_X - 1, int(cx / rw * NB_X))
                tot[by, bx] += 1
                if env.grid[cy, cx] > 0:
                    occ[by, bx] += 1
                    if per_cat and (cx, cy) in cat_grid:
                        cat_occ[cat_grid[(cx, cy)]][by, bx] += 1

    rate = np.divide(occ, tot, out=np.zeros_like(occ), where=tot > 0)
    cat_rate = None
    if per_cat:
        cat_rate = {c: np.divide(cat_occ[c], tot, out=np.zeros_like(occ), where=tot > 0)
                    for c in CATS}
    return rate, cat_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=800)
    ap.add_argument("--out", default="plots/occupancy_data.npz")
    args = ap.parse_args()

    ckpt_dir = Path(f"runs/{RUN}/checkpoints")
    ckpts = sorted(ckpt_dir.glob("model_*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[1]))
    steps = [int(p.stem.split("_")[1]) for p in ckpts]

    # evolution: N_EVO evenly-spaced checkpoints
    idx = np.linspace(0, len(ckpts) - 1, N_EVO).round().astype(int)
    evo_ckpts = [(steps[i], ckpts[i]) for i in idx]

    print(f"occupancy: {N_EVO} evolution checkpoints + final, {args.n_eps} eps each")
    occ_rates = []
    occ_rates_cat = {c: [] for c in CATS}     # per-category occupancy per checkpoint
    for step, ck in evo_ckpts:
        model = load_model(str(ck))
        rate, cat_rate_evo = accumulate(model, args.n_eps, seed0=200_000, per_cat=True)
        occ_rates.append(rate)
        for c in CATS:
            occ_rates_cat[c].append(cat_rate_evo[c])
        print(f"  {step:>9,}  occupied frac = {rate.mean():.3f}")

    # final agent: full per-category breakdown (more episodes)
    final_step, final_ck = steps[-1], ckpts[-1]
    model = load_model(str(final_ck))
    occ_final, cat_rate = accumulate(model, args.n_eps, seed0=300_000, per_cat=True)
    print(f"  final {final_step:,}  per-category map done")

    save = {
        "checkpoints": np.array([s for s, _ in evo_ckpts]),
        "occ_rate": np.stack(occ_rates),
        "occ_final": occ_final,
    }
    for c in CATS:
        save[f"occ_{c}"] = cat_rate[c]                      # final per-category
        save[f"occ_evo_{c}"] = np.stack(occ_rates_cat[c])  # (N_EVO, NB_Y, NB_X)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
