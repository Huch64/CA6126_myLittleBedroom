"""collect_occupancy_styles.py — normalised space-usage maps for the THREE
reward-composition agents (hybrid / additive / multiplicative).

Each agent's best model is rolled out on the SAME room sequence (identical
seeds), so any difference in the occupancy map is attributable to the reward
composition alone — the visual counterpart of the radar ablation.

Output plots/occupancy_styles.npz:
    occ_<style>            (NB_Y, NB_X)   occupancy rate, final model
    occ_<style>_<cat>      (NB_Y, NB_X)   per-category occupancy
    styles                 array of style names

Usage:  python collect_occupancy_styles.py --n-eps 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_occupancy import load_model, accumulate, CATS, NB_X, NB_Y

STYLE_RUNS = {
    "hybrid":         "ppo_2M_v2",
    "additive":       "pilot_additive_2M",
    "multiplicative": "pilot_mult",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=600)
    ap.add_argument("--out", default="plots/occupancy_styles.npz")
    args = ap.parse_args()

    save = {"styles": np.array(list(STYLE_RUNS.keys()))}
    for style, run in STYLE_RUNS.items():
        path = Path(f"runs/{run}/best/best_model.zip")
        if not path.exists():
            path = Path(f"runs/{run}/final.zip")
        model = load_model(str(path))
        # same seed0 across styles → identical rooms → fair comparison
        rate, cat_rate = accumulate(model, args.n_eps, seed0=400_000, per_cat=True)
        save[f"occ_{style}"] = rate
        for c in CATS:
            save[f"occ_{style}_{c}"] = cat_rate[c]
        print(f"  {style:>14} ({run})  occupied frac = {rate.mean():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
