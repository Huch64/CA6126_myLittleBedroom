"""
evaluate.py — Unified evaluation for trained agents.

Compares multiple agents (possibly trained with different reward styles)
on the SAME yardstick: full v5 (hybrid) environment + reward-independent
behavior metrics. Output a per-episode CSV + aggregate CSV ready for radar
plot.

Why force hybrid for eval?
  Each agent was trained to maximize ITS OWN reward formula. But when we
  ask "which agent built better bedrooms?", we must judge under a single
  definition of "good". We pick the hybrid (v5) formula as the reference,
  but also report 6 reward-independent behavior metrics so the radar plot
  doesn't depend on this choice.

Usage:
    python evaluate.py --models pilot_hybrid pilot_additive pilot_mult \
                       --n-eps 500
    # outputs:
    #   runs/<model>/eval_episodes.csv    per-episode metrics
    #   runs/evaluation_summary.csv       aggregated means across agents
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom, CATALOG


# Behavior metrics we record per episode (reward-independent or low-dependence)
BEHAVIOR_KEYS = [
    # Reward components (computed under hybrid = v5)
    "total_v5", "availability", "diversity", "compactness", "shape_coef",
    "privacy", "light", "efficiency",
    # Structural counts
    "n_placed", "n_categories", "bed_placed",
    "furniture_area",
    # Geometric / behavior (reward-independent)
    "wardrobe_wall_dist",      # min dist of wardrobe to any wall (0 = on wall)
    "pillow_exposed_rate",     # fraction of pillow cells visible from door
    "unreachable_cells",
]


def _wardrobe_wall_dist(env) -> float:
    """Minimum distance from any wardrobe cell to the nearest room wall.

    0 = at least one cell touches a wall.
    -1 = no wardrobe placed (sentinel).
    """
    rw, rh = env.room_w, env.room_h
    cells = []
    for p in env.placed:
        if CATALOG[p.fid].cat != "wardrobe":
            continue
        for dy in range(p.fh):
            for dx in range(p.fw):
                cells.append((p.x + dx, p.y + dy))
    if not cells:
        return -1.0
    return float(min(min(x, y, rw - 1 - x, rh - 1 - y) for x, y in cells))


def evaluate_agent(model_path: str, n_eps: int, seed: int = 12345) -> list[dict]:
    """Run model for n_eps episodes in full v5 env; return per-ep metric dicts."""
    model = MaskablePPO.load(model_path)
    env = MyLittleBedroom(seed=seed, reward_style="hybrid")   # unified eval

    out = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=seed + ep)
        while True:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            if term or trunc:
                bd = env._last_breakdown
                placed = env.placed
                pillow_total = max(1, bd.get("total_pillow_cells", 1))
                row = {
                    "ep": ep,
                    "total_v5": bd["total"],
                    "availability": bd["availability"],
                    "diversity": bd.get("diversity", 0.0),
                    "compactness": bd.get("compactness", 0.0),
                    "shape_coef": bd.get("shape_coef", 0.0),
                    "privacy": bd["privacy"],
                    "light": bd["light"],
                    "efficiency": bd["efficiency"],
                    "n_placed": len(placed),
                    "n_categories": bd.get("n_categories", 0),
                    "bed_placed": int(any(CATALOG[p.fid].cat == "bed" for p in placed)),
                    "furniture_area": sum(p.fw * p.fh for p in placed),
                    "wardrobe_wall_dist": _wardrobe_wall_dist(env),
                    "pillow_exposed_rate": bd.get("n_exposed_pillow", 0) / pillow_total,
                    "unreachable_cells": bd["unreachable_cells"],
                }
                out.append(row)
                break
    return out


def aggregate(per_ep: list[dict], agent_name: str) -> dict:
    """Mean + std per metric (ignore -1 sentinel for wardrobe dist)."""
    agg = {"agent": agent_name}
    for k in BEHAVIOR_KEYS:
        vals = [m[k] for m in per_ep if not (k == "wardrobe_wall_dist" and m[k] < 0)]
        if not vals:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
        else:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    agg["n_eps"] = len(per_ep)
    return agg


def save_per_ep_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def save_summary_csv(path: Path, results: list[dict]):
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["agent", "n_eps"] + [f"{k}_mean" for k in BEHAVIOR_KEYS] + \
           [f"{k}_std" for k in BEHAVIOR_KEYS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})


def print_summary(results: list[dict]):
    cols = ["total_v5", "n_categories", "shape_coef",
            "wardrobe_wall_dist", "pillow_exposed_rate", "furniture_area"]
    header = f"{'Agent':<22s} " + " ".join(f"{c:>14s}" for c in cols)
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        cells = []
        for c in cols:
            v = r.get(f"{c}_mean", float("nan"))
            if isinstance(v, float) and np.isnan(v):
                cells.append(f"{'-':>14s}")
            else:
                cells.append(f"{v:>14.2f}")
        print(f"{r['agent']:<22s} " + " ".join(cells))
    print("=" * len(header))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="run names under runs/, e.g. pilot_hybrid pilot_additive pilot_mult")
    p.add_argument("--n-eps", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", default="runs/evaluation_summary.csv",
                   help="aggregate CSV across all agents")
    args = p.parse_args()

    results = []
    for name in args.models:
        # Try best/best_model.zip first, fall back to final.zip
        for candidate in [f"runs/{name}/best/best_model.zip",
                          f"runs/{name}/final.zip",
                          f"runs/{name}/model.zip"]:
            mp = Path(candidate)
            if mp.exists():
                break
        else:
            print(f"[skip] {name}: no model file found")
            continue

        print(f"[eval] {name}  ({mp})  → {args.n_eps} episodes")
        per_ep = evaluate_agent(str(mp), args.n_eps, seed=args.seed)
        save_per_ep_csv(Path(f"runs/{name}/eval_episodes.csv"), per_ep)
        agg = aggregate(per_ep, agent_name=name)
        results.append(agg)

    save_summary_csv(Path(args.output), results)
    print_summary(results)
    print(f"\nSaved: {args.output}")
    print(f"Per-episode CSVs: runs/<model>/eval_episodes.csv")


if __name__ == "__main__":
    main()
