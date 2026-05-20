"""collect_evolution.py — roll out every ppo_2M_v2 checkpoint and cache the
behaviour signals needed for the training-evolution figures in
analysis_plots.ipynb.

Heavy step (run once): loads all 40 checkpoints, plays N deterministic
episodes each in the hybrid env, and records — per checkpoint —
  • furniture centre positions (normalised to room coords) by category
  • wardrobe / bed minimum distance to a wall
  • final-layout reward components
  • category placed at each step (sequential grammar)
  • a random-policy reward baseline (single pass)

Output: plots/evolution_data.json  (read by the notebook plotting cells).

Usage:
    python collect_evolution.py                 # default N=300
    python collect_evolution.py --n-eps 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom, CATALOG
from train import FactoredMaskablePolicy

RUN = "ppo_2M_v2"
CATS = ["bed", "wardrobe", "desk", "cabinet", "nightstand"]


def load_model(path: str) -> MaskablePPO:
    # Models were trained on Python 3.10 / Windows; injecting these via
    # custom_objects skips the cloudpickle blobs that don't survive a
    # cross-version load (policy_class / lr_schedule / clip_range).
    return MaskablePPO.load(path, custom_objects={
        "policy_class":  FactoredMaskablePolicy,
        "learning_rate": 0.0,
        "lr_schedule":   lambda _: 0.0,
        "clip_range":    lambda _: 0.0,
    })


def wall_dist_cells(p, rw, rh) -> float:
    return float(min(min(p.x + dx, p.y + dy, rw - 1 - (p.x + dx), rh - 1 - (p.y + dy))
                     for dy in range(p.fh) for dx in range(p.fw)))


def rollout(model, env, seed) -> None:
    obs, _ = env.reset(seed=seed)
    while True:
        m = env.action_masks()
        a, _ = model.predict(obs, action_masks=m, deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        if term or trunc:
            return


def collect_for_model(model, n_eps: int, seed0: int = 100_000) -> dict:
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    centers = {c: [] for c in CATS}           # normalised (x/rw, y/rh)
    wall = {"bed": [], "wardrobe": []}
    comp = {k: [] for k in ("total", "availability", "privacy", "light",
                            "efficiency", "diversity", "compactness")}
    n_placed, n_categories = [], []
    step_cat = {s: [] for s in range(1, 9)}   # category placed at each step

    for ep in range(n_eps):
        rollout(model, env, seed0 + ep)
        rw, rh = env.room_w, env.room_h
        bd = env._last_breakdown
        for k in comp:
            comp[k].append(bd.get(k, 0.0))
        n_placed.append(len(env.placed))
        n_categories.append(bd.get("n_categories", 0))
        for i, p in enumerate(env.placed):
            cat = CATALOG[p.fid].cat
            cx = (p.x + p.fw / 2) / rw
            cy = (p.y + p.fh / 2) / rh
            centers[cat].append([cx, cy])
            if cat in wall:
                wall[cat].append(wall_dist_cells(p, rw, rh))
            if i + 1 in step_cat:
                step_cat[i + 1].append(cat)

    return {
        "centers": centers,
        "wall": wall,
        "components": {k: [float(np.mean(v)), float(np.std(v))] for k, v in comp.items()},
        "components_raw": comp,
        "n_placed_mean": float(np.mean(n_placed)),
        "n_categories_mean": float(np.mean(n_categories)),
        "step_cat": {str(s): step_cat[s] for s in step_cat},
        "n_eps": n_eps,
    }


def random_baseline(n_eps: int, seed0: int = 999_000) -> float:
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    rng = np.random.default_rng(0)
    rewards = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            m = env.action_masks()
            a = int(rng.choice(np.flatnonzero(m)))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                rewards.append(env._last_breakdown["total"])
                break
    return float(np.mean(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=300)
    ap.add_argument("--out", default="plots/evolution_data.json")
    args = ap.parse_args()

    ckpt_dir = Path(f"runs/{RUN}/checkpoints")
    ckpts = sorted(ckpt_dir.glob("model_*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[1]))
    steps = [int(p.stem.split("_")[1]) for p in ckpts]
    print(f"{len(ckpts)} checkpoints: {steps[0]:,} … {steps[-1]:,}")

    out = {"run": RUN, "n_eps": args.n_eps, "checkpoints": steps, "by_ckpt": {}}

    t_all = time.time()
    for step, ck in zip(steps, ckpts):
        t0 = time.time()
        model = load_model(str(ck))
        out["by_ckpt"][str(step)] = collect_for_model(model, args.n_eps)
        print(f"  {step:>9,} steps  {time.time()-t0:5.1f}s  "
              f"R={out['by_ckpt'][str(step)]['components']['total'][0]:5.2f}")

    print("random baseline …")
    out["random_baseline"] = random_baseline(args.n_eps)
    print(f"  random R = {out['random_baseline']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"saved → {args.out}   (total {time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
