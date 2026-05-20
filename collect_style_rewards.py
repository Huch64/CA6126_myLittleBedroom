"""collect_style_rewards.py — per-episode reward + component records for the
three reward-composition agents, evaluated on the SAME room sequence.

Enables distribution plots (violin / KDE of episode reward) and
component-correlation heatmaps that the mean-only radar can't show.

All agents are scored under the *hybrid* (v5) reward so the numbers are on one
yardstick (the env reward_style only changes the scalar, not the rollout).

Output plots/style_rewards.npz:
    styles                 array of style names
    total_<style>          (N,)      hybrid-yardstick episode reward
    <comp>_<style>         (N,)      component value per episode
                                     comp in availability/privacy/light/
                                     efficiency/diversity/compactness
    room_area_<style>      (N,)      room cell count (for scatter vs area)

Usage:  python collect_style_rewards.py --n-eps 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom
from train import FactoredMaskablePolicy

STYLE_RUNS = {
    "hybrid":         "ppo_2M_v2",
    "additive":       "pilot_additive_2M",
    "multiplicative": "pilot_mult",
}
COMPS = ["availability", "privacy", "light", "efficiency", "diversity", "compactness"]


def load_model(path: str) -> MaskablePPO:
    return MaskablePPO.load(path, custom_objects={
        "policy_class":  FactoredMaskablePolicy,
        "learning_rate": 0.0,
        "lr_schedule":   lambda _: 0.0,
        "clip_range":    lambda _: 0.0,
    })


def eval_model(model, n_eps: int, seed0: int = 500_000) -> dict:
    # unified hybrid yardstick so rewards are comparable across styles
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    rec = {"total": [], "room_area": []}
    for c in COMPS:
        rec[c] = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            m = env.action_masks()
            a, _ = model.predict(obs, action_masks=m, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                bd = env._last_breakdown
                rec["total"].append(bd["total"])
                rec["room_area"].append(env.room_w * env.room_h)
                for c in COMPS:
                    rec[c].append(bd.get(c, 0.0))
                break
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=500)
    ap.add_argument("--out", default="plots/style_rewards.npz")
    args = ap.parse_args()

    save = {"styles": np.array(list(STYLE_RUNS.keys()))}
    for style, run in STYLE_RUNS.items():
        path = Path(f"runs/{run}/best/best_model.zip")
        if not path.exists():
            path = Path(f"runs/{run}/final.zip")
        model = load_model(str(path))
        rec = eval_model(model, args.n_eps)
        save[f"total_{style}"] = np.array(rec["total"])
        save[f"room_area_{style}"] = np.array(rec["room_area"])
        for c in COMPS:
            save[f"{c}_{style}"] = np.array(rec[c])
        print(f"  {style:>14}  mean R = {np.mean(rec['total']):.2f}  "
              f"std = {np.std(rec['total']):.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
