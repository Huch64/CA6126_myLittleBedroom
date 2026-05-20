"""collect_fixed_dw.py — controlled-geometry occupancy maps for the three
reward-composition agents.

Unlike collect_occupancy_styles.py (which uses fully random door/window and so
blurs the relative spatial structure), here we PIN the door and window and vary
only the room size. Door sits near the left of the bottom wall; window is
centred on the top wall. After normalising to room coordinates both become
fixed anchors, so furniture positions are directly comparable across styles.

Output plots/fixed_dw_occupancy.npz:
    occ_<style>            (NB_Y, NB_X)
    occ_<style>_<cat>      (NB_Y, NB_X)
    door_norm, win_norm    normalised (x, y) anchor of door / window
    styles

Usage:  python collect_fixed_dw.py --n-eps 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom, CATALOG, DW, ROOM_W_RANGE, ROOM_H_RANGE
from train import FactoredMaskablePolicy

NB_X, NB_Y = 26, 22
CATS = ["bed", "wardrobe", "desk", "cabinet", "nightstand"]
STYLE_RUNS = {
    "hybrid":         "ppo_2M_v2",
    "additive":       "pilot_additive_2M",
    "multiplicative": "pilot_mult",
}
DOOR_POS = 1   # near the LEFT edge of the bottom wall (fixed)


def load_model(path: str) -> MaskablePPO:
    return MaskablePPO.load(path, custom_objects={
        "policy_class":  FactoredMaskablePolicy,
        "learning_rate": 0.0,
        "lr_schedule":   lambda _: 0.0,
        "clip_range":    lambda _: 0.0,
    })


def accumulate_fixed(model, n_eps: int, seed0: int):
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    rng = np.random.default_rng(seed0)
    occ = np.zeros((NB_Y, NB_X)); tot = np.zeros((NB_Y, NB_X))
    cat_occ = {c: np.zeros((NB_Y, NB_X)) for c in CATS}

    for ep in range(n_eps):
        rw = int(rng.integers(ROOM_W_RANGE[0], ROOM_W_RANGE[1] + 1))
        rh = int(rng.integers(ROOM_H_RANGE[0], ROOM_H_RANGE[1] + 1))
        cfg = {"room_w": rw, "room_h": rh, "door_pos": DOOR_POS, "win_wall": "top"}
        obs, _ = env.reset(seed=seed0 + ep, options={"config": cfg})
        while True:
            m = env.action_masks()
            a, _ = model.predict(obs, action_masks=m, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                break
        cat_grid = {}
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
                    if (cx, cy) in cat_grid:
                        cat_occ[cat_grid[(cx, cy)]][by, bx] += 1

    rate = np.divide(occ, tot, out=np.zeros_like(occ), where=tot > 0)
    cat_rate = {c: np.divide(cat_occ[c], tot, out=np.zeros_like(occ), where=tot > 0)
                for c in CATS}
    return rate, cat_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=800)
    ap.add_argument("--out", default="plots/fixed_dw_occupancy.npz")
    args = ap.parse_args()

    save = {"styles": np.array(list(STYLE_RUNS.keys()))}
    for style, run in STYLE_RUNS.items():
        path = Path(f"runs/{run}/best/best_model.zip")
        if not path.exists():
            path = Path(f"runs/{run}/final.zip")
        model = load_model(str(path))
        rate, cat_rate = accumulate_fixed(model, args.n_eps, seed0=600_000)
        save[f"occ_{style}"] = rate
        for c in CATS:
            save[f"occ_{style}_{c}"] = cat_rate[c]
        print(f"  {style:>14}  occupied frac = {rate.mean():.3f}")

    # normalised anchors (door centre near left of bottom wall; window top-centre)
    # door spans cells [DOOR_POS, DOOR_POS+DW); use mean room width for the anchor
    mean_w = (ROOM_W_RANGE[0] + ROOM_W_RANGE[1]) / 2
    save["door_norm"] = np.array([(DOOR_POS + DW / 2) / mean_w, 1.0])  # bottom
    save["win_norm"] = np.array([0.5, 0.0])                            # top centre

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
