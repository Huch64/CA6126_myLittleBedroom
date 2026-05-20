"""collect_legacy.py — regenerate the two data files the original analysis
cells (§2 radar, §5 spatial, §6 sequential) read but that no script produced.

Makes analysis_plots.ipynb self-contained: run this once and §2/§5/§6 work.

Outputs:
    plots/radar_metrics.json   {agent: {coverage, compactness, n_placed,
                                         privacy, lighting, reachability}}  (all [0,1])
    plots/behavior_data.json   {bed_positions:[[x,y,rw,rh],...],
                                 wardrobe_positions:[...], room_sizes:[[rw,rh],...],
                                 geo_scores:[[area, privacy*light*efficiency],...],
                                 step_categories:{step:[cat,...]}}    (hybrid agent)

Usage:  python collect_legacy.py --n-eps 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from env import MyLittleBedroom, CATALOG
from train import FactoredMaskablePolicy

STYLE_RUNS = {
    "hybrid":         "ppo_2M_v2",
    "additive":       "pilot_additive_2M",
    "multiplicative": "pilot_mult",
}
MAX_PIECES = 6   # 1 each of bed/desk/wardrobe/cabinet + 2 nightstands


def load_model(path: str) -> MaskablePPO:
    return MaskablePPO.load(path, custom_objects={
        "policy_class":  FactoredMaskablePolicy,
        "learning_rate": 0.0,
        "lr_schedule":   lambda _: 0.0,
        "clip_range":    lambda _: 0.0,
    })


def rollout_records(model, n_eps: int, seed0: int, detailed: bool):
    """Return aggregate radar stats; if detailed, also per-episode behavior."""
    env = MyLittleBedroom(seed=0, reward_style="hybrid")
    agg = {k: [] for k in ("coverage", "compactness", "n_placed",
                           "privacy", "lighting", "reachability")}
    beh = {"bed_positions": [], "wardrobe_positions": [], "room_sizes": [],
           "geo_scores": [], "step_categories": {}}

    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed0 + ep)
        step = 0
        while True:
            m = env.action_masks()
            a, _ = model.predict(obs, action_masks=m, deterministic=True)
            prev = len(env.placed)
            obs, r, term, trunc, info = env.step(int(a))
            if len(env.placed) > prev:           # a piece was placed this step
                step += 1
                if detailed:
                    p = env.placed[-1]
                    beh["step_categories"].setdefault(step, []).append(CATALOG[p.fid].cat)
            if term or trunc:
                break
        rw, rh = env.room_w, env.room_h
        bd = env._last_breakdown
        priv, light, eff = bd["privacy"], bd["light"], bd["efficiency"]
        fur_area = sum(p.fw * p.fh for p in env.placed)

        agg["coverage"].append(min(1.0, fur_area / (rw * rh)))
        agg["compactness"].append(bd.get("compactness", 0.0) / 5.0)
        agg["n_placed"].append(len(env.placed) / MAX_PIECES)
        agg["privacy"].append(priv)
        agg["lighting"].append(light)
        agg["reachability"].append(eff)

        if detailed:
            for cat, store in (("bed", "bed_positions"), ("wardrobe", "wardrobe_positions")):
                for p in env.placed:
                    if CATALOG[p.fid].cat == cat:
                        beh[store].append([p.x + p.fw / 2, p.y + p.fh / 2, rw, rh])
            beh["room_sizes"].append([rw, rh])
            beh["geo_scores"].append([rw * rh, priv * light * eff])

    radar = {k: float(np.mean(v)) for k, v in agg.items()}
    return radar, (beh if detailed else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=1000)
    args = ap.parse_args()

    radar = {}
    behavior = None
    for style, run in STYLE_RUNS.items():
        path = Path(f"runs/{run}/best/best_model.zip")
        if not path.exists():
            path = Path(f"runs/{run}/final.zip")
        model = load_model(str(path))
        detailed = (style == "hybrid")     # behavior_data is the hybrid agent only
        radar[style], beh = rollout_records(model, args.n_eps, seed0=700_000, detailed=detailed)
        if detailed:
            behavior = beh
        print(f"  {style:>14}  " + "  ".join(f"{k}={v:.2f}" for k, v in radar[style].items()))

    Path("plots").mkdir(exist_ok=True)
    with open("plots/radar_metrics.json", "w") as f:
        json.dump(radar, f, indent=1)
    # step_categories keys must be JSON strings (cell 16 casts back to int)
    behavior["step_categories"] = {str(k): v for k, v in behavior["step_categories"].items()}
    with open("plots/behavior_data.json", "w") as f:
        json.dump(behavior, f)
    print("saved -> plots/radar_metrics.json, plots/behavior_data.json")


if __name__ == "__main__":
    main()
