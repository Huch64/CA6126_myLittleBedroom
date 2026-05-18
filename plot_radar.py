"""
plot_radar.py — Radar plot for reward-style ablation comparison.

Reads runs/evaluation_summary.csv (produced by evaluate.py) and renders
a single radar plot with one polygon per trained agent. Each axis is a
reward-independent behavior metric, normalized to [0, 1] where OUTER =
better (inverted as needed for "lower-is-better" metrics).

Usage:
    python plot_radar.py
    python plot_radar.py --input runs/evaluation_summary.csv \
                        --output plots/radar.png \
                        --models pilot_hybrid pilot_additive pilot_mult
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Each axis: (csv_key, display_label, higher_is_better, theoretical_max)
# theoretical_max is used as the [0, 1] denominator for that axis.
AXES = [
    # (csv key with _mean suffix,    display label,        higher-is-better, max)
    ("n_categories",                 "Categories\n(0-5)",          True,  5.0),
    ("furniture_area",               "Furniture area\n(cells)",    True, 100.0),
    ("wardrobe_wall_dist",           "Wardrobe-on-wall",           False, 6.0),   # lower better → invert
    ("compactness",                  "Compactness\n(0-5)",         True,  5.0),
    ("pillow_exposed_rate",          "Privacy\n(less exposed)",    False, 1.0),   # lower better → invert
    ("bed_placed",                   "Bed rate",                   True,  1.0),
]


COLORS = ["#C03030", "#3060C0", "#30A040", "#C09030", "#7030A0"]


def load_summary(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def normalize_value(raw_mean: float, higher_better: bool, max_val: float) -> float:
    """Map a raw metric mean to [0, 1] with 1 = best."""
    if max_val <= 0:
        return 0.0
    v = raw_mean / max_val
    v = max(0.0, min(1.0, v))
    return v if higher_better else (1.0 - v)


def plot_radar(rows: list[dict], output: Path, title: str = "Reward Style Comparison"):
    n_axes = len(AXES)
    # close the polygon
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in enumerate(rows):
        agent = row["agent"]
        vals = []
        for key, _, higher_better, max_val in AXES:
            raw = float(row.get(f"{key}_mean", 0.0))
            vals.append(normalize_value(raw, higher_better, max_val))
        vals += vals[:1]   # close polygon

        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, color=color, linewidth=2, label=agent)
        ax.fill(angles, vals, color=color, alpha=0.15)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for (_, label, _, _) in AXES], fontsize=10)
    # Radial grid
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="#888")
    ax.grid(True, alpha=0.4)

    ax.set_title(title, pad=20, fontsize=13, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def print_table(rows: list[dict]):
    """Print a tabular view of the normalized scores."""
    agent_names = [r["agent"] for r in rows]
    print()
    print(f"{'Axis':<24s} " + " ".join(f"{a:>14s}" for a in agent_names))
    print("-" * (26 + 16 * len(agent_names)))
    for key, label, higher_better, max_val in AXES:
        clean_label = label.replace("\n", " ")
        cells = []
        for row in rows:
            raw = float(row.get(f"{key}_mean", 0.0))
            norm = normalize_value(raw, higher_better, max_val)
            cells.append(f"{raw:>6.2f} ({norm:>4.2f})")
        print(f"{clean_label:<24s} " + " ".join(f"{c:>14s}" for c in cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="runs/evaluation_summary.csv")
    p.add_argument("--output", default="plots/radar_reward_styles.png")
    p.add_argument("--models", nargs="*", default=None,
                   help="filter to specific agents (default: all in CSV)")
    p.add_argument("--title", default="Reward Style — Behavior Comparison")
    args = p.parse_args()

    rows = load_summary(Path(args.input))
    if args.models:
        rows = [r for r in rows if r["agent"] in args.models]
    if not rows:
        print("No rows to plot. Check --input / --models.")
        return

    print_table(rows)
    plot_radar(rows, Path(args.output), title=args.title)


if __name__ == "__main__":
    main()
