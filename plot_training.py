"""
plot_training.py — generate report figures from a training run's logs.

Reads from runs/<run_name>/:
    progress.csv        SB3 internals + custom aggregates
    episodes.csv        per-training-episode breakdown (our custom CSV)
    evaluations.npz     MaskableEvalCallback output

Writes to plots/<run_name>/:
    training_curve.png      ep_rew_mean (train) + mean (eval) vs env steps
    reward_breakdown.png    A / D / W as three rolling-mean lines
    ep_length.png           episode length over training
    ppo_internals.png       value_loss / policy_loss / entropy / explained_var
    placement_stats.png     items placed and unique categories over training
    per_category_freq.png   bar chart: how often each cat got placed
    room_difficulty.png     mean reward bucketed by room area

Usage:
    python plot_training.py --run ppo_run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.dpi"] = 110


def _rolling(series: pd.Series, window: int = 50) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_training_curve(progress: pd.DataFrame, eval_data: dict | None,
                        out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if "rollout/ep_rew_mean" in progress.columns:
        ax.plot(progress["time/total_timesteps"],
                progress["rollout/ep_rew_mean"],
                label="train (rollout mean)", color="#4477aa", lw=1.5)

    if eval_data is not None:
        ts = eval_data["timesteps"]
        results = eval_data["results"]          # shape (n_evals, n_eval_eps)
        mean = results.mean(axis=1)
        std = results.std(axis=1)
        ax.plot(ts, mean, label="eval (deterministic)",
                color="#cc6677", lw=2)
        ax.fill_between(ts, mean - std, mean + std,
                        color="#cc6677", alpha=0.18, lw=0)

    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward")
    ax.set_title("Training curve")
    ax.axhline(0, color="#999", lw=0.6, linestyle="--")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reward_breakdown(eps: pd.DataFrame, out_path: Path,
                          window: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eps["step"], _rolling(eps["availability"], window),
            label="Availability", color="#117733", lw=1.6)
    ax.plot(eps["step"], _rolling(eps["discomfort"], window),
            label="Discomfort", color="#cc6677", lw=1.6)
    ax.plot(eps["step"], _rolling(eps["waste"], window),
            label="Waste", color="#ddaa33", lw=1.6)
    ax.plot(eps["step"], _rolling(eps["total"], window),
            label="Total", color="#222", lw=2.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"per-component reward (rolling mean, w={window})")
    ax.set_title("Reward breakdown over training")
    ax.axhline(0, color="#999", lw=0.6, linestyle="--")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ep_length(progress: pd.DataFrame, eps: pd.DataFrame,
                   out_path: Path, window: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if "rollout/ep_len_mean" in progress.columns:
        ax.plot(progress["time/total_timesteps"],
                progress["rollout/ep_len_mean"],
                label="ep_len (SB3 rolling)", color="#4477aa", lw=1.5)
    ax.plot(eps["step"], _rolling(eps["n_placed"], window),
            label=f"items placed (rolling w={window})",
            color="#117733", lw=1.5)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("count")
    ax.set_title("Episode length & items placed")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ppo_internals(progress: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("train/value_loss", "value loss", "#4477aa"),
        ("train/policy_gradient_loss", "policy gradient loss", "#cc6677"),
        ("train/entropy_loss", "entropy loss", "#117733"),
        ("train/explained_variance", "explained variance", "#ddaa33"),
    ]
    present = [(c, label, color) for c, label, color in metrics
               if c in progress.columns]
    n = len(present)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (col, label, color) in zip(axes, present):
        ax.plot(progress["time/total_timesteps"], progress[col],
                color=color, lw=1.5)
        ax.set_ylabel(label)
        ax.axhline(0, color="#999", lw=0.5, linestyle="--")
    axes[-1].set_xlabel("environment steps")
    axes[0].set_title("PPO internals")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_placement_stats(eps: pd.DataFrame, out_path: Path,
                         window: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(eps["step"], _rolling(eps["n_placed"], window),
            label="items placed", color="#117733", lw=1.6)
    ax.plot(eps["step"], _rolling(eps["n_unique_cats"], window),
            label="unique categories", color="#4477aa", lw=1.6)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"count (rolling mean, w={window})")
    ax.set_title("Placement diversity over training")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_category_freq(eps: pd.DataFrame, out_path: Path,
                           last_fraction: float = 0.2) -> None:
    """Bar: how often each category appears in the LAST `last_fraction` of episodes."""
    tail = eps.tail(int(len(eps) * last_fraction))
    cats_all = ["bed", "desk", "wardrobe", "cabinet", "nightstand"]
    counts = {c: 0 for c in cats_all}
    for s in tail["cats_placed"].dropna():
        for c in str(s).split("|"):
            if c in counts:
                counts[c] += 1
    n_eps = len(tail)
    freqs = {c: counts[c] / n_eps for c in cats_all}

    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = {"bed": "#45D468", "desk": "#30D4A0", "wardrobe": "#30A8D4",
              "cabinet": "#5050D4", "nightstand": "#9530D4"}
    bars = ax.bar(cats_all, [freqs[c] for c in cats_all],
                  color=[colors[c] for c in cats_all])
    for bar, c in zip(bars, cats_all):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#999", lw=0.6, linestyle="--",
               label="placed every episode")
    ax.set_ylim(0, max(1.2, max(freqs.values()) + 0.15))
    ax.set_ylabel(f"avg placements per episode (last {int(last_fraction*100)}%)")
    ax.set_title("Per-category placement frequency (final policy)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_room_difficulty(eps: pd.DataFrame, out_path: Path,
                         last_fraction: float = 0.2) -> None:
    """Heatmap of mean reward by (room_w, room_h) using the last slice."""
    tail = eps.tail(int(len(eps) * last_fraction))
    pivot = tail.pivot_table(index="room_h", columns="room_w",
                             values="total", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("room width (cells)")
    ax.set_ylabel("room height (cells)")
    ax.set_title(f"Mean reward by room size (last {int(last_fraction*100)}% of training)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v < pivot.values.mean() else "black")
    fig.colorbar(im, ax=ax, label="mean total reward")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name under runs/")
    p.add_argument("--window", type=int, default=200,
                   help="rolling-window for the per-episode plots")
    p.add_argument("--last-frac", type=float, default=0.2,
                   help="use last fraction of episodes for the final-policy plots")
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")

    out_dir = Path("plots") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading from {run_dir}")
    progress = pd.read_csv(run_dir / "progress.csv")
    eps = pd.read_csv(run_dir / "episodes.csv")

    eval_data = None
    npz_path = run_dir / "evaluations.npz"
    if npz_path.exists():
        eval_data = dict(np.load(npz_path))

    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"  {len(progress)} progress rows  |  {len(eps)} episodes  |  "
          f"trained {cfg.get('timesteps')} steps")

    print(f"writing to {out_dir}/")
    plot_training_curve(progress, eval_data, out_dir / "training_curve.png")
    plot_reward_breakdown(eps, out_dir / "reward_breakdown.png", args.window)
    plot_ep_length(progress, eps, out_dir / "ep_length.png", args.window)
    plot_ppo_internals(progress, out_dir / "ppo_internals.png")
    plot_placement_stats(eps, out_dir / "placement_stats.png", args.window)
    plot_per_category_freq(eps, out_dir / "per_category_freq.png", args.last_frac)
    plot_room_difficulty(eps, out_dir / "room_difficulty.png", args.last_frac)

    print("\ngenerated:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
