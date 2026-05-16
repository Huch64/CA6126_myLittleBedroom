"""
reward_audit.py — profile the reward landscape *before* committing to a
multi-hour training run.

For each of several policies (random under action mask + DONE-immediate +
hand-written greedy) we run N rollouts and dump:

  • histogram of total reward
  • mean / std / quantiles per component
    (availability / privacy_loss / light_loss / waste_loss)
  • DONE-trap gap: mean(greedy) − reward(DONE)
  • per-cell sensitivity: shift one bed placement by ±1, ±2, ±3 cells and
    report how much the reward moves (verifies continuity)

The aim is to validate that the reward is "learnable":
  - distribution is roughly unimodal, not clumped at a single value
  - no single component dominates the others by 10× under typical play
  - moving one cell changes the reward by a small, smooth amount
    (not "fall off a cliff" like the v1 binary lumps)

Run:
    python reward_audit.py                       # audits the *current* env.py
    python reward_audit.py --n 5000 --save plots/reward_audit_v2.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env import (ACTION_DONE, CATALOG, GW, GH, N_ORI, MyLittleBedroom,
                 encode_action, get_footprint)


# ─── policies ──────────────────────────────────────────────────────

def policy_random(env, rng):
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    # Force at least one placement to avoid pure DONE under the random policy.
    non_done = valid[valid != ACTION_DONE]
    if non_done.size and rng.random() < 0.85:
        return int(rng.choice(non_done))
    return int(rng.choice(valid))


def policy_done_immediate(env, rng):
    return ACTION_DONE


def policy_greedy(env, rng):
    """Pick the action with the largest immediate per-piece value. Naive:
    doesn't consider position. Underestimates a 'good' policy's ceiling."""
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    if valid.size == 1:
        return ACTION_DONE
    best_a, best_v = ACTION_DONE, 0.0
    from env import _value as eval_value
    for a in valid:
        if a == ACTION_DONE:
            continue
        fid = a // (GW * GH * N_ORI)
        v = eval_value(CATALOG[fid])
        if v > best_v:
            best_v, best_a = v, int(a)
    if rng.random() < 0.10:
        return int(rng.choice(valid))
    return best_a


def policy_edge_greedy(env, rng):
    """Greedy by item value, then break ties by 'closest to a wall'. Stand-in
    for a sensible human player who pushes furniture against walls to leave
    a contiguous walking area in the middle."""
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    if valid.size == 1:
        return ACTION_DONE
    from env import _value as eval_value

    best_score = -1e9
    best_a = ACTION_DONE
    rw, rh = env.room_w, env.room_h
    for a in valid:
        if a == ACTION_DONE:
            continue
        fid = a // (GW * GH * N_ORI)
        rem = a % (GW * GH * N_ORI)
        x = rem // (GH * N_ORI)
        rem = rem % (GH * N_ORI)
        y = rem // N_ORI
        spec = CATALOG[fid]
        fw, fh = get_footprint(spec, rem % N_ORI)
        # wall-affinity bonus: smaller distance to any wall is better
        dx = min(x, rw - (x + fw))
        dy = min(y, rh - (y + fh))
        edge_bonus = -min(dx, dy)        # 0 if hugging a wall, negative inside
        score = eval_value(spec) + 0.05 * edge_bonus
        if score > best_score:
            best_score, best_a = score, int(a)
    if rng.random() < 0.10:
        return int(rng.choice(valid))
    return best_a


POLICIES = {
    "random":         policy_random,
    "done_immediate": policy_done_immediate,
    "greedy":         policy_greedy,
    "edge_greedy":    policy_edge_greedy,
}


# ─── rollouts ──────────────────────────────────────────────────────

def rollout(env, policy, rng) -> dict:
    obs, info = env.reset()
    bd = None
    for _ in range(env.max_steps + 1):
        a = policy(env, rng)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            bd = info.get("breakdown") or env._last_breakdown
            break
    if bd is None:
        bd = env._last_breakdown
    return {
        "total":         bd["total"],
        "availability":  bd["availability"],
        "privacy_loss":  bd["privacy_loss"],
        "light_loss":    bd["light_loss"],
        "waste_loss":    bd["waste_loss"],
        "privacy":       bd["privacy"],
        "light":         bd["light"],
        "efficiency":    bd["efficiency"],
        "n_placed":      info.get("placed", 0),
    }


def collect(policy_name: str, n: int, seed: int) -> dict[str, np.ndarray]:
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        env = MyLittleBedroom(seed=seed + i, max_steps=8)
        rows.append(rollout(env, policy, rng))
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


# ─── continuity test ──────────────────────────────────────────────

def continuity_test() -> list[tuple[str, int, float]]:
    """Place Bed 1.2 at a known position, then shift x by ±1..±3 and report
    how the *final* reward changes. A healthy reward function gives smooth,
    monotonic changes — not cliffs."""
    rows = []
    base_x = 3
    for dx in (-3, -2, -1, 0, 1, 2, 3):
        env = MyLittleBedroom(seed=0, max_steps=8)
        env.reset(options={"config": {
            "room_w": 20, "room_h": 20, "door_pos": 7, "win_wall": "top",
        }})
        a = encode_action(1, base_x + dx, 1, 0)     # Bed 1.2 head-left
        if not env.action_masks()[a]:
            rows.append((f"dx={dx:+d}", None, float("nan")))
            continue
        env.step(a)
        env.step(ACTION_DONE)
        bd = env._last_breakdown
        rows.append((f"dx={dx:+d}", base_x + dx, bd["total"]))
    return rows


# ─── reporting ─────────────────────────────────────────────────────

def describe(name: str, data: dict[str, np.ndarray]) -> None:
    t = data["total"]
    print(f"\n── {name:>15s}  (n={len(t)}) ──")
    print(f"  total:        mean={t.mean():+6.2f}  std={t.std():5.2f}  "
          f"min={t.min():+6.2f}  median={np.median(t):+6.2f}  max={t.max():+6.2f}")
    for k in ("availability", "privacy_loss", "light_loss", "waste_loss"):
        v = data[k]
        print(f"  {k:<14s}  mean={v.mean():6.2f}  std={v.std():5.2f}  "
              f"max={v.max():6.2f}")
    print(f"  n_placed:     mean={data['n_placed'].mean():.2f}   "
          f"frac(n_placed==0): {(data['n_placed']==0).mean():.1%}")
    print(f"  frac total≥0: {(t>=0).mean():.1%}     "
          f"frac total≥+3: {(t>=3).mean():.1%}")


def plot_audit(per_policy: dict[str, dict], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110)

    colors = {"random": "#cc6677", "done_immediate": "#999",
              "greedy":  "#88aa44", "edge_greedy": "#117733"}

    # 1) Total reward histogram
    ax = axes[0, 0]
    bins = np.linspace(-20, 15, 60)
    for name, d in per_policy.items():
        ax.hist(d["total"], bins=bins, alpha=0.55,
                label=name, color=colors.get(name, "#777"))
    ax.set_title("Total reward distribution")
    ax.set_xlabel("episode reward"); ax.set_ylabel("count")
    ax.axvline(0, color="#333", lw=0.6, linestyle="--")
    ax.legend(frameon=False)

    # 2) Component means (bar comparison)
    ax = axes[0, 1]
    components = ["availability", "privacy_loss", "light_loss", "waste_loss"]
    names = list(per_policy.keys())
    x = np.arange(len(components))
    width = 0.25
    for i, name in enumerate(names):
        means = [per_policy[name][k].mean() for k in components]
        # losses are subtracted from availability in `total`; plot signed
        signed = [means[0], -means[1], -means[2], -means[3]]
        ax.bar(x + i * width, signed, width, label=name,
               color=colors.get(name, "#777"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(["+A", "−privacy", "−light", "−waste"])
    ax.set_title("Mean component contribution to total")
    ax.axhline(0, color="#333", lw=0.6, linestyle="--")
    ax.legend(frameon=False)

    # 3) Per-factor discount under random policy
    ax = axes[1, 0]
    rand = per_policy["random"]
    parts = {
        "privacy":    rand["privacy"],
        "light":      rand["light"],
        "efficiency": rand["efficiency"],
    }
    labels = list(parts.keys())
    means = [parts[k].mean() for k in labels]
    stds  = [parts[k].std() for k in labels]
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4477aa")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#999", lw=0.6, linestyle="--")
    ax.set_title("Per-factor discount under random policy (mean ± std)")
    ax.set_ylabel("× factor (1 = no discount)")
    ax.tick_params(axis="x", rotation=15)

    # 4) Continuity test: reward vs bed x-offset
    ax = axes[1, 1]
    cont = continuity_test()
    xs = [r[1] for r in cont if r[1] is not None]
    ys = [r[2] for r in cont if r[1] is not None]
    ax.plot(xs, ys, "o-", color="#cc6677", lw=2)
    ax.set_title("Continuity: reward vs Bed-1.2 x position")
    ax.set_xlabel("bed x (cells)"); ax.set_ylabel("episode total")
    ax.axhline(0, color="#333", lw=0.6, linestyle="--")

    fig.suptitle("Reward landscape audit", fontsize=13, weight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000,
                   help="rollouts per policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="plots/reward_audit.png")
    args = p.parse_args()

    print(f"running {args.n} rollouts each for: {list(POLICIES.keys())}")
    per_policy = {}
    for name in POLICIES:
        per_policy[name] = collect(name, args.n, args.seed)
        describe(name, per_policy[name])

    print("\n── continuity test (Bed 1.2 at y=1, varying x) ──")
    for label, x, total in continuity_test():
        print(f"  {label}  x={x}  total={total:+.2f}")

    # DONE-trap gaps
    dd = per_policy["done_immediate"]["total"].mean()
    print()
    for name in ("random", "greedy", "edge_greedy"):
        if name in per_policy:
            mean = per_policy[name]["total"].mean()
            print(f"DONE-trap gap ({name:<11s} − DONE): {mean - dd:+.2f}   "
                  f"(positive ⇒ better to play)")

    plot_audit(per_policy, Path(args.save))
    print(f"\nplot → {args.save}")


if __name__ == "__main__":
    main()
