"""Sanity check for env.py.

Runs three smoke tests:
  1. Reset/step/mask shapes and types match the spec.
  2. A deterministic scripted episode (bed + paired nightstand) prints the
     reward breakdown — compare against my_little_bedroom.html with the same
     room/door/window/items.
  3. A random-agent rollout under the action mask completes without error.

Run:
    python sanity_check.py
"""

from __future__ import annotations

import numpy as np

from env import (
    ACTION_DONE,
    CATALOG,
    DW,
    GH,
    GW,
    MyLittleBedroom,
    N_ACTIONS,
    encode_action,
    get_footprint,
)


def smoke_shapes() -> None:
    env = MyLittleBedroom(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, GH, GW), obs.shape
    assert obs.dtype == np.int8, obs.dtype
    mask = env.action_masks()
    assert mask.shape == (N_ACTIONS,), mask.shape
    assert mask.dtype == np.bool_
    assert mask[ACTION_DONE], "DONE must always be available"
    print(f"[ok] shapes: obs={obs.shape} mask={mask.shape} "
          f"valid_actions={int(mask.sum())} info={info}")


def scripted_episode() -> None:
    """Hand-crafted placement; prints availability + each (1 − ratio) factor."""
    env = MyLittleBedroom(seed=42, max_steps=8)
    obs, info = env.reset(seed=42)
    rw, rh = info["room"]
    print(f"[scripted] room={rw}x{rh} door_pos={info['door_pos']} "
          f"window={info['win_wall']}@{info['win_pos']}+{info['win_w']}")

    # Bed 1.2 (fid=1) head-left at (0, 0); footprint 14x8.
    # We pick the largest furniture we can fit by walking the catalog top-down.
    actions: list[int] = []
    placed_idx_set: set[int] = set()
    grid_free = np.ones((rh, rw), dtype=bool)  # naive: just for picking spots
    for fid in [1, 16, 4, 13, 7]:  # bed, nightstand, desk, cabinet, wardrobe
        spec = CATALOG[fid]
        for ori in range(4):
            fw, fh = get_footprint(spec, ori)
            if fw > rw or fh > rh:
                continue
            placed = False
            for y in range(rh - fh + 1):
                for x in range(rw - fw + 1):
                    a = encode_action(fid, x, y, ori)
                    if env.action_masks()[a]:
                        actions.append(a)
                        env.step(a)
                        placed_idx_set.add(fid)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

    obs, reward, term, trunc, info = env.step(ACTION_DONE)
    bd = env._last_breakdown
    print(f"[scripted] placed {len(placed_idx_set)} items -> reward={reward}")
    print(f"           availability={bd['availability']}  "
          f"×privacy={bd['privacy']:.2f}  ×light={bd['light']:.2f}  "
          f"×efficiency={bd['efficiency']:.2f}")
    print(f"           per_item={bd['per_item']}")
    print(f"           exposed={bd['exposed_cells']}/{bd['total_bed_cells']} "
          f"pillow_seen={bd['pillow_seen']} unreachable={bd['unreachable_cells']}c")


def random_rollout(n_eps: int = 3) -> None:
    env = MyLittleBedroom(seed=123)
    rng = np.random.default_rng(123)
    rewards = []
    for ep in range(n_eps):
        env.reset(seed=ep)
        total = 0.0
        for _ in range(env.max_steps + 1):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            a = int(rng.choice(valid))
            _, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    print(f"[random] {n_eps} eps: rewards={rewards} mean={np.mean(rewards):.2f}")


if __name__ == "__main__":
    smoke_shapes()
    scripted_episode()
    random_rollout()
    print("\nAll checks passed.")
