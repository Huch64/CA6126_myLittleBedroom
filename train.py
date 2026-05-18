"""
train.py — MaskablePPO training for my_little_bedroom.

Logs everything we need for the report's training-process section:

    runs/<run_name>/
        config.json          hyperparameters and run metadata
        progress.csv         SB3 progress (loss, KL, lr, ep_rew_mean, etc.)
        episodes.csv         one row per completed training episode with our
                             A/D/W breakdown, room config, placements
        evaluations.npz      MaskableEvalCallback output (timesteps + reward
                             per-seed across eval episodes)
        events.out.tfevents  TensorBoard event file
        final.zip            policy at the end of training
        best/best_model.zip  policy with highest eval reward (best so far)

The episodes.csv schema is what plot_training.py reads to slice the data by
room config / category / step count when making report figures.

Usage:
    python train.py                                       # default 500K steps
    python train.py --timesteps 50_000 --name sanity      # quick sanity run
    python train.py --n-envs 4 --max-steps 10             # tweak knobs
    tensorboard --logdir runs/                            # live training curves
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import (CATALOG, DW, GH, GW, GRID_M, MAX_PER_CAT, N_ACTIONS, N_FURN,
                 N_ORI, WCOEFF, MyLittleBedroom)


# ────────────────────────────────────────────────────────────────────────
# Factored output head: 64-dim latent → fid/x/y/ori/done logits separately,
# then combined into the joint 41185-action logit space. Compared to a flat
# 64→41185 linear (≈ 2.66 M params), this is 64×(18+26+22+4+1) ≈ 4.5 K
# params — every parameter sees ~10³ more effective updates per sample.
# Trade-off: assumes independence P(a) = P(fid)·P(x)·P(y)·P(ori), but the
# action mask still filters invalid combinations at sampling time.
# ────────────────────────────────────────────────────────────────────────
class FactoredActionNet(nn.Module):
    def __init__(self, in_dim: int,
                 n_furn: int = N_FURN, gw: int = GW,
                 gh: int = GH, n_ori: int = N_ORI):
        super().__init__()
        self.fid_head  = nn.Linear(in_dim, n_furn)
        self.x_head    = nn.Linear(in_dim, gw)
        self.y_head    = nn.Linear(in_dim, gh)
        self.ori_head  = nn.Linear(in_dim, n_ori)
        self.done_head = nn.Linear(in_dim, 1)
        self.n_furn, self.gw, self.gh, self.n_ori = n_furn, gw, gh, n_ori

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, in_dim) → (B, N_FURN * GW * GH * N_ORI + 1)
        B = features.shape[0]
        fid  = self.fid_head(features).view(B, self.n_furn, 1, 1, 1)
        x    = self.x_head(features).view(B, 1, self.gw,   1, 1)
        y    = self.y_head(features).view(B, 1, 1, self.gh, 1)
        ori  = self.ori_head(features).view(B, 1, 1, 1, self.n_ori)
        done = self.done_head(features)                        # (B, 1)
        joint = fid + x + y + ori                              # (B, F, W, H, O)
        # Flatten in the same order as encode_action():
        #   idx = fid*GW*GH*N_ORI + x*GH*N_ORI + y*N_ORI + ori
        joint_flat = joint.reshape(B, -1)                      # (B, 41184)
        return torch.cat([joint_flat, done], dim=1)            # (B, 41185)


class FactoredMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy with our FactoredActionNet replacing the dense head."""
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.action_net = FactoredActionNet(latent_dim)
        # Re-build the optimizer so it sees the new parameters.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


def _mask_fn(env):
    return env.action_masks()


class CompletionBonusWrapper(gym.Wrapper):
    """Adds a small dense reward each time a placement succeeds (non-DONE).

    Pure reward shaping — leaves env.py / verify.py / HTML reward parity
    intact. Use --placement-bonus to enable; bonus value is recorded in
    config.json so the run is fully reproducible.
    """

    def __init__(self, env, bonus: float = 0.1):
        super().__init__(env)
        self.bonus = bonus
        self._prev_placed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_placed = info.get("placed", 0)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        n_now = info.get("placed", 0)
        if n_now > self._prev_placed:
            r += self.bonus
        self._prev_placed = n_now
        return obs, r, term, trunc, info


def make_env(seed: int | None = None, max_steps: int = 8,
             placement_bonus: float = 0.0):
    """Factory: MyLittleBedroom → ActionMasker → [CompletionBonusWrapper] → Monitor.

    Monitor is outermost so it sees the *shaped* reward used for training.
    ActionMasker exposes .action_masks() which MaskablePPO calls each step.
    """
    def _init():
        env = MyLittleBedroom(seed=seed, max_steps=max_steps)
        env = ActionMasker(env, _mask_fn)
        if placement_bonus > 0:
            env = CompletionBonusWrapper(env, bonus=placement_bonus)
        env = Monitor(env)
        return env
    return _init


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


class EpisodeBreakdownLogger(BaseCallback):
    """Writes one CSV row per completed episode + aggregates to TensorBoard.

    Per-row columns capture everything a report figure could want: the
    multiplicative reward factors, the "points lost" per factor, what got
    placed, and the room/door/window config. Plot scripts can slice on any
    of these.
    """

    HEADER = [
        "step", "ep_idx",
        "total", "availability",
        "privacy", "light", "efficiency",
        "privacy_loss", "light_loss", "waste_loss",
        "n_placed", "n_unique_cats",
        "room_w", "room_h", "door_pos", "win_wall",
        "exposed_cells", "total_bed_cells",
        "pillow_seen", "window_blocked", "unreachable_cells",
        "cats_placed",
    ]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._fh = None
        self._writer = None
        self._ep_count = 0

    def _on_training_start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            bd = info.get("breakdown")
            cfg = info.get("config")
            cats = info.get("cats_placed", [])
            if bd is None or cfg is None:
                continue
            self._ep_count += 1
            self._writer.writerow([
                self.num_timesteps, self._ep_count,
                bd["total"], bd["availability"],
                bd["privacy"], bd["light"], bd["efficiency"],
                bd["privacy_loss"], bd["light_loss"], bd["waste_loss"],
                len(cats), len(set(cats)),
                cfg["room_w"], cfg["room_h"], cfg["door_pos"], cfg["win_wall"],
                bd["exposed_cells"], bd["total_bed_cells"],
                int(bd["pillow_seen"]), int(bd["window_blocked"]),
                bd["unreachable_cells"],
                "|".join(cats),
            ])
            # flush so `tail -f episodes.csv` shows new rows immediately
            # (csv writer otherwise buffers and tailing looks frozen).
            self._fh.flush()
            # TensorBoard aggregates (these show up under custom/ in TB)
            self.logger.record_mean("custom/availability", bd["availability"])
            self.logger.record_mean("custom/privacy",      bd["privacy"])
            self.logger.record_mean("custom/light",        bd["light"])
            self.logger.record_mean("custom/efficiency",   bd["efficiency"])
            self.logger.record_mean("custom/n_placed", len(cats))
            self.logger.record_mean("custom/n_unique_cats", len(set(cats)))
            self.logger.record_mean("custom/unreachable", bd["unreachable_cells"])
        return True

    def _on_training_end(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()


class LiveProgressCallback(BaseCallback):
    """Prints a rolling-window summary every N episodes — gives a clear
    "is training going well?" signal directly in stdout, so teammates don't
    need a separate `tail -f` window. Prints look like:

      [live] step=  5624  ep=  1000  total=2.49  priv=0.64  light=0.81 ...

    where `total` is the rolling mean of the last `window` episodes.
    """

    HEADER = ("[live] step=  step  ep=     ep  total=##.##  "
              "priv=#.##  light=#.##  eff=#.##  n_pl=#.#  bed=##%")

    def __init__(self, window: int = 500, every: int = 500):
        super().__init__()
        self.window = window
        self.every = every
        self.recent: list[dict] = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            bd = info.get("breakdown")
            if bd is None:
                continue
            cats = info.get("cats_placed", [])
            self._ep_count += 1
            self.recent.append({
                "total": bd["total"],
                "priv":  bd["privacy"],
                "light": bd["light"],
                "eff":   bd["efficiency"],
                "n_pl":  len(cats),
                "bed":   1 if "bed" in cats else 0,
            })
            if len(self.recent) > self.window:
                self.recent.pop(0)
            if self._ep_count % self.every == 0:
                w = self.recent
                n = len(w)
                avg = lambda k: sum(r[k] for r in w) / n
                print(f"[live] step={self.num_timesteps:>6}  ep={self._ep_count:>5}  "
                      f"total={avg('total'):.2f}  priv={avg('priv'):.2f}  "
                      f"light={avg('light'):.2f}  eff={avg('eff'):.2f}  "
                      f"n_pl={avg('n_pl'):.1f}  bed={avg('bed'):.0%}",
                      flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None,
                   help="run name; default: ppo_<timestamp>")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512,
                   help="PPO rollout length per env (×n_envs per update)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="entropy bonus; higher → more exploration")
    p.add_argument("--placement-bonus", type=float, default=0.0,
                   help="dense reward per successful placement (0 = pure sparse)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="run eval every N total env steps (sum across n_envs)")
    p.add_argument("--eval-eps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()

    run_name = args.name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best").mkdir(exist_ok=True)

    # persist hyperparameters + env constants + git commit so the run is
    # fully reproducible even if env.py later changes
    cfg = vars(args).copy()
    cfg["run_name"] = run_name
    cfg["started_at"] = datetime.now().isoformat()
    cfg["git_commit"] = _git_commit_hash()
    cfg["env_constants"] = {
        "WCOEFF": WCOEFF,
        "GRID_M": GRID_M,
        "GW": GW, "GH": GH, "DW": DW,
        "N_ACTIONS": N_ACTIONS,
        "MAX_PER_CAT": MAX_PER_CAT,
    }
    cfg["reward_shaping"] = {
        "placement_bonus": args.placement_bonus,
        "note": ("dense reward per successful placement (non-DONE). 0 = pure "
                 "sparse, matches env.py / verify.py / HTML exactly."),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── envs ─────────────────────────────────────────────
    env_fns = [make_env(args.seed + i, args.max_steps, args.placement_bonus)
               for i in range(args.n_envs)]
    if args.n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    # Eval env: same bonus so eval reward matches training reward; use a
    # disjoint seed so eval distribution isn't a subset of training.
    eval_env = DummyVecEnv([make_env(args.seed + 100_000, args.max_steps,
                                     args.placement_bonus)])

    # ── model ────────────────────────────────────────────
    # MLP backbone widened to 128-128 (was 64-64). Factored output head saves
    # ~2.6 M params on the output side, so we can afford a richer backbone.
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    # Linear LR decay: starts at args.lr, ramps to 0 over training.
    # SB3 calls the schedule with `progress_remaining` ∈ [1, 0].
    lr_schedule = lambda progress_remaining: args.lr * progress_remaining
    model = MaskablePPO(
        FactoredMaskablePolicy, vec_env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
    )

    # Direct logger: progress.csv + TB events both land in run_dir/.
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # ── callbacks ────────────────────────────────────────
    ep_logger = EpisodeBreakdownLogger(run_dir / "episodes.csv")
    live_cb = LiveProgressCallback(window=500, every=500)
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir),                                # → evaluations.npz
        eval_freq=max(args.eval_freq // args.n_envs, 1),      # per-env counter
        n_eval_episodes=args.eval_eps,
        deterministic=True,
        render=False,
    )

    print(f"\n=== Training {run_name} ===")
    print(f"  timesteps  : {args.timesteps:,}")
    print(f"  n_envs     : {args.n_envs}")
    print(f"  n_steps    : {args.n_steps}  (rollout = {args.n_steps * args.n_envs} env steps)")
    print(f"  lr         : {args.lr}")
    print(f"  ent_coef   : {args.ent_coef}")
    print(f"  bonus      : {args.placement_bonus}  (per-placement dense reward)")
    print(f"  eval every : {args.eval_freq:,} steps  ({args.eval_eps} eps)")
    print(f"  logs       : {run_dir}/")
    print(f"  view live  : tensorboard --logdir runs/\n")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[ep_logger, live_cb, eval_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final model before exit ...")

    elapsed = time.time() - t0
    model.save(run_dir / "final.zip")
    (run_dir / "training_time.txt").write_text(
        f"{elapsed:.1f} seconds ({elapsed / 60:.2f} min)\n"
    )

    # ── summary ──────────────────────────────────────────
    print(f"\n=== Done in {elapsed / 60:.1f} min ===")
    print(f"final model  → {run_dir}/final.zip")
    print(f"best model   → {run_dir}/best/best_model.zip")
    print(f"episodes     → {run_dir}/episodes.csv")
    print(f"progress     → {run_dir}/progress.csv")
    print(f"evaluations  → {run_dir}/evaluations.npz")
    print(f"tensorboard  → tensorboard --logdir {run_dir}")
    print(f"\nNext: python plot_training.py --run {run_name}    (or)")
    print(f"      python render.py --episodes 5 --seed 0 \\")
    print(f"          --model {run_dir}/best/best_model.zip --save videos/trained.mp4")


if __name__ == "__main__":
    main()
