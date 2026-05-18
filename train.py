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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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
        "total", "availability", "diversity", "n_categories",
        "compactness", "shape_coef",
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
                bd.get("diversity", 0.0), bd.get("n_categories", 0),
                bd.get("compactness", 0.0), bd.get("shape_coef", 0.0),
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
            self.logger.record_mean("custom/diversity",    bd.get("diversity", 0.0))
            self.logger.record_mean("custom/n_categories", bd.get("n_categories", 0))
            self.logger.record_mean("custom/compactness",  bd.get("compactness", 0.0))
            self.logger.record_mean("custom/shape_coef",   bd.get("shape_coef", 0.0))
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
    """Prints rolling-window training stats. Shows:

      [live] 12% step=  5624 ep= 1000  total=2.49 (max 5.8)  priv=0.64
             light=0.81  eff=0.59  n_pl=4.6  bed=100%  Δ+0.15

    where:
      - 12%       = fraction of total timesteps completed
      - total     = rolling mean over last `window` episodes
      - max       = best total seen in window (probe of upper bound)
      - Δ         = change vs previous window (trend indicator)
      - priv/light/eff = rolling means of those factors
      - bed       = fraction of window episodes with a bed
    """

    def __init__(self, window: int = 200, every: int = 200,
                 total_timesteps: int | None = None):
        super().__init__()
        self.window = window
        self.every = every
        self.total_timesteps = total_timesteps
        self.recent: list[dict] = []
        self._ep_count = 0
        self._last_mean: float | None = None
        self._best_seen = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            bd = info.get("breakdown")
            if bd is None:
                continue
            cats = info.get("cats_placed", [])
            self._ep_count += 1
            tot = bd["total"]
            self._best_seen = max(self._best_seen, tot)
            self.recent.append({
                "total": tot,
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
                mean_total = avg("total")
                max_total = max(r["total"] for r in w)
                pct = ""
                if self.total_timesteps:
                    pct = f"{self.num_timesteps / self.total_timesteps * 100:>4.0f}% "
                trend = ""
                if self._last_mean is not None:
                    delta = mean_total - self._last_mean
                    arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
                    trend = f"  Δ{arrow}{delta:+.2f}"
                self._last_mean = mean_total
                print(f"[live] {pct}step={self.num_timesteps:>7}  "
                      f"ep={self._ep_count:>5}  "
                      f"total={mean_total:.2f} (max {max_total:.1f}, best {self._best_seen:.1f})  "
                      f"priv={avg('priv'):.2f}  light={avg('light'):.2f}  eff={avg('eff'):.2f}  "
                      f"n_pl={avg('n_pl'):.1f}  bed={avg('bed'):.0%}"
                      f"{trend}",
                      flush=True)
        return True


class MilestoneCallback(BaseCallback):
    """Prints a banner at every 10% of training progress with a summary
    of recent performance. Useful for "set it and forget it" runs to spot
    plateaus or drift at a glance."""

    def __init__(self, total_timesteps: int, n_milestones: int = 10):
        super().__init__()
        self.total = total_timesteps
        self.step_per_milestone = max(total_timesteps // n_milestones, 1)
        self._next_milestone = self.step_per_milestone
        self._t0 = time.time()
        self._recent: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            bd = info.get("breakdown")
            if bd is None:
                continue
            cats = info.get("cats_placed", [])
            self._recent.append({
                "total": bd["total"], "priv": bd["privacy"],
                "light": bd["light"], "eff": bd["efficiency"],
                "n_pl": len(cats),
                "bed":  1 if "bed" in cats else 0,
            })
            if len(self._recent) > 500:
                self._recent.pop(0)
        if self.num_timesteps >= self._next_milestone:
            pct = self.num_timesteps / self.total * 100
            elapsed = time.time() - self._t0
            eta = elapsed * (self.total - self.num_timesteps) / max(self.num_timesteps, 1)
            mins_elapsed = elapsed / 60
            mins_eta = eta / 60
            if self._recent:
                w = self._recent
                n = len(w)
                a = lambda k: sum(r[k] for r in w) / n
                summary = (f"total={a('total'):.2f}  priv={a('priv'):.2f}  "
                           f"light={a('light'):.2f}  eff={a('eff'):.2f}  "
                           f"bed={a('bed'):.0%}")
            else:
                summary = "(no episodes yet)"
            print(f"\n{'═'*70}\n"
                  f"  ★ MILESTONE  {pct:>4.0f}%  step {self.num_timesteps:>8,}/{self.total:,}"
                  f"   elapsed {mins_elapsed:>5.1f}m   ETA {mins_eta:>5.1f}m\n"
                  f"  └─ last 500 eps:  {summary}\n"
                  f"{'═'*70}\n",
                  flush=True)
            self._next_milestone += self.step_per_milestone
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
    p.add_argument("--checkpoint-freq", type=int, default=50_000,
                   help="save a checkpoint every N total env steps "
                        "(0 = disabled). Files saved as model_<step>_steps.zip")
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
    # window=200 / every=200 → progress line every ~1.5 min (vs old 4 min).
    live_cb = LiveProgressCallback(window=200, every=200,
                                   total_timesteps=args.timesteps)
    # Every 10% of training: print a milestone banner with summary.
    milestone_cb = MilestoneCallback(total_timesteps=args.timesteps,
                                     n_milestones=10)
    # Periodic checkpoint snapshot (every checkpoint_freq env steps).
    # Useful for crash recovery and analyzing agent at different stages.
    callbacks = [ep_logger, live_cb, milestone_cb]
    if args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),  # per-env counter
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
            verbose=0,
        )
        callbacks.append(ckpt_cb)
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
    if args.checkpoint_freq > 0:
        print(f"  ckpt every : {args.checkpoint_freq:,} steps  → runs/{run_name}/checkpoints/")
    else:
        print(f"  ckpt every : disabled")
    print(f"  logs       : {run_dir}/")
    print(f"  view live  : tensorboard --logdir runs/\n")

    t0 = time.time()
    try:
        callbacks.append(eval_cb)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
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
