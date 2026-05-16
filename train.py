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
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import CATALOG, MyLittleBedroom


def _mask_fn(env):
    return env.action_masks()


def make_env(seed: int | None = None, max_steps: int = 8):
    """Factory returning a fresh env: MyLittleBedroom → ActionMasker → Monitor.

    Monitor is outermost so it sees the unmodified reward; ActionMasker exposes
    .action_masks() which MaskablePPO calls each step through the VecEnv.
    """
    def _init():
        env = MyLittleBedroom(seed=seed, max_steps=max_steps)
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        return env
    return _init


class EpisodeBreakdownLogger(BaseCallback):
    """Writes one CSV row per completed episode + aggregates to TensorBoard.

    Per-row columns capture everything a report figure could want: the three
    reward components, what got placed, the room/door/window config, and the
    discomfort sub-flags. Plot scripts can slice on any of these.
    """

    HEADER = [
        "step", "ep_idx",
        "total", "availability", "discomfort", "waste",
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
                bd["total"], bd["availability"], bd["discomfort"], bd["waste"],
                len(cats), len(set(cats)),
                cfg["room_w"], cfg["room_h"], cfg["door_pos"], cfg["win_wall"],
                bd["exposed_cells"], bd["total_bed_cells"],
                int(bd["pillow_seen"]), int(bd["window_blocked"]),
                bd["unreachable_cells"],
                "|".join(cats),
            ])
            # TensorBoard aggregates (these show up under custom/ in TB)
            self.logger.record_mean("custom/availability", bd["availability"])
            self.logger.record_mean("custom/discomfort", bd["discomfort"])
            self.logger.record_mean("custom/waste", bd["waste"])
            self.logger.record_mean("custom/n_placed", len(cats))
            self.logger.record_mean("custom/n_unique_cats", len(set(cats)))
            self.logger.record_mean("custom/unreachable", bd["unreachable_cells"])
        return True

    def _on_training_end(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()


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

    # persist hyperparameters
    cfg = vars(args).copy()
    cfg["run_name"] = run_name
    cfg["started_at"] = datetime.now().isoformat()
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── envs ─────────────────────────────────────────────
    env_fns = [make_env(args.seed + i, args.max_steps) for i in range(args.n_envs)]
    if args.n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    # Separate seed for eval so it doesn't overlap training distribution.
    eval_env = DummyVecEnv([make_env(args.seed + 100_000, args.max_steps)])

    # ── model ────────────────────────────────────────────
    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate=args.lr,
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
    )

    # Direct logger: progress.csv + TB events both land in run_dir/.
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # ── callbacks ────────────────────────────────────────
    ep_logger = EpisodeBreakdownLogger(run_dir / "episodes.csv")
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
    print(f"  eval every : {args.eval_freq:,} steps  ({args.eval_eps} eps)")
    print(f"  logs       : {run_dir}/")
    print(f"  view live  : tensorboard --logdir runs/\n")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[ep_logger, eval_cb],
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
