# 🏠 My Little Bedroom — CA6126 RL Final Project

> A MaskablePPO agent learns to furnish a randomized bedroom.
> Reward: `R = Availability × privacy × light × efficiency + diversity`
> &nbsp;&nbsp;privacy / light: linear remap to [0.3, 1]; efficiency: full [0, 1] for strong wall-hugging signal — see [spec](my_little_bedroom_spec.md)
> &nbsp;&nbsp;diversity: `+1` per distinct furniture category placed (max +5), added outside the product.
> &nbsp;&nbsp;Semantic gate: `R = 0` if no bed (bedroom must have a bed).

Full MDP spec: [`my_little_bedroom_spec.md`](my_little_bedroom_spec.md)
Interactive reward reference (open in a browser): [`my_little_bedroom.html`](my_little_bedroom.html)
Assignment brief: [`CA6126 final project.pdf`](CA6126%20final%20project.pdf)

---

## 🚀 Quick Start

```bash
# 1. create env + install deps (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. smoke-test the env (~5 s)
python sanity_check.py

# 3. spot-check that env reward matches the HTML preview (~5 s)
python verify.py                # prints HTML setup steps + env's A/D/W

# 4. record a random-agent video (random.mp4 is required for submission)
python render.py --episodes 2 --seed 0 --save videos/random.mp4

# 5. train MaskablePPO (default 500K steps, ~25–40 min on CPU)
python train.py

# 6. record the trained agent on the SAME seeds as random
python render.py --episodes 5 --seed 0 \
    --model runs/<run_name>/best/best_model.zip \
    --save videos/trained.mp4
```

While training, watch curves live:
```bash
tensorboard --logdir runs/
```

---

## 📁 Files

| File | Purpose | Status |
|---|---|---|
| 🎮 `my_little_bedroom.html` | Interactive preview — the visual + reward reference | ✅ |
| 📄 `my_little_bedroom_spec.md` | Full MDP / reward spec | ✅ |
| 🏗️ `env.py` | Gymnasium env (v3 reward), action mask, RGB render | ✅ |
| 🧪 `sanity_check.py` | 3 smoke tests (shapes, scripted episode, random rollout) | ✅ |
| ✅ `verify.py` | Hand-crafted cases for cross-checking against the HTML | ✅ |
| 🔬 `reward_audit.py` | Profile reward distribution before training (continuity, DONE-trap gap, per-component health) | ✅ |
| 🚂 `train.py` | MaskablePPO training + CSV/TB logging + best-model saving | ✅ |
| 🎬 `render.py` | Record agent playing to mp4 (random or trained policy) | ✅ |
| 📈 `plot_training.py` | Generate report figures from `runs/<name>/` logs | ✅ |
| 👥 `TEAMMATE.md` | Quick-start for collaborators — what to run, what to monitor, what to tune | ✅ |
| 📊 `report.pptx` | Slides (≤ 20 pages) | ⬜ TODO |

Generated at runtime (gitignored):
- `runs/<run_name>/` — training logs (`progress.csv`, `episodes.csv`, `evaluations.npz`, TB events, `final.zip`, `best/best_model.zip`)
- `videos/*.mp4` — recorded agent playthroughs

---

## 🧭 Workflow

```
┌────────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────────┐
│ verify.py  │ →  │ sanity_    │ →  │  train.py    │ →  │  render.py   │
│ vs HTML    │    │ check.py   │    │  → runs/...  │    │  → videos/.. │
└────────────┘    └────────────┘    └──────────────┘    └──────────────┘
   reward            env API           policy +              showcase
   correctness       sanity            logs + ckpt           videos
```

---

## 🏋️ Training Details

- **Algorithm**: MaskablePPO (`sb3-contrib`) — vanilla PPO + action masking so the agent only samples from the ~16 K valid placements at each step.
- **Policy**: `MlpPolicy`, observation flattened from `(3, 22, 26)` → 1 716 features.
- **Parallel envs**: 8 (`SubprocVecEnv`), each samples a fresh random room every reset.
- **Total steps**: 500 K (≈ 25–40 min on CPU).
- **Eval**: every 10 K steps, 20 deterministic episodes; best model auto-saved.

Logs written per run under `runs/<run_name>/`:

| File | What's in it |
|---|---|
| `config.json` | Hyperparameters + start time |
| `progress.csv` | SB3 internals: ep_rew_mean, value/policy loss, KL, lr, entropy, … |
| `episodes.csv` | One row per training episode: A/D/W, room config, items placed |
| `evaluations.npz` | Eval rewards over time (per-seed reward across eval episodes) |
| `events.out.tfevents.*` | TensorBoard events |
| `final.zip`, `best/best_model.zip` | Policies |

---

## 👥 Team Workflow

- 🌿 Main branch stays runnable — feature work goes on branches (`<initial>/<topic>`, e.g. `hcw/plot-training`).
- 🧪 Before pushing: `python sanity_check.py` and (if env changed) `python verify.py`.
- 🚫 Don't commit `runs/`, `checkpoints/`, `videos/`, or `*.zip` — already gitignored.
- 📝 PR description: what changed + how you verified it (1-paragraph).

```bash
git checkout -b hcw/plot-training
# edit
python sanity_check.py            # quick check
git add plot_training.py
git commit -m "feat: training-curve + reward-breakdown plots"
git push -u origin hcw/plot-training
# open PR
```

---

## 📤 Submission Checklist

Per assignment brief (PDF):

- [ ] 📊 Report (PPT / PDF, ≤ 20 pages) — **must include group members on title page**
- [ ] 🎬 `videos/random.mp4` — random agent (clearly bad)
- [ ] 🎬 `videos/trained.mp4` — trained agent (clearly better)
- [ ] 💾 Source code zip — **without** `runs/`, `checkpoints/`, `videos/` (they're large)

Report sections expected by the rubric:
- Title page + group members
- RL game description and formulation (MDP states / actions / transition / reward + state-space size estimate)
- RL solution (algorithm choice, tricks like masking & reward shaping, training-curve plot, eval results)

Grading: 20 pts total — 5 novelty / 2 formalism / 3 env / 5 showcase / 5 training process.

---

## 🧪 Reward audit before training

`reward_audit.py` profiles the reward landscape under random / greedy / edge-greedy / DONE policies + a continuity sweep, **before** committing to a 1-3 h training run. Plots the distribution, per-component contributions, and DONE-trap gap. This is how we converged on the v3 reward (multiplicative + ratio-based) — see the iteration history in the spec.

```bash
python reward_audit.py --n 2000 --save plots/audit.png
```

Healthy signs to look for:
- Reward distribution roughly unimodal (no clumps at single value)
- DONE-trap gap > 0 (any "play" policy ≥ DONE-immediate)
- No single factor dominates the discount (privacy / light / efficiency means in the same order of magnitude)
- Continuity sweep smooth (no cliffs)

## ❓ Open Issues to Watch

- **PPO failure modes** even with v3:
  - If `episodes.csv` shows `n_placed` stuck low, try `--ent-coef 0.05` (more exploration)
  - The training pipeline still supports `--placement-bonus X` (dense per-placement reward as further insurance — kept for backwards compat with v1/v2 runs)
- **Action mask cost**: ≈ 3–4 ms per env step (vectorized). Fine for 500 K steps.
