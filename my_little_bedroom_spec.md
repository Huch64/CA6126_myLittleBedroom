# 🏠 My Little Bedroom — Final Spec

> CA6126 RL Final Project · MaskablePPO agent learns to furnish a randomized bedroom
>
> `R = A × privacy × light × efficiency`
> &nbsp;&nbsp;`privacy = 1 − pillow_ratio` · `light = 1 − window_ratio` · `efficiency = 1 − waste_ratio`
>
> Both `env.py` and the interactive HTML preview implement this v3 reward.
> Every penalty is a fraction of its relevant resource in [0, 1], so each
> `(1 − ratio)` is an independent "discount factor" in [0, 1] — zero
> weights, zero τ, every term has one physical meaning. Earlier additive
> forms (v1, v2) are documented in the *v1 → v3 evolution* table for the
> report's reward-engineering section.

---

## 📋 Assignment Requirements

**Task**: Formulate a novel RL problem and solve it.

**Deliverables** (1 submission per group):
- 📊 Report (PPT/PDF, ≤20 pages) — clean slides, to the point
- 🎬 Videos of agent playing (random vs trained)
- 💾 Source code (zip, no model checkpoints)

**Grading (20 pts)**:

| Category | Points | What |
|----------|--------|------|
| 🆕 Novelty | 5 | Original RL formulation, not from existing envs |
| 📐 Formalism | 2 | Clear MDP statement |
| 💻 Environment | 3 | Gymnasium-style env code |
| 🎬 Showcase | 5 | Videos of agent playing |
| 📈 Training | 5 | Document training process |

---

## 🎯 MDP Formulation

### State *S*

Fixed 26×22 multi-channel grid (0.15m/cell).

| Channel | Content |
|---------|---------|
| ch0 | Occupancy: 0=empty, 1=wall, 2-19=furniture ID |
| ch1 | Door position |
| ch2 | Window position |

Per-episode randomization:

| Element | Range |
|---------|-------|
| 🏠 Room | 14-26 × 18-22 cells, outside = wall |
| 🚪 Door | Bottom wall, random position, 6 cells, 90° swing, hinge nearest corner |
| 🪟 Window | Random wall (top/left/right), centered, width = 50% wall length |

### Action *A*

One step = place one complete furniture or DONE:

```
a = (furniture_id, x, y, orientation) | DONE

Flat index: fid × (26×22×4) + x × (22×4) + y × 4 + ori
Total: 18 × 26 × 22 × 4 + 1 = 41,185
Most actions masked (overlap / out of bounds / door swing)
```

### Transition *T*

Deterministic. Furniture placed permanently. DONE ends episode.

### 📏 State / action space size

**Initial state space** (room configurations sampled at every `reset()`):

| Element | Domain | Count |
|---|---|---|
| `room_w` | 14–26 (integer) | 13 |
| `room_h` | 18–22 (integer) | 5 |
| `door_pos` | 0 … room_w − 6 | avg ≈ 15 |
| `win_wall` | {top, left, right} | 3 |

Window pos/width are derived (centered, 50% of wall length), so they don't
add cardinality. Total initial configurations: **13 × 5 × ~15 × 3 ≈ 2 900**.

**Reachable state space** (after placements) — combinatorial in the number
of placements and their positions. With up to 6 items × ~thousand valid
positions each, the trajectory space is astronomical (~10²⁰ episodes), but
the agent only needs to learn a policy over the ~2 900 root configs.

**Action space**: `Discrete(41 185)` = 18 furniture × 26 × 22 grid × 4
orientations + 1 DONE. Per step, the action mask typically leaves
**10 000 – 16 000 valid actions**; PPO samples only from these via
`sb3-contrib`'s `MaskablePPO`.

### Reward *R*  (v3: multiplicative, ratio-based)

Computed once at episode end:

```
R = Availability × privacy × light × efficiency

  privacy    = 1 − pillow_ratio    (door can't peek at pillow)
  light      = 1 − window_ratio    (window not blocked)
  efficiency = 1 − waste_ratio     (room is walkable)
```

Four factors, zero weights, zero τ, no transcendentals. Each ratio is a
fraction of its relevant resource in [0, 1], so the corresponding
`(1 − ratio)` is automatically a discount in [0, 1] with **one physical
meaning per factor**:

- pillow fully exposed (`pillow_ratio = 1`) → privacy = 0 → R = 0
- half the window blocked (`window_ratio = 0.5`) → ×0.5
- 20 % dead space (`waste_ratio = 0.2`) → efficiency = ×0.8
- all three combine multiplicatively, no clamping needed

Two key properties:

1. **Scale-invariant**: ratios cancel cell counts, so the same ratio means
   the same effect across all room sizes — no per-room tuning.
2. **No DONE-trap**: R = 0 iff Availability = 0 (DONE-immediate) or some
   ratio hits 1. Any sensible placement → R > 0, so "do nothing" is never
   the locally safe choice.

Defaults in `env.py`:

```python
AVAIL_FACTOR = {"bed": 0.32, "desk": 0.43, "wardrobe": 0.33,
                "cabinet": 0.37, "nightstand": 0.29}
NIGHTSTAND_PAIR_BONUS = 1.0   # absolute bonus when paired with headboard
# No weights, no τ — each (1 − ratio) is an independent discount factor.
```

---

## 💰 Reward Detail

### ✅ Availability = Σ usable furniture value

`value = √(area in cells) × cat_factor` (sub-linear: big furniture has
diminishing returns). Each piece scores only if ALL conditions hold:

| Condition | Rule |
|-----------|------|
| No duplicate | 2nd of same category → both score 0 (nightstand allows 2) |
| Zone clear | Functional zone cells empty + in bounds |
| Reachable | Furniture neighbor in flood-fill swept set |
| Bed special | ≥ 1 long side with cells in swept (not just any 1 cell) |

Functional zones:

| Furniture | Faces | Depth | Notes |
|-----------|-------|-------|-------|
| 🛏️ Bed | 3 (two long sides + foot) | 3 cells | ≥ 1 long side accessible, nightstand allowed |
| 🖥️ Desk | 1 front | 5 cells | Fully clear (includes chair space) |
| 👔 Wardrobe | 1 front | 4 cells | Fully clear |
| 🗄️ Cabinet | 1 front | 3 cells | Fully clear |
| 🛋️ Nightstand | 1 front | 3 cells | Fully clear |

**Nightstand pairing**: on bed's long side + aligned with headboard → **+1 bonus** (absolute), unpaired → ×0.5 multiplier.

### 👀 Privacy  →  privacy = (1 − pillow_ratio)

```
pillow_ratio = exposed_pillow_cells / total_pillow_cells
```

Fraction of pillow cells visible from door's 90° cone, unblocked by
wardrobe. Pillow fully hidden → privacy = 1 (no discount). Pillow fully
exposed → privacy = 0 → R = 0.

**Bed body exposure (excluding pillow) is intentionally NOT penalised** —
in real design, privacy concerns the head end (where the pillow is), not
the side of the bed.

Only **wardrobe** blocks line of sight to bed (other furniture too short).

### 🪟 Light  →  light = (1 − window_ratio)

```
window_ratio = blocked_window_cells / window_strip_cells
```

Fraction of the 2-deep window strip occupied by tall furniture
(bed/wardrobe). Window strip fully clear → light = 1. Half blocked →
light = 0.5.

### 🗑️ Efficiency  →  efficiency = (1 − waste_ratio)

```
waste_ratio = unreachable_cells / total_empty_cells
```

`unreachable_cells` = empty cells the 3×3-brush flood fill from the door
cannot reach. Expressed as a fraction of total empty cells, so a tightly
packed room with a few stranded corners ranks just as well as a sparse
room with a similar proportion unreachable.

No furniture placed → ratio = 0 → efficiency = 1 (no penalty), but
availability = 0 → R = 0 anyway.

### 📜 v1 → v3 evolution (recorded for the report)

| Version | R formula | Key change |
|---------|-----------|------------|
| **v1** (initial) | A − D − W (additive, binary lumps `+4` for pillow / `+3` for window) | Original hand-tuned reward; binary cliffs created the DONE-trap |
| **v2** | A − D − W (unified `cells × factor × CELL_REWARD`) | Removed binary discontinuities; everything per-cell |
| **v3** (current — env.py + HTML) | A × privacy × light × efficiency, each factor `1 − ratio` | Each penalty is an independent discount in [0, 1] with one physical meaning; zero weights, zero τ, no transcendentals; eliminates DONE-trap; scale-invariant across room sizes |

Audit-driven derivation: `python reward_audit.py` runs random / greedy /
edge-greedy policies + a continuity sweep on the current env, plotting the
reward distribution and per-component contributions to verify the design
is "healthy" before launching a multi-hour training run.

---

## 🪑 Catalog — 5 categories, 18 pieces

| Category | Variant | Grid w×h | Value | Limit |
|----------|---------|----------|-------|-------|
| 🛏️ Bed | 0.9 | 14×6 | 3 | pick 1 |
| | 1.2 | 14×8 | 3.5 | |
| | 1.5 | 14×10 | 4 | |
| | 1.8 | 14×12 | 4 | |
| 🖥️ Desk | S / L / XL | 6-12 × 4 | 2.5-3 | pick 1 (includes chair 4×4) |
| 👔 Wardrobe | S-XXL | 6-14 × 4 | 1.5-2.5 | pick 1 |
| 🗄️ Cabinet | S-XL | 4-10 × 3 | 1-2 | pick 1 |
| 🛋️ Nightstand | A / B | 4×3 / 3×3 | 1 | up to 2 |

Value ≈ √area, encourages bigger pieces when space allows.

---

## 🔧 Key Algorithms

### Flood Fill (3×3 brush + swept expansion)

```python
# 1. Each cell: check if 3×3 neighborhood all empty → passable center
# 2. BFS from door on passable centers
# 3. Expand centers ±1 → swept (cells person can actually stand on)
# Min passage width: 3 cells = 0.45m
```

### Door Swing

```python
# Hinge at nearest corner (doorPos < wallLen/2 → left hinge, else right)
# 90° arc, radius = door width (6 cells)
# Check cell CENTER in arc (not corner) → tighter, fewer locked cells
# Swing cells blocked for furniture placement
```

### Bresenham Vision (Privacy)

```python
# 90° cone (±45°) from door center
# For each bed cell in cone: trace ray, check if wardrobe blocks it
# Only wardrobe blocks (tall enough), all other furniture transparent
```

### Nightstand Pairing

```python
# Paired if: adjacent to bed's LONG side AND aligned with headboard column/row
# ori=0/2: check above/below bed, overlapping headboard x position
# ori=1/3: check left/right of bed, overlapping headboard y position
```

---

## 🏋️ Training Plan

Single run: full reward + random room/door/window per episode.

```python
algorithm: MaskablePPO (sb3-contrib)
n_envs: 8
total_timesteps: ~500K
observation: multi-channel 26×22 grid
action: Discrete(41185) with action_masks()
```

Deliverables: 📈 training curve (x=steps, y=reward) + 🎬 random vs trained video

---

## 📁 Files

| File | Purpose | Status |
|------|---------|--------|
| `my_little_bedroom.html` | 🎮 Interactive preview (reward reference) | ✅ Done |
| `my_little_bedroom_spec.md` | 📄 This document | ✅ Done |
| `env.py` | 🏗️ Gymnasium environment + action mask + RGB render | ✅ Done |
| `sanity_check.py` | 🧪 Smoke tests (shapes / scripted ep / random rollout) | ✅ Done |
| `verify.py` | ✅ Hand-crafted cases for cross-checking against the HTML | ✅ Done |
| `reward_audit.py` | 🔬 Reward landscape profiling (continuity / DONE-trap / per-component) | ✅ Done |
| `train.py` | 🚂 MaskablePPO training + CSV/TB logging + best-model save | ✅ Done |
| `render.py` | 🎬 Record agent playing to mp4 (random or trained) | ✅ Done |
| `plot_training.py` | 📈 Generate report figures from `runs/<name>/` logs | ✅ Done |
| `TEAMMATE.md` | 👥 Quick-start for collaborators (run training, knobs, what to monitor) | ✅ Done |
| `report.pptx` | 📊 Slides (≤20 pages) | ⬜ TODO |

---

## 🏗️ Implementation Notes (for env.py)

```python
class MyLittleBedroom(gym.Env):
    """
    Observation: Box(0, 19, shape=(3, 22, 26), dtype=int8)
    Action: Discrete(41185) with action_masks()
    
    Grid constants: GW=26, GH=22, CELL=0.15m
    Room: random 14-26 × 18-22 per reset
    Door: bottom wall, random pos, 6 cells wide
    Window: random wall (T/L/R), centered, width=50% wall
    """
    
    def reset(self):
        # Randomize room_w, room_h, door_pos, win_wall
        # Initialize empty grid, mark walls
        # Compute door swing cells
        # Return observation, info
    
    def step(self, action):
        # Decode: fid, x, y, ori = decode(action) or DONE
        # Place furniture on grid
        # If DONE or max_steps: compute reward, done=True
        # Return obs, reward, done, truncated, info
    
    def calc_reward(self):
        # Availability: Σ (√area × cat_factor) × pair/mult adjustments
        # privacy    = 1 − pillow_ratio
        # light      = 1 − window_ratio
        # efficiency = 1 − waste_ratio
        # Return A × privacy × light × efficiency
    
    def action_masks(self):
        # For each (fid, x, y, ori): check bounds + no overlap + not in swing
        # Return bool array [41185]
    
    def flood_fill(self):
        # 3×3 brush passability → BFS from door → expand → swept
    
    def render(self):
        # Return RGB image of current room state
```

### Reward calc — same geometric primitives as HTML, different combination
- `flood()` / 3×3 brush passability → reachability set      (matches HTML)
- `zoneRects()` → functional zones for each piece            (matches HTML)
- Bresenham vision cone, wardrobe-only line-of-sight block   (matches HTML)
- Nightstand pairing (long side + headboard alignment)       (matches HTML)
- **Combined multiplicatively (v3) instead of additively (HTML v1)** — see § Reward (v3)

---

## ❓ Open Questions

- 🛏️ Will the agent skip bed in tiny rooms where pillow can't be hidden? Worst case `pillow_ratio = 1` → privacy = 0 → R = 0. Watch training for "skip-bed" behaviour
- ⏹️ Early stopping after 1-2 items? v3 eliminates the DONE-trap (R = 0 only when A = 0 or some `(1 − ratio) = 0`), but agent could still settle for partial play if the marginal availability of the next item < the marginal discount it incurs. Audit shows greedy = +0.35, optimal likely +5 to +10
- 🏗️ Bed accessibility: currently "partial OK" (≥1 cell of long side in swept); may want ≥50% for stricter realism
