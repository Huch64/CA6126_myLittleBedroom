# 🏠 My Little Bedroom — Final Spec

> CA6126 RL Final Project · MaskablePPO agent learns to furnish a randomized bedroom
>
> `R = A × privacy × light × efficiency`
> &nbsp;&nbsp;`privacy = 1 − 0.7 × pillow_ratio`   (linear remap to [0.3, 1])
> · `light = 1 − 0.7 × window_ratio`             (linear remap to [0.3, 1])
> · `efficiency = 1 − waste_ratio`                (full [0, 1], no floor)
>
> Privacy and light use a linear remap to **[0.3, 1.0]** so every ratio
> change produces a smooth factor change (no flat zone), but extreme
> misses don't wipe out the rest of the reward. Efficiency uses the
> **full [0, 1]** range — agent has the strongest incentive to use space
> well (e.g. push furniture against walls instead of leaving dead corners).
>
> **Semantic gate**: `R = 0` if no bed is placed (a bedroom must have a bed).
> Enforced by both (a) action mask blocking DONE until bed exists, and
> (b) reward gate returning 0 at episode end if no bed.
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
```

**Action mask** (computed per step, ~10K-16K valid out of 41,185):

1. **Geometric**: no overlap, in-bounds, not in door swing
2. **Bed-first**: step 0 restricted to bed actions only (avoids "bed unplaceable late")
3. **Per-category limit**: `MAX_PER_CAT` enforced (1 bed, 1 desk, 1 wardrobe, 1 cabinet, 2 nightstand)
4. **Zone clearance**: new placement can't fall in any non-bed piece's functional zone
5. **Own zone validity**: bed uses `partial=True` (≥ 1 zone cell accessible);
   other furniture requires the full functional zone clear + in-bounds
6. **DONE gate**: blocked until ≥ 1 bed has been placed (escape valve if no
   valid placement remains, with `R=0` reward override at end)

The interactive HTML preview (`my_little_bedroom.html`) mirrors all five
rules — try selecting a non-bed piece on a fresh room and the catalog
button greys out, hovering shows red highlights.

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

  privacy    = 1 − (1 − FACTOR_FLOOR) × pillow_ratio    (door can't peek at pillow)
  light      = 1 − (1 − FACTOR_FLOOR) × window_ratio    (window not blocked)
  efficiency = 1 − waste_ratio                          (room walkable, no floor)

  FACTOR_FLOOR = 0.3   (soft factors range [0.3, 1] linearly; efficiency full [0, 1])
```

Four factors, zero weights, zero τ, no transcendentals. Each ratio is a
fraction of its relevant resource in [0, 1], so the corresponding
`(1 − ratio)` is automatically a discount in [0, 1] with **one physical
meaning per factor**:

- pillow fully exposed (`pillow_ratio = 1`) → privacy = 0.3 (floor)
- pillow at cone edge (`pillow_ratio = 0`) → privacy = 1.0 (full)
- half the window blocked (`window_ratio = 0.5`) → light = 0.65
- 30 % dead space (`waste_ratio = 0.3`) → efficiency = 0.70
- privacy and light always retain ≥ 0.3 (smooth floor), efficiency can hit 0

Two key properties:

1. **Scale-invariant**: ratios cancel cell counts, so the same ratio means
   the same effect across all room sizes — no per-room tuning.
2. **No DONE-trap**: R = 0 iff Availability = 0 (DONE-immediate) or some
   ratio hits 1. Any sensible placement → R > 0, so "do nothing" is never
   the locally safe choice.
3. **Bed-required semantic gate**: even if all four factors are positive,
   `R` is overridden to 0 if no bed is in the final layout. Combined with
   the DONE-mask constraint, this closes the "skip bed for stability"
   local optimum that emerged in earlier training runs.
4. **Asymmetric ranges for stability + signal**: `privacy` and `light` are
   linearly remapped to [0.3, 1.0] — every ratio change produces a smooth
   factor change (no flat zone), and a severe miss only halves the reward
   instead of wiping it. `efficiency` keeps the full [0, 1] range — agent
   has the strongest incentive to eliminate dead space (e.g. push
   furniture against walls).
5. **Hard/soft separation**: validity is in the *action mask* (mandatory
   bed, no overlap, zone clear, no out-of-bounds zone). Optimization is
   in the *reward factors* (privacy, light, efficiency, nightstand pair
   bonus). Every legal placement gets full base value — no cliffs hidden
   inside availability.

Defaults in `env.py`:

```python
CELL_REWARD = 0.05            # value per cell of furniture occupancy
NIGHTSTAND_PAIR_BONUS = 1.0   # absolute bonus when paired with headboard
# No per-category factor, no weights, no τ — single knob total.
```

---

## 💰 Reward Detail

### ✅ Availability = Σ usable furniture value

`value = (area in cells) × CELL_REWARD` (linear, single knob; no per-category
factor). Big furniture (bed = 168 cells) outweighs small (nightstand = 12
cells) naturally.

**Hard validity is enforced by the action mask, not by reward**: a placement
is in the mask iff (a) it doesn't overlap walls/other furniture/door swing,
(b) it doesn't fall in any other non-bed piece's functional zone, and (c)
its own functional zone is clear (bed uses `partial=True` semantics — any
one accessible zone cell suffices; other furniture requires the full zone
clear). This means every *legal* placement gets full base value — there is
no "placed but invalid" cliff at reward time.

**Nightstand pairing**: continuous distance-decay bonus.

```
bonus = NIGHTSTAND_PAIR_BONUS × max(0, 1 − d / D_PAIR)
```

where `d` is the Manhattan distance from the nightstand centre to the
nearest pillow (headboard) cell, `D_PAIR = 4 cells`. Adjacent = full bonus
of `+1.0`; far away = base value only. Smooth gradient, no binary cliff.

Functional zones:

| Furniture | Faces | Depth | Notes |
|-----------|-------|-------|-------|
| 🛏️ Bed | 3 (two long sides + foot) | 3 cells | ≥ 1 long side accessible, nightstand allowed |
| 🖥️ Desk | 1 front | 5 cells | Fully clear (includes chair space) |
| 👔 Wardrobe | 1 front | 4 cells | Fully clear |
| 🗄️ Cabinet | 1 front | 3 cells | Fully clear |
| 🛋️ Nightstand | 1 front | 3 cells | Fully clear |

**Nightstand pairing**: on bed's long side + aligned with headboard → **+1 bonus** (absolute), unpaired → ×0.5 multiplier.

### 👀 Privacy  →  privacy = 1 − 0.7 × pillow_ratio  (∈ [0.3, 1])

```
angle_dev   = |angle from door's facing direction to pillow centroid|
pillow_ratio = max(0, 1 − angle_dev / (π/4))
```

Treat the pillow as one point (its centroid). Measure how far that point is
from the door's facing direction. Inside the door's ±45° cone, exposure
falls off linearly with angle (0° → full exposure, 45° → no exposure).
Outside the cone, no exposure.

One angle → one ratio. No per-cell iteration, no Bresenham line-of-sight,
no distance term. Matches the simplicity of `efficiency = unreachable /
total_empty` so all three soft factors fit one slide:

| Factor | Formula |
|--------|---------|
| Privacy | `1 − angle_to_pillow / (π/4)` (clamped to cone) |
| Light | `blocked_window_cells / window_strip_cells` |
| Efficiency | `unreachable / total_empty` |

Wardrobes affect reward only through **light** and **efficiency** in this
simplified design. The "wardrobe-as-privacy-shield" emergent strategy is
sacrificed for one-line interpretability.

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

| Category | Variant | Grid w×h | Value (area × 0.05) | Limit |
|----------|---------|----------|---------------------|-------|
| 🛏️ Bed | 0.9 / 1.2 / 1.5 / 1.8 | 14×6 / 14×8 / 14×10 / 14×12 | 4.2 / 5.6 / 7.0 / **8.4** | pick 1 |
| 🖥️ Desk | S / L / XL | 6×4 / 8×4 / 12×4 | 1.2 / 1.6 / 2.4 | pick 1 (includes chair 4×4) |
| 👔 Wardrobe | S / M / L / XL / XXL | 6-14 × 4 | 1.2 / 1.6 / 2.0 / 2.4 / 2.8 | pick 1 |
| 🗄️ Cabinet | S / M / L / XL | 4-10 × 3 | 0.6 / 0.9 / 1.2 / 1.5 | pick 1 |
| 🛋️ Nightstand | A / B | 4×3 / 3×3 | 0.6 / 0.45 (+1 paired) | up to 2 |

Value = area × 0.05, linear in occupancy. Bed dominates by design — Bed 1.8
alone (8.4) is ≈50 % of the catalog's max sum, so the agent can't reach high
reward without learning to place the bed well.

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

### Privacy (angular)

```python
# Pillow centroid = mean of pillow cells (depends on bed orientation)
# angle_dev = |angle from door's facing direction to pillow centroid|
# pillow_ratio = max(0, 1 − angle_dev / (π/4))
#   = 1 when pillow dead-centre of door's ±45° cone
#   = 0 when pillow at or beyond cone edge
# privacy = 1 − (1 − FACTOR_FLOOR) × pillow_ratio
```

Earlier versions used per-cell ray tracing + wardrobe Bresenham blocking;
simplified to single-angle for PPT clarity (see Reward Detail).

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
        # Availability: Σ (area × CELL_REWARD) + nightstand distance-decay bonus
        # privacy    = 1 − (1 − FACTOR_FLOOR) × pillow_ratio    (∈ [0.3, 1])
        # light      = 1 − (1 − FACTOR_FLOOR) × window_ratio    (∈ [0.3, 1])
        # efficiency = 1 − waste_ratio                           (∈ [0, 1])
        # Return A × privacy × light × efficiency, gated to 0 if no bed
    
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

- 🛏️ Will the agent skip bed in tiny rooms where pillow can't be hidden? Worst case `pillow_ratio = 1` → privacy = 0.3 (floor), so reward is reduced but not zero. Bed-first mask + reward gate guarantee a bed is always placed
- ⏹️ Early stopping after 1-2 items? v3 eliminates the DONE-trap (R = 0 only when A = 0 or some `(1 − ratio) = 0`), but agent could still settle for partial play if the marginal availability of the next item < the marginal discount it incurs. Audit shows greedy = +0.35, optimal likely +5 to +10
- 🏗️ Bed accessibility: currently "partial OK" (≥1 cell of long side in swept); may want ≥50% for stricter realism
