# 🏠 My Little Bedroom — Final Spec

> CA6126 RL Final Project · MaskablePPO agent learns to furnish a randomized bedroom
>
> `R = Availability × privacy × light × efficiency + diversity`
> &nbsp;&nbsp;`privacy = 1 − 0.7 × exposure_ratio`   (linear remap to [0.3, 1])
> · `light = 1 − 0.7 × window_ratio`             (linear remap to [0.3, 1])
> · `efficiency = 1 − waste_ratio`                (full [0, 1], no floor)
> · `diversity = +1 per distinct category`        (max +5, added outside product)
>
> Privacy and light use a linear remap to **[0.3, 1.0]** so every ratio
> change produces a smooth factor change (no flat zone), but extreme
> misses don't wipe out the rest of the reward. Efficiency uses the
> **full [0, 1]** range — agent has the strongest incentive to use space
> well (e.g. push furniture against walls instead of leaving dead corners).
> Diversity is **added outside** the multiplicative core so the bonus for
> placing all 5 furniture categories is never wiped out by a single
> bad-quality placement (e.g. a fully-exposed bed).
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

Fixed 26×22 multi-channel grid (0.15m/cell) flattened and concatenated
with a 5-dim category-placed binary vector, total observation size
`3·22·26 + 5 = 1721`.

| Channel | Content |
|---------|---------|
| ch0 | Occupancy: 0=empty, 1=wall, 2-19=furniture ID |
| ch1 | Door position |
| ch2 | Window position |
| cats[5] | Binary flags (bed, desk, wardrobe, cabinet, nightstand placed) |

The 5 category-placed flags are an explicit signal for the diversity head;
without them, a flat MLP would have to scan the grid for furniture IDs
to count distinct categories, which is hard under the factored output
architecture.

Per-episode randomization:

| Element | Range |
|---------|-------|
| 🏠 Room | 16-26 × 18-22 cells, outside = wall |
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
5. **Own zone validity**: bed has 2 fixed **1.2m × 0.45m (8×3 cells)**
   foot-anchored zones on each long side; at least one must be in-bounds
   AND fully clear at placement time. Other furniture requires its single
   functional zone fully clear + in-bounds. Nightstand has a bespoke
   mask (2 slots on bed's long sides at the head end, `ns_ori = bed.ori^1`).
6. **Bed corridor invariant**: when only 1 bed zone remains clear of
   non-bed furniture, that zone auto-locks (no new placement may enter
   it). Backed by a reward-time safety net that zeroes the bed's
   availability if both corridors get blocked.
7. **DONE gate**: blocked until ≥ 1 bed has been placed (escape valve if no
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
| `room_w` | 16–26 (integer) | 11 |
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
R = Availability × privacy × light × efficiency + diversity

  privacy    = 1 − (1 − FACTOR_FLOOR) × exposure_ratio    (door can't peek at pillow)
  light      = 1 − (1 − FACTOR_FLOOR) × window_ratio    (window not blocked)
  efficiency = 1 − waste_ratio                          (room walkable, no floor)
  diversity  = DIVERSITY_BONUS_PER_CAT × n_distinct_categories  (max +5)

  FACTOR_FLOOR = 0.3              (soft factors [0.3, 1] linearly; efficiency [0, 1])
  DIVERSITY_BONUS_PER_CAT = 1.0   (added OUTSIDE the product so the bonus isn't
                                   multiplied away by a single bad placement)
```

Three multiplicative discount factors + one additive bonus. Each ratio is
a fraction of its relevant resource in [0, 1], so the corresponding
`(1 − ratio)` is automatically a discount in [0, 1] with **one physical
meaning per factor**:

- pillow fully exposed (`exposure_ratio = 1`) → privacy = 0.3 (floor)
- pillow at cone edge (`exposure_ratio = 0`) → privacy = 1.0 (full)
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
   bed, no overlap, zone clear, no out-of-bounds zone, **nightstand
   restricted to the 2 natural slots at bed headboard with ns_ori =
   bed.ori**). Optimization is in the *reward factors* (privacy, light,
   efficiency, diversity). Every legal placement gets full base value —
   no cliffs hidden inside availability.

Defaults in `env.py`:

```python
CELL_REWARD = 0.05                  # value per cell of furniture occupancy
DIVERSITY_BONUS_PER_CAT = 1.0       # +1 per distinct category (max +5)
FACTOR_FLOOR = 0.3                  # soft-factor floor for privacy / light
# No per-category factor, no weights, no τ — single knob per term.
```

---

## 💰 Reward Detail

### ✅ Availability = Σ usable furniture value

`value = (area in cells) × CELL_REWARD` (linear, single knob; no per-category
factor). Big furniture (bed = 168 cells) outweighs small (nightstand = 12
cells) naturally.

**Hard validity is enforced by the action mask, not by reward**: a placement
is in the mask iff (a) it doesn't overlap walls/other furniture/door swing,
(b) it doesn't fall in any **non-bed** piece's functional zone (bed zones
are LOOSE by default), and (c) its own functional zone is clear. Bed has
exactly **2 fixed zones**, each `1.2m × 0.45m` (8×3 cells), foot-anchored
on each long side. At least one zone must be in-bounds AND fully clear at
placement time. Other furniture requires the full zone clear; nightstand
allows bed cells in its zone but blocks on any other furniture.

**Bed corridor lifetime invariant**: even though bed zones are loose by
default (other furniture may sit inside one of the 2 corridors), the mask
**auto-locks the last clear zone** whenever only one remains — no further
placement can break it. As a safety net, `_reward()` also re-checks at
episode end: if both bed zones are blocked, the bed's availability is
zeroed (catches any edge case the mask might miss). Net result: every
*legal* placement gets full base value, and the bed always retains at
least one usable access corridor.

**Nightstand placement is mask-enforced** (no reward bonus needed): the
mask exposes exactly **2 candidate (x, y) slots per ns size**, both on
the bed's **two long sides at the head end** (above and below for
horizontal beds, left and right for vertical beds). The NS is aligned
with the pillow column/row so it sits right beside the headboard.
`ns_ori = bed.ori ^ 1` — the nightstand's front/drawer faces **along**
the bed's long axis from the pillow side toward the foot side (matching
how a person in bed reaches over and opens it). NS own-zone (3 cells
deep, drawer-side) must be clear of any non-bed furniture — so a
wardrobe blocking the drawer area prevents the nightstand.

Functional zones:

| Furniture | Faces | Depth | Notes |
|-----------|-------|-------|-------|
| 🛏️ Bed | 2 (long sides, foot-anchored 8×3 each) | 3 cells | At least one zone in-bounds + fully clear at placement; lifetime invariant locks the last clear zone |
| 🖥️ Desk | 1 front | 5 cells | Fully clear (includes chair space) |
| 👔 Wardrobe | 1 front | 4 cells | Fully clear |
| 🗄️ Cabinet | 1 front | 3 cells | Fully clear |
| 🛋️ Nightstand | n/a (mask-restricted) | — | accessed via bed; zone check bypassed |

### 🎨 Diversity  →  diversity = +1 × n_distinct_categories  (max +5)

Flat bonus added **outside** the multiplicative product so the bonus for
placing all 5 furniture categories (bed / desk / wardrobe / cabinet /
nightstand) is **never multiplied away** by a single poor-quality
placement. Counter-balances the bed-dominates-area bias of pure
availability — a 5-category layout always beats a bed-only layout, all
else equal.

### 👀 Privacy  →  privacy = 1 − 0.7 × exposure_ratio  (∈ [0.3, 1])

```
For each bed cell c:
    w = 10.0 if c is pillow else 1.0
    total_weight += w
    if c in door cone AND Bresenham line from door isn't blocked by wardrobe:
        exposed_weight += w
exposure_ratio = exposed_weight / total_weight
```

Each bed cell contributes to exposure with **weight 2 for pillow cells, 1
for body cells**. A cell counts as exposed iff (a) it's in the door's
±45° cone AND (b) the Bresenham line of sight from the door isn't blocked
by a wardrobe.

This reflects "how much of the bed (pillow weighted 10x) is actually
visible from the door". Wardrobe-as-privacy-shield is a meaningful
emergent strategy — agent learns to place wardrobe in the door cone
to break the line of sight to the bed.

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
| **v3** (current — env.py + HTML) | A × privacy × light × efficiency + diversity | Each factor `1 − ratio` is an independent discount in [0, 1] with one physical meaning; diversity is an outside-the-product `+1 per category` bonus (max +5) so full-set layouts always beat bed-only; eliminates DONE-trap; scale-invariant across room sizes |

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
| 🛋️ Nightstand | A / B | 4×3 / 3×3 | 0.6 / 0.45 | up to 2 (mask-restricted to bed headboard slot) |

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

### Privacy (per-cell weighted exposure)

```python
# For each bed cell:
#   w = 10.0 if pillow cell else 1.0
#   total_weight += w
#   if cell in door's ±45° cone AND
#      Bresenham line from door to cell isn't blocked by a wardrobe:
#       exposed_weight += w
# exposure_ratio = exposed_weight / total_weight
# privacy = 1 − (1 − FACTOR_FLOOR) × exposure_ratio
```

Wardrobe-as-privacy-shield is a meaningful emergent strategy — the agent
can park a wardrobe in the door cone to break line of sight to the bed.

### Nightstand Slot (mask-restricted)

```python
# 2 slots on bed's LONG SIDES at the head end. ns_ori = bed.ori ^ 1
# (drawer faces along bed's long axis from pillow toward foot):
#   bed.ori=0 (horiz, pillow LEFT):
#     above bed: (bed.x, bed.y − ns.fh)        below bed: (bed.x, bed.y + bed.fh)
#   bed.ori=2 (horiz, pillow RIGHT):
#     above bed: (bed.x + bed.fw − ns.fw, bed.y − ns.fh)
#     below bed: (bed.x + bed.fw − ns.fw, bed.y + bed.fh)
#   bed.ori=1 (vert,  pillow TOP):
#     left of bed: (bed.x − ns.fw, bed.y)      right of bed: (bed.x + bed.fw, bed.y)
#   bed.ori=3 (vert,  pillow BOTTOM):
#     left of bed: (bed.x − ns.fw, bed.y + bed.fh − ns.fh)
#     right of bed: (bed.x + bed.fw, bed.y + bed.fh − ns.fh)
# NS own-zone must be clear of any non-bed furniture (bed cells allowed
# but unlikely to appear in NS zone with this layout).
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
    Observation: Box(0, 19, shape=(3*22*26 + 5,), dtype=int8)
    Action: Discrete(41185) with action_masks()
    
    Grid constants: GW=26, GH=22, CELL=0.15m
    Room: random 16-26 × 18-22 per reset
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
        # Availability: Σ (area × CELL_REWARD)
        # diversity  : +1 per distinct category placed (max +5)
        # privacy    = 1 − (1 − FACTOR_FLOOR) × exposure_ratio    (∈ [0.3, 1])
        # light      = 1 − (1 − FACTOR_FLOOR) × window_ratio      (∈ [0.3, 1])
        # efficiency = 1 − waste_ratio                            (∈ [0, 1])
        # Return Availability × privacy × light × efficiency + diversity,
        # gated to 0 if no bed.
    
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

- 🛏️ Will the agent skip bed in tiny rooms where pillow can't be hidden? Worst case `exposure_ratio = 1` → privacy = 0.3 (floor), so reward is reduced but not zero. Bed-first mask + reward gate guarantee a bed is always placed
- ⏹️ Early stopping after 1-2 items? v3 eliminates the DONE-trap (R = 0 only when A = 0 or some `(1 − ratio) = 0`), but agent could still settle for partial play if the marginal availability of the next item < the marginal discount it incurs. Audit shows greedy = +0.35, optimal likely +5 to +10
- 🏗️ Bed accessibility: currently "partial OK" (≥1 cell of long side in swept); may want ≥50% for stricter realism
