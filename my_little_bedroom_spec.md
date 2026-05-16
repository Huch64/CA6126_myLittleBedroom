# 🏠 My Little Bedroom — Final Spec

> CA6126 RL Final Project · MaskablePPO agent learns to furnish a randomized bedroom
> 
> `R = Availability − Discomfort − Waste`

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

### Reward *R*

Computed once at episode end:

```
R = Availability − Discomfort − Waste
```

---

## 💰 Reward Detail

### ✅ Availability = Σ usable furniture value

Each piece scores if ALL conditions met:

| Condition | Rule |
|-----------|------|
| No duplicate | 2nd of same category → both 0 (nightstand allows 2) |
| Zone clear | Functional zone cells empty + in bounds |
| Reachable | Furniture neighbor in flood fill swept set |
| Bed special | ≥1 long side with cells in swept (not just any 1 cell) |

Functional zones:

| Furniture | Faces | Depth | Notes |
|-----------|-------|-------|-------|
| 🛏️ Bed | 3 (two long sides + foot) | 3 cells | ≥1 long side accessible, nightstand allowed |
| 🖥️ Desk | 1 front | 5 cells | Fully clear (includes chair space) |
| 👔 Wardrobe | 1 front | 4 cells | Fully clear |
| 🗄️ Cabinet | 1 front | 4 cells | Fully clear |
| 🛋️ Nightstand | 1 front | 3 cells | Fully clear |

**Nightstand pairing**: on bed's long side + aligned with headboard → +1 bonus, unpaired → ×0.5

### 😰 Discomfort = exposure + pillow + window

| Component | Formula | Range |
|-----------|---------|-------|
| Bed exposure | 90° cone from door center, Bresenham per bed cell | 0-5 |
| Pillow visible | Any pillow cell in cone + unblocked → +4 | 0 or 4 |
| Window blocked | Bed/wardrobe directly in front of window (width only, 2 cells deep) → +3 | 0 or 3 |

Only **wardrobe** blocks line of sight (other furniture too short).

### 🗑️ Waste = unreachable cells × 0.2

3×3 brush flood fill from door → passable centers → expand ±1 → swept set.
Empty cells not in swept = dead space. No furniture placed → waste = 0.

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
| `train.py` | 🚂 MaskablePPO training + CSV/TB logging + best-model save | ✅ Done |
| `render.py` | 🎬 Record agent playing to mp4 (random or trained) | ✅ Done |
| `plot_training.py` | 📈 Generate report figures from `runs/<name>/` logs | ⬜ TODO |
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
        # Availability: check each placed piece
        # Discomfort: vision cone + window check
        # Waste: flood fill
        # Return A - D - W
    
    def action_masks(self):
        # For each (fid, x, y, ori): check bounds + no overlap + not in swing
        # Return bool array [41185]
    
    def flood_fill(self):
        # 3×3 brush passability → BFS from door → expand → swept
    
    def render(self):
        # Return RGB image of current room state
```

### Reward calc mirrors HTML `calc()` function exactly:
- flood() → flood fill
- zoneRects() → functional zones
- Bresenham → vision cone
- Nightstand pairing → long side + headboard alignment

---

## ❓ Open Questions

- 🛏️ Will agent skip bed in tiny rooms? Highest value (3-4) should prevent, but watch training
- ⏹️ Early stopping after 1-2 items? May need completion bonus if observed
- ⚖️ Value weights are heuristic, tune after training
- 📏 Waste ×0.2 — adjust if needed
- 🏗️ Bed accessibility: currently "partial OK", may want ≥50% of one long side in swept
