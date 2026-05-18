"""
my_little_bedroom — CA6126 RL Final Project.

Gymnasium environment for the "My Little Bedroom" furniture-placement task.
Mirrors the reward in `my_little_bedroom.html` so rewards can be
sanity-checked side-by-side with the interactive preview.

MDP recap (see my_little_bedroom_spec.md):
  - State:  multi-channel 26x22 grid (occupancy / door / window).
  - Action: Discrete(41185) = (fid, x, y, ori) flattened + DONE.
            DONE is mask-blocked until a bed has been placed.
            Step 0 is mask-restricted to bed actions only (bed-first).
  - Reward: 0 every step except final, where
        R = Availability × privacy × light × efficiency
    with each factor a (1 − ratio) discount in [0, 1].
    Semantic gate: if no bed at episode end, R = 0
    (a bedroom isn't a bedroom without a bed).

Public API:
  env = MyLittleBedroom(seed=0)
  obs, info = env.reset()
  obs, r, terminated, truncated, info = env.step(action)
  mask = env.action_masks()              # for sb3-contrib MaskablePPO
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ------------------------------------------------------------------ constants

GRID_M = 0.15
GW, GH = 26, 22
DW = 6
ROOM_W_RANGE = (16, 26)
ROOM_H_RANGE = (18, 22)
WINDOW_WALLS = ("top", "left", "right")
N_ORI = 4

# ── reward v3 (multiplicative, ratio-based, scale-invariant) ──────
#
# R = Availability × privacy × light × efficiency + diversity + compactness
#
#   • Availability  = Σ (area_cells × CELL_REWARD)
#                       — linear in area, single knob, no per-category factor.
#                       Big furniture (bed) dominates naturally because it
#                       occupies the most space.
#   • diversity     = n_distinct_categories_placed² / 5   ∈ {0.2, 0.8, 1.8, 3.2, 5.0}
#                       — quadratic in distinct-category count: 1 cat→0.2,
#                       2→0.8, 3→1.8, 4→3.2, 5→5.0. Max +5 when all 5
#                       furniture types (bed/desk/wardrobe/cabinet/
#                       nightstand) are placed. The 5-category jump is the
#                       biggest reward (5.0 − 3.2 = +1.8) so agent strongly
#                       prefers the full set. Added OUTSIDE the product so
#                       the bonus is never wiped out by a poor-quality
#                       placement. Counter-balances the bed-dominates-area
#                       bias of pure availability.
#   • compactness   = 5 × (1 − (perimeter/√area − 4) / 8)  ∈ [0, 5]
#                       — shape coefficient of the remaining empty space.
#                       Door swing is transparent (ignored — fixed at reset
#                       and not controllable by agent, no point baking it
#                       into the signal). Rewards layouts where furniture
#                       clusters and the leftover space is a clean
#                       rectangle. Penalizes scattered furniture, 1-cell
#                       narrow strips, dead-end fingers, etc. Empty room
#                       gives shape_coef ≈ 4 → compactness = 5.
#   • privacy       = 1 − (1 − FACTOR_FLOOR) × exposure_ratio
#                         exposure_ratio = angle_dev_to_pillow / (π/4)
#                         linear remap to [FACTOR_FLOOR, 1] — no flat zone
#   • light         = 1 − (1 − FACTOR_FLOOR) × window_ratio
#                         window_ratio = blocked window cells / window strip
#                         linear remap to [FACTOR_FLOOR, 1] — no flat zone
#   • efficiency    = 1 − waste_ratio                         (full [0, 1])
#                         waste_ratio  = unreachable / empty cells
#                         NO floor — strongest signal for wall-hugging
#
# Properties:
#   • Scale-invariant across room sizes (ratios cancel cell count).
#   • privacy / light always have gradient (smooth linear in their range).
#   • efficiency can range fully [0, 1] — agent strongly incentivized to
#     eliminate dead space (push furniture against walls).
#   • Bed-required semantic gate: if no bed placed at episode end,
#     R is overridden to 0 in _reward().

CELL_REWARD = 0.05                  # reward per cell of furniture occupancy
# Diversity bonus: quadratic — n² / 5 → {0.2, 0.8, 1.8, 3.2, 5.0} for n=1..5.
# Constant kept for back-compat but no longer used directly in formula.
FACTOR_FLOOR = 0.3                  # soft-factor minimum (privacy / light range
                                    # linearly from FACTOR_FLOOR to 1.0 — no
                                    # flat zone, gradient always alive).
                                    # Efficiency has no floor (full [0, 1]).
                                    # privacy/light/efficiency ∈ [FACTOR_FLOOR, 1].
                                    # Keeps every factor's gradient alive even
                                    # in worst-case placements — bed/wardrobe/etc.
                                    # always retain ≥ 20 % of their functional
                                    # value, matching the intuition "a partially
                                    # exposed bed is still a bed, a blocked
                                    # window is still a window".

# Back-compat shims (kept so old code that imports these doesn't break).
WASTE_FACTOR = 1.0
WCOEFF = WASTE_FACTOR * CELL_REWARD


@dataclass(frozen=True)
class FurnSpec:
    name: str
    cat: str
    w: int
    h: int
    v: float
    zd: int
    pl: int = 1
    z3: bool = False


CATALOG: list[FurnSpec] = [
    FurnSpec("Bed 0.9",      "bed",        14,  6, 3.0, 3, pl=1, z3=True),
    FurnSpec("Bed 1.2",      "bed",        14,  8, 3.5, 3, pl=1, z3=True),
    FurnSpec("Bed 1.5",      "bed",        14, 10, 4.0, 3, pl=2, z3=True),
    FurnSpec("Bed 1.8",      "bed",        14, 12, 4.0, 3, pl=2, z3=True),
    FurnSpec("Desk S",       "desk",        6,  4, 2.5, 5),
    FurnSpec("Desk L",       "desk",        8,  4, 3.0, 5),
    FurnSpec("Desk XL",      "desk",       12,  4, 3.0, 5),
    FurnSpec("Wardrobe S",   "wardrobe",    6,  4, 1.5, 4),
    FurnSpec("Wardrobe M",   "wardrobe",    8,  4, 2.0, 4),
    FurnSpec("Wardrobe L",   "wardrobe",   10,  4, 2.0, 4),
    FurnSpec("Wardrobe XL",  "wardrobe",   12,  4, 2.5, 4),
    FurnSpec("Wardrobe XXL", "wardrobe",   14,  4, 2.5, 4),
    FurnSpec("Cabinet S",    "cabinet",     4,  3, 1.0, 3),
    FurnSpec("Cabinet M",    "cabinet",     6,  3, 1.5, 3),
    FurnSpec("Cabinet L",    "cabinet",     8,  3, 1.5, 3),
    FurnSpec("Cabinet XL",   "cabinet",    10,  3, 2.0, 3),
    FurnSpec("Nightstand A", "nightstand",  4,  3, 1.0, 3),
    FurnSpec("Nightstand B", "nightstand",  3,  3, 1.0, 3),
]
N_FURN = len(CATALOG)
ACTION_DONE = N_FURN * GW * GH * N_ORI
N_ACTIONS = ACTION_DONE + 1

MAX_PER_CAT = {"bed": 1, "desk": 1, "wardrobe": 1, "cabinet": 1, "nightstand": 2}

CAT_COLORS = {
    "bed":        (69, 212, 104),
    "desk":       (48, 212, 160),
    "wardrobe":   (48, 168, 212),
    "cabinet":    (80, 80, 212),
    "nightstand": (149, 48, 212),
}


@dataclass
class Placement:
    fid: int
    x: int
    y: int
    fw: int
    fh: int
    ori: int


# ------------------------------------------------------------------ utilities

def get_footprint(spec: FurnSpec, ori: int) -> tuple[int, int]:
    return (spec.h, spec.w) if ori in (1, 3) else (spec.w, spec.h)


def _value(spec: FurnSpec) -> float:
    """Availability per piece: area × CELL_REWARD (linear, single knob).

    Big furniture (bed = 168 cells) naturally outweighs small (nightstand =
    12 cells) without per-category fudge factors — a bedroom without a bed
    isn't a bedroom, and the math reflects that.
    """
    return spec.w * spec.h * CELL_REWARD


def encode_action(fid: int, x: int, y: int, ori: int) -> int:
    return fid * GW * GH * N_ORI + x * GH * N_ORI + y * N_ORI + ori


def decode_action(a: int) -> tuple[int, int, int, int]:
    fid = a // (GW * GH * N_ORI)
    rem = a % (GW * GH * N_ORI)
    x = rem // (GH * N_ORI)
    rem = rem % (GH * N_ORI)
    y = rem // N_ORI
    ori = rem % N_ORI
    return fid, x, y, ori


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# ------------------------------------------------------------------ env

class MyLittleBedroom(gym.Env):
    """Gymnasium environment — see module docstring."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        max_steps: int = 8,
        strict_mask: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.strict_mask = strict_mask
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_ACTIONS)
        # Observation = flat[(3 occupancy/door/window channels) ++ 5 cat-placed flags]
        # The 5 extra binary features give the diversity head a direct
        # readable signal of "which categories have already been placed",
        # which the MLP would otherwise have to infer by scanning the grid
        # for furniture IDs (slow under flat MLP, see network audit).
        self.observation_space = spaces.Box(
            low=0, high=19, shape=(3 * GH * GW + 5,), dtype=np.int8
        )

        self._rng = np.random.default_rng(seed)

        self.room_w: int = 0
        self.room_h: int = 0
        self.door_pos: int = 0
        self.win_wall: str = "top"
        self.win_pos: int = 0
        self.win_w: int = 0
        self.grid: np.ndarray = np.zeros((GH, GW), dtype=np.int8)
        self.placed: list[Placement] = []
        self.swing: set[tuple[int, int]] = set()
        self.steps: int = 0

    # --- gym API ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset the env. Pass ``options={"config": {...}}`` to pin the room
        for deterministic verification — keys: ``room_w``, ``room_h``,
        ``door_pos``, ``win_wall``. Window pos/width stay derived from the
        wall length to match the HTML preview exactly.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = (options or {}).get("config")
        if cfg is not None:
            self.room_w = int(cfg["room_w"])
            self.room_h = int(cfg["room_h"])
            self.door_pos = int(cfg["door_pos"])
            self.win_wall = cfg["win_wall"]
        else:
            self.room_w = int(self._rng.integers(ROOM_W_RANGE[0], ROOM_W_RANGE[1] + 1))
            self.room_h = int(self._rng.integers(ROOM_H_RANGE[0], ROOM_H_RANGE[1] + 1))
            self.door_pos = int(self._rng.integers(0, self.room_w - DW + 1))
            self.win_wall = WINDOW_WALLS[int(self._rng.integers(0, len(WINDOW_WALLS)))]
        wall_len = self.room_w if self.win_wall == "top" else self.room_h
        self.win_w = _round_half_up(wall_len * 0.5 / 2) * 2
        self.win_pos = (wall_len - self.win_w) // 2

        self.grid = np.zeros((GH, GW), dtype=np.int8)
        self.placed = []
        self.swing = _make_swing(self.door_pos, self.room_w, self.room_h)
        self.steps = 0

        return self._observation(), self._info()

    def step(self, action: int):
        assert self.action_space.contains(action), action
        terminated = False
        truncated = False
        reward = 0.0

        if action == ACTION_DONE:
            terminated = True
        else:
            fid, x, y, ori = decode_action(action)
            if self._can_place(fid, x, y, ori):
                fw, fh = get_footprint(CATALOG[fid], ori)
                self.placed.append(Placement(fid, x, y, fw, fh, ori))
                self.grid[y:y + fh, x:x + fw] = np.int8(fid + 2)
            # Invalid action: no-op (MaskablePPO should never sample these).

        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True
        if terminated or truncated:
            reward += self._reward()

        info = self._info()
        if terminated or truncated:
            # Episode-end payload for training callbacks (drop big fields like
            # swept/exposed sets so the dict stays cheap to pickle through
            # SubprocVecEnv pipes).
            info["breakdown"] = {
                k: v for k, v in self._last_breakdown.items()
                if k not in ("swept", "exposed", "door_center")
            }
            info["config"] = {
                "room_w": self.room_w, "room_h": self.room_h,
                "door_pos": self.door_pos,
                "win_wall": self.win_wall,
                "win_pos": self.win_pos, "win_w": self.win_w,
            }
            info["cats_placed"] = [CATALOG[p.fid].cat for p in self.placed]
            info["placements"] = [(p.fid, p.x, p.y, p.ori, p.fw, p.fh)
                                  for p in self.placed]

        return self._observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Vectorized mask of valid actions.

        Now enforces TWO levels of validity:

        1. Geometric: footprint can't overlap walls / occupied cells / door
           swing / OR functional zones of already-placed NON-BED pieces.
           Bed zones are intentionally LOOSE — other furniture may occupy
           bed's access corridor (kept playable in tight rooms; the bed's
           placement-time corridor check is a one-shot guarantee, not a
           lifetime invariant).

        2. Functional (own zone): the candidate's own functional zone must
           currently be clear and in-bounds. Beds have 2 fixed 8×3 (or 3×8)
           foot-anchored zones on each long side (1.2m × 0.45m corridor);
           at least one must be in-bounds AND fully clear at placement
           time. Other pieces require the full zone clear. Nightstand has
           a bespoke check (handled separately below).

        Semantic gate: DONE allowed only after a bed has been placed.
        Escape valve: if no placement is valid either, DONE is unblocked so
        the env always has at least one valid action.
        """
        mask = np.zeros(N_ACTIONS, dtype=bool)
        rw, rh = self.room_w, self.room_h

        # ── Map A: base blocked (walls outside room + occupied + door swing) ──
        base_blocked = np.zeros((GH, GW), dtype=bool)
        base_blocked[:, :] = True
        base_blocked[:rh, :rw] = self.grid[:rh, :rw] > 0
        if self.swing:
            sx, sy = zip(*self.swing)
            base_blocked[np.asarray(sy), np.asarray(sx)] = True

        # ── Map B: non-bed pieces' functional zones (new placements can't enter) ──
        # Bed zones are LOOSE by default — other furniture may sit inside
        # one of bed's 2 access corridors, but the LIFETIME INVARIANT kicks
        # in when only one bed zone remains clear: that last zone is
        # auto-locked (added to other_zone_blocked) so no new placement can
        # break it. Combined with reward-time validation (see _reward), bed
        # always retains ≥ 1 corridor or its availability is zeroed.
        other_zone_blocked = np.zeros((GH, GW), dtype=bool)
        for p in self.placed:
            if CATALOG[p.fid].cat == "bed":
                continue   # bed zones loose; handled by invariant below
            for zx, zy, zw, zh in _zone_rects(p):
                x0, x1 = max(0, zx), min(GW, zx + zw)
                y0, y1 = max(0, zy), min(GH, zy + zh)
                if x0 < x1 and y0 < y1:
                    other_zone_blocked[y0:y1, x0:x1] = True

        # ── Bed corridor invariant: lock the last clear bed zone ──
        beds_placed = [p for p in self.placed if CATALOG[p.fid].cat == "bed"]
        if beds_placed:
            clear_bed_zones = _bed_clear_zones(beds_placed[0], self.grid, self.placed, rw, rh)
            if len(clear_bed_zones) == 1:
                zx, zy, zw, zh = clear_bed_zones[0]
                x0, x1 = max(0, zx), min(GW, zx + zw)
                y0, y1 = max(0, zy), min(GH, zy + zh)
                if x0 < x1 and y0 < y1:
                    other_zone_blocked[y0:y1, x0:x1] = True

        # Combined: any new footprint must not overlap either.
        footprint_blocked = base_blocked | other_zone_blocked

        # ── integral images ──
        integ_fp = np.zeros((GH + 1, GW + 1), dtype=np.int32)
        integ_fp[1:, 1:] = footprint_blocked.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
        # Own-zone check only cares about walls/occupied/swing (not other zones).
        integ_base = np.zeros((GH + 1, GW + 1), dtype=np.int32)
        integ_base[1:, 1:] = base_blocked.astype(np.int32).cumsum(axis=0).cumsum(axis=1)

        def _rect_sum(integ, y1, x1, y2, x2):
            return (integ[y2[:, None], x2[None, :]]
                    - integ[y1[:, None], x2[None, :]]
                    - integ[y2[:, None], x1[None, :]]
                    + integ[y1[:, None], x1[None, :]])

        cat_counts: dict[str, int] = {}
        for p in self.placed:
            c = CATALOG[p.fid].cat
            cat_counts[c] = cat_counts.get(c, 0) + 1

        # Bed-first constraint: the very first action must place a bed.
        # Rationale: with zone-in-mask, placing non-bed pieces first often
        # eats up the wall space a bed needs (bed is 14×6 = the widest piece).
        # Forcing bed at step 0 guarantees a bedroom always has a bed and
        # lets other pieces fit around it, instead of failing as a no-bed
        # truncation (R=0).
        restrict_to_bed = (len(self.placed) == 0)

        for fid, spec in enumerate(CATALOG):
            if restrict_to_bed and spec.cat != "bed":
                continue
            if spec.cat == "nightstand":
                # Nightstands are mask-restricted to two natural slots next
                # to the placed bed's headboard (handled below the loop).
                continue
            if self.strict_mask and cat_counts.get(spec.cat, 0) >= MAX_PER_CAT[spec.cat]:
                continue
            for ori in range(N_ORI):
                fw, fh = get_footprint(spec, ori)
                if fw > rw or fh > rh:
                    continue
                xs = np.arange(0, GW - fw + 1)
                ys = np.arange(0, GH - fh + 1)
                if xs.size == 0 or ys.size == 0:
                    continue
                x2 = xs + fw
                y2 = ys + fh

                # ── footprint clear (geometry + others' zones) ──
                sub = _rect_sum(integ_fp, ys, xs, y2, x2)
                valid = (sub == 0)

                # ── own-zone check (per (fid, ori) the offsets are fixed) ──
                hypo = Placement(fid, 0, 0, fw, fh, ori)
                zone_offsets = _zone_rects(hypo)   # at (0,0), so these are (dx, dy, dw, dh)
                if zone_offsets:
                    if spec.z3:
                        # Bed: each zone is exactly 8×3 (or 3×8) = 1.2m × 0.45m
                        # foot-anchored on a long side. At least ONE zone must
                        # be in-bounds AND fully clear of walls / swing.
                        any_zone_clear = np.zeros((ys.size, xs.size), dtype=bool)
                        for zdx, zdy, zdw, zdh in zone_offsets:
                            zx1 = xs + zdx
                            zy1 = ys + zdy
                            zxe = zx1 + zdw
                            zye = zy1 + zdh
                            in_x = (zx1 >= 0) & (zxe <= GW)
                            in_y = (zy1 >= 0) & (zye <= GH)
                            in_bounds = in_y[:, None] & in_x[None, :]
                            zx1c = np.clip(zx1, 0, GW)
                            zxec = np.clip(zxe, 0, GW)
                            zy1c = np.clip(zy1, 0, GH)
                            zyec = np.clip(zye, 0, GH)
                            blocked_sum = _rect_sum(integ_base, zy1c, zx1c, zyec, zxec)
                            any_zone_clear |= in_bounds & (blocked_sum == 0)
                        valid &= any_zone_clear
                    else:
                        # non-bed: partial=False. All zone rects must be in-bounds AND all-clear.
                        zone_ok = np.ones((ys.size, xs.size), dtype=bool)
                        for zdx, zdy, zdw, zdh in zone_offsets:
                            zx1 = xs + zdx
                            zy1 = ys + zdy
                            zxe = zx1 + zdw
                            zye = zy1 + zdh
                            in_x = (zx1 >= 0) & (zxe <= GW)
                            in_y = (zy1 >= 0) & (zye <= GH)
                            in_bounds = in_y[:, None] & in_x[None, :]
                            zx1c = np.clip(zx1, 0, GW)
                            zxec = np.clip(zxe, 0, GW)
                            zy1c = np.clip(zy1, 0, GH)
                            zyec = np.clip(zye, 0, GH)
                            blocked_sum = _rect_sum(integ_base, zy1c, zx1c, zyec, zxec)
                            zone_ok &= in_bounds & (blocked_sum == 0)
                        valid &= zone_ok

                vy, vx = np.where(valid)
                if vy.size == 0:
                    continue
                idxs = (
                    fid * GW * GH * N_ORI
                    + xs[vx] * GH * N_ORI
                    + ys[vy] * N_ORI
                    + ori
                )
                mask[idxs] = True

        # Nightstand: hard-restricted to TWO natural slots adjacent to the
        # placed bed's headboard SHORT side, with ns_ori = bed.ori ^ 1
        # (drawer faces along bed long axis, pillow → foot direction). Both
        # sizes A (4×3) and B (3×3) are offered for each slot, anchored so
        # the NS top/bottom-flushes with bed's long edge.
        #
        # NS validity: (a) NS footprint clear (walls / occupied cells / door
        # swing / other pieces' zones); (b) NS's OWN functional zone must
        # not be blocked by ANY non-bed furniture. The NS zone always
        # overlaps the bed footprint (NS drawer faces into bed), so bed
        # cells inside the NS zone are explicitly allowed — but a wardrobe
        # or cabinet sitting in the NS zone would block placement.
        beds_placed = [p for p in self.placed if CATALOG[p.fid].cat == "bed"]
        if (beds_placed and not restrict_to_bed and self.strict_mask
                and cat_counts.get("nightstand", 0) < MAX_PER_CAT["nightstand"]):
            bed = beds_placed[0]
            for fid in (16, 17):
                spec = CATALOG[fid]
                if spec.cat != "nightstand":
                    continue
                ns_ori = bed.ori ^ 1
                fw, fh = get_footprint(spec, ns_ori)
                if fw > rw or fh > rh:
                    continue
                for nx, ny in _nightstand_slots(bed, fw, fh):
                    if nx < 0 or ny < 0 or nx + fw > rw or ny + fh > rh:
                        continue
                    if footprint_blocked[ny:ny + fh, nx:nx + fw].any():
                        continue
                    # NS own-zone check (allow bed cells, block everything else)
                    hypo_ns = Placement(fid, nx, ny, fw, fh, ns_ori)
                    zone_ok = True
                    for zx, zy, zw, zh in _zone_rects(hypo_ns):
                        if zx < 0 or zy < 0 or zx + zw > rw or zy + zh > rh:
                            zone_ok = False
                            break
                        for yy in range(zy, zy + zh):
                            if not zone_ok:
                                break
                            for xx in range(zx, zx + zw):
                                if self.grid[yy, xx] == 0:
                                    continue
                                occ = _furniture_at(self.placed, xx, yy)
                                if occ is None or CATALOG[occ.fid].cat != "bed":
                                    zone_ok = False
                                    break
                    if not zone_ok:
                        continue
                    mask[encode_action(fid, nx, ny, ns_ori)] = True

        # DONE gate: allowed iff (a) a bed has been placed, or (b) no
        # placement action is valid at all (escape valve so mask is never
        # all-False — _reward() still gates no-bed episodes to R = 0).
        has_bed = any(CATALOG[p.fid].cat == "bed" for p in self.placed)
        mask[ACTION_DONE] = has_bed or not bool(mask[:ACTION_DONE].any())
        return mask

    def render(self) -> np.ndarray:
        return self._render_rgb()

    # --- internals --------------------------------------------------

    def _observation(self) -> np.ndarray:
        grid_obs = np.zeros((3, GH, GW), dtype=np.int8)
        grid_obs[0].fill(1)  # walls outside room
        grid_obs[0, :self.room_h, :self.room_w] = self.grid[:self.room_h, :self.room_w]
        # door cells on the bottom interior row
        grid_obs[1, self.room_h - 1, self.door_pos:self.door_pos + DW] = 1
        if self.win_wall == "top":
            grid_obs[2, 0, self.win_pos:self.win_pos + self.win_w] = 1
        elif self.win_wall == "left":
            grid_obs[2, self.win_pos:self.win_pos + self.win_w, 0] = 1
        else:
            grid_obs[2, self.win_pos:self.win_pos + self.win_w, self.room_w - 1] = 1
        # 5 cat-placed binary flags (bed, desk, wardrobe, cabinet, nightstand)
        placed_set = {CATALOG[p.fid].cat for p in self.placed}
        cats_flag = np.array(
            [int(c in placed_set) for c in ("bed", "desk", "wardrobe", "cabinet", "nightstand")],
            dtype=np.int8,
        )
        return np.concatenate([grid_obs.flatten(), cats_flag])

    def _info(self) -> dict:
        return {
            "room": (self.room_w, self.room_h),
            "door_pos": self.door_pos,
            "win_wall": self.win_wall,
            "win_pos": self.win_pos,
            "win_w": self.win_w,
            "placed": len(self.placed),
            "steps": self.steps,
        }

    def _can_place(self, fid: int, x: int, y: int, ori: int) -> bool:
        spec = CATALOG[fid]
        fw, fh = get_footprint(spec, ori)
        if x < 0 or y < 0 or x + fw > self.room_w or y + fh > self.room_h:
            return False
        for yy in range(y, y + fh):
            for xx in range(x, x + fw):
                if self.grid[yy, xx] > 0:
                    return False
                if (xx, yy) in self.swing:
                    return False
        return True

    def _reward(self) -> float:
        """Compute final reward (v3: multiplicative, ratio-based).

        R = availability × privacy × light × efficiency + diversity + compactness
          diversity   = (n_distinct_categories_placed²) / 5    ∈ [0.2, 5.0]
                        quadratic in n_cats: 1→0.2, 2→0.8, 3→1.8, 4→3.2,
                        5→5.0. Biggest jump at full set (3.2→5.0 = +1.8).
                        Added outside the product — never multiplied away.
          compactness = 5 × (1 − (perim/√area − 4)/8) ∈ [0, 5]
                        shape coefficient of remaining empty space
                        (swing transparent; lower perim/√area = more
                        integrated, less fragmented)
          privacy     = 1 − (1 − FACTOR_FLOOR) × exposure_ratio
          light       = 1 − (1 − FACTOR_FLOOR) × window_ratio
          efficiency  = 1 − waste_ratio

        where every penalty is a continuous geometric measure:
            exposure_ratio  = exposed_weight / total_weight
                              (per-cell weighted, pillow = 10× body, with
                              Bresenham occlusion through wardrobes)
            window_ratio  = blocked_window_cells / window_strip_cells
            waste_ratio   = unreachable_cells   / total_empty_cells

        Properties:
            • All ratios ∈ [0, 1] → scale-invariant across room sizes
            • All ratios = 0 → R = availability + diversity + compactness
            • diversity counterbalances area-weighted availability so the
              agent isn't biased toward bed-only layouts.
            • compactness penalizes fragmented / scattered layouts where
              empty space wraps around stranded furniture in the middle.
            • No weights, no τ — each (1 − ratio) is an independent discount.
        """
        rw, rh, pl = self.room_w, self.room_h, self.placed
        swept = _flood(self.grid, self.door_pos, rw, rh) if pl else set()

        beds = [p for p in pl if CATALOG[p.fid].cat == "bed"]

        # ── availability  (area × CELL_REWARD) ──
        # Hard validity (no overlap / no zone violation / no out-of-room zone) is
        # enforced by action_masks(); per-item value here is pure base value.
        # Nightstand position is mask-restricted to the bed-headboard slot —
        # no distance-decay bonus needed since invalid placements are impossible.
        # Safety net: a bed whose 2 access zones are both blocked by non-bed
        # furniture contributes 0 (action_masks() already guards this via
        # lock-last-zone, but this catches any leak / mask=False rollout).
        bed_corridor_ok = (not beds) or (len(_bed_clear_zones(beds[0], self.grid, pl, rw, rh)) >= 1)
        availability = 0.0
        per_item: list[tuple[str, float]] = []
        for p in pl:
            spec = CATALOG[p.fid]
            val = _value(spec)
            if spec.cat == "bed" and not bed_corridor_ok:
                val = 0.0   # corridor blocked → bed no longer usable
            availability += val
            per_item.append((spec.name, round(val * 10) / 10))

        # ── diversity bonus  (quadratic in n_categories, max +5) ──
        # Counterbalances bed-area dominance. Quadratic scaling
        # (1→0.2, 2→0.8, 3→1.8, 4→3.2, 5→5.0) so the 5-category jump is
        # the biggest reward, but the gradient stays smooth.
        cats_placed = {CATALOG[p.fid].cat for p in pl}
        n_categories = len(cats_placed)
        diversity = (n_categories ** 2) / 5.0

        # ── privacy: per-cell weighted exposure with wardrobe occlusion ──
        # For each bed cell, accumulate a weight (pillow cells = 2.0, body
        # cells = 1.0). A cell contributes to exposure if (a) it's inside
        # the door's ±45° cone AND (b) the Bresenham line of sight from the
        # door isn't blocked by a wardrobe. The ratio reflects "how much
        # of the bed (with pillow weighted 10x) is actually visible from
        # the door". Wardrobe-as-privacy-shield is a meaningful gameplay
        # strategy under this formulation.
        dcx, dcy, fac = _door_center("bottom", self.door_pos, rw, rh)
        exposed: list[tuple[int, int]] = []
        total_bed = 0
        exposed_pillow_n = 0
        total_pillow_n = 0
        total_weight = 0.0
        exposed_weight = 0.0
        cone_half = math.pi / 4
        # Pillow cells weighted 10× body cells so the pillow's privacy
        # actually dominates the ratio. Without the boost, body cells
        # (12×of pillow count for a 1.2m bed) drown out pillow exposure.
        # Ratio stays in [0, 1] because we normalize by total weight.
        PILLOW_W, BODY_W = 10.0, 1.0
        if beds:
            pillow_set = {c for b in beds for c in _pillow_cells(b)}
            total_pillow_n = len(pillow_set)
            for b in beds:
                for by in range(b.fh):
                    for bx in range(b.fw):
                        total_bed += 1
                        gx, gy = b.x + bx, b.y + by
                        is_pillow = (gx, gy) in pillow_set
                        w = PILLOW_W if is_pillow else BODY_W
                        total_weight += w
                        ang = math.atan2(gy + 0.5 - dcy, gx + 0.5 - dcx)
                        df = (ang - fac + math.pi) % (2 * math.pi) - math.pi
                        if abs(df) < cone_half:
                            if not _bresenham_blocked(round(dcx), round(dcy), gx, gy,
                                                      self.grid, pl, rw, rh):
                                exposed.append((gx, gy))
                                exposed_weight += w
                                if is_pillow:
                                    exposed_pillow_n += 1
            exposure_ratio = exposed_weight / max(total_weight, 1.0)
        else:
            exposure_ratio = 0.0
        n_window_blocked = _window_blocked_cells(
            self.win_wall, self.win_pos, self.win_w, rw, rh, self.grid, pl)
        window_strip_cells = self.win_w * 2          # 2-deep strip in front of window
        window_ratio = n_window_blocked / max(window_strip_cells, 1)

        unreachable = 0
        total_empty = rw * rh
        waste_ratio = 0.0
        if pl:
            total_empty = int((self.grid[:rh, :rw] == 0).sum())
            unreachable = max(0, total_empty - len(swept))
            waste_ratio = unreachable / max(total_empty, 1)

        # ── three independent multiplicative discounts ──
        # privacy / light: linear remap to [FACTOR_FLOOR, 1] — every ratio
        # change moves the factor (no flat zone like a hard floor), but
        # extreme ratios still don't wipe out the rest of the reward.
        # efficiency: full [0, 1] — wall-hugging gets the strongest signal.
        soft = 1.0 - FACTOR_FLOOR
        privacy    = 1.0 - soft * exposure_ratio       # ∈ [FACTOR_FLOOR, 1]
        light      = 1.0 - soft * window_ratio       # ∈ [FACTOR_FLOOR, 1]
        efficiency = 1.0 - waste_ratio                # ∈ [0, 1]
        product    = availability * privacy * light * efficiency
        # Compactness bonus: shape coefficient of the remaining empty space
        # (lower perimeter/√area = more integrated, less fragmented).
        compactness, shape_coef = _compactness(self.grid, rw, rh)
        total      = round((product + diversity + compactness) * 10) / 10

        # ── semantic gate: a bedroom isn't a bedroom without a bed ──
        # Closes the truncation loophole: even if max_steps runs out before
        # agent places a bed, the episode reward is forced to 0.
        has_bed = any(CATALOG[p.fid].cat == "bed" for p in self.placed)
        if not has_bed:
            total = 0.0

        # Express each discount as "points lost" for the right-panel display.
        # privacy_loss + light_loss + waste_loss + total = availability.
        privacy_loss = round(availability * (1.0 - privacy)                       * 10) / 10
        light_loss   = round(availability * privacy * (1.0 - light)               * 10) / 10
        waste_loss   = round(availability * privacy * light * (1.0 - efficiency)  * 10) / 10

        self._last_breakdown = {
            # Top-level
            "availability":      round(availability * 10) / 10,
            "diversity":         round(diversity * 10) / 10,
            "n_categories":      n_categories,
            "compactness":       round(compactness * 10) / 10,
            "shape_coef":        round(shape_coef * 100) / 100,
            "total":             total,
            "per_item":          per_item,
            # v3 native factors (∈ [0, 1])
            "privacy":           round(privacy    * 1000) / 1000,
            "light":             round(light      * 1000) / 1000,
            "efficiency":        round(efficiency * 1000) / 1000,
            # "Points lost" decomposition (sums to A − product_of_factors)
            "privacy_loss":      privacy_loss,
            "light_loss":        light_loss,
            "waste_loss":        waste_loss,
            # Raw ratios
            "exposure_ratio":    round(exposure_ratio * 1000) / 1000,
            "window_ratio":      round(window_ratio * 1000) / 1000,
            "waste_ratio":       round(waste_ratio * 1000) / 1000,
            # Counts (used for in-panel "X / Y" display)
            "n_exposed_pillow":  exposed_pillow_n,
            "total_pillow_cells": total_pillow_n,
            "n_window_blocked":  n_window_blocked,
            "window_strip_cells": window_strip_cells,
            "unreachable_cells": unreachable,
            "total_empty_cells": total_empty,
            # Bed-cone visualization (used by render.py overlay)
            "exposed_cells":     len(exposed),
            "exposed":           exposed,
            "total_bed_cells":   total_bed,
            # Booleans
            "pillow_seen":       exposed_pillow_n > 0,
            "window_blocked":    n_window_blocked > 0,
            # Cone overlay
            "swept":             swept,
            "door_center":       (dcx, dcy, fac),
        }
        return float(total)

    def _render_rgb(self) -> np.ndarray:
        cell = 18
        img = np.full((GH * cell, GW * cell, 3), 255, dtype=np.uint8)
        wall = np.array([200, 196, 190], dtype=np.uint8)
        for y in range(GH):
            for x in range(GW):
                if y >= self.room_h or x >= self.room_w:
                    img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = wall
        for p in self.placed:
            c = CAT_COLORS[CATALOG[p.fid].cat]
            img[p.y * cell:(p.y + p.fh) * cell, p.x * cell:(p.x + p.fw) * cell] = c
        # door (yellow strip)
        img[self.room_h * cell - 2:self.room_h * cell,
            self.door_pos * cell:(self.door_pos + DW) * cell] = (255, 230, 0)
        # window (light blue strip)
        win = (218, 237, 248)
        if self.win_wall == "top":
            img[0:2, self.win_pos * cell:(self.win_pos + self.win_w) * cell] = win
        elif self.win_wall == "left":
            img[self.win_pos * cell:(self.win_pos + self.win_w) * cell, 0:2] = win
        else:
            xpos = self.room_w * cell - 2
            img[self.win_pos * cell:(self.win_pos + self.win_w) * cell, xpos:xpos + 2] = win
        return img


# ------------------------------------------------------------------ pure helpers

def _make_swing(dp: int, rw: int, rh: int) -> set[tuple[int, int]]:
    """Door-swing cells. Door is always on the bottom wall; hinge at nearer corner."""
    swing: set[tuple[int, int]] = set()
    r = DW
    hinge_left = dp < rw / 2
    hx = dp if hinge_left else dp + DW
    sign = 1 if hinge_left else -1
    for i in range(r + 1):
        for j in range(r + 1):
            cx, cy = hx + j * sign, (rh - 1) - i
            if 0 <= cx < rw and 0 <= cy < rh:
                if (j + 0.5) ** 2 + (i + 0.5) ** 2 <= r * r:
                    swing.add((cx, cy))
    return swing


BED_ZONE_LEN = 8         # 1.2 m corridor length (along bed long axis)
BED_ZONE_DEPTH = 3       # 0.45 m corridor depth (perpendicular to bed)


def _zone_rects(p: Placement) -> list[tuple[int, int, int, int]]:
    spec = CATALOG[p.fid]
    d, o = spec.zd, p.ori
    rs: list[tuple[int, int, int, int]] = []
    if spec.z3:
        # Bed: 2 fixed zones, both 1.2 m × 0.45 m (8 × 3 cells), foot-anchored
        # on each long side. Lifetime invariant: at least one zone must be
        # fully clear of non-bed furniture for the bed to count its score.
        L, D = BED_ZONE_LEN, BED_ZONE_DEPTH
        if o == 0:    # horizontal, pillow LEFT, foot RIGHT
            zx = p.x + p.fw - L            # 8 cells anchored at foot (right end)
            rs.append((zx, p.y - D, L, D))         # top long side
            rs.append((zx, p.y + p.fh, L, D))      # bottom long side
        elif o == 2:  # horizontal, pillow RIGHT, foot LEFT
            zx = p.x                       # 8 cells anchored at foot (left end)
            rs.append((zx, p.y - D, L, D))         # top long side
            rs.append((zx, p.y + p.fh, L, D))      # bottom long side
        elif o == 1:  # vertical, pillow TOP, foot BOTTOM
            zy = p.y + p.fh - L            # 8 cells anchored at foot (bottom)
            rs.append((p.x - D, zy, D, L))         # left long side
            rs.append((p.x + p.fw, zy, D, L))      # right long side
        else:         # o == 3, vertical, pillow BOTTOM, foot TOP
            zy = p.y                       # 8 cells anchored at foot (top)
            rs.append((p.x - D, zy, D, L))         # left long side
            rs.append((p.x + p.fw, zy, D, L))      # right long side
    else:
        if o == 0:   rs.append((p.x, p.y + p.fh, p.fw, d))
        elif o == 1: rs.append((p.x + p.fw, p.y, d, p.fh))
        elif o == 2: rs.append((p.x, p.y - d, p.fw, d))
        else:        rs.append((p.x - d, p.y, d, p.fh))
    return rs


def _pillow_cells(p: Placement) -> list[tuple[int, int]]:
    o = p.ori
    if o == 0:   return [(p.x, p.y + i) for i in range(p.fh)]
    if o == 1:   return [(p.x + i, p.y) for i in range(p.fw)]
    if o == 2:   return [(p.x + p.fw - 1, p.y + i) for i in range(p.fh)]
    return [(p.x + i, p.y + p.fh - 1) for i in range(p.fw)]


def _bed_clear_zones(bed: Placement, grid: np.ndarray, pl: list[Placement],
                     rw: int, rh: int) -> list[tuple[int, int, int, int]]:
    """List of bed zones currently in-bounds AND fully clear of non-bed
    furniture. Used for the lifetime corridor invariant (mask) and the
    reward-time safety net (bed availability gates to 0 if list is empty).
    """
    out: list[tuple[int, int, int, int]] = []
    for zx, zy, zw, zh in _zone_rects(bed):
        if zx < 0 or zy < 0 or zx + zw > rw or zy + zh > rh:
            continue
        clear = True
        for yy in range(zy, zy + zh):
            if not clear:
                break
            for xx in range(zx, zx + zw):
                if grid[yy, xx] == 0:
                    continue
                occ = _furniture_at(pl, xx, yy)
                if occ is None or CATALOG[occ.fid].cat != "bed":
                    clear = False
                    break
        if clear:
            out.append((zx, zy, zw, zh))
    return out


def _nightstand_slots(bed: Placement, ns_fw: int, ns_fh: int) -> list[tuple[int, int]]:
    """Two nightstand positions on the LONG SIDES of the bed at the head
    end (one above/left, one below/right of the bed). NS sits beside the
    pillow on the long side, anchored so it covers the pillow column/row.

    ns_ori = bed.ori ^ 1 so the drawer faces along bed's long axis from
    pillow toward foot.
    """
    o = bed.ori
    if o == 0:    # horizontal, pillow LEFT — NS at top/bottom long side, x=bed.x
        return [(bed.x, bed.y - ns_fh),
                (bed.x, bed.y + bed.fh)]
    if o == 2:    # horizontal, pillow RIGHT — NS at top/bottom long side, right-aligned
        x = bed.x + bed.fw - ns_fw
        return [(x, bed.y - ns_fh),
                (x, bed.y + bed.fh)]
    if o == 1:    # vertical, pillow TOP — NS at left/right long side, top-aligned
        return [(bed.x - ns_fw, bed.y),
                (bed.x + bed.fw, bed.y)]
    # o == 3, vertical, pillow BOTTOM — NS at left/right long side, bottom-aligned
    y = bed.y + bed.fh - ns_fh
    return [(bed.x - ns_fw, y),
            (bed.x + bed.fw, y)]


def _has_reachable_neighbor(p: Placement, swept: set[tuple[int, int]]) -> bool:
    for dy in range(-1, p.fh + 1):
        for dx in range(-1, p.fw + 1):
            if 0 <= dy < p.fh and 0 <= dx < p.fw:
                continue
            if (p.x + dx, p.y + dy) in swept:
                return True
    return False


def _furniture_at(pl: list[Placement], x: int, y: int) -> Optional[Placement]:
    for p in pl:
        if p.x <= x < p.x + p.fw and p.y <= y < p.y + p.fh:
            return p
    return None


def _zone_ok(p: Placement, grid: np.ndarray, pl: list[Placement],
             rw: int, rh: int, allowed_cats: set[str], partial: bool) -> bool:
    any_good = False
    any_bad = False
    for zx, zy, zw, zh in _zone_rects(p):
        for yy in range(zy, zy + zh):
            for xx in range(zx, zx + zw):
                if not (0 <= xx < rw and 0 <= yy < rh):
                    any_bad = True
                    continue
                if grid[yy, xx] != 0:
                    occ = _furniture_at(pl, xx, yy)
                    if occ is not None and CATALOG[occ.fid].cat in allowed_cats:
                        any_good = True
                    else:
                        any_bad = True
                else:
                    any_good = True
    return any_good if partial else not any_bad


def _flood(grid: np.ndarray, dp: int, rw: int, rh: int) -> set[tuple[int, int]]:
    """3x3-brush passability -> BFS from door -> expand +-1 -> swept set.

    Door swing cells are NOT blocked here — the door is open as the agent
    enters, so swing area is physically walkable. (The mask separately
    blocks placing furniture in swing cells.) Swing-as-non-empty is only
    relevant to compactness, where it affects the empty-region perimeter.
    """
    passable = np.zeros((rh, rw), dtype=bool)
    for y in range(rh):
        for x in range(rw):
            if grid[y, x] != 0:
                continue
            ok = True
            for dy in (-1, 0, 1):
                if not ok:
                    break
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < rw and 0 <= ny < rh) or grid[ny, nx] != 0:
                        ok = False
                        break
            passable[y, x] = ok

    centers: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []
    # Door is on bottom wall -> seed centers one row up from the bottom edge.
    for i in range(DW):
        sx, sy = dp + i, rh - 2
        if 0 <= sx < rw and 0 <= sy < rh and passable[sy, sx] and (sx, sy) not in centers:
            centers.add((sx, sy))
            queue.append((sx, sy))
    head = 0
    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < rw and 0 <= ny < rh and passable[ny, nx] and (nx, ny) not in centers:
                centers.add((nx, ny))
                queue.append((nx, ny))

    swept: set[tuple[int, int]] = set()
    for cx, cy in centers:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x2, y2 = cx + dx, cy + dy
                if 0 <= x2 < rw and 0 <= y2 < rh and grid[y2, x2] == 0:
                    swept.add((x2, y2))
    return swept


def _compactness(grid: np.ndarray, rw: int, rh: int) -> tuple[float, float]:
    """Spatial integrity of the remaining empty space.

    Computes the *shape coefficient* `perimeter / sqrt(area)` of the empty
    region (door swing is treated as transparent — only walls and furniture
    contribute to the non-empty side). Lower shape_coef = more compact /
    less fragmented empty region.

    Returns (compactness_bonus, shape_coef). bonus ∈ [0, 5]:
        empty rectangle  : shape_coef ≈ 4    → bonus ≈ 5
        fragmented layout: shape_coef ≥ 12   → bonus = 0
    """
    perimeter = 0
    area = 0
    for y in range(rh):
        for x in range(rw):
            if grid[y, x] != 0:
                continue
            area += 1
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < rw and 0 <= ny < rh) or grid[ny, nx] != 0:
                    perimeter += 1
    if area == 0:
        return 0.0, 0.0
    shape_coef = perimeter / math.sqrt(area)
    bonus = max(0.0, 5.0 * (1.0 - (shape_coef - 4.0) / 8.0))
    return bonus, shape_coef


def _door_center(wall: str, dp: int, rw: int, rh: int) -> tuple[float, float, float]:
    if wall == "top":    return dp + DW / 2, 0.5, math.pi / 2
    if wall == "bottom": return dp + DW / 2, rh - 0.5, -math.pi / 2
    if wall == "left":   return 0.5, dp + DW / 2, 0.0
    return rw - 0.5, dp + DW / 2, math.pi


def _bresenham_blocked(x0: int, y0: int, x1: int, y1: int,
                       grid: np.ndarray, pl: list[Placement],
                       rw: int, rh: int) -> bool:
    adx, ady = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = adx - ady
    cx, cy = x0, y0
    for _ in range(100):
        if cx == x1 and cy == y1:
            return False
        e2 = 2 * err
        if e2 > -ady:
            err -= ady
            cx += sx
        if e2 < adx:
            err += adx
            cy += sy
        if cx == x1 and cy == y1:
            return False
        if 0 <= cx < rw and 0 <= cy < rh and grid[cy, cx] != 0:
            occ = _furniture_at(pl, cx, cy)
            if occ is not None and CATALOG[occ.fid].cat == "wardrobe":
                return True
    return False


def _window_blocked_cells(wall: str, wp: int, ww: int, rw: int, rh: int,
                          grid: np.ndarray, pl: list[Placement]) -> int:
    """Count cells in the 2-deep window strip occupied by bed/wardrobe."""
    n = 0
    for wi in range(ww):
        for wd in range(2):
            if wall == "top":      wx, wy = wp + wi, wd
            elif wall == "bottom": wx, wy = wp + wi, rh - 1 - wd
            elif wall == "left":   wx, wy = wd, wp + wi
            else:                  wx, wy = rw - 1 - wd, wp + wi
            if 0 <= wx < rw and 0 <= wy < rh and grid[wy, wx] != 0:
                occ = _furniture_at(pl, wx, wy)
                if occ is not None and CATALOG[occ.fid].cat in ("bed", "wardrobe"):
                    n += 1
    return n


def _window_blocked(wall, wp, ww, rw, rh, grid, pl) -> bool:
    """Back-compat wrapper: True iff any cell is blocked."""
    return _window_blocked_cells(wall, wp, ww, rw, rh, grid, pl) > 0
