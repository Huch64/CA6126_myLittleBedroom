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
ROOM_W_RANGE = (14, 26)
ROOM_H_RANGE = (18, 22)
WINDOW_WALLS = ("top", "left", "right")
N_ORI = 4

# ── reward v3 (multiplicative, ratio-based, scale-invariant) ──────
#
# R = Availability × privacy × light × efficiency
#
#   • Availability  = Σ (area_cells × CELL_REWARD)
#                       — linear in area, single knob, no per-category factor.
#                       Big furniture (bed) dominates naturally because it
#                       occupies the most space.
#   • privacy       = 1 − (1 − FACTOR_FLOOR) × pillow_ratio
#                         pillow_ratio = angle_dev_to_pillow / (π/4)
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
NIGHTSTAND_PAIR_BONUS = 1.0         # max bonus when nightstand is adjacent to headboard
D_PAIR = 4                          # nightstand bonus decays linearly to 0 over D_PAIR cells
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
        self.observation_space = spaces.Box(
            low=0, high=19, shape=(3, GH, GW), dtype=np.int8
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
           swing / OR functional zones of already-placed non-bed pieces. The
           non-bed-zone block prevents agent from breaking those pieces'
           accessibility (no zone_ok cliff at reward time).

        2. Functional (own zone): the candidate's own functional zone (in
           front of desk, around bed, etc.) must currently be clear and
           in-bounds. Beds use partial=True semantics: any one zone cell
           accessible is enough (loose). Other pieces require all zone cells
           clear (strict).

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
        other_zone_blocked = np.zeros((GH, GW), dtype=bool)
        for p in self.placed:
            if CATALOG[p.fid].cat == "bed":
                continue   # bed zones are loose; allow placements (esp. nightstand)
            for zx, zy, zw, zh in _zone_rects(p):
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
                        # bed: partial=True. Need at least one empty cell summed across zones.
                        total_empty = np.zeros((ys.size, xs.size), dtype=np.int32)
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
                            empty_in = np.where(in_bounds, zdw * zdh - blocked_sum, 0)
                            total_empty += empty_in
                        valid &= (total_empty >= 1)
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
        obs = np.zeros((3, GH, GW), dtype=np.int8)
        obs[0].fill(1)  # walls outside room
        obs[0, :self.room_h, :self.room_w] = self.grid[:self.room_h, :self.room_w]
        # door cells on the bottom interior row
        obs[1, self.room_h - 1, self.door_pos:self.door_pos + DW] = 1
        if self.win_wall == "top":
            obs[2, 0, self.win_pos:self.win_pos + self.win_w] = 1
        elif self.win_wall == "left":
            obs[2, self.win_pos:self.win_pos + self.win_w, 0] = 1
        else:
            obs[2, self.win_pos:self.win_pos + self.win_w, self.room_w - 1] = 1
        return obs

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

        R = availability × privacy × light × efficiency
          privacy    = 1 − pillow_ratio
          light      = 1 − window_ratio
          efficiency = 1 − waste_ratio

        where every penalty is a continuous geometric measure:
            pillow_ratio  = 1 − angle_to_pillow_centroid / (π/2)
                            (0° from door = fully exposed, 90° = fully private)
            window_ratio  = blocked_window_cells / window_strip_cells
            waste_ratio   = unreachable_cells   / total_empty_cells

        Privacy uses the pillow centroid (not per-cell exposure) and pure
        angular deviation from the door's facing direction. Wardrobe-as-
        privacy-shield is not modelled here (kept formula PPT-friendly);
        wardrobes affect reward only through light and efficiency.

        Properties:
            • All ratios ∈ [0, 1] → scale-invariant across room sizes
            • R = 0 iff availability = 0 (DONE-immediate) OR any ratio = 1
            • All ratios = 0 → R = availability (full credit, no discount)
            • No weights, no τ — each (1 − ratio) is an independent discount
        """
        rw, rh, pl = self.room_w, self.room_h, self.placed
        swept = _flood(self.grid, self.door_pos, rw, rh) if pl else set()

        beds = [p for p in pl if CATALOG[p.fid].cat == "bed"]

        # ── availability  (area × CELL_REWARD, plus nightstand distance-decay bonus) ──
        # Hard validity (no overlap / no zone violation / no out-of-room zone) is
        # enforced by action_masks(); per-item value here is pure base value plus
        # a smooth pairing bonus for nightstands.
        availability = 0.0
        per_item: list[tuple[str, float]] = []
        for p in pl:
            spec = CATALOG[p.fid]
            val = _value(spec)
            if spec.cat == "nightstand" and beds:
                # Manhattan distance from nightstand centre to nearest headboard
                # cell, decayed linearly over D_PAIR cells. Adjacent = full bonus.
                d_min = min(_dist_to_headboard(p, b) for b in beds)
                val += NIGHTSTAND_PAIR_BONUS * max(0.0, 1.0 - d_min / D_PAIR)
            availability += val
            per_item.append((spec.name, round(val * 10) / 10))

        # ── privacy: simple 1-D angular deviation from door's facing direction ──
        # Treat the pillow as a single point (its centroid). Compute angle from
        # door's facing direction to that centroid; map to exposure via the
        # door's vision cone (±45°):
        #
        #   angle_dev ∈ [0, π/4]:  0 = pillow dead-centre of door cone (worst)
        #                          π/4 = pillow at edge of cone (no exposure)
        #   angle_dev ≥ π/4:       pillow outside cone (no exposure)
        #
        #   pillow_ratio = max(0, 1 − angle_dev / (π/4))
        #
        # No per-cell loop, no Bresenham, no distance — one angle → one ratio.
        # Trades wardrobe-as-privacy-shield as an emergent strategy for a
        # formula clean enough to fit on one PPT slide alongside
        # efficiency = unreachable / total_empty.
        dcx, dcy, fac = _door_center("bottom", self.door_pos, rw, rh)
        exposed: list[tuple[int, int]] = []           # kept for HUD overlay
        total_bed = 0
        exposed_pillow_n = 0                          # display-only stub
        total_pillow_n = 0
        if beds:
            pillow_set = {c for b in beds for c in _pillow_cells(b)}
            total_pillow_n = len(pillow_set)
            # Pillow centroid (cell-centre coordinates)
            pcx = sum(c[0] for c in pillow_set) / len(pillow_set) + 0.5
            pcy = sum(c[1] for c in pillow_set) / len(pillow_set) + 0.5
            ang = math.atan2(pcy - dcy, pcx - dcx)
            angle_dev = abs((ang - fac + math.pi) % (2 * math.pi) - math.pi)
            pillow_ratio = max(0.0, 1.0 - angle_dev / (math.pi / 4))
            # for HUD: still trace which bed cells the door can see (visual only)
            cone_half = math.pi / 4
            for b in beds:
                for by in range(b.fh):
                    for bx in range(b.fw):
                        total_bed += 1
                        gx, gy = b.x + bx, b.y + by
                        a = math.atan2(gy + 0.5 - dcy, gx + 0.5 - dcx)
                        df = (a - fac + math.pi) % (2 * math.pi) - math.pi
                        if abs(df) < cone_half:
                            exposed.append((gx, gy))
                            if (gx, gy) in pillow_set:
                                exposed_pillow_n += 1
        else:
            pillow_ratio = 0.0
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
        privacy    = 1.0 - soft * pillow_ratio       # ∈ [FACTOR_FLOOR, 1]
        light      = 1.0 - soft * window_ratio       # ∈ [FACTOR_FLOOR, 1]
        efficiency = 1.0 - waste_ratio                # ∈ [0, 1]
        total = round(availability * privacy * light * efficiency * 10) / 10

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
            "total":             total,
            "per_item":          per_item,
            # v3 native factors (∈ [0, 1])
            "privacy":           round(privacy    * 1000) / 1000,
            "light":             round(light      * 1000) / 1000,
            "efficiency":        round(efficiency * 1000) / 1000,
            # "Points lost" decomposition (sums to A − R)
            "privacy_loss":      privacy_loss,
            "light_loss":        light_loss,
            "waste_loss":        waste_loss,
            # Back-compat aliases (so old runs / scripts that read these keep working)
            "comfort":           round(privacy * light * 1000) / 1000,
            "waste_eff":         round(efficiency * 1000) / 1000,
            "discomfort":        round((privacy_loss + light_loss) * 10) / 10,
            "waste":             waste_loss,
            # Raw ratios
            "pillow_ratio":      round(pillow_ratio * 1000) / 1000,
            "window_ratio":      round(window_ratio * 1000) / 1000,
            "waste_ratio":       round(waste_ratio * 1000) / 1000,
            # Counts (used for in-panel "X / Y" display)
            "n_exposed_pillow":  exposed_pillow_n,
            "total_pillow_cells": total_pillow_n,
            "n_window_blocked":  n_window_blocked,
            "window_strip_cells": window_strip_cells,
            "unreachable_cells": unreachable,
            "total_empty_cells": total_empty,
            # Bed-cone visualization (still used by render.py overlay)
            "exposed_cells":     len(exposed),
            "exposed":           exposed,
            "total_bed_cells":   total_bed,
            "bed_exposure_score": 0.0,                # deprecated under v3
            # Booleans (back-compat)
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


def _zone_rects(p: Placement) -> list[tuple[int, int, int, int]]:
    spec = CATALOG[p.fid]
    d, o = spec.zd, p.ori
    rs: list[tuple[int, int, int, int]] = []
    if spec.z3:
        if o in (0, 2):
            rs.append((p.x, p.y - d, p.fw, d))
            rs.append((p.x, p.y + p.fh, p.fw, d))
            rs.append((p.x + p.fw, p.y, d, p.fh) if o == 0 else (p.x - d, p.y, d, p.fh))
        else:
            rs.append((p.x - d, p.y, d, p.fh))
            rs.append((p.x + p.fw, p.y, d, p.fh))
            rs.append((p.x, p.y + p.fh, p.fw, d) if o == 1 else (p.x, p.y - d, p.fw, d))
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


def _dist_to_headboard(ns: Placement, bed: Placement) -> float:
    """Manhattan distance from a nightstand's centre to the nearest pillow
    (headboard) cell of the bed. Used for the continuous pairing bonus."""
    nsx = ns.x + ns.fw / 2.0
    nsy = ns.y + ns.fh / 2.0
    best = float("inf")
    for hx, hy in _pillow_cells(bed):
        d = abs(nsx - (hx + 0.5)) + abs(nsy - (hy + 0.5))
        if d < best:
            best = d
    return best


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


def _paired_nightstands(beds: list[Placement], nss: list[Placement]) -> set[int]:
    paired: set[int] = set()
    for ns in nss:
        for bd in beds:
            o = bd.ori
            if o in (0, 2):
                above = ns.y + ns.fh == bd.y and ns.x < bd.x + bd.fw and ns.x + ns.fw > bd.x
                below = ns.y == bd.y + bd.fh and ns.x < bd.x + bd.fw and ns.x + ns.fw > bd.x
                hb_x = bd.x if o == 0 else bd.x + bd.fw - 1
                ok = (above or below) and ns.x <= hb_x < ns.x + ns.fw
            else:
                left = ns.x + ns.fw == bd.x and ns.y < bd.y + bd.fh and ns.y + ns.fh > bd.y
                right = ns.x == bd.x + bd.fw and ns.y < bd.y + bd.fh and ns.y + ns.fh > bd.y
                hb_y = bd.y if o == 1 else bd.y + bd.fh - 1
                ok = (left or right) and ns.y <= hb_y < ns.y + ns.fh
            if ok:
                paired.add(id(ns))
                break
    return paired


def _flood(grid: np.ndarray, dp: int, rw: int, rh: int) -> set[tuple[int, int]]:
    """3x3-brush passability -> BFS from door -> expand +-1 -> swept set."""
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
