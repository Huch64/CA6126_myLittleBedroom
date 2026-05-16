"""Render and/or record an agent placing furniture.

Examples:
    python render.py                                  # random agent → videos/random.mp4
    python render.py --show                           # live matplotlib window
    python render.py --episodes 3 --save videos/r.mp4 # 3 episodes back-to-back
    python render.py --model checkpoints/ppo.zip      # trained MaskablePPO

The random policy samples uniformly from valid actions (action_masks).
The trained policy lazy-imports sb3_contrib, so a missing install only
blocks `--model`, not the default random run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import (
    CAT_COLORS,
    CATALOG,
    DW,
    GH,
    GW,
    MyLittleBedroom,
    _pillow_cells,
    _zone_rects,
)


# ----------------------------------------------------------- policies

def random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def step(obs, env):
        mask = env.action_masks()
        return int(rng.choice(np.flatnonzero(mask)))

    return step


def model_policy(model_path: str):
    from sb3_contrib import MaskablePPO   # lazy
    model = MaskablePPO.load(model_path)

    def step(obs, env):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return step


# ----------------------------------------------------------- rendering

_WALL_C = "#C8C4BE"
_GRID_C = "#E8E4DE"
_GRAY = "#999"
_LEM = "#FFE500"
_WIN_FILL = "#DAEDF8"
WALL_T = 0.8         # wall thickness in cells — door/window strips match this


def _draw_door(ax, env):
    """Yellow opening (full wall thickness) + dashed swing arc + hinge line."""
    from matplotlib.patches import Arc, Rectangle
    rh, rw = env.room_h, env.room_w
    # White opening cut out of the wall — full wall thickness.
    ax.add_patch(Rectangle((env.door_pos, rh), DW, WALL_T,
                           facecolor="white", edgecolor="none", zorder=5))
    # Yellow threshold tint for visual emphasis.
    ax.add_patch(Rectangle((env.door_pos, rh), DW, WALL_T * 0.25,
                           facecolor=_LEM, alpha=0.35, edgecolor="none", zorder=5.5))
    # Door swing arc + hinge line.
    hinge_left = env.door_pos < rw / 2
    hx = env.door_pos if hinge_left else env.door_pos + DW
    if hinge_left:
        theta1, theta2 = 270, 360
    else:
        theta1, theta2 = 180, 270
    ax.add_patch(Arc((hx, rh), 2 * DW, 2 * DW, angle=0,
                     theta1=theta1, theta2=theta2,
                     edgecolor=_GRAY, lw=0.8, linestyle=":", alpha=0.5))
    ax.plot([hx, hx], [rh, rh - DW], color=_GRAY, lw=1.2, alpha=0.45)


def _draw_window(ax, env):
    from matplotlib.patches import Rectangle
    rh, rw = env.room_h, env.room_w
    t = WALL_T   # window depth = wall thickness
    glass_inset = 0.18
    if env.win_wall == "top":
        x, y, w, h = env.win_pos, -t, env.win_w, t
        gx, gy, gw, gh = x + 0.10, y + glass_inset, w - 0.20, h - 2 * glass_inset
    elif env.win_wall == "left":
        x, y, w, h = -t, env.win_pos, t, env.win_w
        gx, gy, gw, gh = x + glass_inset, y + 0.10, w - 2 * glass_inset, h - 0.20
    else:
        x, y, w, h = rw, env.win_pos, t, env.win_w
        gx, gy, gw, gh = x + glass_inset, y + 0.10, w - 2 * glass_inset, h - 0.20
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor=_WIN_FILL, alpha=0.7,
                           edgecolor="none", zorder=5.5))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=_GRAY, lw=1.0, zorder=6))


def _draw_furniture(ax, p, spec, c_rgb):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    # Body.
    ax.add_patch(FancyBboxPatch((p.x + 0.03, p.y + 0.03),
                                p.fw - 0.06, p.fh - 0.06,
                                boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=c_rgb, edgecolor="none", zorder=3))
    # Label.
    font = max(6, min(12, min(p.fw, p.fh) * 0.95))
    ax.text(p.x + p.fw / 2, p.y + p.fh / 2, spec.name,
            ha="center", va="center", color="white",
            fontsize=font, weight="bold", zorder=6)
    # Front-face line (white) — for non-z3 categories. ori encodes face dir:
    #   0=down  1=right  2=up  3=left  (matches zoneRects in env.py)
    if not spec.z3:
        if p.ori == 0:
            f = (p.x, p.y + p.fh, p.x + p.fw, p.y + p.fh)
        elif p.ori == 1:
            f = (p.x + p.fw, p.y, p.x + p.fw, p.y + p.fh)
        elif p.ori == 2:
            f = (p.x, p.y, p.x + p.fw, p.y)
        else:
            f = (p.x, p.y, p.x, p.y + p.fh)
        ax.plot([f[0], f[2]], [f[1], f[3]], color="white",
                lw=1.8, alpha=0.7, solid_capstyle="round", zorder=7)
    else:
        # Bed: outline three "long sides + foot" with thin white lines.
        ox, oy = p.x, p.y
        ow, oh = p.fw, p.fh
        if p.ori in (0, 2):
            ax.plot([ox, ox + ow], [oy, oy], color="white", lw=1.2, alpha=0.6, zorder=7)
            ax.plot([ox, ox + ow], [oy + oh, oy + oh], color="white", lw=1.2, alpha=0.6, zorder=7)
            fx = ox + ow if p.ori == 0 else ox
            ax.plot([fx, fx], [oy, oy + oh], color="white", lw=1.5, alpha=0.7, zorder=7)
        else:
            ax.plot([ox, ox], [oy, oy + oh], color="white", lw=1.2, alpha=0.6, zorder=7)
            ax.plot([ox + ow, ox + ow], [oy, oy + oh], color="white", lw=1.2, alpha=0.6, zorder=7)
            fy = oy + oh if p.ori == 1 else oy
            ax.plot([ox, ox + ow], [fy, fy], color="white", lw=1.5, alpha=0.7, zorder=7)


def _draw_detail(ax, p, spec):
    """Draw the symbolic hint inside a furniture body. Proportions mirror the
    HTML Detail() function: 1 cell == 18 px in HTML, so HTML pixel constants
    are divided by 18 here."""
    from matplotlib.patches import FancyBboxPatch, Circle
    WB = (1, 1, 1, 0.60)
    W  = (1, 1, 1, 0.45)
    cx = p.x + p.fw / 2
    cy = p.y + p.fh / 2

    if spec.cat == "bed":
        n = spec.pl                                  # 1 or 2 pillows
        t = max(8 / 18, min(p.fw, p.fh) * 0.18)      # HTML: max(8, min(pw,ph)*0.18) px
        edge = 4 / 18                                # HTML px inset → cells
        gap = 3 / 18                                 # half-gap between two pillows
        if p.ori in (0, 2):
            spine_x = p.x + p.fw * (0.42 if p.ori == 0 else 0.58)
            ax.plot([spine_x, spine_x], [p.y + edge, p.y + p.fh - edge],
                    color=W, lw=1.0, zorder=8)
            px_p = p.x + edge if p.ori == 0 else p.x + p.fw - edge - t
            if n == 1:
                pL = p.fh * 0.45
                ax.add_patch(FancyBboxPatch((px_p, cy - pL / 2), t, pL,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
            else:
                pL = p.fh * 0.30
                ax.add_patch(FancyBboxPatch((px_p, cy - pL - gap), t, pL,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
                ax.add_patch(FancyBboxPatch((px_p, cy + gap), t, pL,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
        else:
            spine_y = p.y + p.fh * (0.42 if p.ori == 1 else 0.58)
            ax.plot([p.x + edge, p.x + p.fw - edge], [spine_y, spine_y],
                    color=W, lw=1.0, zorder=8)
            py_p = p.y + edge if p.ori == 1 else p.y + p.fh - edge - t
            if n == 1:
                pL = p.fw * 0.45
                ax.add_patch(FancyBboxPatch((cx - pL / 2, py_p), pL, t,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
            else:
                pL = p.fw * 0.30
                ax.add_patch(FancyBboxPatch((cx - pL - gap, py_p), pL, t,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
                ax.add_patch(FancyBboxPatch((cx + gap, py_p), pL, t,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=WB, edgecolor="none", zorder=9))
        return

    if spec.cat == "desk":
        # monitor strip on the back edge
        mH = 4 / 18
        edge = 4 / 18
        if p.ori in (0, 2):
            mW = min(p.fw * 0.75, 4)               # HTML: min(pw*0.75, 4*CELL)
            mY = p.y + edge if p.ori == 0 else p.y + p.fh - edge - mH
            ax.add_patch(FancyBboxPatch((cx - mW / 2, mY), mW, mH,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=WB, edgecolor="none", zorder=8))
        else:
            mW = min(p.fh * 0.75, 4)
            mX = p.x + edge if p.ori == 1 else p.x + p.fw - edge - mH
            ax.add_patch(FancyBboxPatch((mX, cy - mW / 2), mH, mW,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=WB, edgecolor="none", zorder=8))
        return

    if spec.cat == "wardrobe":
        # one handle line; orientation chosen by aspect ratio (matches HTML)
        edge = 4 / 18
        if p.fw >= p.fh:
            yline = p.y + p.fh * 0.4
            ax.plot([p.x + edge, p.x + p.fw - edge], [yline, yline],
                    color=W, lw=1.5, zorder=8)
        else:
            xline = p.x + p.fw * 0.4
            ax.plot([xline, xline], [p.y + edge, p.y + p.fh - edge],
                    color=W, lw=1.5, zorder=8)
        return

    if spec.cat == "cabinet":
        # shelf dividers — HTML: floor(max(pw, ph) / 20) lines, perpendicular to long axis
        long_cells = max(p.fw, p.fh)
        n = max(1, int(long_cells * 18 // 20))
        edge = 3 / 18
        if p.fw >= p.fh:
            for i in range(n):
                f = (i + 1) / (n + 1)
                xline = p.x + p.fw * f
                ax.plot([xline, xline], [p.y + edge, p.y + p.fh - edge],
                        color=W, lw=1.5, zorder=8)
        else:
            for i in range(n):
                f = (i + 1) / (n + 1)
                yline = p.y + p.fh * f
                ax.plot([p.x + edge, p.x + p.fw - edge], [yline, yline],
                        color=W, lw=1.5, zorder=8)
        return

    if spec.cat == "nightstand":
        r = min(p.fw, p.fh) * 0.15
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=WB,
                            lw=1.0, zorder=8))


def _draw_desk_chair(ax, p, c_rgb):
    from matplotlib.patches import FancyBboxPatch
    csz = 4
    if p.ori == 0:    cx, cy = p.x + (p.fw - csz) / 2, p.y + p.fh
    elif p.ori == 1:  cx, cy = p.x + p.fw, p.y + (p.fh - csz) / 2
    elif p.ori == 2:  cx, cy = p.x + (p.fw - csz) / 2, p.y - csz
    else:             cx, cy = p.x - csz, p.y + (p.fh - csz) / 2
    ax.add_patch(FancyBboxPatch((cx + 0.06, cy + 0.06), csz - 0.12, csz - 0.12,
                                boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=c_rgb, alpha=0.35,
                                edgecolor="none", zorder=2))
    ax.text(cx + csz / 2, cy + csz / 2, "Chair",
            ha="center", va="center", color="white",
            fontsize=8, alpha=0.7, zorder=6)


def _draw_zones(ax, p, c_rgb):
    from matplotlib.patches import Rectangle
    for zx, zy, zw, zh in _zone_rects(p):
        ax.add_patch(Rectangle((zx + 0.06, zy + 0.06),
                               zw - 0.12, zh - 0.12,
                               fill=False, edgecolor=c_rgb,
                               lw=0.7, linestyle=(0, (2.2, 2)),
                               alpha=0.32, zorder=1))


# ── typography (only 4 sizes, mono + serif title) ────────────────
MONO = "DejaVu Sans Mono"
SERIF = "serif"
SZ_BRAND = 18    # brand title
SZ_HEADER = 11   # section headers (ENVIRONMENT, PLACEMENTS, REWARD)
SZ_BODY = 10     # key labels and data values
SZ_DETAIL = 9    # secondary info, hints, units
SZ_TOTAL = 28    # the big TOTAL number

# ── figure layout constants (everything aligns to these) ──────
_LM = 0.04           # left margin (figure coords)
_RM = 0.04           # right margin
_GAP = 0.03          # gap between left column and right column
_LEFT_W = 0.55       # left column width
_RIGHT_X = _LM + _LEFT_W + _GAP                # = 0.62
_RIGHT_W = 1.0 - _RM - _RIGHT_X                # = 0.34

# Vertical bands (figure coords, y measured from bottom):
_TITLE_BAND = (0.94, 0.99)        # title bar
_ENV_BAND   = (0.74, 0.93)        # env_ax + placements_ax top edge
_ROOM_BAND  = (0.03, 0.72)        # room_ax + reward_details_ax top edge
_TOTALS_H   = 0.28                # totals_ax height (fixed at bottom)


def render_frame(env, step_idx: int, total_steps: int, breakdown: dict | None):
    """One matplotlib frame -> RGB ndarray. Visuals mirror my_little_bedroom.html.

    Layout (figure coords, 0=bottom):
        ┌──────────────────────────────────────────────────┐
        │  TITLE BAR (brand + step)        0.94 - 0.99      │
        ├──────────────────────┬───────────────────────────┤
        │  ENVIRONMENT         │  PLACEMENTS               │   0.74 - 0.93
        ├──────────────────────┤                           │
        │                      ├───────────────────────────┤
        │  ROOM VIEW           │  REWARD (auto-fill)       │   0.32 - 0.72
        │                      │                           │
        │  legend strip        ├───────────────────────────┤
        │                      │  TOTALS (FIXED bottom)    │   0.03 - 0.31
        └──────────────────────┴───────────────────────────┘
    Every column shares left/right edges with the title bar.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(14, 8.2), dpi=110, facecolor="white")
    canvas = FigureCanvasAgg(fig)

    # ── TITLE BAR (full width, spans both columns) ──
    title_ax = fig.add_axes([_LM, _TITLE_BAND[0],
                             1.0 - _LM - _RM,
                             _TITLE_BAND[1] - _TITLE_BAND[0]])
    title_ax.axis("off"); title_ax.set_xlim(0, 1); title_ax.set_ylim(0, 1)
    title_ax.text(0.0, 0.5, "my little bedroom", color="#C03030",
                  fontsize=SZ_BRAND, weight="bold", style="italic",
                  family=SERIF, va="center")
    title_ax.text(1.0, 0.5, f"step {step_idx} / {total_steps}",
                  fontsize=SZ_BODY, color="#999", family=MONO,
                  va="center", ha="right")

    # ── LEFT TOP: ENVIRONMENT block (same x as room below) ──
    env_ax = fig.add_axes([_LM, _ENV_BAND[0],
                           _LEFT_W,
                           _ENV_BAND[1] - _ENV_BAND[0]])
    env_ax.axis("off"); env_ax.set_xlim(0, 1); env_ax.set_ylim(0, 1)
    _draw_env_block(env_ax, env)

    # ── LEFT MAIN: room view; wall outer edge aligns with env text at x=0 ──
    room_ax = fig.add_axes([_LM, _ROOM_BAND[0],
                            _LEFT_W,
                            _ROOM_BAND[1] - _ROOM_BAND[0]])
    _draw_room(room_ax, env, breakdown)

    # ── RIGHT TOP: PLACEMENTS (aligned with ENVIRONMENT band) ──
    placements_ax = fig.add_axes([_RIGHT_X, _ENV_BAND[0],
                                  _RIGHT_W,
                                  _ENV_BAND[1] - _ENV_BAND[0]])
    placements_ax.axis("off")
    placements_ax.set_xlim(0, 1); placements_ax.set_ylim(0, 1)
    _draw_placements(placements_ax, env)

    # ── RIGHT BOTTOM (fixed): TOTALS box, always pinned to figure bottom ──
    totals_y0 = _ROOM_BAND[0]
    totals_ax = fig.add_axes([_RIGHT_X, totals_y0, _RIGHT_W, _TOTALS_H])
    totals_ax.axis("off")
    totals_ax.set_xlim(0, 1); totals_ax.set_ylim(0, 1)
    _draw_totals(totals_ax, breakdown)

    # ── RIGHT MIDDLE (auto-fill): REWARD details between placements and totals ──
    details_y0 = totals_y0 + _TOTALS_H + 0.01
    details_h = _ENV_BAND[0] - details_y0 - 0.01
    reward_details_ax = fig.add_axes([_RIGHT_X, details_y0, _RIGHT_W, details_h])
    reward_details_ax.axis("off")
    reward_details_ax.set_xlim(0, 1); reward_details_ax.set_ylim(0, 1)
    _draw_reward_details(reward_details_ax, breakdown)

    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return img


def _draw_env_block(ax, env):
    """Top-left clear, labeled environment configuration block."""
    from env import GRID_M
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.plot([0.0, 1.0], [0.97, 0.97], color="#ddd", lw=0.8,
            transform=ax.transAxes)
    ax.text(0.0, 0.92, "ENVIRONMENT", fontsize=SZ_HEADER, weight="bold",
            color="#888", family=MONO, va="top")

    hinge = "left" if env.door_pos < env.room_w / 2 else "right"
    rows = [
        ("Room",   f"{env.room_w} × {env.room_h} cells",
                   f"({env.room_w * GRID_M:.2f} × {env.room_h * GRID_M:.2f} m)"),
        ("Door",   f"bottom wall, x={env.door_pos}, w={DW}",
                   f"hinge {hinge}"),
        ("Window", f"{env.win_wall} wall, x={env.win_pos}, w={env.win_w}",
                   "(centered, 50% of wall)"),
        ("Swing",  f"{len(env.swing)} cells reserved",
                   "(door's 90° arc)"),
    ]
    y = 0.74
    for key, val, note in rows:
        ax.text(0.00, y, key, fontsize=SZ_BODY, color="#333", family=MONO,
                weight="bold", va="top")
        ax.text(0.14, y, val, fontsize=SZ_BODY, color="#222", family=MONO,
                va="top")
        ax.text(0.62, y, note, fontsize=SZ_DETAIL, color="#999", family=MONO,
                va="top", style="italic")
        y -= 0.17


def _draw_room(ax, env, breakdown):
    """The main room view — walls, door, window, swing, furniture + overlays
    (cone / flood-fill / exposed cells) on the final frame."""
    from matplotlib.patches import Rectangle

    rw, rh = env.room_w, env.room_h

    for i in range(1, GH):
        ax.plot([0, GW], [i, i], color=_GRID_C, lw=0.4, zorder=0)
    for j in range(1, GW):
        ax.plot([j, j], [0, GH], color=_GRID_C, lw=0.4, zorder=0)

    if rw < GW:
        ax.add_patch(Rectangle((rw, 0), GW - rw, GH, facecolor=_WALL_C,
                               alpha=0.5, edgecolor="none", zorder=0.5))
    if rh < GH:
        ax.add_patch(Rectangle((0, rh), rw, GH - rh, facecolor=_WALL_C,
                               alpha=0.5, edgecolor="none", zorder=0.5))

    t = WALL_T
    ax.add_patch(Rectangle((-t, -t), rw + 2 * t, t, facecolor=_WALL_C, edgecolor="none"))
    ax.add_patch(Rectangle((-t, rh), rw + 2 * t, t, facecolor=_WALL_C, edgecolor="none"))
    ax.add_patch(Rectangle((-t, 0), t, rh, facecolor=_WALL_C, edgecolor="none"))
    ax.add_patch(Rectangle((rw, 0), t, rh, facecolor=_WALL_C, edgecolor="none"))
    ax.add_patch(Rectangle((-t, -t), rw + 2 * t, rh + 2 * t,
                           fill=False, edgecolor=_GRAY, lw=1.2))

    for sx, sy in env.swing:
        ax.add_patch(Rectangle((sx, sy), 1, 1, facecolor=_GRAY,
                               alpha=0.06, edgecolor="none", zorder=0.7))

    _draw_door(ax, env)
    _draw_window(ax, env)

    # Overlays go BEHIND furniture so labels stay readable.
    if breakdown is not None:
        _draw_overlays(ax, env, breakdown)

    for p in env.placed:
        c_rgb = tuple(v / 255 for v in CAT_COLORS[CATALOG[p.fid].cat])
        _draw_zones(ax, p, c_rgb)
    for p in env.placed:
        spec = CATALOG[p.fid]
        c_rgb = tuple(v / 255 for v in CAT_COLORS[spec.cat])
        if spec.cat == "desk":
            _draw_desk_chair(ax, p, c_rgb)
        _draw_furniture(ax, p, spec, c_rgb)
        _draw_detail(ax, p, spec)

    if breakdown is not None:
        _draw_legend(ax, env)

    # Wall outer edge sits at axes x = 0 so it aligns with the ENVIRONMENT
    # block above (whose text is also at axes x = 0). The anchor='NW'
    # pins the axes box to the top-left of its allotted area so the box
    # doesn't drift when matplotlib shrinks it to satisfy aspect='equal'.
    ax.set_xlim(-WALL_T, GW + WALL_T + 0.3)
    ax.set_ylim(GH + 3.5, -WALL_T - 0.3)
    ax.set_aspect("equal", adjustable="box", anchor="NW")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_overlays(ax, env, breakdown):
    """Three reward visualizations overlaid on the room (final frame only):
       - flood-fill swept set  (light yellow tint)
       - door 90° cone lines   (dashed yellow)
       - exposed bed cells     (strong yellow tint)
    Functional zones are already drawn separately as dashed colored rects.
    """
    import math
    from matplotlib.patches import Rectangle

    for sx, sy in breakdown["swept"]:
        ax.add_patch(Rectangle((sx + 0.08, sy + 0.08), 0.84, 0.84,
                               facecolor=_LEM, alpha=0.18,
                               edgecolor="none", zorder=1.2))

    dcx, dcy, fac = breakdown["door_center"]
    cone_len = max(env.room_w, env.room_h) * 1.3
    for sign in (-1, 1):
        ang = fac + sign * math.pi / 4
        ax.plot([dcx, dcx + math.cos(ang) * cone_len],
                [dcy, dcy + math.sin(ang) * cone_len],
                color=_LEM, lw=1.2, alpha=0.55,
                linestyle=(0, (4, 3)), zorder=9)

    for ex_x, ex_y in breakdown["exposed"]:
        ax.add_patch(Rectangle((ex_x + 0.12, ex_y + 0.12), 0.76, 0.76,
                               facecolor=_LEM, alpha=0.6,
                               edgecolor="none", zorder=9.5))


def _draw_legend(ax, env):
    """Color-chip legend strip just under the room view (fixed-column layout)."""
    from matplotlib.patches import Rectangle

    y0 = GH + 1.7
    chip_w = 0.9
    col_w = 8.5     # cells per legend slot; 3 slots fit GW=26
    entries = [
        ("fill", _LEM,   0.60, "exposed cells (cone)"),
        ("fill", _LEM,   0.18, "reachable  (flood)"),
        ("dash", "#888", None, "functional zone"),
    ]
    for i, (kind, color, alpha, label) in enumerate(entries):
        x = i * col_w
        if kind == "fill":
            ax.add_patch(Rectangle((x, y0), chip_w, chip_w,
                                   facecolor=color, alpha=alpha,
                                   edgecolor="#aaa", lw=0.4))
        else:
            ax.plot([x + 0.05, x + chip_w - 0.05],
                    [y0 + chip_w / 2, y0 + chip_w / 2],
                    color=color, lw=1.6, linestyle=(0, (2.5, 2)))
        ax.text(x + chip_w + 0.35, y0 + chip_w / 2, label,
                fontsize=SZ_DETAIL, color="#666",
                family=MONO, va="center")


def _draw_placements(ax, env):
    """Right top: numbered list of placements, top-aligned with ENVIRONMENT."""
    from matplotlib.patches import Rectangle

    ax.plot([0.0, 1.0], [0.97, 0.97], color="#ddd", lw=0.8,
            transform=ax.transAxes)
    ax.text(0.0, 0.93, "PLACEMENTS", fontsize=SZ_HEADER, weight="bold",
            color="#888", family=MONO, va="top")

    if not env.placed:
        ax.text(0.04, 0.65, "(none yet)", fontsize=SZ_DETAIL, color="#bbb",
                va="center", style="italic", family=MONO)
        return

    row_h = 0.13         # axes y per row — fixed
    y_top = 0.72
    for idx, p in enumerate(env.placed):
        y = y_top - idx * row_h
        spec = CATALOG[p.fid]
        c_rgb = tuple(v / 255 for v in CAT_COLORS[spec.cat])
        arrow = "↑→↓←"[p.ori]
        ax.add_patch(Rectangle((0.02, y - 0.05), 0.025, 0.10,
                               facecolor=c_rgb, edgecolor="none",
                               transform=ax.transAxes))
        ax.text(0.07, y, f"{idx + 1}.", fontsize=SZ_DETAIL, color="#aaa",
                va="center", family=MONO)
        ax.text(0.12, y, f"{spec.name:<13}", fontsize=SZ_BODY,
                color="#222", va="center", family=MONO)
        ax.text(0.62, y, f"({p.x:>2},{p.y:>2}) {arrow}  {p.fw}×{p.fh}",
                fontsize=SZ_DETAIL, color="#888",
                va="center", family=MONO)


def _draw_reward_details(ax, breakdown):
    """Right middle: per-item / discomfort / waste breakdown. Auto-fills
    the band between PLACEMENTS and the fixed TOTALS box."""
    ax.plot([0.0, 1.0], [0.97, 0.97], color="#ddd", lw=0.8,
            transform=ax.transAxes)
    ax.text(0.0, 0.93, "REWARD", fontsize=SZ_HEADER, weight="bold",
            color="#888", family=MONO, va="top")

    if breakdown is None:
        ax.text(0.0, 0.78, "(computed at episode end)", fontsize=SZ_DETAIL,
                color="#bbb", family=MONO, va="top", style="italic")
        return

    y = 0.78
    ax.text(0.02, y, "Per-item availability", fontsize=SZ_DETAIL, color="#888",
            va="top", family=MONO)
    y -= 0.055
    for name, val in breakdown["per_item"]:
        ax.text(0.04, y, f"{name:<14}", fontsize=SZ_DETAIL, color="#444",
                va="top", family=MONO)
        c = "#2a9" if val > 0 else "#bbb"
        ax.text(1.0, y, f"+{val}", fontsize=SZ_DETAIL, color=c,
                va="top", ha="right", family=MONO)
        y -= 0.045

    y -= 0.025
    ax.text(0.02, y, "Discomfort details", fontsize=SZ_DETAIL, color="#888",
            va="top", family=MONO)
    y -= 0.055
    exp = breakdown["exposed_cells"]; tot = breakdown["total_bed_cells"]
    bes = breakdown["bed_exposure_score"]
    pillow = breakdown["pillow_seen"]; winblk = breakdown["window_blocked"]
    for k, v, contrib in [
        ("bed exposed",  f"{exp}/{tot}",            f"+{bes}"),
        ("pillow seen",  "yes" if pillow else "no", "+4.0" if pillow else "+0.0"),
        ("window block", "yes" if winblk else "no", "+3.0" if winblk else "+0.0"),
    ]:
        ax.text(0.04, y, f"{k:<13}", fontsize=SZ_DETAIL, color="#444",
                va="top", family=MONO)
        ax.text(0.52, y, v, fontsize=SZ_DETAIL, color="#666",
                va="top", family=MONO)
        ax.text(1.0, y, contrib, fontsize=SZ_DETAIL,
                color="#c44" if contrib != "+0.0" else "#bbb",
                va="top", ha="right", family=MONO)
        y -= 0.045

    y -= 0.025
    ax.text(0.02, y, "Waste details", fontsize=SZ_DETAIL, color="#888",
            va="top", family=MONO)
    y -= 0.055
    unr = breakdown["unreachable_cells"]
    ax.text(0.04, y, f"unreachable    {unr} cells × 0.2",
            fontsize=SZ_DETAIL, color="#444", va="top", family=MONO)
    ax.text(1.0, y, f"+{breakdown['waste']}", fontsize=SZ_DETAIL,
            color="#c44" if breakdown["waste"] > 0 else "#bbb",
            va="top", ha="right", family=MONO)


def _draw_totals(ax, breakdown):
    """Right bottom: A / D / W rows + big TOTAL number, all at FIXED y
    positions regardless of how many reward details are above."""
    if breakdown is None:
        return

    # Top divider — sits clear of the rows below.
    ax.plot([0.0, 1.0], [0.96, 0.96], color="#ccc", lw=1.0,
            transform=ax.transAxes)

    rows = [
        ("Availability", f"+{breakdown['availability']}", "#222"),
        ("Discomfort",   f"-{breakdown['discomfort']}",   "#c44"),
        ("Waste",        f"-{breakdown['waste']}",        "#c44"),
    ]
    y_positions = [0.82, 0.68, 0.54]
    for (k, v, c), y in zip(rows, y_positions):
        ax.text(0.02, y, k, fontsize=SZ_BODY, color="#333",
                va="center", weight="bold", family=MONO)
        ax.text(1.0, y, v, fontsize=SZ_BODY, color=c,
                va="center", ha="right", weight="bold", family=MONO)

    # Divider above TOTAL — placed in the empty band between rows and TOTAL
    # so the line never overlaps with text.
    ax.plot([0.0, 1.0], [0.40, 0.40], color="#bbb", lw=1.0,
            transform=ax.transAxes)

    # TOTAL label + big number share a baseline at a fixed y. The big serif
    # number rises about 0.10 axes units above the baseline (28pt ≈ 35 px
    # in a ~245 px tall axes), well below the divider at 0.40.
    BASELINE = 0.16
    ax.text(0.02, BASELINE, "TOTAL", fontsize=SZ_BODY + 2, color="#333",
            va="baseline", weight="bold", family=MONO)
    ax.text(1.0, BASELINE, f"{breakdown['total']:+}", fontsize=SZ_TOTAL,
            color="#111", family=SERIF, weight="bold",
            va="baseline", ha="right")


# ----------------------------------------------------------- episode loop

def rollout(env, policy_fn, label: str):
    frames = []
    obs, info = env.reset()
    frames.append(render_frame(env, 0, env.max_steps, None))
    bd = None
    for step in range(env.max_steps + 1):
        action = policy_fn(obs, env)
        obs, r, term, trunc, info = env.step(action)
        bd = getattr(env, "_last_breakdown", None) if (term or trunc) else None
        frames.append(render_frame(env, step + 1, env.max_steps, bd))
        if term or trunc:
            return frames, r, bd
    return frames, 0.0, bd


# ----------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="MaskablePPO .zip; omit → random agent")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--save", default=None,
                   help="output mp4 path; default: videos/<label>.mp4")
    p.add_argument("--show", action="store_true", help="live matplotlib window")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()

    if args.model:
        policy_fn = model_policy(args.model)
        label = "trained_" + Path(args.model).stem
    else:
        policy_fn = random_policy(args.seed)
        label = "random"

    all_frames = []
    rewards = []
    for ep in range(args.episodes):
        env = MyLittleBedroom(seed=args.seed + ep, max_steps=args.max_steps)
        frames, r, bd = rollout(env, policy_fn, label)
        rewards.append(r)
        all_frames.extend(frames)
        info = env._info()
        bd_str = (f"A={bd['availability']} D={bd['discomfort']} W={bd['waste']}"
                  if bd else "(no breakdown)")
        print(f"[{label}] ep {ep + 1}/{args.episodes}: "
              f"reward={r}  {bd_str}  room={info['room']}")

    print(f"[{label}] mean reward over {len(rewards)} eps = {np.mean(rewards):.2f}")

    save_path = args.save
    if save_path is None and not args.show:
        save_path = f"videos/{label}.mp4"
    if save_path:
        import imageio.v2 as imageio  # mp4 via ffmpeg
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_path, all_frames, fps=args.fps)
        print(f"saved → {save_path}  ({len(all_frames)} frames @ {args.fps} fps)")

    if args.show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        for f in all_frames:
            ax.clear(); ax.axis("off"); ax.imshow(f)
            plt.pause(1.0 / args.fps)
        plt.show()


if __name__ == "__main__":
    main()
