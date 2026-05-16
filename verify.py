"""Verify env.py matches my_little_bedroom.html.

For each case below the script prints (a) exact HTML setup steps and
(b) the env's computed Availability / Discomfort / Waste. Open the HTML
in a browser, replicate the setup, click each reward row to highlight,
and compare the three numbers.

Run:
    python verify.py              # run all cases
    python verify.py --case 2     # just one case

Catalog index reference (fid):
   0 Bed 0.9     4 Desk S       7 Wardrobe S    12 Cabinet S    16 Nightstand A
   1 Bed 1.2     5 Desk L       8 Wardrobe M    13 Cabinet M    17 Nightstand B
   2 Bed 1.5     6 Desk XL      9 Wardrobe L    14 Cabinet L
   3 Bed 1.8                   10 Wardrobe XL   15 Cabinet XL
                               11 Wardrobe XXL

Orientation (matches HTML wheel-scroll / arrow buttons):
    ori 0  head/face → LEFT     ori 2  → RIGHT
    ori 1            → TOP      ori 3  → BOTTOM
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from env import ACTION_DONE, CATALOG, MyLittleBedroom, encode_action


@dataclass
class Case:
    name: str
    description: str
    room_w: int
    room_h: int
    door_pos: int
    win_wall: str
    placements: list[tuple[int, int, int, int]]  # (fid, x, y, ori)


CASES: list[Case] = [
    Case(
        name="empty room",
        description="No furniture. Expect A=0, D=0, W=0 (waste only counts when ≥1 item).",
        room_w=20, room_h=20, door_pos=7, win_wall="top",
        placements=[],
    ),
    Case(
        name="bed + paired nightstand",
        description=(
            "Bed 1.2 horizontal head-LEFT at (3,1); Nightstand A above the "
            "headboard at (3,-2)? No — must stay in room. Place NS at (3,9) "
            "(below bed, aligned with headboard column x=3)."
        ),
        room_w=20, room_h=20, door_pos=7, win_wall="top",
        placements=[
            (1,  3, 1, 0),   # Bed 1.2  (14×8) head-left
            (16, 3, 9, 0),   # Nightstand A (4×3) right below headboard
        ],
    ),
    Case(
        name="wardrobe blocking the window",
        description=(
            "Wardrobe M (8×4) at top wall y=0..3. Window is centered on top "
            "wall (50% of room_w). Expect window penalty +3 in Discomfort."
        ),
        room_w=20, room_h=20, door_pos=7, win_wall="top",
        placements=[
            (8, 6, 0, 0),    # Wardrobe M (8×4) covering window strip
            (1, 3, 5, 0),    # Bed 1.2 elsewhere, head-left
        ],
    ),
    Case(
        name="bed exposed to door (no wardrobe shield)",
        description=(
            "Bed 1.2 in upper area with head on RIGHT, pillow column inside "
            "door's 90° cone. Expect non-zero bed exposure and pillow-seen +4."
        ),
        room_w=20, room_h=20, door_pos=7, win_wall="left",
        placements=[
            (1, 3, 2, 2),    # Bed 1.2 head-right; pillow at x=16
        ],
    ),
]


def run_case(case: Case) -> None:
    print(f"\n=== {case.name} ===")
    print(case.description)
    print(f"HTML setup: Room {case.room_w}×{case.room_h} · "
          f"Door slider → {case.door_pos} · Window button → {case.win_wall}")
    arrows = ["↑", "→", "↓", "←"]   # matches HTML's ori button row
    for fid, x, y, ori in case.placements:
        spec = CATALOG[fid]
        print(f"  place fid={fid:>2} {spec.name:<14} at ({x:>2},{y:>2}) "
              f"ori={ori} ({arrows[ori]} in HTML)")

    env = MyLittleBedroom()
    obs, info = env.reset(options={"config": {
        "room_w": case.room_w, "room_h": case.room_h,
        "door_pos": case.door_pos, "win_wall": case.win_wall,
    }})
    for fid, x, y, ori in case.placements:
        a = encode_action(fid, x, y, ori)
        mask = env.action_masks()
        if not mask[a]:
            print(f"  ! action ({fid},{x},{y},{ori}) is masked invalid — "
                  f"check overlap with door swing or out of bounds")
            return
        env.step(a)
    env.step(ACTION_DONE)
    bd = env._last_breakdown
    print(f"env result:  A={bd['availability']:<5}  D={bd['discomfort']:<5}  "
          f"W={bd['waste']:<5}  total={bd['total']}")
    print(f"             per_item={bd['per_item']}")
    print(f"             bed_exposed={bd['exposed_cells']}/{bd['total_bed_cells']} "
          f"pillow_seen={bd['pillow_seen']} unreachable={bd['unreachable_cells']}")
    print("verify in HTML: click each reward row (Availability / Discomfort / "
          "Waste) — yellow highlights should match the counts above.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, default=None,
                   help="1-indexed case number; omit to run all")
    args = p.parse_args()
    cases = CASES if args.case is None else [CASES[args.case - 1]]
    for c in cases:
        run_case(c)


if __name__ == "__main__":
    main()
