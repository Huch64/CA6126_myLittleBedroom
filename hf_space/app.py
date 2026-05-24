"""
app.py — Hugging Face Spaces demo for "My Little Bedroom".

Loads the trained MaskablePPO agent once at startup, then lets a visitor
generate a brand-new random room and watch the agent furnish it step by step,
with the final design's score broken down by what the agent was rewarded for.

The heavy lifting (env dynamics, action masking, matplotlib rendering) is
reused verbatim from the research code (env.py / render.py) so the demo always
matches the trained model exactly.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering on the Space

import random
import tempfile

import gradio as gr
import imageio.v2 as imageio

from env import MyLittleBedroom
from render import model_policy, rollout

MODEL_PATH = "model.zip"

# Load the policy a single time — ~4s — when the Space boots, not per click.
_policy = model_policy(MODEL_PATH)


def _score_markdown(reward: float, seed: int, bd: dict | None) -> str:
    """Format the final reward and its multiplicative breakdown for display."""
    if not bd:
        return f"### Score: {reward:.1f}"
    lines = [
        f"### 🏆 Score: {reward:.1f}  \n<sub>random room · seed {seed}</sub>",
        "",
        "The score is **multiplicative** — a strong room has to do well on "
        "*every* term at once, not just pile on furniture:",
        "",
        "| term | value | what it rewards |",
        "|---|---|---|",
        f"| availability | {bd.get('availability', 0):.1f} | useful furniture placed |",
        f"| × privacy | {bd.get('privacy', 1):.2f} | bed hidden from the door |",
        f"| × light | {bd.get('light', 1):.2f} | window left unblocked |",
        f"| × efficiency | {bd.get('efficiency', 1):.2f} | floor stays walkable |",
        f"| + diversity | {bd.get('diversity', 0):.1f} | variety of furniture types |",
        f"| + compactness | {bd.get('compactness', 0):.1f} | tidy use of leftover space |",
    ]
    return "\n".join(lines)


def furnish(seed):
    """Run one episode of the trained agent and return (process gif, score)."""
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 99_999)
    seed = int(seed)

    env = MyLittleBedroom(seed=seed, max_steps=8)
    frames, reward, bd = rollout(env, _policy, "demo")

    # Linger on the finished room so the gif doesn't snap back to empty.
    seq = frames + [frames[-1]] * 4
    out_path = os.path.join(tempfile.gettempdir(), f"room_{seed}.gif")
    imageio.mimsave(out_path, seq, fps=1.5, loop=0)

    return out_path, _score_markdown(reward, seed, bd)


INTRO = """
# 🏠 My Little Bedroom

A reinforcement-learning agent that **furnishes a bedroom**. Every room below is
brand new — random shape, door and window — so the agent can't memorise layouts.
One piece at a time it picks *what* furniture to place, *where*, and *which way
it faces* (out of ~41,000 moves each step), aiming for a single design score.

Nobody told it the rules of interior design. Keeping the bed private from the
door, leaving the window unblocked, and the floor walkable are habits it picked
up from the reward alone. **Click below to watch it furnish a fresh room.**
"""

with gr.Blocks(title="My Little Bedroom", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=2):
            out_img = gr.Image(label="The agent furnishing the room", type="filepath")
        with gr.Column(scale=1):
            seed_in = gr.Number(
                label="Room seed (−1 = random)", value=-1, precision=0
            )
            go = gr.Button("🪑 Furnish a new room", variant="primary")
            score_md = gr.Markdown()

    go.click(furnish, inputs=seed_in, outputs=[out_img, score_md])
    # Furnish one room on load so the page is never empty.
    demo.load(furnish, inputs=seed_in, outputs=[out_img, score_md])

if __name__ == "__main__":
    demo.launch()
