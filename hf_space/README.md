---
title: My Little Bedroom
emoji: 🏠
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: An RL agent that furnishes a random bedroom from scratch.
---

# 🏠 My Little Bedroom — live demo

A [MaskablePPO](https://sb3-contrib.readthedocs.io/) agent that furnishes a
randomly generated bedroom one piece of furniture at a time, trained only on a
single design score (no example layouts). Click **Furnish a new room** to watch
it lay out a brand-new room and see the score broken down.

Trained with `sb3-contrib` (2M steps, 20 parallel rooms). The environment,
action masking and rendering here are the same code used for training and
evaluation, so the demo matches the trained model exactly.
