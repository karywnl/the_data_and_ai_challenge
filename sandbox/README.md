---
title: FitSignal
emoji: 🎯
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# FitSignal — sandbox

Hackathon sandbox/demo for the Redrob "Intelligent Candidate Discovery & Ranking"
challenge. Upload a small candidate sample (`.jsonl` or `.json`, ≤100 candidates)
and the app ranks them end-to-end on CPU for the **Senior AI Engineer — Founding
Team** role.

## Deploy to HuggingFace Spaces

1. Create a new **Gradio** Space.
2. Upload these files into the Space root:
   - `app.py`, `requirements.txt` (this folder)
   - `common.py`, `embed.py` (from the repo root)
3. The Space builds and runs automatically.

The ranking logic is identical to the repo's `rank.py`; only the input path
differs (inline embedding instead of a precomputed cache, since the sample is
small).
