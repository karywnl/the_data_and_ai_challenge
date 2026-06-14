---
title: FitSignal
emoji: 🎯
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# FitSignal — sandbox

Hackathon sandbox/demo for the Redrob "Intelligent Candidate Discovery & Ranking"
challenge. Upload a small candidate sample (`.jsonl` or `.json`, ≤100 candidates)
and the app ranks them end-to-end on CPU for the **Senior AI Engineer — Founding
Team** role.

## Deploy to HuggingFace Spaces

1. Create a new **Gradio** Space (free CPU tier is fine).
2. Upload these **5 files** into the Space root:
   - `app.py`            (this folder)
   - `requirements.txt`  (this folder)
   - `README.md`         (this folder — its YAML header tells HF it's a Gradio app)
   - `common.py`         (from the repo root)
   - `embed.py`          (from the repo root)
3. Optionally also upload `sample_candidates.jsonl` (this folder) so reviewers
   have a ready demo input to try.
4. The Space builds and runs automatically. First launch downloads the embedding
   model (~90 MB), which is fine — the no-network rule only applies to the
   contest ranking step, not this demo.

The ranking logic is identical to the repo's `rank.py`; only the input path
differs (inline embedding instead of a precomputed cache, since the sample is
small).
