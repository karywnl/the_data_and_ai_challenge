#!/usr/bin/env python3
"""HuggingFace Spaces sandbox for FitSignal.

Upload a small JSONL/JSON candidate sample (<=100 candidates) and the app ranks
it end-to-end on CPU, exactly like rank.py but embedding inline (no precomputed
cache needed for a small sample). Satisfies the hackathon sandbox requirement.

Deploy: create a Gradio Space, copy this folder's files (app.py, requirements.txt)
plus common.py and embed.py from the repo root into the Space.
"""
from __future__ import annotations

import json
import tempfile

import gradio as gr
import numpy as np

from common import (
    build_doc, extract_features, final_score, make_reasoning,
    JD_POSITIVE_FACETS, JD_NEGATIVE_FACETS,
)
from embed import embed_texts


def _load(file_obj):
    text = open(file_obj.name, "r", encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def rank(file_obj, topk):
    cands = _load(file_obj)
    if not cands:
        return "No candidates found.", None
    topk = int(min(topk, len(cands)))

    feats = [extract_features(c) for c in cands]
    docs = [build_doc(c) for c in cands]
    emb = embed_texts(docs)
    pos = emb @ embed_texts(JD_POSITIVE_FACETS).T
    neg = emb @ embed_texts(JD_NEGATIVE_FACETS).T
    sem = np.sort(pos, axis=1)[:, -2:].mean(axis=1) - 0.6 * neg.max(axis=1)
    lo, hi = np.percentile(sem, 5), np.percentile(sem, 99)
    sem_n = np.clip((sem - lo) / max(hi - lo, 1e-6), 0, 1)

    scored = [(final_score(f, float(sem_n[i])), f, float(sem_n[i]))
              for i, f in enumerate(feats)]
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    rows = [[r + 1, f["candidate_id"], round(s, 4), make_reasoning(f, sn)]
            for r, (s, f, sn) in enumerate(scored[:topk])]

    out = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    out.write("candidate_id,rank,score,reasoning\n")
    for rank_, cid, score, reason in rows:
        reason = reason.replace('"', "'")
        out.write(f'{cid},{rank_},{score},"{reason}"\n')
    out.close()
    return rows, out.name


demo = gr.Interface(
    fn=rank,
    inputs=[gr.File(label="Candidate sample (.jsonl or .json, <=100)"),
            gr.Slider(1, 100, value=10, step=1, label="Top-K")],
    outputs=[gr.Dataframe(headers=["rank", "candidate_id", "score", "reasoning"],
                          label="Ranking"),
             gr.File(label="Download CSV")],
    title="FitSignal — Senior AI Engineer Candidate Ranker",
    description="Ranks candidates by reading title + career history + behavioral "
                "signals (skills are ignored — they're noise by design). CPU-only, "
                "no LLM calls.",
)

if __name__ == "__main__":
    demo.launch()
