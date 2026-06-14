#!/usr/bin/env python3
"""Streamlit sandbox for FitSignal.

Upload a small JSONL/JSON candidate sample (<=100 candidates) and the app ranks
it end-to-end on CPU, exactly like rank.py but embedding inline (no precomputed
cache needed for a small sample). Satisfies the hackathon sandbox requirement.

Streamlit is used instead of Gradio to avoid Gradio's heavy dependency chain
(pydub/audioop, huggingface_hub, gradio_client schema bugs) on hosted runtimes.

Deploy: create a Streamlit Space and copy this folder's files (app.py,
requirements.txt) plus common.py and embed.py from the repo root into the Space.
"""
from __future__ import annotations

import json
import io

import numpy as np
import streamlit as st

from common import (
    build_doc, extract_features, final_score, make_reasoning,
    JD_POSITIVE_FACETS, JD_NEGATIVE_FACETS,
)
from embed import embed_texts

st.set_page_config(page_title="FitSignal", page_icon="🎯", layout="wide")
st.title("🎯 FitSignal — Senior AI Engineer Candidate Ranker")
st.caption(
    "Ranks candidates by reading title + career history + behavioral signals "
    "(skills are ignored — they're noise by design). CPU-only, no LLM calls."
)


def load(raw_bytes):
    text = raw_bytes.decode("utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def rank(cands, topk):
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
    rows = [{"rank": r + 1, "candidate_id": f["candidate_id"],
             "score": round(s, 4), "reasoning": make_reasoning(f, sn)}
            for r, (s, f, sn) in enumerate(scored[:topk])]
    return rows


uploaded = st.file_uploader(
    "Candidate sample (.jsonl or .json, up to 100 candidates)",
    type=["jsonl", "json"])
topk = st.slider("How many to rank (Top-K)", 1, 100, 10)

if uploaded is not None:
    cands = load(uploaded.read())
    st.write(f"Loaded **{len(cands)}** candidates.")
    if st.button("Rank candidates", type="primary"):
        with st.spinner("Embedding and ranking on CPU…"):
            rows = rank(cands, min(topk, len(cands)))
        st.success(f"Ranked top {len(rows)}.")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        buf.write("candidate_id,rank,score,reasoning\n")
        for r in rows:
            reason = r["reasoning"].replace('"', "'")
            buf.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{reason}"\n')
        st.download_button("Download CSV", buf.getvalue(),
                           file_name="ranking.csv", mime="text/csv")
else:
    st.info("Upload a candidate sample to begin. "
            "A ready demo file (sample_candidates.jsonl) is in the repo's "
            "sandbox/ folder.")
