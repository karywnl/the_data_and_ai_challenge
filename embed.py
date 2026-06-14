#!/usr/bin/env python3
"""Embedding helper. CPU sentence-transformer (all-MiniLM-L6-v2): small, fast,
384-dim. Used offline in precompute.py; also used as an inline fallback in
rank.py for the small sandbox sample when no precomputed cache is present."""
from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed_texts(texts, batch_size=256, show_progress=False):
    import numpy as np
    model = get_model()
    emb = model.encode(
        list(texts), batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=show_progress,
    )
    return emb.astype(np.float32)
