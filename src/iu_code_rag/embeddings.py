"""Local embeddings via sentence-transformers (no API key, runs on CPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """Return a cached, L2-normalised HuggingFace embedding model.

    Because the vectors are normalised, the inner product equals the cosine
    similarity, which is what the FAISS index and the similarity tests use.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
