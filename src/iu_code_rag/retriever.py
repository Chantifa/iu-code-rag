"""Hybrid retriever: dense (FAISS, cosine) + sparse (BM25), fused with weighted RRF."""

from __future__ import annotations

import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def code_tokenize(text: str) -> list[str]:
    """Tokenizer for BM25 that also splits camelCase / snake_case identifiers."""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        tokens.append(tok.lower())
        # camelCase -> camel case ; KElbow -> K Elbow ; snake_case -> snake case
        sub = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
        sub = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", sub)
        sub = sub.replace("_", " ").lower().split()
        if len(sub) > 1:
            tokens.extend(sub)
    return tokens


class HybridRetriever(BaseRetriever):
    """Reciprocal-rank fusion of a FAISS similarity search and a BM25 keyword search."""

    vector_store: Any
    bm25: Any
    k: int = 6
    vector_weight: float = 0.6
    candidate_multiplier: int = 3
    rrf_k: int = 60

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_chunks(
        cls, vector_store: FAISS, chunks: list[Document], k: int = 6, vector_weight: float = 0.6
    ) -> HybridRetriever:
        bm25 = BM25Retriever.from_documents(chunks, preprocess_func=code_tokenize, k=k * 3)
        return cls(vector_store=vector_store, bm25=bm25, k=k, vector_weight=vector_weight)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        n = self.k * self.candidate_multiplier
        dense = self.vector_store.similarity_search_with_score(query, k=n)
        self.bm25.k = n
        sparse = self.bm25.invoke(query)

        fused: dict[str, float] = {}
        docs: dict[str, Document] = {}
        cosine: dict[str, float] = {}

        for rank, (doc, score) in enumerate(dense):
            cid = doc.metadata.get("chunk_id", doc.page_content[:64])
            docs[cid] = doc
            cosine[cid] = float(score)
            fused[cid] = fused.get(cid, 0.0) + self.vector_weight / (self.rrf_k + rank + 1)
        for rank, doc in enumerate(sparse):
            cid = doc.metadata.get("chunk_id", doc.page_content[:64])
            docs.setdefault(cid, doc)
            fused[cid] = fused.get(cid, 0.0) + (1.0 - self.vector_weight) / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.k]
        out: list[Document] = []
        for cid, score in ranked:
            doc = docs[cid]
            meta = {**doc.metadata, "score": round(score, 6)}
            if cid in cosine:
                meta["cosine"] = round(cosine[cid], 4)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out
