"""FAISS vector index persistence plus the chunk corpus needed for BM25."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("no chunks to index")
    return FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


def save_index(vs: FAISS, chunks: list[Document], index_dir: Path, meta: dict | None = None) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    (index_dir / META_FILE).write_text(json.dumps(meta or {}, indent=2), encoding="utf-8")
    log.info("saved index with %d chunks to %s", len(chunks), index_dir)


def load_chunks(index_dir: Path) -> list[Document]:
    path = Path(index_dir) / CHUNKS_FILE
    docs: list[Document] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                docs.append(Document(page_content=rec["page_content"], metadata=rec["metadata"]))
    return docs


def load_index(index_dir: Path, embeddings: Embeddings) -> tuple[FAISS, list[Document], dict]:
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"no index at {index_dir}; run `iu-rag ingest` first")
    vs = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # the pickle was written by this very application
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    chunks = load_chunks(index_dir)
    meta_path = index_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return vs, chunks, meta
