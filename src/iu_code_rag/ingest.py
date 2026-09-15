"""Ingestion: GitHub -> local files -> Documents -> chunks -> FAISS index."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from .chunking import chunk_documents
from .config import Settings
from .embeddings import get_embeddings
from .github_loader import GitHubClient, SourcesConfig, load_documents_from_dir
from .vectorstore import build_vector_store, save_index

log = logging.getLogger(__name__)


def fetch_repositories(
    settings: Settings, only: list[str] | None = None, limit: int | None = None, refresh: bool = False
) -> list[str]:
    """Download the configured repositories. Returns the list of owner/name fetched or reused."""
    config = SourcesConfig.load(settings.sources_file)
    client = GitHubClient(settings.github_token)
    only_set = {o.lower() for o in only} if only else None
    fetched: list[str] = []
    for spec in config.sources:
        for ref in client.list_repos(spec):
            if only_set and ref.full_name.lower() not in only_set:
                continue
            if limit is not None and len(fetched) >= limit:
                return fetched
            dest = settings.repos_dir / ref.owner / ref.name
            if dest.exists() and not refresh:
                log.info("reusing %s", ref.full_name)
                fetched.append(ref.full_name)
                continue
            t0 = time.time()
            n = client.download_repo(ref, dest, config, settings.max_file_bytes)
            log.info("fetched %s: %d files in %.1fs", ref.full_name, n, time.time() - t0)
            fetched.append(ref.full_name)
    return fetched


def build_index(settings: Settings) -> dict:
    config = SourcesConfig.load(settings.sources_file)
    docs = load_documents_from_dir(settings.repos_dir, config, settings.max_file_bytes)
    if not docs:
        raise RuntimeError(f"no documents found under {settings.repos_dir}")
    log.info("loaded %d documents", len(docs))
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    log.info("split into %d chunks; embedding with %s ...", len(chunks), settings.embedding_model)
    t0 = time.time()
    embeddings = get_embeddings(settings.embedding_model, settings.embedding_device)
    vs = build_vector_store(chunks, embeddings)
    repos = sorted({d.metadata["owner"] + "/" + d.metadata["repo"] for d in docs})
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "documents": len(docs),
        "chunks": len(chunks),
        "repositories": repos,
        "embedding_seconds": round(time.time() - t0, 1),
    }
    save_index(vs, chunks, settings.index_dir, meta)
    return meta
