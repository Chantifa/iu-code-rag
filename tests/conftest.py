"""Shared fixtures: an in-memory RAG pipeline built from the small real-code corpus."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iu_code_rag.chain import RagPipeline
from iu_code_rag.config import Settings
from iu_code_rag.embeddings import get_embeddings
from iu_code_rag.evaluation import load_golden
from iu_code_rag.github_loader import SourcesConfig, load_documents_from_dir

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Test settings: extractive mode, local embeddings, top_k=5, no index on disk.

    Tests never talk to an LLM unless they are marked ``llm`` and use ``llm_pipeline``.
    """
    return Settings(
        llm_provider="none",
        sources_file=ROOT / "config" / "sources.yaml",
        data_dir=ROOT / "tests" / ".tmp_data",
        top_k=5,
    )


@pytest.fixture(scope="session")
def sources_config(settings) -> SourcesConfig:
    """The real ``config/sources.yaml`` so the tests use the production file filters."""
    return SourcesConfig.load(settings.sources_file)


@pytest.fixture(scope="session")
def documents(settings, sources_config):
    """All fixture files loaded as LangChain Documents (21 real files from both GitHub accounts)."""
    docs = load_documents_from_dir(CORPUS, sources_config, settings.max_file_bytes)
    assert docs, "fixture corpus is empty"
    return docs


@pytest.fixture(scope="session")
def embeddings(settings):
    """The local sentence-transformers embedding model, loaded once per test session."""
    return get_embeddings(settings.embedding_model, settings.embedding_device)


@pytest.fixture(scope="session")
def pipeline(documents, settings) -> RagPipeline:
    """In-memory RAG pipeline over the fixture corpus, built once per test session."""
    return RagPipeline.from_documents(documents, settings)


@pytest.fixture(scope="session")
def golden():
    """The golden queries from ``tests/golden/queries.jsonl``."""
    return load_golden(ROOT / "tests" / "golden" / "queries.jsonl")


@pytest.fixture(scope="session")
def llm_pipeline(documents, settings):
    """Pipeline with a real LLM; skipped unless one is reachable/configured."""
    provider = os.environ.get("LLM_PROVIDER", "none")
    if provider == "none":
        pytest.skip("set LLM_PROVIDER=ollama|anthropic to run LLM tests")
    llm_settings = Settings(
        llm_provider=provider, sources_file=settings.sources_file, data_dir=settings.data_dir, top_k=5
    )
    if provider == "ollama":
        import httpx

        try:
            httpx.get(llm_settings.ollama_base_url + "/api/tags", timeout=3)
        except Exception:  # noqa: BLE001
            pytest.skip("Ollama is not reachable")
    if provider == "anthropic" and not (llm_settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return RagPipeline.from_documents(documents, llm_settings)
