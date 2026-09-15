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
    # Tests never talk to an LLM unless explicitly marked; they always use local embeddings.
    return Settings(
        llm_provider="none",
        sources_file=ROOT / "config" / "sources.yaml",
        data_dir=ROOT / "tests" / ".tmp_data",
        top_k=5,
    )


@pytest.fixture(scope="session")
def sources_config(settings) -> SourcesConfig:
    return SourcesConfig.load(settings.sources_file)


@pytest.fixture(scope="session")
def documents(settings, sources_config):
    docs = load_documents_from_dir(CORPUS, sources_config, settings.max_file_bytes)
    assert docs, "fixture corpus is empty"
    return docs


@pytest.fixture(scope="session")
def embeddings(settings):
    return get_embeddings(settings.embedding_model, settings.embedding_device)


@pytest.fixture(scope="session")
def pipeline(documents, settings) -> RagPipeline:
    return RagPipeline.from_documents(documents, settings)


@pytest.fixture(scope="session")
def golden():
    return load_golden()


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
