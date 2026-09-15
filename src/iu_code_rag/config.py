"""Runtime configuration, read from environment variables / a `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- sources -------------------------------------------------------------
    github_token: str | None = Field(default=None, description="Optional GitHub token (raises API rate limits).")
    sources_file: Path = Field(default=Path("config/sources.yaml"))
    data_dir: Path = Field(default=Path("data"), description="Where cloned repos and the index live.")

    # --- embeddings ----------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model, run locally on CPU.",
    )
    embedding_device: str = "cpu"

    # --- chunking / retrieval ------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_file_bytes: int = 200_000
    top_k: int = 6
    vector_weight: float = Field(
        default=0.6, description="Weight of dense retrieval in the hybrid fusion (BM25 gets 1 - this)."
    )

    # --- generation ----------------------------------------------------------
    llm_provider: Literal["ollama", "anthropic", "none"] = "none"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:1.5b"
    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None
    llm_max_tokens: int = 2048

    # --- api -----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


def get_settings() -> Settings:
    return Settings()
