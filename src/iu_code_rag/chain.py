"""The RAG pipeline: retriever -> prompt -> LLM, built with LangChain (LCEL)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .chunking import chunk_documents
from .config import Settings
from .embeddings import get_embeddings
from .retriever import HybridRetriever
from .vectorstore import build_vector_store, load_index

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer answering questions about a collection of GitHub
repositories: the personal projects of the developer "Chantifa" and the course material of the
IU International University of Applied Sciences (GitHub organisation "iubh").

Answer using ONLY the context snippets below. Every snippet starts with a line "# File: owner/repo/path".
When you use a snippet, cite it in square brackets like [owner/repo/path].
If the context does not contain the answer, say that you could not find it in the indexed repositories.
Answer in the language of the question. Keep code excerpts short and accurate."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class SourceHit:
    source: str
    url: str
    score: float
    snippet: str
    cosine: float | None = None
    chunk: int = 0


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[SourceHit] = field(default_factory=list)
    provider: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "provider": self.provider,
            "sources": [s.__dict__ for s in self.sources],
        }


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _extractive_answer(inputs: dict) -> str:
    """Fallback "generator" when no LLM is configured: return the best snippets verbatim."""
    docs: list[Document] = inputs["docs"]
    if not docs:
        return "No relevant code was found in the indexed repositories."
    lines = ["No LLM is configured (LLM_PROVIDER=none), so here are the most relevant snippets:", ""]
    for i, d in enumerate(docs, 1):
        body = d.page_content.split("\n", 1)[1] if "\n" in d.page_content else d.page_content
        lines.append(f"[{i}] {d.metadata['source']}")
        lines.append("```")
        lines.append(body.strip()[:700])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build_llm(settings: Settings) -> Runnable | None:
    """Return a LangChain chat model for the configured provider, or None for extractive mode."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_predict=settings.llm_max_tokens,
        )
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {"model": settings.anthropic_model, "max_tokens": settings.llm_max_tokens}
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key
        return ChatAnthropic(**kwargs)
    return None


def build_generation_chain(llm: Runnable | None) -> Runnable:
    """Input: {"context": str, "question": str, "docs": list[Document]} -> answer string."""
    if llm is None:
        return RunnableLambda(_extractive_answer)
    return PROMPT | llm | StrOutputParser()


class RagPipeline:
    def __init__(self, retriever: HybridRetriever, settings: Settings, index_meta: dict | None = None):
        self.retriever = retriever
        self.settings = settings
        self.index_meta = index_meta or {}
        self.llm = build_llm(settings)
        self.generate = build_generation_chain(self.llm)
        self.chain = (
            RunnableLambda(lambda q: {"question": q, "docs": retriever.invoke(q)})
            | RunnableLambda(lambda x: {**x, "context": format_docs(x["docs"])})
            | RunnableLambda(lambda x: {**x, "answer": self.generate.invoke(x)})
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_index(cls, settings: Settings) -> RagPipeline:
        embeddings = get_embeddings(settings.embedding_model, settings.embedding_device)
        vs, chunks, meta = load_index(settings.index_dir, embeddings)
        retriever = HybridRetriever.from_chunks(vs, chunks, k=settings.top_k, vector_weight=settings.vector_weight)
        return cls(retriever, settings, meta)

    @classmethod
    def from_documents(cls, docs: list[Document], settings: Settings) -> RagPipeline:
        """Build an in-memory pipeline (used by the tests and by `--fixtures` evaluation)."""
        embeddings = get_embeddings(settings.embedding_model, settings.embedding_device)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        vs = build_vector_store(chunks, embeddings)
        retriever = HybridRetriever.from_chunks(vs, chunks, k=settings.top_k, vector_weight=settings.vector_weight)
        return cls(retriever, settings, {"chunks": len(chunks), "documents": len(docs)})

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[Document]:
        if k is not None:
            self.retriever.k = k
        return self.retriever.invoke(query)

    def ask(self, question: str, k: int | None = None) -> RagAnswer:
        if k is not None:
            self.retriever.k = k
        result = self.chain.invoke(question)
        sources = [
            SourceHit(
                source=d.metadata["source"],
                url=d.metadata.get("url", ""),
                score=float(d.metadata.get("score", 0.0)),
                cosine=d.metadata.get("cosine"),
                chunk=int(d.metadata.get("chunk", 0)),
                snippet=d.page_content[:400],
            )
            for d in result["docs"]
        ]
        return RagAnswer(
            question=question, answer=result["answer"], sources=sources, provider=self.settings.llm_provider
        )


def index_exists(index_dir: Path) -> bool:
    return (Path(index_dir) / "index.faiss").exists()
