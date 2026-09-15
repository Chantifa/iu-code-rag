"""Language-aware chunking of source documents with LangChain text splitters."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_LANGUAGE_MAP: dict[str, Language] = {
    "python": Language.PYTHON,
    "java": Language.JAVA,
    "kotlin": Language.KOTLIN,
    "js": Language.JS,
    "ts": Language.TS,
    "php": Language.PHP,
    "csharp": Language.CSHARP,
    "markdown": Language.MARKDOWN,
    "rst": Language.RST,
    "html": Language.HTML,
}


def _splitter_for(language: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    lang = _LANGUAGE_MAP.get(language)
    if lang is None:
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def chunk_documents(docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Document]:
    """Split documents into chunks.

    Each chunk gets a header line naming the repository and file, so that the
    embedding also captures *where* the code lives (this noticeably helps
    queries like "the heap implementation from the IU algorithms course").
    """
    chunks: list[Document] = []
    splitters: dict[str, RecursiveCharacterTextSplitter] = {}
    for doc in docs:
        language = doc.metadata.get("language", "text")
        splitter = splitters.setdefault(language, _splitter_for(language, chunk_size, chunk_overlap))
        pieces = splitter.split_text(doc.page_content)
        source = doc.metadata["source"]
        header = f"# File: {source}\n"
        for i, piece in enumerate(pieces):
            chunk_id = hashlib.sha1(f"{source}#{i}".encode()).hexdigest()[:16]
            chunks.append(
                Document(
                    page_content=header + piece,
                    metadata={**doc.metadata, "chunk": i, "chunk_id": chunk_id, "n_chunks": len(pieces)},
                )
            )
    return chunks
