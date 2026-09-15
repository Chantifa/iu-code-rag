from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from iu_code_rag.chunking import chunk_documents
from iu_code_rag.github_loader import load_documents_from_dir, notebook_to_text
from iu_code_rag.retriever import code_tokenize


def test_loader_reads_both_owners(documents):
    """The fixture corpus yields documents from both owners with language and GitHub URL set."""
    owners = {d.metadata["owner"] for d in documents}
    assert owners == {"Chantifa", "iubh"}
    langs = {d.metadata["language"] for d in documents}
    assert {"python", "js", "markdown"} <= langs
    for d in documents:
        assert d.metadata["url"].startswith("https://github.com/")
        assert d.metadata["source"].count("/") >= 2


def test_sources_config_filters(sources_config):
    """Include-extension and exclude-glob rules from sources.yaml behave as documented."""
    assert sources_config.accepts("src/main.py")
    assert sources_config.accepts("README.md")
    assert not sources_config.accepts("node_modules/x/index.js")
    assert not sources_config.accepts("assets/logo.png")
    assert not sources_config.accepts("package-lock.json")
    # committed virtual environments must never be indexed
    assert not sources_config.accepts("DLBDSMLSL01/lib/python3.12/site-packages/scipy/signal/_filter_design.py")
    assert not sources_config.accepts(".venv/Lib/site-packages/numpy/__init__.py")
    assert not sources_config.accepts("venv/bin/activate_this.py")
    assert sources_config.accepts("src/env/config.py") is False  # any folder literally named env
    assert sources_config.accepts("src/environment/config.py")


def test_chunks_carry_header_and_metadata(documents):
    """Every chunk starts with its ``# File:`` header, respects the size limit and has a unique id."""
    chunks = chunk_documents(documents, chunk_size=600, chunk_overlap=50)
    assert len(chunks) > len(documents)
    for c in chunks:
        assert c.page_content.startswith("# File: " + c.metadata["source"])
        assert "chunk_id" in c.metadata and len(c.metadata["chunk_id"]) == 16
        assert len(c.page_content) <= 600 + len("# File: " + c.metadata["source"]) + 1 + 50
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_python_chunks_split_on_definitions():
    """The Python-aware splitter cuts at function definitions instead of mid-function."""
    code = "\n\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(40))
    doc = Document(page_content=code, metadata={"source": "a/b/c.py", "language": "python"})
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=0)
    assert len(chunks) > 3
    assert all("def f" in c.page_content for c in chunks)


def test_notebook_to_text():
    """Notebook cells become markdown plus fenced code blocks; empty cells are dropped."""
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "Some text"]},
            {"cell_type": "code", "source": "import numpy as np\nnp.zeros(3)"},
            {"cell_type": "code", "source": ""},
        ]
    }
    text = notebook_to_text(json.dumps(nb))
    assert "# Title" in text
    assert "```python\nimport numpy as np" in text
    assert text.count("```python") == 1


def test_binary_and_oversized_files_are_skipped(tmp_path: Path, sources_config):
    """Files containing NUL bytes or exceeding max_file_bytes are not loaded."""
    repo = tmp_path / "owner" / "repo"
    repo.mkdir(parents=True)
    (repo / "ok.py").write_text("print('hi')\n", encoding="utf-8")
    (repo / "bin.py").write_bytes(b"\x00\x01\x02")
    (repo / "big.py").write_text("x = 1\n" * 100, encoding="utf-8")
    docs = load_documents_from_dir(tmp_path, sources_config, max_file_bytes=100)
    assert [d.metadata["path"] for d in docs] == ["ok.py"]


def test_code_tokenizer_splits_identifiers():
    """The BM25 tokenizer emits the whole identifier plus its camelCase / snake_case parts."""
    toks = code_tokenize("KElbowVisualizer max_requests isAllowed")
    assert {"kelbowvisualizer", "k", "elbow", "visualizer", "max", "requests", "is", "allowed"} <= set(toks)
