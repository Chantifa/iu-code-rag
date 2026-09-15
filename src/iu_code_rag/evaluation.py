"""Golden-set evaluation: retrieval hit@k / MRR and answer similarity to golden answers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .chain import RagPipeline
from .embeddings import cosine_similarity, get_embeddings

_TESTS_REL = Path("tests")


def _find_tests_dir() -> Path:
    """Locate the ``tests/`` folder whether the package is installed editable or into site-packages.

    Order: ``$IU_RAG_TESTS_DIR``, ``<cwd>/tests`` (the Docker image runs from /app), then the
    source checkout next to this package.
    """
    env = os.environ.get("IU_RAG_TESTS_DIR")
    candidates = [Path(env)] if env else []
    candidates += [Path.cwd() / _TESTS_REL, Path(__file__).resolve().parents[2] / _TESTS_REL]
    for c in candidates:
        if (c / "golden" / "queries.jsonl").exists():
            return c
    raise FileNotFoundError("tests/golden/queries.jsonl not found; set IU_RAG_TESTS_DIR or run from the repo root")


def default_golden_file() -> Path:
    """Path of the default golden query set, ``tests/golden/queries.jsonl``."""
    return _find_tests_dir() / "golden" / "queries.jsonl"


def default_fixture_corpus() -> Path:
    """Path of the small real-code corpus used by the tests, ``tests/fixtures/corpus``."""
    return _find_tests_dir() / "fixtures" / "corpus"


@dataclass
class GoldenQuery:
    """One entry of the golden set.

    ``expected_sources`` are substrings that must occur in a retrieved ``owner/repo/path``,
    ``expected_keywords`` must appear in the answer text, ``golden_answer`` is the reference
    answer used for similarity scoring and ``paraphrase`` a differently worded question.
    """

    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    golden_answer: str = ""
    paraphrase: str | None = None
    tags: list[str] = field(default_factory=list)


def load_golden(path: Path | None = None) -> list[GoldenQuery]:
    """Read golden queries from a JSON-lines file (blank lines and ``#`` comments are ignored)."""
    path = Path(path) if path else default_golden_file()
    out: list[GoldenQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(GoldenQuery(**json.loads(line)))
    return out


def source_matches(source: str, expected: list[str]) -> bool:
    """True if any expected substring occurs in ``source`` (case-insensitive)."""
    return any(e.lower() in source.lower() for e in expected)


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first source matching ``expected``, or ``None`` if none matched."""
    for i, s in enumerate(sources, 1):
        if source_matches(s, expected):
            return i
    return None


def evaluate(pipeline: RagPipeline, golden: list[GoldenQuery], k: int = 5, with_answers: bool = True) -> dict:
    """Run every golden query through the pipeline and compute retrieval and answer metrics.

    Returns ``{"summary": {...}, "results": [...]}`` with hit rate and MRR at ``k``. With
    ``with_answers`` it also records the missing keywords per query, the cosine similarity
    between the generated and the golden answer, and the mean of those similarities.
    """
    emb = get_embeddings(pipeline.settings.embedding_model, pipeline.settings.embedding_device)
    rows = []
    for g in golden:
        docs = pipeline.search(g.question, k=k)
        sources = [d.metadata["source"] for d in docs]
        rank = first_hit_rank(sources, g.expected_sources)
        row = {
            "id": g.id,
            "question": g.question,
            "hit": rank is not None,
            "rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "top_sources": sources,
        }
        if with_answers:
            ans = pipeline.ask(g.question, k=k)
            row["answer"] = ans.answer
            missing = [kw for kw in g.expected_keywords if kw.lower() not in ans.answer.lower()]
            row["missing_keywords"] = missing
            if g.golden_answer:
                row["answer_similarity"] = round(
                    cosine_similarity(emb.embed_query(ans.answer), emb.embed_query(g.golden_answer)), 4
                )
        rows.append(row)

    n = len(rows) or 1
    summary = {
        "k": k,
        "queries": len(rows),
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        "provider": pipeline.settings.llm_provider,
    }
    if with_answers:
        sims = [r["answer_similarity"] for r in rows if "answer_similarity" in r]
        summary["mean_answer_similarity"] = round(sum(sims) / len(sims), 4) if sims else None
        summary["keyword_coverage"] = round(sum(1 for r in rows if not r["missing_keywords"]) / n, 4)
    return {"summary": summary, "results": rows}
