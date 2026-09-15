"""Golden-set evaluation: retrieval hit@k / MRR and answer similarity to golden answers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .chain import RagPipeline
from .embeddings import cosine_similarity, get_embeddings

GOLDEN_FILE = Path(__file__).resolve().parents[2] / "tests" / "golden" / "queries.jsonl"


@dataclass
class GoldenQuery:
    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    golden_answer: str = ""
    paraphrase: str | None = None
    tags: list[str] = field(default_factory=list)


def load_golden(path: Path = GOLDEN_FILE) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(GoldenQuery(**json.loads(line)))
    return out


def source_matches(source: str, expected: list[str]) -> bool:
    return any(e.lower() in source.lower() for e in expected)


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    for i, s in enumerate(sources, 1):
        if source_matches(s, expected):
            return i
    return None


def evaluate(pipeline: RagPipeline, golden: list[GoldenQuery], k: int = 5, with_answers: bool = True) -> dict:
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
