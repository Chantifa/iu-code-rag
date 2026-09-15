"""Golden tests: every golden question must retrieve its known source file.

The golden set lives in tests/golden/queries.jsonl. Each entry names the file
that contains the answer; the hybrid retriever must return a chunk of that file
in the top-k results. Aggregate metrics (hit rate, MRR) guard against slow
regressions that a single per-query test would not catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iu_code_rag.evaluation import evaluate, first_hit_rank, load_golden

GOLDEN = load_golden(Path(__file__).parent / "golden" / "queries.jsonl")
K = 5


@pytest.mark.parametrize("g", GOLDEN, ids=[g.id for g in GOLDEN])
def test_golden_source_is_retrieved(pipeline, g):
    """Golden test: the expected file appears in the top-k results for the question."""
    docs = pipeline.search(g.question, k=K)
    sources = [d.metadata["source"] for d in docs]
    rank = first_hit_rank(sources, g.expected_sources)
    assert rank is not None, f"expected one of {g.expected_sources} in top-{K}, got {sources}"


@pytest.mark.parametrize("g", [g for g in GOLDEN if g.paraphrase], ids=[g.id for g in GOLDEN if g.paraphrase])
def test_paraphrase_is_retrieved(pipeline, g):
    """Golden test: the expected file is also found for the paraphrased question."""
    docs = pipeline.search(g.paraphrase, k=K)
    sources = [d.metadata["source"] for d in docs]
    assert first_hit_rank(sources, g.expected_sources) is not None, f"paraphrase missed: {sources}"


@pytest.mark.parametrize("g", GOLDEN, ids=[g.id for g in GOLDEN])
def test_extractive_answer_contains_expected_keywords(pipeline, g):
    """With LLM_PROVIDER=none the answer is the retrieved context itself, so the golden
    keywords must be present in it - this verifies the *right chunk* (not only the right
    file) was retrieved."""
    answer = pipeline.ask(g.question, k=K).answer.lower()
    missing = [kw for kw in g.expected_keywords if kw.lower() not in answer]
    assert not missing, f"keywords {missing} not in retrieved context"


def test_aggregate_retrieval_metrics(pipeline):
    """Hit rate and MRR over the whole golden set must stay above their floors."""
    report = evaluate(pipeline, GOLDEN, k=K, with_answers=False)
    summary = report["summary"]
    assert summary["hit_rate"] >= 0.9, summary
    assert summary["mrr"] >= 0.7, summary


def test_results_are_ranked_and_deduplicated(pipeline):
    """Fused scores come back in descending order and no chunk id repeats."""
    docs = pipeline.search("rate limiter sliding window", k=8)
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == len(set(ids))
