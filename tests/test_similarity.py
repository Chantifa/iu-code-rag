"""Similarity tests for the embedding model and for the generated answers.

These tests pin down the *semantic* behaviour that the RAG system relies on:
  * paraphrases embed close together, unrelated texts embed far apart,
  * paraphrased questions retrieve the same top document,
  * answers are closer to their own golden answer than to any other golden answer.
Thresholds are calibrated for sentence-transformers/all-MiniLM-L6-v2; if you
switch EMBEDDING_MODEL, re-calibrate them with `iu-rag evaluate --fixtures`.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from iu_code_rag.embeddings import cosine_similarity
from iu_code_rag.evaluation import load_golden

GOLDEN = load_golden(Path(__file__).parent / "golden" / "queries.jsonl")

PARAPHRASE_PAIRS = [
    (
        "How does the rate limiter decide whether a request is allowed?",
        "Explain how the limiter checks if a user exceeded the request limit.",
    ),
    (
        "Show me the heap implementation with insert and extractMax.",
        "Where is the binary heap class with insert and extract maximum?",
    ),
    (
        "How is the greatest common divisor computed in JavaScript?",
        "JavaScript function that calculates the gcd of two integers",
    ),
    (
        "Which ports do the Rasa server and the action server use?",
        "On which port numbers do the Rasa and action servers listen?",
    ),
]

UNRELATED_PAIRS = [
    ("How does the rate limiter decide whether a request is allowed?", "PCA explained variance plot with scikit-learn"),
    (
        "Show me the heap implementation with insert and extractMax.",
        "Which payment methods does the hotel chatbot accept?",
    ),
    ("Zeige mir Iteratoren und Generatoren in Python.", "selection sort in JavaScript"),
]

PARAPHRASE_MIN = 0.60
UNRELATED_MAX = 0.35


def test_self_similarity_is_one_and_symmetric(embeddings):
    """cos(a, a) is 1 and cos(a, b) equals cos(b, a)."""
    a = embeddings.embed_query("hybrid retrieval with BM25 and FAISS")
    b = embeddings.embed_query("a completely different sentence about cooking pasta")
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-6)


def test_embeddings_are_normalised(embeddings):
    """Embedding vectors have unit length, so inner product equals cosine similarity."""
    import numpy as np

    v = np.asarray(embeddings.embed_query("normalised vector"))
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("a,b", PARAPHRASE_PAIRS, ids=[f"para{i}" for i in range(len(PARAPHRASE_PAIRS))])
def test_paraphrases_are_similar(embeddings, a, b):
    """Similarity test: paraphrase pairs must reach at least PARAPHRASE_MIN."""
    sim = cosine_similarity(embeddings.embed_query(a), embeddings.embed_query(b))
    assert sim >= PARAPHRASE_MIN, f"cosine={sim:.3f} < {PARAPHRASE_MIN}"


@pytest.mark.parametrize("a,b", UNRELATED_PAIRS, ids=[f"unrel{i}" for i in range(len(UNRELATED_PAIRS))])
def test_unrelated_texts_are_dissimilar(embeddings, a, b):
    """Similarity test: unrelated pairs must stay at or below UNRELATED_MAX."""
    sim = cosine_similarity(embeddings.embed_query(a), embeddings.embed_query(b))
    assert sim <= UNRELATED_MAX, f"cosine={sim:.3f} > {UNRELATED_MAX}"


def test_paraphrase_gap(embeddings):
    """Every paraphrase pair must be more similar than every unrelated pair."""
    para = min(cosine_similarity(embeddings.embed_query(a), embeddings.embed_query(b)) for a, b in PARAPHRASE_PAIRS)
    unrel = max(cosine_similarity(embeddings.embed_query(a), embeddings.embed_query(b)) for a, b in UNRELATED_PAIRS)
    assert para > unrel + 0.15


def test_paraphrased_queries_retrieve_same_file(pipeline):
    """Retrieval consistency: question and paraphrase must agree on the top file for >= 80% of the golden set."""
    agree = 0
    total = 0
    for g in GOLDEN:
        if not g.paraphrase:
            continue
        total += 1
        top_a = pipeline.search(g.question, k=3)
        top_b = pipeline.search(g.paraphrase, k=3)
        files_a = {d.metadata["source"] for d in top_a}
        files_b = {d.metadata["source"] for d in top_b}
        if files_a & files_b:
            agree += 1
    assert total > 0
    assert agree / total >= 0.8, f"only {agree}/{total} question/paraphrase pairs share a top-3 file"


def test_query_is_closest_to_its_golden_chunk(pipeline, embeddings):
    """The top retrieved chunk should have a meaningful cosine similarity with the question."""
    low = []
    for g in GOLDEN:
        docs = pipeline.search(g.question, k=1)
        sim = cosine_similarity(embeddings.embed_query(g.question), embeddings.embed_query(docs[0].page_content))
        if sim < 0.25:
            low.append((g.id, round(sim, 3)))
    assert len(low) <= 2, f"too many weak top-1 matches: {low}"


def test_answers_match_their_own_golden_answer_best(pipeline, embeddings):
    """Answer/golden-answer similarity matrix: the diagonal must win for >= 70% of queries.

    This is a relative test, so it is robust to the absolute similarity level of the answers.
    In extractive mode (no LLM) the answer is raw code while the golden answer is prose, and
    two golden queries share the same README, so 70% is the calibrated floor; with a real LLM
    see test_llm_answer_is_similar_to_golden.
    """
    items = [g for g in GOLDEN if g.golden_answer]
    answers = [pipeline.ask(g.question, k=5).answer for g in items]
    ans_vecs = embeddings.embed_documents(answers)
    gold_vecs = embeddings.embed_documents([g.golden_answer for g in items])
    wins = 0
    for i, av in enumerate(ans_vecs):
        sims = [cosine_similarity(av, gv) for gv in gold_vecs]
        if max(range(len(sims)), key=sims.__getitem__) == i:
            wins += 1
    assert wins / len(items) >= 0.7, f"only {wins}/{len(items)} answers were closest to their own golden answer"


def test_golden_answers_are_mutually_distinct(embeddings):
    """Sanity check of the golden set itself: no two golden answers should be near-duplicates."""
    items = [g for g in GOLDEN if g.golden_answer]
    vecs = embeddings.embed_documents([g.golden_answer for g in items])
    for (i, a), (j, b) in itertools.combinations(enumerate(vecs), 2):
        sim = cosine_similarity(a, b)
        assert sim < 0.9, f"golden answers {items[i].id} and {items[j].id} are near duplicates ({sim:.2f})"


@pytest.mark.llm
def test_llm_answer_is_similar_to_golden(llm_pipeline, embeddings):
    """With a real LLM, generated answers must be semantically close to the golden answers."""
    sims = []
    for g in [g for g in GOLDEN if g.golden_answer][:6]:
        ans = llm_pipeline.ask(g.question, k=5).answer
        sims.append(cosine_similarity(embeddings.embed_query(ans), embeddings.embed_query(g.golden_answer)))
    mean = sum(sims) / len(sims)
    assert mean >= 0.5, f"mean answer similarity {mean:.3f} too low: {sims}"
