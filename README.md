# iu-code-rag

A **Retrieval-Augmented Generation (RAG)** system, built with **LangChain**, that answers
questions about two collections of GitHub repositories:

* the personal projects of [Chantifa](https://github.com/Chantifa), and
* the course repositories of the IU International University of Applied Sciences
  ([github.com/iubh](https://github.com/iubh)).

Everything that touches your data runs locally: embeddings are computed on your CPU with a
sentence-transformers model, the vector index is FAISS on disk, and the default chat model is
served by Ollama inside Docker. No API key is required (Claude via Anthropic is an optional
alternative generator).

The project ships with a **golden test set** (known questions with the file that contains the
answer) and **similarity tests** (embedding-level and answer-level cosine checks) so that the
quality of retrieval and generation is measured, not assumed.

---

## Contents

1. [Architecture](#architecture)
2. [Quick start with Docker](#quick-start-with-docker)
3. [Local development without Docker](#local-development-without-docker)
4. [CLI reference](#cli-reference)
5. [HTTP API](#http-api)
6. [Configuration](#configuration)
7. [How the pipeline works, step by step](#how-the-pipeline-works-step-by-step)
8. [Testing: golden tests and similarity tests](#testing-golden-tests-and-similarity-tests)
9. [Evaluation report](#evaluation-report)
10. [Project layout](#project-layout)
11. [Design decisions](#design-decisions)

---

## Architecture

```
                 ┌────────────────────┐
  GitHub API     │  github_loader.py  │  tarball download, file filter, notebook flattening
  (users/orgs) ─▶│  Documents         │
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │  chunking.py       │  language-aware RecursiveCharacterTextSplitter
                 │  1200 chars / 150  │  + "# File: owner/repo/path" header per chunk
                 └─────────┬──────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌─────────────────┐                 ┌──────────────────┐
│ embeddings.py   │                 │ BM25 (rank-bm25) │
│ all-MiniLM-L6-v2│                 │ code tokenizer   │
│ FAISS (cosine)  │                 └────────┬─────────┘
└────────┬────────┘                          │
         └──────────────┬─────────────────────┘
                        ▼
              ┌───────────────────┐
              │ retriever.py      │  weighted Reciprocal Rank Fusion  →  top-k chunks
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ chain.py (LCEL)   │  prompt | ChatOllama / ChatAnthropic | StrOutputParser
              │                   │  or extractive fallback when LLM_PROVIDER=none
              └─────────┬─────────┘
                        ▼
          CLI (typer)  ·  FastAPI (/ask, /search)  ·  tiny web page at /
```

---

## Quick start with Docker

Requirements: Docker Desktop (or Docker Engine + Compose v2). About 4 GB of free disk for the
image plus the Ollama model.

```bash
git clone https://github.com/Chantifa/iu-code-rag.git
cd iu-code-rag
cp .env.example .env        # optional: put a GITHUB_TOKEN in it to avoid rate limits
```

**1. Build the image and start Ollama + the API**

```bash
docker compose up --build -d
```

This starts three services: `ollama` (LLM server), `ollama-pull` (a one-shot job that downloads
`qwen2.5-coder:1.5b`, about 1 GB, on the first run) and `rag` (the API on port 8000). The
embedding model is baked into the image at build time, so the API container does not need
internet access to embed.

**2. Ingest the repositories (one-time, re-run to refresh)**

```bash
docker compose run --rm rag iu-rag ingest
```

This lists every non-fork repository of `Chantifa` and of the `iubh` organisation via the GitHub
API, downloads each as a tarball, keeps only source/text files (see `config/sources.yaml`), chunks
and embeds them, and writes the FAISS index to `./data/index/`. The `./data` folder is mounted into
the container, so the index survives restarts. For a quick smoke test use
`--limit 3` or `--repo Chantifa/Interviews --repo iubh/DLBCSL01_DataStructures_and_Algorithms`.

Without a token GitHub allows 60 API requests per hour; listing both accounts and downloading ~80
repositories needs about 90 requests, so set `GITHUB_TOKEN` in `.env` (any classic or fine-grained
token with public repository read access works).

**3. Restart the API so it loads the new index, then ask questions**

```bash
docker compose restart rag
```

* Web page: <http://localhost:8000>
* API docs (Swagger): <http://localhost:8000/docs>

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How does the rate limiter in the Interviews repo decide if a request is allowed?"}' | jq
```

```bash
docker compose run --rm rag iu-rag ask "Wie funktioniert das Elbow-Kriterium im IU Clustering Kurs?"
```

**4. Run the tests inside the container**

```bash
docker compose run --rm -e LLM_PROVIDER=none rag sh -c "pip install pytest && pytest -q"
```

Add `-e LLM_PROVIDER=ollama -e OLLAMA_BASE_URL=http://ollama:11434` to also run the LLM-backed
similarity test.

**Using Claude instead of Ollama**

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` in `.env` and restart. The model defaults to
`claude-opus-5` (see `ANTHROPIC_MODEL`). The Ollama services can then be left out with
`docker compose up rag`.

---

## Local development without Docker

Python 3.11 or 3.12 (PyTorch has no wheels for 3.13+ at the time of writing).

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"   # CPU-only torch
pip install -e ".[dev]"
```

```bash
iu-rag ingest --limit 5          # small index for a first try
iu-rag search "binary heap insert"
iu-rag ask "Which payment methods does the hotel booking chatbot support?"
iu-rag serve                     # http://localhost:8000
pytest -q                        # golden + similarity tests
iu-rag evaluate --fixtures       # evaluation report on the test corpus
```

If you have Ollama installed natively: `ollama pull qwen2.5-coder:1.5b` and set
`LLM_PROVIDER=ollama` in `.env`.

> If Hugging Face downloads fail with a 401 although the model is public, an expired token is
> stored in `~/.cache/huggingface/token`. Either log in again with `hf auth login` or run with
> `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

---

## CLI reference

| Command | Purpose |
|---|---|
| `iu-rag ingest [--refresh] [--limit N] [--repo owner/name ...] [--skip-fetch]` | Download repositories and (re)build the index |
| `iu-rag ask "question" [--k 6]` | Retrieve + generate an answer, print sources |
| `iu-rag search "query" [--k 6]` | Retrieval only, print the best chunks |
| `iu-rag serve [--host] [--port] [--reload]` | Start the FastAPI server |
| `iu-rag evaluate [--k 5] [--fixtures] [--no-answers] [--output file]` | Golden-set evaluation, JSON report |

---

## HTTP API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{"status","index_loaded","provider"}` |
| GET | `/stats` | – | index metadata (repos, chunk count, embedding model, build time) |
| POST | `/search` | `{"query": str, "k": int}` | ranked chunks with `source`, `url`, fused `score`, `cosine` |
| POST | `/ask` | `{"question": str, "k": int}` | `{"answer", "provider", "sources": [...]}` |
| GET | `/` | – | minimal HTML page to ask questions |

The API answers `503` on `/ask` and `/search` until an index exists.

---

## Configuration

All settings are environment variables (or `.env`), read by `src/iu_code_rag/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `GITHUB_TOKEN` | – | Optional token for the GitHub API |
| `SOURCES_FILE` | `config/sources.yaml` | Which accounts/repos and file types to index |
| `DATA_DIR` | `data` | Cache of downloaded repos + the FAISS index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `150` | Characters per chunk / shared characters between neighbours |
| `TOP_K` | `6` | Chunks handed to the LLM |
| `VECTOR_WEIGHT` | `0.6` | Weight of dense retrieval in the fusion (BM25 gets the rest) |
| `LLM_PROVIDER` | `none` (`ollama` in compose) | `ollama` \| `anthropic` \| `none` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5-coder:1.5b` | Ollama settings |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | – / `claude-opus-5` | Claude settings |
| `LLM_MAX_TOKENS` | `2048` | Max generated tokens |

`config/sources.yaml` controls which repositories are indexed (`user`, `org` or single `repo`
entries, with per-source `exclude` lists and `include_forks`), which file extensions are kept and
which paths are skipped (`node_modules`, build output, lock files, ...).

---

## How the pipeline works, step by step

1. **Listing.** For each source in `sources.yaml` the GitHub REST API is paginated
   (`/users/{name}/repos`, `/orgs/{name}/repos`). Forks and empty repositories are skipped.
2. **Download.** Each repository is fetched as a tarball of its default branch and streamed through
   `tarfile`. Only files whose extension is on the include list, that are below `MAX_FILE_BYTES`
   (200 kB) and that contain no NUL byte (binary) are written to `data/repos/<owner>/<repo>/`. A
   `.repo.json` next to them records branch and URL so that every chunk can link back to GitHub.
3. **Documents.** Files become LangChain `Document`s with metadata `source` (`owner/repo/path`),
   `owner`, `repo`, `path`, `language` and `url`. Jupyter notebooks are flattened into markdown
   cells plus fenced code cells first.
4. **Chunking.** `RecursiveCharacterTextSplitter.from_language` is chosen per language (Python,
   Java, Kotlin, JS/TS, PHP, C#, Markdown, HTML, ...), so splits prefer `class`/`def`/`function`
   boundaries. Chunks are 1200 characters with 150 characters overlap. Each chunk is prefixed with
   `# File: owner/repo/path`, which makes both the embedding and the LLM aware of the file
   identity, and gets a stable `chunk_id` (SHA-1 of source + index).
5. **Embedding.** `HuggingFaceEmbeddings` (langchain-huggingface) runs
   `all-MiniLM-L6-v2` on the CPU with `normalize_embeddings=True`. Normalised vectors make the
   inner product equal to the cosine similarity.
6. **Indexing.** `FAISS.from_documents(..., distance_strategy=MAX_INNER_PRODUCT)` builds the
   dense index; it is saved with `save_local`. The chunks are also written to `chunks.jsonl` so the
   BM25 index can be rebuilt at load time, and `index_meta.json` stores build statistics.
7. **Retrieval.** `HybridRetriever` (a `BaseRetriever`) queries FAISS and a `BM25Retriever` (with a
   tokenizer that splits `camelCase`, `KElbow` and `snake_case` identifiers) for `3·k` candidates
   each and merges them with weighted Reciprocal Rank Fusion. Dense similarity captures meaning and
   paraphrases; BM25 makes sure exact identifiers such as `KElbowVisualizer` or port `5055` are
   never missed.
8. **Generation.** An LCEL chain `prompt | llm | StrOutputParser()` receives the formatted context
   and the question. The system prompt restricts the model to the context and asks it to cite
   `[owner/repo/path]`. With `LLM_PROVIDER=none` the "generator" is a `RunnableLambda` that returns
   the top snippets verbatim, which is what the tests use so they never depend on a model.

---

## Testing: golden tests and similarity tests

All tests run offline against a small **real** corpus in `tests/fixtures/corpus/` (21 files copied
from the actual repositories of both accounts: rate limiter, heap, elbow criterion, chatbot README,
SQLAlchemy models, gcd in JavaScript, German iterator lesson, ...). The corpus is indexed in memory
once per test session. Only the embedding model is downloaded (once, ~90 MB).

```bash
pytest -q            # 78 tests, ~20 s after the model is cached
pytest -m llm        # the LLM-backed test (needs LLM_PROVIDER=ollama or anthropic)
```

### Golden tests (`tests/test_golden_retrieval.py`, data in `tests/golden/queries.jsonl`)

Each golden entry has an `id`, a `question`, a `paraphrase`, the `expected_sources` (file paths that
contain the answer), `expected_keywords` that must appear in the retrieved context, and a prose
`golden_answer`. There are 16 entries spanning both accounts, Python/JavaScript/Markdown and an
English/German query.

| Test | What it guarantees |
|---|---|
| `test_golden_source_is_retrieved[<id>]` | the expected file is in the top-5 for the question |
| `test_paraphrase_is_retrieved[<id>]` | ... and for a differently worded paraphrase |
| `test_extractive_answer_contains_expected_keywords[<id>]` | the *right chunk* was retrieved, not only the right file |
| `test_aggregate_retrieval_metrics` | hit@5 ≥ 0.9 and MRR ≥ 0.7 over the whole set |
| `test_results_are_ranked_and_deduplicated` | fused scores are descending, no duplicate chunk ids |

### Similarity tests (`tests/test_similarity.py`)

| Test | What it guarantees |
|---|---|
| `test_self_similarity_is_one_and_symmetric`, `test_embeddings_are_normalised` | the embedding space behaves as the index assumes |
| `test_paraphrases_are_similar` | cosine ≥ 0.60 for paraphrase pairs |
| `test_unrelated_texts_are_dissimilar` | cosine ≤ 0.35 for unrelated pairs |
| `test_paraphrase_gap` | every paraphrase pair beats every unrelated pair by ≥ 0.15 |
| `test_paraphrased_queries_retrieve_same_file` | question and paraphrase share a top-3 file for ≥ 80 % of the golden set |
| `test_query_is_closest_to_its_golden_chunk` | top-1 chunk has cosine ≥ 0.25 with the question (≤ 2 exceptions) |
| `test_answers_match_their_own_golden_answer_best` | answer/golden-answer similarity matrix: the diagonal wins for ≥ 70 % (extractive mode) |
| `test_golden_answers_are_mutually_distinct` | the golden set itself has no near-duplicate answers |
| `test_llm_answer_is_similar_to_golden` (`-m llm`) | mean cosine(LLM answer, golden answer) ≥ 0.5 |

Thresholds are calibrated for `all-MiniLM-L6-v2`. If you change `EMBEDDING_MODEL`, run
`iu-rag evaluate --fixtures` and adjust them.

### Other tests

`tests/test_chunking_and_loader.py` covers the file filter, notebook flattening, binary/oversized
file skipping, chunk headers/ids and the identifier tokenizer. `tests/test_api.py` exercises the
FastAPI endpoints with the in-memory pipeline, including validation errors and the 503 without an
index.

---

## Evaluation report

`iu-rag evaluate` runs the golden set against the **real** index (or `--fixtures` for the test
corpus) and prints a table plus a JSON report (`data/eval_report.json`) with hit rate, MRR, keyword
coverage and the mean cosine similarity between generated and golden answers. Use it after changing
the chunk size, the embedding model or the fusion weight to see whether quality moved.

---

## Project layout

```
config/sources.yaml            which GitHub accounts / file types to index
src/iu_code_rag/
  config.py                    pydantic-settings (env / .env)
  github_loader.py             GitHub API, tarball extraction, Documents, notebook flattening
  chunking.py                  language-aware splitting + chunk headers
  embeddings.py                local HuggingFace embeddings, cosine helper
  vectorstore.py               FAISS build/save/load + chunks.jsonl for BM25
  retriever.py                 HybridRetriever (FAISS + BM25, RRF), code tokenizer
  chain.py                     LCEL RAG chain, LLM factory, extractive fallback, RagPipeline
  ingest.py                    fetch repositories and build the index
  evaluation.py                golden-set metrics (hit@k, MRR, answer similarity)
  api.py                       FastAPI app + HTML page
  cli.py                       typer CLI
tests/
  fixtures/corpus/             21 real files from both GitHub accounts
  golden/queries.jsonl         16 golden queries
  test_golden_retrieval.py     golden tests
  test_similarity.py           similarity tests
  test_chunking_and_loader.py  unit tests
  test_api.py                  API tests
Dockerfile, docker-compose.yml, .env.example, .github/workflows/ci.yml
```

---

## Design decisions

* **LangChain, but only the stable pieces.** `langchain-core` runnables and prompts, the
  text splitters, `langchain-huggingface`, `langchain-ollama`, `langchain-anthropic`, and the FAISS
  and BM25 integrations. The hybrid fusion is a small `BaseRetriever` subclass instead of the
  legacy `EnsembleRetriever`, so the project does not depend on `langchain-classic`.
* **FAISS instead of a vector database server.** The corpus is a few hundred thousand chunks at
  most; a flat inner-product index answers in milliseconds, needs no extra container, and the
  identical code path runs in tests, CI and Docker. Swapping to Chroma/Qdrant/pgvector only touches
  `vectorstore.py`.
* **Local embeddings.** `all-MiniLM-L6-v2` is small (22 M parameters, 384 dimensions), fast on CPU
  and good enough for code + prose; it is baked into the Docker image. A stronger code model such
  as `jinaai/jina-embeddings-v2-base-code` can be set through `EMBEDDING_MODEL`.
* **Hybrid retrieval.** Pure dense retrieval misses exact identifiers and numbers; pure BM25 misses
  paraphrases and other languages. RRF needs no score calibration between the two.
* **Extractive fallback.** The whole system is testable and usable without any LLM.
* **Golden + similarity tests in CI.** `.github/workflows/ci.yml` runs lint, the tests, the
  fixture evaluation and a Docker build on every push.

## License

MIT
