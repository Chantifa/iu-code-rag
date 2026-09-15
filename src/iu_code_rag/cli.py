"""Command line interface: `iu-rag ingest | ask | search | serve | evaluate`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings

app = typer.Typer(help="LangChain RAG over Chantifa's and IU's GitHub repositories.", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def ingest(
    refresh: bool = typer.Option(False, help="Re-download repositories that are already cached."),
    repo: list[str] | None = typer.Option(None, "--repo", help="Only these repositories (owner/name). Repeatable."),
    limit: int | None = typer.Option(None, help="Stop after N repositories (useful for a quick smoke test)."),
    skip_fetch: bool = typer.Option(False, help="Do not talk to GitHub; only rebuild the index from data/repos."),
    verbose: bool = typer.Option(False, "-v"),
):
    """Download the configured GitHub repositories and build the vector index."""
    _setup_logging(verbose)
    from .ingest import build_index, fetch_repositories

    settings = get_settings()
    if not skip_fetch:
        fetched = fetch_repositories(settings, only=repo, limit=limit, refresh=refresh)
        console.print(f"[green]fetched/reused {len(fetched)} repositories[/green]")
    meta = build_index(settings)
    console.print_json(json.dumps(meta))


@app.command()
def ask(question: str, k: int = typer.Option(6, help="Number of chunks to retrieve.")):
    """Ask a question; prints the answer and the sources."""
    _setup_logging(False)
    from .chain import RagPipeline

    pipeline = RagPipeline.from_index(get_settings())
    result = pipeline.ask(question, k=k)
    console.rule("Answer")
    console.print(result.answer)
    console.rule("Sources")
    for s in result.sources:
        console.print(f"[bold]{s.source}[/bold]  score={s.score}  {s.url}")


@app.command()
def search(query: str, k: int = typer.Option(6)):
    """Retrieval only: show the best matching chunks."""
    _setup_logging(False)
    from .chain import RagPipeline

    pipeline = RagPipeline.from_index(get_settings())
    for i, d in enumerate(pipeline.search(query, k=k), 1):
        console.rule(f"[{i}] {d.metadata['source']}  score={d.metadata.get('score')}")
        console.print(d.page_content[:800])


@app.command()
def serve(host: str | None = None, port: int | None = None, reload: bool = False):
    """Start the HTTP API (FastAPI / uvicorn)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("iu_code_rag.api:app", host=host or settings.api_host, port=port or settings.api_port, reload=reload)


@app.command()
def evaluate(
    k: int = typer.Option(5),
    fixtures: bool = typer.Option(False, help="Evaluate against the small test corpus instead of the real index."),
    corpus: Path | None = typer.Option(None, help="Custom corpus directory (<owner>/<repo>/...) to evaluate against."),
    golden: Path | None = typer.Option(None, help="Golden queries file (default: tests/golden/queries.jsonl)."),
    no_answers: bool = typer.Option(False, help="Only measure retrieval (faster, no LLM calls)."),
    output: Path = typer.Option(Path("data/eval_report.json")),
):
    """Run the golden-query evaluation and write a JSON report."""
    _setup_logging(False)
    from .chain import RagPipeline
    from .evaluation import default_fixture_corpus, load_golden
    from .evaluation import evaluate as run_eval
    from .github_loader import SourcesConfig, load_documents_from_dir

    settings = get_settings()
    if fixtures or corpus:
        root = corpus or default_fixture_corpus()
        docs = load_documents_from_dir(root, SourcesConfig.load(settings.sources_file))
        pipeline = RagPipeline.from_documents(docs, settings)
    else:
        pipeline = RagPipeline.from_index(settings)

    report = run_eval(pipeline, load_golden(golden), k=k, with_answers=not no_answers)
    table = Table(title=f"Golden evaluation (k={k})")
    table.add_column("id")
    table.add_column("hit")
    table.add_column("rank")
    table.add_column("answer sim")
    table.add_column("missing keywords")
    for r in report["results"]:
        table.add_row(
            r["id"],
            "yes" if r["hit"] else "NO",
            str(r["rank"] or "-"),
            str(r.get("answer_similarity", "-")),
            ", ".join(r.get("missing_keywords", [])) or "-",
        )
    console.print(table)
    console.print_json(json.dumps(report["summary"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"report written to {output}")


if __name__ == "__main__":
    app()
