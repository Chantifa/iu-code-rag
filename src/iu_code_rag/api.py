"""FastAPI service exposing search and question answering over the index."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .chain import RagPipeline, index_exists
from .config import Settings, get_settings

log = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Request body of ``POST /search``: the free-text query and how many chunks to return."""

    query: str = Field(..., min_length=1)
    k: int = Field(default=6, ge=1, le=50)


class AskRequest(BaseModel):
    """Request body of ``POST /ask``: the question and how many chunks to hand to the LLM."""

    question: str = Field(..., min_length=1)
    k: int = Field(default=6, ge=1, le=20)


def _load_pipeline(settings: Settings) -> RagPipeline | None:
    """Load the RAG pipeline from the on-disk index, or return ``None`` if no index was built yet."""
    if not index_exists(settings.index_dir):
        log.warning("no index at %s - run `iu-rag ingest`", settings.index_dir)
        return None
    return RagPipeline.from_index(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: resolve the settings and load the index once when the server starts.

    The tests inject a ready pipeline through :func:`create_app`, so nothing is loaded from disk there.
    """
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = _load_pipeline(settings)
    yield


def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    """Build the FastAPI application with all routes.

    ``settings`` and ``pipeline`` can be passed explicitly (tests, embedding into another app);
    otherwise they are resolved from the environment when the server starts.
    """
    app = FastAPI(title="iu-code-rag", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline

    def pipe() -> RagPipeline:
        """Return the loaded pipeline, or answer HTTP 503 while no index exists."""
        p = app.state.pipeline
        if p is None:
            raise HTTPException(status_code=503, detail="index not built yet - run `iu-rag ingest`")
        return p

    @app.get("/health")
    def health():
        """Liveness endpoint: reports whether an index is loaded and which LLM provider is active."""
        p = app.state.pipeline
        return {"status": "ok", "index_loaded": p is not None, "provider": app.state.settings.llm_provider}

    @app.get("/stats")
    def stats():
        """Metadata of the loaded index: repositories, chunk count, embedding model, build time."""
        return pipe().index_meta

    @app.post("/search")
    def search(req: SearchRequest):
        """Retrieval only: return the top-k chunks for ``query`` with their scores and GitHub links."""
        docs = pipe().search(req.query, k=req.k)
        return {
            "query": req.query,
            "results": [
                {
                    "source": d.metadata["source"],
                    "url": d.metadata.get("url"),
                    "score": d.metadata.get("score"),
                    "cosine": d.metadata.get("cosine"),
                    "chunk": d.metadata.get("chunk"),
                    "content": d.page_content,
                }
                for d in docs
            ],
        }

    @app.post("/ask")
    def ask(req: AskRequest):
        """Full RAG: retrieve the best chunks, generate an answer and return it with the cited sources."""
        return pipe().ask(req.question, k=req.k).to_dict()

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        """Serve the minimal HTML page for asking questions from a browser."""
        return _HTML

    return app


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>iu-code-rag</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:4rem}pre{background:#f4f4f4;padding:.75rem;overflow:auto;white-space:pre-wrap}
.src{font-size:.9rem;color:#444}button{padding:.5rem 1rem}</style></head><body>
<h1>iu-code-rag</h1><p>Ask a question about Chantifa's repositories or the IU (iubh) course repositories.</p>
<textarea id="q" placeholder="e.g. How does the rate limiter decide whether a request is allowed?"></textarea><br>
<button onclick="ask()">Ask</button> <span id="st"></span>
<h3>Answer</h3><pre id="a"></pre><h3>Sources</h3><div id="s"></div>
<script>
async function ask(){const q=document.getElementById('q').value;document.getElementById('st').textContent='thinking...';
const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q})});
const j=await r.json();document.getElementById('st').textContent='';
document.getElementById('a').textContent=j.answer||JSON.stringify(j);
document.getElementById('s').innerHTML=(j.sources||[]).map(s=>`<div class="src"><a href="${s.url}" target="_blank">${s.source}</a> (score ${s.score})</div>`).join('');}
</script></body></html>"""


app = create_app()
