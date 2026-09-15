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
    query: str = Field(..., min_length=1)
    k: int = Field(default=6, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=6, ge=1, le=20)


def _load_pipeline(settings: Settings) -> RagPipeline | None:
    if not index_exists(settings.index_dir):
        log.warning("no index at %s - run `iu-rag ingest`", settings.index_dir)
        return None
    return RagPipeline.from_index(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = _load_pipeline(settings)
    yield


def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="iu-code-rag", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline

    def pipe() -> RagPipeline:
        p = app.state.pipeline
        if p is None:
            raise HTTPException(status_code=503, detail="index not built yet - run `iu-rag ingest`")
        return p

    @app.get("/health")
    def health():
        p = app.state.pipeline
        return {"status": "ok", "index_loaded": p is not None, "provider": app.state.settings.llm_provider}

    @app.get("/stats")
    def stats():
        return pipe().index_meta

    @app.post("/search")
    def search(req: SearchRequest):
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
        return pipe().ask(req.question, k=req.k).to_dict()

    @app.get("/", response_class=HTMLResponse)
    def index_page():
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
