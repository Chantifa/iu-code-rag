from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iu_code_rag.api import create_app


@pytest.fixture(scope="module")
def client(pipeline, settings):
    """FastAPI test client with the in-memory pipeline injected (no index on disk needed)."""
    app = create_app(settings=settings, pipeline=pipeline)
    with TestClient(app) as c:
        yield c


def test_health(client):
    """``/health`` reports a loaded index and the extractive provider."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "index_loaded": True, "provider": "none"}


def test_stats(client):
    """``/stats`` exposes the index metadata, including the chunk count."""
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json()["chunks"] > 0


def test_search(client):
    """``/search`` returns exactly k ranked chunks with URLs and scores and finds the elbow file."""
    r = client.post("/search", json={"query": "elbow criterion KMeans", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert any("ElbowCriterion" in x["source"] for x in body["results"])
    assert all("url" in x and "score" in x for x in body["results"])


def test_ask(client):
    """``/ask`` answers the port question from the chatbot README and cites that file."""
    r = client.post("/ask", json={"question": "Which ports do the Rasa server and action server use?"})
    assert r.status_code == 200
    body = r.json()
    assert "5005" in body["answer"]
    assert body["sources"] and any("chatbot_room_booking" in s["source"] for s in body["sources"])


def test_validation(client):
    """Empty questions and k=0 are rejected with HTTP 422."""
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422


def test_index_page(client):
    """The root page serves the HTML question form."""
    r = client.get("/")
    assert r.status_code == 200 and "iu-code-rag" in r.text


def test_503_without_index(settings):
    """Without an index the API starts, reports index_loaded=false and answers 503 on /ask."""
    app = create_app(settings=settings, pipeline=None)
    with TestClient(app) as c:
        # lifespan tries to load data/index which does not exist for the test settings
        assert c.get("/health").json()["index_loaded"] is False
        assert c.post("/ask", json={"question": "hi"}).status_code == 503
