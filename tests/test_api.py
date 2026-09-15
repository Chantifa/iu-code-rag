from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iu_code_rag.api import create_app


@pytest.fixture(scope="module")
def client(pipeline, settings):
    app = create_app(settings=settings, pipeline=pipeline)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "index_loaded": True, "provider": "none"}


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json()["chunks"] > 0


def test_search(client):
    r = client.post("/search", json={"query": "elbow criterion KMeans", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert any("ElbowCriterion" in x["source"] for x in body["results"])
    assert all("url" in x and "score" in x for x in body["results"])


def test_ask(client):
    r = client.post("/ask", json={"question": "Which ports do the Rasa server and action server use?"})
    assert r.status_code == 200
    body = r.json()
    assert "5005" in body["answer"]
    assert body["sources"] and any("chatbot_room_booking" in s["source"] for s in body["sources"])


def test_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "iu-code-rag" in r.text


def test_503_without_index(settings):
    app = create_app(settings=settings, pipeline=None)
    with TestClient(app) as c:
        # lifespan tries to load data/index which does not exist for the test settings
        assert c.get("/health").json()["index_loaded"] is False
        assert c.post("/ask", json={"question": "hi"}).status_code == 503
