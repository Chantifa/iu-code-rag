"""Tests for the MCP server: tool registration, in-process mode and remote (HTTP proxy) mode."""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

pytest.importorskip("mcp")

from iu_code_rag import mcp_server  # noqa: E402


@pytest.fixture(autouse=True)
def in_process(monkeypatch, pipeline):
    """Point the MCP tools at the in-memory test pipeline and make sure remote mode is off."""
    monkeypatch.delenv("IU_RAG_API_URL", raising=False)
    monkeypatch.setattr(mcp_server, "_pipeline", lambda: pipeline)


def test_tools_are_registered():
    """The server advertises exactly the three documented tools with their docstrings as descriptions."""
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"search_code", "ask_code", "index_stats"}
    search = next(t for t in tools if t.name == "search_code")
    assert "query" in search.input_schema["properties"]
    assert "GitHub" in search.description


def test_search_code_returns_linked_chunks():
    """search_code returns numbered chunks with source, GitHub link and content."""
    out = mcp_server.search_code("elbow criterion for KMeans", k=3)
    assert "[1] " in out and out.count("\n\n") >= 2
    assert "iubh/DLBDSMLUSL01/Unit_2_Clustering/2_1_ElbowCriterion.py" in out
    assert "<https://github.com/iubh/DLBDSMLUSL01/blob/" in out
    assert "KElbowVisualizer" in out


def test_search_code_clamps_k():
    """k is clamped to the 1..20 range instead of failing."""
    headers = re.compile(r"^\[\d+\] ", re.MULTILINE)
    assert len(headers.findall(mcp_server.search_code("heap insert", k=0))) == 1
    assert 1 < len(headers.findall(mcp_server.search_code("heap insert", k=999))) <= 20


def test_ask_code_returns_answer_and_sources():
    """ask_code returns the answer text followed by a Sources list."""
    out = mcp_server.ask_code("Which ports do the Rasa server and the action server use?", k=3)
    assert "5005" in out
    assert "Sources:" in out
    assert "Chantifa/chatbot_room_booking" in out


def test_index_stats_is_json():
    """index_stats returns the index metadata as JSON."""
    meta = json.loads(mcp_server.index_stats())
    assert meta["chunks"] > 0 and meta["documents"] > 0


def test_remote_mode_proxies_to_api(monkeypatch):
    """With IU_RAG_API_URL set, tool calls are forwarded to the HTTP API."""
    monkeypatch.setenv("IU_RAG_API_URL", "http://rag.test:8000/")
    calls: list[tuple[str, dict]] = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        payload = {
            "results": [
                {"source": "o/r/a.py", "url": "https://github.com/o/r/blob/main/a.py", "score": 0.5, "content": "x = 1"}
            ]
        }
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(mcp_server.httpx, "post", fake_post)
    out = mcp_server.search_code("anything", k=2)
    assert calls == [("http://rag.test:8000/search", {"query": "anything", "k": 2})]
    assert "[1] o/r/a.py <https://github.com/o/r/blob/main/a.py> (score 0.5)\nx = 1" == out
