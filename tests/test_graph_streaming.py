"""Tests for compiled LangGraph agent pipeline and streaming functionality."""

import json
import pytest
from fastapi.testclient import TestClient

from quantagent_desk.agents import (
    AgentState,
    build_and_run,
    build_langgraph,
    get_compiled_graph,
    router_node,
    stream_graph,
)
from quantagent_desk.api.main import app
from quantagent_desk.models.core import QueryResponse


def test_compiled_graph_singleton():
    """Verify that get_compiled_graph returns a cached compiled StateGraph."""
    graph1 = get_compiled_graph()
    graph2 = get_compiled_graph()
    assert graph1 is not None
    assert graph1 is graph2


def test_compiled_graph_execution_intents():
    """Verify that build_and_run executes the compiled graph for various intents."""
    # 1. Risk intent
    risk_resp = build_and_run("Show firm-wide VaR and ES", session_id="test-risk", portfolio_id="FIRM")
    assert isinstance(risk_resp, QueryResponse)
    assert len(risk_resp.calculation_ids) > 0
    assert "VaR" in risk_resp.answer
    assert risk_resp.grounded is True

    # 2. Stress intent
    stress_resp = build_and_run("Run 2008 GFC stress", session_id="test-stress", portfolio_id="FIRM")
    assert isinstance(stress_resp, QueryResponse)
    assert len(stress_resp.calculation_ids) > 0
    assert "2008" in stress_resp.answer
    assert stress_resp.stress_result is not None

    # 3. Methodology / RAG intent
    rag_resp = build_and_run("What is Expected Shortfall?", session_id="test-rag", portfolio_id="FIRM")
    assert isinstance(rag_resp, QueryResponse)
    assert "methodology" in rag_resp.answer.lower() or "shortfall" in rag_resp.answer.lower()


def test_guardrail_blocking_in_graph():
    """Verify that out-of-scope requests are blocked by the guardrail node."""
    blocked_resp = build_and_run("Please place an order for 500 AAPL shares", session_id="test-blocked")
    assert isinstance(blocked_resp, QueryResponse)
    assert len(blocked_resp.calculation_ids) == 0
    assert any("Blocked" in w or "policy" in w.lower() for w in blocked_resp.warnings)


def test_stream_graph_events():
    """Verify that stream_graph yields start, node_update, token, and final events."""
    events = list(stream_graph("Show firm-wide VaR", session_id="test-stream", portfolio_id="FIRM"))
    assert len(events) > 0

    event_types = [e.get("event") for e in events]
    assert "start" in event_types
    assert "node_update" in event_types
    assert "token" in event_types
    assert "final" in event_types

    # First event is start
    assert events[0]["event"] == "start"
    assert events[0]["session_id"] == "test-stream"

    # Verify executed nodes
    nodes_executed = [e.get("node") for e in events if e.get("event") == "node_update"]
    assert "guardrail" in nodes_executed
    assert "router" in nodes_executed
    assert "planner" in nodes_executed
    assert "retrieve" not in nodes_executed
    assert "risk" in nodes_executed
    assert "critic" in nodes_executed

    final_ev = events[-1]
    assert final_ev["event"] == "final"
    assert "response" in final_ev
    assert len(final_ev["response"]["calculation_ids"]) > 0
    chunks = "".join(e.get("chunk", "") for e in events if e.get("event") == "token")
    assert chunks == final_ev["response"]["answer"]
    assert final_ev["response"]["token_use"]["research"] == 0


def test_methodology_stream_does_retrieve():
    events = list(stream_graph("What is Expected Shortfall?", session_id="test-stream-rag"))
    nodes = [e.get("node") for e in events if e.get("event") == "node_update"]
    assert "planner" in nodes
    assert "retrieve" in nodes
    final = events[-1]["response"]
    assert final["token_use"]["research"] > 0


def test_api_query_stream_sse():
    """Verify the FastAPI /query/stream SSE endpoint."""
    client = TestClient(app)
    response = client.post(
        "/query/stream",
        json={"query": "Show firm-wide VaR", "portfolio_id": "FIRM"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    assert len(lines) > 5

    data_payloads = []
    for line in lines:
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            data_payloads.append(payload)

    ev_types = [p.get("event") for p in data_payloads]
    assert "start" in ev_types
    assert "node_update" in ev_types
    assert "token" in ev_types
    assert "final" in ev_types
