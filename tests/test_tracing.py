"""Local traces stay short, and export sinks see the clipped payload."""

from fastapi.testclient import TestClient

from quantagent_desk.api.main import app
from quantagent_desk.utils.tokens import TRACE_META_TOKENS
from quantagent_desk.utils.tracing import (
    clear_exporters,
    clear_traces,
    get_active_backends,
    recent_traces,
    register_exporter,
    span,
)


class _Sink:
    def __init__(self):
        self.started = []
        self.ended = []

    def start(self, current):
        self.started.append(current)
        return current.id

    def end(self, current, handle, error=None):
        self.ended.append((current, handle, error))


def test_trace_fields_are_clipped_and_nested():
    clear_traces()
    sink = _Sink()
    register_exporter(sink)
    try:
        long_text = "methodology " * 200
        with span("root", query=long_text) as root:
            with span("child", note=long_text) as child:
                child.finish(output=long_text)
            root.finish(output={"intent": "risk", "tokens": {"research": 0}})
    finally:
        clear_exporters()

    assert sink.started[0].parent_id is None
    assert sink.started[1].parent_id == sink.started[0].id
    assert len(sink.started[0].meta["query"]) <= TRACE_META_TOKENS * 4 + 1
    assert sink.ended[0][1] == sink.started[1].id
    assert sink.ended[1][1] == sink.started[0].id

    traces = recent_traces(1)
    assert traces[0]["name"] == "root"
    assert traces[0]["output"]["tokens"]["research"] == 0
    assert traces[0]["children"][0]["name"] == "child"
    assert len(traces[0]["children"][0]["output"]) <= TRACE_META_TOKENS * 4 + 1
    assert get_active_backends()["local"] is True


def test_traces_endpoint_returns_clipped_trees():
    clear_traces()
    with span("desk", query="x" * 500):
        pass
    client = TestClient(app)
    response = client.get("/traces/recent")
    assert response.status_code == 200
    body = response.json()
    assert body["backends"]["local"] is True
    assert body["traces"]
    assert "id" in body["traces"][-1]
    assert len(body["traces"][-1]["meta"].get("query", "")) <= TRACE_META_TOKENS * 4 + 1
