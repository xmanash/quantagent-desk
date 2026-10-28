"""Local traces, with optional Langfuse and Phoenix export.

Exports carry clipped fields only: intent, token counts, and short status.
Full answers, research excerpts, and prompts stay off the wire.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Protocol
from uuid import uuid4

from quantagent_desk.utils.tokens import TRACE_META_TOKENS, clip_payload

_TRACES: List["TraceSpan"] = []
_ACTIVE_SPAN: contextvars.ContextVar[Optional["TraceSpan"]] = contextvars.ContextVar(
    "active_span", default=None
)
_EXPORTERS: List["SpanExporter"] = []

_OTEL_TRACER = None
_PHOENIX_INITIALIZED = False
_LANGFUSE_CLIENT = None
_LANGFUSE_DISABLED = False


class SpanExporter(Protocol):
    def start(self, span: "TraceSpan") -> Any: ...

    def end(self, span: "TraceSpan", handle: Any, error: Optional[BaseException] = None) -> None: ...


def register_exporter(exporter: SpanExporter) -> None:
    """Add a sink. Tests use this; Langfuse and Phoenix register themselves."""
    _EXPORTERS.append(exporter)


def clear_exporters() -> None:
    _EXPORTERS.clear()


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _phoenix_enabled() -> bool:
    return bool(
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("PHOENIX_ENABLE", "").lower() in ("true", "1", "yes")
    )


def _langfuse_client():
    """One client per process. Construction is skipped when keys are absent."""
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED
    if _LANGFUSE_DISABLED or not _langfuse_enabled():
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
        kwargs: Dict[str, Any] = {"timeout": 2}
        if host:
            kwargs["base_url"] = host
        _LANGFUSE_CLIENT = Langfuse(**kwargs)
    except Exception:
        _LANGFUSE_DISABLED = True
        _LANGFUSE_CLIENT = None
    return _LANGFUSE_CLIENT


def _init_phoenix_otel():
    """Private Phoenix tracer. It does not replace the process-wide provider."""
    global _OTEL_TRACER, _PHOENIX_INITIALIZED
    if _PHOENIX_INITIALIZED:
        return _OTEL_TRACER
    _PHOENIX_INITIALIZED = True
    if not _phoenix_enabled():
        return None
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "quantagent-desk"}))
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=2)))
        _OTEL_TRACER = provider.get_tracer("quantagent_desk")
    except Exception:
        _OTEL_TRACER = None
    return _OTEL_TRACER


class TraceSpan:
    """Hierarchical trace span with parent-child relationships and metadata."""

    def __init__(
        self,
        name: str,
        meta: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ):
        self.id = str(uuid4())[:8]
        self.name = name
        self.parent_id = parent_id
        self.meta = clip_payload(dict(meta or {}), TRACE_META_TOKENS)
        self.start = time.time()
        self.end: Optional[float] = None
        self.status = "ok"
        self.output: Any = None
        self.children: List[TraceSpan] = []

    def finish(self, status: str = "ok", output: Any = None) -> None:
        self.end = time.time()
        self.status = status
        if output is not None:
            self.output = clip_payload(output, TRACE_META_TOKENS)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return max(0.0, (time.time() - self.start) * 1000)
        return (self.end - self.start) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "meta": self.meta,
            "output": self.output,
            "children": [c.to_dict() for c in self.children],
        }


def get_active_backends() -> Dict[str, bool]:
    """Which sinks will receive spans. Local is always on."""
    return {
        "local": True,
        "langfuse": _langfuse_enabled() and not _LANGFUSE_DISABLED,
        "phoenix": _phoenix_enabled(),
    }


def _otel_attributes(meta: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in list(meta.items())[:8]:
        if isinstance(value, (int, float, bool)):
            flat[str(key)[:40]] = str(value)
        else:
            flat[str(key)[:40]] = str(clip_payload(value, TRACE_META_TOKENS))[:160]
    return flat


@contextmanager
def span(name: str, **meta: Any) -> Generator[TraceSpan, None, None]:
    """
    Context manager creating hierarchical spans.
    Local tree always. Langfuse and Phoenix only when configured.
    """
    parent = _ACTIVE_SPAN.get()
    current = TraceSpan(name, meta=meta, parent_id=parent.id if parent else None)

    if parent:
        parent.children.append(current)
    else:
        _TRACES.append(current)
        if len(_TRACES) > 200:
            _TRACES.pop(0)

    handles: List[tuple] = []
    for exporter in list(_EXPORTERS):
        try:
            handles.append((exporter, exporter.start(current)))
        except Exception:
            continue

    lf_cm = None
    lf_obs = None
    client = _langfuse_client()
    if client is not None:
        try:
            lf_cm = client.start_as_current_observation(
                as_type="span",
                name=name,
                input=current.meta,
                metadata={"clipped": True},
            )
            lf_obs = lf_cm.__enter__()
        except Exception:
            lf_cm = None
            lf_obs = None

    otel_tracer = _init_phoenix_otel()
    otel_span = None
    if otel_tracer is not None:
        try:
            otel_span = otel_tracer.start_span(name)
            for key, value in _otel_attributes(current.meta).items():
                otel_span.set_attribute(key, value)
            if current.parent_id:
                otel_span.set_attribute("parent_id", current.parent_id)
        except Exception:
            otel_span = None

    token = _ACTIVE_SPAN.set(current)
    error: Optional[BaseException] = None
    try:
        yield current
        current.finish("ok")
    except Exception as exc:
        error = exc
        current.finish("error", output=str(exc))
        raise
    finally:
        if lf_obs is not None:
            try:
                if error is not None:
                    lf_obs.update(output=current.output, level="ERROR")
                else:
                    lf_obs.update(output=current.output)
            except Exception:
                pass
        if lf_cm is not None:
            try:
                lf_cm.__exit__(type(error) if error else None, error, None)
            except Exception:
                pass
        if otel_span is not None:
            try:
                if error is not None:
                    otel_span.record_exception(error)
                otel_span.end()
            except Exception:
                pass
        for exporter, handle in handles:
            try:
                exporter.end(current, handle, error)
            except Exception:
                continue
        try:
            _ACTIVE_SPAN.reset(token)
        except ValueError:
            # The SSE generator can be closed on another worker thread.
            _ACTIVE_SPAN.set(None)


def recent_traces(limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent root trace trees with clipped fields."""
    return [item.to_dict() for item in _TRACES[-limit:]]


def clear_traces() -> None:
    """Clear in-memory traces."""
    _TRACES.clear()
