"""FastAPI application for QuantAgent-Desk."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from quantagent_desk.engines import (
    LimitsEngine,
    MarketRiskEngine,
    PortfolioEngine,
    StressEngine,
)
from quantagent_desk.marketdata.feed import MarketDataUnavailable, load_market_snapshot
from quantagent_desk.models.core import (
    QueryRequest,
    QueryResponse,
    RiskSnapshot,
    StressResult,
    StressScenario,
)

app = FastAPI(
    title="QuantAgent-Desk API",
    description="Unified Agentic Risk Dashboard + Generative Stress Testing",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines (in production use dependency injection / proper state)
portfolio_engine = PortfolioEngine()
market_risk_engine = MarketRiskEngine()
stress_engine = StressEngine()
limits_engine = LimitsEngine()

# Seed synthetic data on startup
@app.on_event("startup")
async def startup():
    portfolio_engine.create_synthetic_firm()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/market/snapshot")
async def market_snapshot() -> Dict[str, Any]:
    """Last factor levels on or before the pinned as-of date. Source is cache or live."""
    try:
        snap = load_market_snapshot()
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = snap.to_public_dict()
    payload["available"] = True
    return payload


@app.get("/dashboard/summary")
async def dashboard_summary(portfolio_id: str = "FIRM") -> Dict[str, Any]:
    """Firm / desk level risk summary for the UI."""
    port = portfolio_engine.get(portfolio_id)
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Compute risk on flattened positions
    # Create a temporary portfolio with all positions
    from quantagent_desk.models.core import Portfolio

    flat_positions = portfolio_engine.flatten_positions(portfolio_id)
    temp = Portfolio(
        portfolio_id=portfolio_id,
        name=port.name,
        positions=flat_positions,
    )
    snapshot = market_risk_engine.from_portfolio_positions(temp)
    breaches = limits_engine.check(snapshot, portfolio_id)

    children = portfolio_engine.get_children(portfolio_id)
    child_summaries = []
    for child in children:
        child_pos = portfolio_engine.flatten_positions(child.portfolio_id)
        child_temp = Portfolio(portfolio_id=child.portfolio_id, name=child.name, positions=child_pos)
        child_snap = market_risk_engine.from_portfolio_positions(child_temp)
        child_summaries.append(
            {
                "portfolio_id": child.portfolio_id,
                "name": child.name,
                "var_95": child_snap.var_95,
                "es_97_5": child_snap.es_97_5,
                "n_positions": len(child_pos),
            }
        )

    return {
        "portfolio_id": portfolio_id,
        "name": port.name,
        "snapshot": snapshot.model_dump(),
        "breaches": [b.model_dump() for b in breaches],
        "children": child_summaries,
        "utilization": snapshot.limit_utilization,
    }


@app.get("/risk/snapshot/{portfolio_id}")
async def get_risk_snapshot(portfolio_id: str) -> RiskSnapshot:
    port = portfolio_engine.get(portfolio_id)
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    flat = portfolio_engine.flatten_positions(portfolio_id)
    from quantagent_desk.models.core import Portfolio
    temp = Portfolio(portfolio_id=portfolio_id, name=port.name, positions=flat)
    return market_risk_engine.from_portfolio_positions(temp)


@app.get("/stress/historical")
async def list_historical_scenarios() -> List[Dict[str, Any]]:
    return [s.model_dump() for s in stress_engine.list_historical()]


@app.post("/stress/run")
async def run_stress(
    scenario_id: str,
    portfolio_id: str = "FIRM",
) -> StressResult:
    """Run a pre-defined historical scenario."""
    scenario = stress_engine.get_historical(scenario_id)
    if not scenario:
        # try by name match
        for s in stress_engine.list_historical():
            if s.scenario_id == scenario_id or s.name.lower().replace(" ", "_") == scenario_id.lower():
                scenario = s
                break
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    port = portfolio_engine.get(portfolio_id)
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    flat = portfolio_engine.flatten_positions(portfolio_id)
    from quantagent_desk.models.core import Portfolio
    temp = Portfolio(portfolio_id=portfolio_id, name=port.name, positions=flat)
    base_snap = market_risk_engine.from_portfolio_positions(temp)
    return stress_engine.apply_stress(temp, scenario, base_var=base_snap.var_95)


@app.post("/query", response_model=QueryResponse)
async def query_desk(request: QueryRequest) -> QueryResponse:
    """Multi-agent risk desk query (LangGraph pipeline + RAG + Memory)."""
    from quantagent_desk.agents.graph import build_and_run
    return build_and_run(
        query=request.query,
        session_id=request.session_id,
        portfolio_id=request.portfolio_id or "FIRM",
    )


@app.post("/query/stream")
async def query_desk_stream(request: QueryRequest):
    """Stream LangGraph multi-agent execution events, token chunks, and final response via SSE."""
    import json
    from quantagent_desk.agents.graph import stream_graph

    def event_generator():
        for event in stream_graph(
            query=request.query,
            session_id=request.session_id,
            portfolio_id=request.portfolio_id or "FIRM",
        ):
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )




@app.get("/eval/run")
async def run_evals():
    """Run accuracy/grounding eval suite (CI / demo)."""
    from quantagent_desk.evals.suite import run_eval_suite
    return run_eval_suite()


@app.get("/traces/recent")
async def recent_traces_endpoint(limit: int = 10):
    """Local trace tree. Fields are clipped; Langfuse and Phoenix stay opt-in."""
    from quantagent_desk.utils.tracing import get_active_backends, recent_traces
    return {
        "backends": get_active_backends(),
        "traces": recent_traces(max(1, min(limit, 50))),
    }

# Make the package importable when running uvicorn from project root
# PYTHONPATH=src uvicorn quantagent_desk.api.main:app --reload
