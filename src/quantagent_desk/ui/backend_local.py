"""In-process backend so Streamlit can run without a separate API process.

Used automatically when http://localhost:8000 is unreachable
(e.g. Streamlit Community Cloud single-app deploy).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from quantagent_desk.agents.graph import build_and_run, stream_graph
from quantagent_desk.engines import LimitsEngine, MarketRiskEngine, PortfolioEngine, StressEngine
from quantagent_desk.models.core import Portfolio

_portfolio = PortfolioEngine()
_market = MarketRiskEngine()
_stress = StressEngine()
_limits = LimitsEngine()

if not _portfolio.get("FIRM"):
    _portfolio.create_synthetic_firm()


def dashboard_summary(portfolio_id: str = "FIRM") -> Dict[str, Any]:
    port = _portfolio.get(portfolio_id)
    if not port:
        return {"error": "Portfolio not found"}
    flat = _portfolio.flatten_positions(portfolio_id)
    temp = Portfolio(portfolio_id=portfolio_id, name=port.name, positions=flat)
    snapshot = _market.from_portfolio_positions(temp)
    breaches = _limits.check(snapshot, portfolio_id)
    children = _portfolio.get_children(portfolio_id)
    child_summaries = []
    for child in children:
        child_pos = _portfolio.flatten_positions(child.portfolio_id)
        child_temp = Portfolio(portfolio_id=child.portfolio_id, name=child.name, positions=child_pos)
        child_snap = _market.from_portfolio_positions(child_temp)
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


def market_snapshot() -> Dict[str, Any]:
    from quantagent_desk.marketdata.feed import (
        PINNED_AS_OF,
        MarketDataUnavailable,
        load_market_snapshot,
    )

    try:
        snap = load_market_snapshot()
    except MarketDataUnavailable as exc:
        return {
            "as_of": None,
            "requested_as_of": PINNED_AS_OF.isoformat(),
            "source": "cache",
            "levels": {},
            "labels": {},
            "notes": {},
            "sessions": 0,
            "available": False,
            "error": str(exc),
        }
    payload = snap.to_public_dict()
    payload["available"] = True
    return payload


def list_historical_scenarios() -> List[Dict[str, Any]]:
    return [s.model_dump() for s in _stress.list_historical()]


def run_stress(scenario_id: str, portfolio_id: str = "FIRM") -> Dict[str, Any]:
    scenario = _stress.get_historical(scenario_id)
    if not scenario:
        for s in _stress.list_historical():
            if s.scenario_id == scenario_id or s.name.lower().replace(" ", "_") == scenario_id.lower():
                scenario = s
                break
    if not scenario:
        return {"error": "Scenario not found"}
    port = _portfolio.get(portfolio_id)
    flat = _portfolio.flatten_positions(portfolio_id)
    temp = Portfolio(portfolio_id=portfolio_id, name=port.name if port else portfolio_id, positions=flat)
    base = _market.from_portfolio_positions(temp)
    result = _stress.apply_stress(temp, scenario, base_var=base.var_95)
    return result.model_dump()


def query_desk(query: str, portfolio_id: str = "FIRM", session_id: Optional[str] = None) -> Dict[str, Any]:
    resp = build_and_run(query=query, session_id=session_id, portfolio_id=portfolio_id)
    return resp.model_dump()


def stream_query_desk(query: str, portfolio_id: str = "FIRM", session_id: Optional[str] = None):
    """Stream LangGraph execution events and tokens in-process."""
    yield from stream_graph(query=query, session_id=session_id, portfolio_id=portfolio_id)
