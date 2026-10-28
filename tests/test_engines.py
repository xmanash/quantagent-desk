"""Basic unit tests for risk engines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from quantagent_desk.engines.market_risk import MarketRiskEngine
from quantagent_desk.engines.stress import StressEngine
from quantagent_desk.engines.portfolio import PortfolioEngine
from quantagent_desk.engines.limits import LimitsEngine


def test_historical_var():
    engine = MarketRiskEngine()
    rng = np.random.default_rng(42)
    pnl = rng.normal(-1000, 50000, 1000)
    snap = engine.historical_var_es(pnl, portfolio_id="TEST")
    assert snap.var_95 > 0
    assert snap.var_99 >= snap.var_95
    assert snap.es_97_5 >= snap.var_95


def test_synthetic_portfolio():
    pe = PortfolioEngine()
    firm = pe.create_synthetic_firm()
    assert firm.portfolio_id == "FIRM"
    assert len(firm.children_ids) == 3
    positions = pe.flatten_positions("FIRM")
    assert len(positions) > 0


def test_stress_historical():
    se = StressEngine()
    scenarios = se.list_historical()
    assert len(scenarios) >= 3
    s = se.get_historical("2008_GFC")
    assert s is not None
    assert len(s.shocks) > 0


def test_limits():
    le = LimitsEngine()
    from quantagent_desk.models.core import RiskSnapshot
    snap = RiskSnapshot(portfolio_id="T", var_95=60_000_000, var_99=90_000_000, es_97_5=75_000_000)
    breaches = le.check(snap)
    assert any(b.severity == "Breach" for b in breaches)
