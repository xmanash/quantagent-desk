"""Factor cache and historical VaR. These tests never call Yahoo or FRED."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from quantagent_desk.engines.market_risk import (
    MarketRiskEngine,
    direction_sign,
    position_pnl,
)
from quantagent_desk.marketdata import feed
from quantagent_desk.models.core import Portfolio, Position


def _move_panel(spx: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2026-09-28", periods=len(spx))
    return pd.DataFrame(
        {
            "SPX": spx,
            "USD_10Y": np.zeros(len(spx)),
            "EURUSD": np.zeros(len(spx)),
            "WTI": np.zeros(len(spx)),
            "IG_OAS": np.zeros(len(spx)),
        },
        index=dates,
    )


def _equity(direction: str) -> Position:
    return Position(
        instrument="EQ",
        asset_class="Equity",
        notional=1_000_000,
        direction=direction,  # type: ignore[arg-type]
        sensitivity=0.01,
    )


def test_fixture_var_matches_known_quantile():
    """20 ordered equity moves. VaR is the historical quantile of one shared vector."""
    spx = np.arange(-0.05, 0.15, 0.01)
    assert len(spx) == 20
    moves = _move_panel(spx)
    engine = MarketRiskEngine()
    snap = engine.from_portfolio_positions(
        Portfolio(portfolio_id="BOOK", name="Book", positions=[_equity("Long")]),
        returns=moves,
    )
    # pnl = +1 * 1_000_000 * 0.01 * move → -500, -400, ..., 1400
    # n=20, 95% index = floor(0.05*20) = 1 → loss of 400
    assert snap.var_95 == pytest.approx(400.0)
    assert snap.var_99 == pytest.approx(500.0)
    assert snap.methodology == f"Historical factors as of {moves.index.max().date().isoformat()}"


def test_short_flips_sign_and_changes_var():
    spx = np.arange(-0.05, 0.15, 0.01)
    moves = _move_panel(spx)
    long_pos = _equity("Long")
    short_pos = _equity("Short")
    long_pnl = position_pnl(long_pos, moves)
    short_pnl = position_pnl(short_pos, moves)
    assert long_pnl is not None and short_pnl is not None
    np.testing.assert_allclose(short_pnl, -long_pnl)
    assert direction_sign("Short") == -1.0
    assert direction_sign("Pay") == -1.0
    assert direction_sign("Receive") == 1.0

    engine = MarketRiskEngine()
    short = engine.from_portfolio_positions(
        Portfolio(portfolio_id="BOOK", name="Book", positions=[short_pos]),
        returns=moves,
    )
    # Negated P&L worst losses are 1400, 1300, ... so 95% VaR is 1300.
    assert short.var_95 == pytest.approx(1300.0)

    pay = Position(
        instrument="RATE",
        asset_class="Rates",
        notional=1_000_000,
        direction="Pay",
        sensitivity=0.01,
    )
    receive = pay.model_copy(update={"direction": "Receive"})
    rate_moves = moves.copy()
    rate_moves["USD_10Y"] = 0.001
    pay_pnl = position_pnl(pay, rate_moves)
    receive_pnl = position_pnl(receive, rate_moves)
    assert pay_pnl is not None and receive_pnl is not None
    np.testing.assert_allclose(pay_pnl, -receive_pnl)


def test_two_positions_share_one_factor_vector():
    spx = np.arange(-0.05, 0.15, 0.01)
    moves = _move_panel(spx)
    engine = MarketRiskEngine()
    one = engine.from_portfolio_positions(
        Portfolio(portfolio_id="BOOK", name="Book", positions=[_equity("Long")]),
        returns=moves,
    )
    two = engine.from_portfolio_positions(
        Portfolio(
            portfolio_id="BOOK",
            name="Book",
            positions=[_equity("Long"), _equity("Long")],
        ),
        returns=moves,
    )
    assert two.var_95 == pytest.approx(one.var_95 * 2)


def test_synthetic_fallback_when_cache_missing(monkeypatch):
    def _missing(*_args, **_kwargs):
        raise feed.MarketDataUnavailable("no cache")

    monkeypatch.setattr(
        "quantagent_desk.engines.market_risk.load_market_snapshot",
        _missing,
    )
    engine = MarketRiskEngine()
    snap = engine.from_portfolio_positions(
        Portfolio(portfolio_id="BOOK", name="Book", positions=[_equity("Long")])
    )
    assert snap.methodology == "Synthetic"
    assert snap.var_95 > 0


def test_level_cache_roundtrip_and_move_units(tmp_path):
    dates = pd.bdate_range("2026-10-01", periods=4)
    levels = pd.DataFrame(
        {
            "SPX": [100.0, 110.0, 121.0, 121.0],
            "VIX": [15.0, 16.0, 14.0, 13.0],
            "EURUSD": [1.10, 1.10, 1.21, 1.21],
            "WTI": [70.0, 77.0, 77.0, 70.0],
            "USD_10Y": [0.040, 0.041, 0.041, 0.039],
            "IG_OAS": [0.010, 0.010, 0.012, 0.011],
        },
        index=dates,
    )
    path = tmp_path / "factors.csv"
    feed.write_factor_cache(levels, path)
    loaded = feed.read_factor_cache(path)
    assert loaded is not None
    moves = feed.factor_moves(loaded)
    assert moves["SPX"].iloc[0] == pytest.approx(0.10)
    assert moves["USD_10Y"].iloc[0] == pytest.approx(0.001)
    assert moves["IG_OAS"].iloc[1] == pytest.approx(0.002)
    assert "VIX" not in moves.columns


def test_offline_refuses_network(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTAGENT_MARKET_OFFLINE", "1")

    def _boom(*_args, **_kwargs):
        raise AssertionError("network download was called")

    monkeypatch.setattr(feed, "download_factor_levels", _boom)
    monkeypatch.setattr(feed, "_download_yahoo", _boom)
    monkeypatch.setattr(feed, "_download_fred", _boom)
    with pytest.raises(feed.MarketDataUnavailable):
        feed.load_market_snapshot(path=tmp_path / "missing.csv")


def test_refresh_script_offline(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_MARKET_OFFLINE", "1")
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "refresh_market_data.py"
    spec = importlib.util.spec_from_file_location("refresh_market_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--as-of", "2026-10-27"]) == 2


def test_live_source_when_cache_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("QUANTAGENT_MARKET_OFFLINE", raising=False)
    dates = pd.bdate_range("2026-09-01", periods=12)
    levels = pd.DataFrame(
        {
            "SPX": np.linspace(5000, 5100, 12),
            "VIX": np.linspace(15, 16, 12),
            "EURUSD": np.linspace(1.08, 1.09, 12),
            "WTI": np.linspace(70, 72, 12),
            "USD_10Y": np.linspace(0.04, 0.041, 12),
            "IG_OAS": np.linspace(0.009, 0.010, 12),
        },
        index=dates,
    )

    def _fake_download(as_of):
        assert as_of.isoformat() == "2026-10-27"
        return levels

    monkeypatch.setattr(feed, "download_factor_levels", _fake_download)
    snap = feed.load_market_snapshot(as_of=feed.PINNED_AS_OF, path=tmp_path / "missing.csv")
    assert snap.source == "live"
    assert snap.as_of == dates.max().date()
    assert set(feed.LEVEL_COLUMNS) <= set(snap.levels)


def test_committed_cache_covers_pinned_as_of():
    snap = feed.load_market_snapshot()
    assert snap.source == "cache"
    assert snap.as_of is not None
    assert snap.as_of <= feed.PINNED_AS_OF
    assert snap.requested_as_of == feed.PINNED_AS_OF
    assert len(snap.moves) >= 200
    assert set(feed.LEVEL_COLUMNS) <= set(snap.levels)
    assert snap.levels["USD_10Y"] < 1  # stored as a decimal, not a percent
    assert snap.levels["IG_OAS"] < 1


def test_market_snapshot_endpoint():
    from fastapi.testclient import TestClient

    from quantagent_desk.api.main import app

    response = TestClient(app).get("/market/snapshot")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "cache"
    assert body["as_of"] <= "2026-10-27"
    assert body["requested_as_of"] == "2026-10-27"
    assert "SPX" in body["levels"]
    assert "VIX" in body["levels"]
    assert "IG OAS" in body["labels"].values() or body["labels"].get("IG_OAS") == "IG OAS"
    assert "credit proxy" in body["notes"]["IG_OAS"].lower()
