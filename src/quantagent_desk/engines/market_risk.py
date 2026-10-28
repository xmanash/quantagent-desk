"""Simple Historical and Parametric VaR / ES engine.

This is intentionally lightweight and educational.
Replace with full revaluation or production QuantLib-based engines later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from quantagent_desk.marketdata.feed import MarketDataUnavailable, factor_moves, load_market_snapshot
from quantagent_desk.models.core import Portfolio, Position, RiskSnapshot
from .registry import get_registry

# Long and Receive take the factor move as-is. Short and Pay flip it.
_DIRECTION_SIGN = {"Long": 1.0, "Receive": 1.0, "Short": -1.0, "Pay": -1.0}

# Same primary factor as the stress engine. IG OAS stands in for CDX IG.
ASSET_FACTOR = {
    "Equity": "SPX",
    "Rates": "USD_10Y",
    "FX": "EURUSD",
    "Credit": "IG_OAS",
    "Commodity": "WTI",
}


class MarketRiskEngine:
    """Compute VaR and Expected Shortfall from P&L vectors or synthetic returns."""

    def __init__(self, confidence_levels: Optional[List[float]] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.es_level = 0.975

    def historical_var_es(
        self,
        pnl_vector: np.ndarray,
        portfolio_id: str = "unknown",
        methodology: str = "Historical",
    ) -> RiskSnapshot:
        """Compute Historical VaR / ES from a 1-day P&L vector."""
        if len(pnl_vector) < 10:
            raise ValueError("Need at least 10 observations for Historical VaR")

        sorted_pnl = np.sort(pnl_vector)  # ascending (worst losses first)
        n = len(sorted_pnl)

        def _var(alpha: float) -> float:
            idx = int(np.floor((1 - alpha) * n))
            idx = max(0, min(idx, n - 1))
            return float(-sorted_pnl[idx])  # positive number = loss

        var_95 = _var(0.95)
        var_99 = _var(0.99)

        # Expected Shortfall at 97.5%
        es_idx = int(np.floor((1 - self.es_level) * n))
        es_97_5 = float(-np.mean(sorted_pnl[: es_idx + 1])) if es_idx >= 0 else var_95

        snap = RiskSnapshot(
            portfolio_id=portfolio_id,
            as_of=datetime.now(timezone.utc),
            var_95=var_95,
            var_99=var_99,
            es_97_5=es_97_5,
            methodology=methodology,
            calculation_id=str(uuid4()),
        )
        get_registry().register(snap.calculation_id, "risk_snapshot", portfolio_id, snap.model_dump())
        return snap

    def parametric_var_es(
        self,
        mean_pnl: float,
        std_pnl: float,
        portfolio_id: str = "unknown",
    ) -> RiskSnapshot:
        """Simple parametric (normal) VaR / ES."""
        from scipy.stats import norm

        var_95 = float(-(mean_pnl + norm.ppf(0.05) * std_pnl))
        var_99 = float(-(mean_pnl + norm.ppf(0.01) * std_pnl))
        # ES for normal: mean + std * phi(ppf(alpha)) / alpha
        es_97_5 = float(
            -(mean_pnl + std_pnl * norm.pdf(norm.ppf(0.025)) / 0.025)
        )

        snap = RiskSnapshot(
            portfolio_id=portfolio_id,
            as_of=datetime.now(timezone.utc),
            var_95=max(0.0, var_95),
            var_99=max(0.0, var_99),
            es_97_5=max(0.0, es_97_5),
            methodology="Parametric-Normal",
            calculation_id=str(uuid4()),
        )
        get_registry().register(snap.calculation_id, "risk_snapshot", portfolio_id, snap.model_dump())
        return snap

    def from_portfolio_positions(
        self,
        portfolio: Portfolio,
        returns: Optional[pd.DataFrame] = None,
        horizon_days: int = 1,
    ) -> RiskSnapshot:
        """Sensitivity P&L on one shared factor-move vector.

        ``pnl = sign(direction) * notional * sensitivity * factor_move``.

        ``returns``, when passed, is that aligned move panel (or a level
        panel, which is converted). Otherwise the pinned factor cache is
        used. A missing cache keeps the seeded normal simulation and labels
        the snapshot ``Synthetic``. ``horizon_days`` is accepted for callers
        and does not rescale the 1-day history.
        """
        del horizon_days
        if not portfolio.positions:
            return RiskSnapshot(portfolio_id=portfolio.portfolio_id)

        if returns is not None:
            moves, as_of_label = _moves_from_returns(returns)
            if moves is None or len(moves) < 10:
                return self._synthetic_from_positions(portfolio)
            return self._historical_from_moves(portfolio, moves, as_of_label)

        try:
            market = load_market_snapshot()
        except MarketDataUnavailable:
            return self._synthetic_from_positions(portfolio)
        if market.as_of is None or len(market.moves) < 10:
            return self._synthetic_from_positions(portfolio)
        return self._historical_from_moves(portfolio, market.moves, market.as_of.isoformat())

    def _historical_from_moves(
        self,
        portfolio: Portfolio,
        moves: pd.DataFrame,
        as_of_label: Optional[str],
    ) -> RiskSnapshot:
        pnl = np.zeros(len(moves), dtype=float)
        used = False
        for pos in portfolio.positions:
            leg = position_pnl(pos, moves)
            if leg is None:
                continue
            pnl = pnl + leg
            used = True
        if not used:
            return self._synthetic_from_positions(portfolio)
        methodology = (
            f"Historical factors as of {as_of_label}" if as_of_label else "Historical factors"
        )
        return self.historical_var_es(
            pnl,
            portfolio_id=portfolio.portfolio_id,
            methodology=methodology,
        )

    def _synthetic_from_positions(self, portfolio: Portfolio) -> RiskSnapshot:
        """Independent normal shocks at a flat 1% daily vol. Seeded for demos."""
        rng = np.random.default_rng(42)
        n_obs = 500
        pnl_list = []
        for pos in portfolio.positions:
            sens = pos.sensitivity if pos.sensitivity is not None else 0.01
            daily_vol = 0.01 * abs(pos.notional) * abs(sens)
            pnl_list.append(rng.normal(0, daily_vol, n_obs))
        total_pnl = np.sum(pnl_list, axis=0)
        return self.historical_var_es(
            total_pnl,
            portfolio_id=portfolio.portfolio_id,
            methodology="Synthetic",
        )

    def aggregate_snapshots(
        self, snapshots: List[RiskSnapshot], portfolio_id: str = "FIRM"
    ) -> RiskSnapshot:
        """Simple sum aggregation (no diversification benefit)."""
        if not snapshots:
            return RiskSnapshot(portfolio_id=portfolio_id)

        snap = RiskSnapshot(
            portfolio_id=portfolio_id,
            var_95=sum(s.var_95 for s in snapshots),
            var_99=sum(s.var_99 for s in snapshots),
            es_97_5=sum(s.es_97_5 for s in snapshots),
            peak_pfe=sum(s.peak_pfe or 0 for s in snapshots) or None,
            cva=sum(s.cva or 0 for s in snapshots) or None,
            methodology="Aggregated",
            calculation_id=str(uuid4()),
        )
        get_registry().register(snap.calculation_id, "risk_snapshot", portfolio_id, snap.model_dump())
        return snap


def direction_sign(direction: str) -> float:
    """Long and Receive are +1. Short and Pay are -1."""
    return _DIRECTION_SIGN.get(direction, 1.0)


def position_pnl(position: Position, moves: pd.DataFrame) -> Optional[np.ndarray]:
    """P&L of one position on the shared factor-move dates. None if unmapped."""
    column = ASSET_FACTOR.get(position.asset_class)
    if column is None or column not in moves.columns:
        return None
    move = pd.to_numeric(moves[column], errors="coerce").to_numpy(dtype=float)
    sens = position.sensitivity if position.sensitivity is not None else 0.01
    return direction_sign(position.direction) * float(position.notional) * float(sens) * move


def _moves_from_returns(returns: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Accept a move panel, or a level panel (SPX prices are well above 1)."""
    frame = returns.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except (TypeError, ValueError, pd.errors.ParserError):
            pass
    as_of_label = None
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
        as_of_label = pd.Timestamp(frame.index.max()).date().isoformat()
    if frame.empty:
        return None, as_of_label
    if "SPX" in frame.columns and float(frame["SPX"].abs().median()) > 1.0:
        frame = factor_moves(frame)
        if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
            as_of_label = pd.Timestamp(frame.index.max()).date().isoformat()
    factor_columns = [column for column in frame.columns if column in set(ASSET_FACTOR.values())]
    if factor_columns:
        frame = frame[factor_columns].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if frame.empty:
        return None, as_of_label
    return frame, as_of_label
