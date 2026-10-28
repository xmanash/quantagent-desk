"""Stress testing engine – historical replay + simple generative shocks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from quantagent_desk.models.core import (
    Portfolio,
    RiskFactorShock,
    StressResult,
    StressScenario,
)
from .registry import get_registry


# Pre-defined historical-style scenarios (public knowledge / educational)
HISTORICAL_SCENARIOS: Dict[str, StressScenario] = {
    "2008_GFC": StressScenario(
        name="2008 Global Financial Crisis",
        description="Severe equity crash, credit spreads blow-out, rates collapse, FX volatility spike.",
        type="Historical",
        shocks=[
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.40),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=-0.015),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.015),
            RiskFactorShock(factor="VIX", shock_type="absolute", value=40.0),
        ],
        grounded_sources=["Public market data 2008-2009"],
        tags=["crisis", "credit", "equity"],
    ),
    "COVID_2020": StressScenario(
        name="COVID-19 March 2020",
        description="Rapid equity sell-off, rates to zero, extreme vol, oil crash.",
        type="Historical",
        shocks=[
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.34),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=-0.012),
            RiskFactorShock(factor="WTI", shock_type="relative", value=-0.60),
            RiskFactorShock(factor="VIX", shock_type="absolute", value=50.0),
        ],
        grounded_sources=["Public market data March 2020"],
        tags=["pandemic", "liquidity"],
    ),
    "RATES_SPIKE_2022": StressScenario(
        name="2022 Rates Shock",
        description="Aggressive hiking cycle, equity correction, credit widening.",
        type="Historical",
        shocks=[
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=0.025),
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.20),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.008),
        ],
        grounded_sources=["Public market data 2022"],
        tags=["rates", "inflation"],
    ),
}


class StressEngine:
    """Apply shocks to portfolios and estimate P&L / risk impact."""

    def __init__(self):
        self.historical = HISTORICAL_SCENARIOS

    def list_historical(self) -> List[StressScenario]:
        return list(self.historical.values())

    def get_historical(self, key: str) -> Optional[StressScenario]:
        return self.historical.get(key)

    def create_generative_scenario(
        self,
        name: str,
        description: str,
        shocks: List[RiskFactorShock],
        sources: Optional[List[str]] = None,
    ) -> StressScenario:
        return StressScenario(
            name=name,
            description=description,
            type="Generative",
            shocks=shocks,
            generated_by="agent",
            grounded_sources=sources or ["LLM + historical analogues"],
            tags=["generative"],
        )

    def apply_stress(
        self,
        portfolio: Portfolio,
        scenario: StressScenario,
        base_var: float = 0.0,
    ) -> StressResult:
        """
        Very simple linear impact model.
        In production replace with full revaluation under shocked market data.
        """
        pnl_impact = 0.0
        breakdown: Dict[str, float] = {}

        for pos in portfolio.positions:
            # crude mapping: equity-like positions sensitive to SPX, rates to USD_10Y etc.
            factor_map = {
                "Equity": "SPX",
                "Rates": "USD_10Y",
                "Credit": "CDX_IG",
                "FX": "EURUSD",
                "Commodity": "WTI",
            }
            primary = factor_map.get(pos.asset_class, "SPX")
            sens = pos.sensitivity if pos.sensitivity is not None else 0.01

            impact = 0.0
            for shock in scenario.shocks:
                if shock.factor == primary or (
                    pos.asset_class == "Rates" and "USD" in shock.factor
                ):
                    if shock.shock_type == "relative":
                        impact += pos.notional * sens * shock.value
                    else:  # absolute
                        impact += pos.notional * sens * shock.value * 100  # rough scaling

            pnl_impact += impact
            breakdown[pos.instrument] = impact

        # Approximate VaR impact (assume VaR scales with stress severity)
        severity = np.mean([abs(s.value) for s in scenario.shocks]) if scenario.shocks else 0.1
        var_impact = base_var * (1 + severity * 2) - base_var

        res = StressResult(
            scenario_id=scenario.scenario_id,
            portfolio_id=portfolio.portfolio_id,
            pnl_impact=float(pnl_impact),
            var_impact=float(var_impact),
            breakdown=breakdown,
            calculation_id=str(uuid4()),
            as_of=datetime.now(timezone.utc),
        )
        get_registry().register(res.calculation_id, "stress_result", portfolio.portfolio_id, res.model_dump())
        return res

    def hybrid_scenario(
        self,
        base_key: str,
        extra_shocks: List[RiskFactorShock],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> StressScenario:
        """Combine a historical scenario with additional generative shocks."""
        base = self.get_historical(base_key)
        if not base:
            raise ValueError(f"Unknown historical scenario: {base_key}")

        combined_shocks = list(base.shocks) + extra_shocks
        return StressScenario(
            name=name or f"Hybrid: {base.name}",
            description=description or f"{base.description} + additional generative shocks",
            type="Hybrid",
            shocks=combined_shocks,
            generated_by="agent",
            grounded_sources=base.grounded_sources + ["Generative overlay"],
            tags=base.tags + ["hybrid"],
        )
