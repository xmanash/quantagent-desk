"""Simple limit monitoring engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from quantagent_desk.models.core import LimitBreach, RiskSnapshot


class LimitsEngine:
    """Track utilization against soft / hard limits and raise breaches."""

    def __init__(self, default_limits: Optional[Dict[str, float]] = None):
        # Example soft limits (USD)
        self.limits = default_limits or {
            "VaR_95": 50_000_000,
            "VaR_99": 80_000_000,
            "ES_97_5": 70_000_000,
            "Peak_PFE": 200_000_000,
            "CVA": 15_000_000,
        }
        self.warning_threshold = 0.80  # 80% = warning
        self.breach_threshold = 1.00

    def update_limits(self, new_limits: Dict[str, float]) -> None:
        self.limits.update(new_limits)

    def check(
        self,
        snapshot: RiskSnapshot,
        portfolio_id: Optional[str] = None,
    ) -> List[LimitBreach]:
        """Return list of warnings / breaches for a risk snapshot."""
        pid = portfolio_id or snapshot.portfolio_id
        breaches: List[LimitBreach] = []

        metrics = {
            "VaR_95": snapshot.var_95,
            "VaR_99": snapshot.var_99,
            "ES_97_5": snapshot.es_97_5,
            "Peak_PFE": snapshot.peak_pfe or 0.0,
            "CVA": snapshot.cva or 0.0,
        }

        utilization: Dict[str, float] = {}

        for name, value in metrics.items():
            limit = self.limits.get(name)
            if limit is None or limit <= 0:
                continue
            util = value / limit
            utilization[name] = util

            if util >= self.breach_threshold:
                severity = "Breach"
            elif util >= self.warning_threshold:
                severity = "Warning"
            else:
                continue

            breaches.append(
                LimitBreach(
                    portfolio_id=pid,
                    limit_type=name,
                    current_value=value,
                    limit_value=limit,
                    severity=severity,
                    timestamp=datetime.now(timezone.utc),
                    message=f"{name} utilization {util:.1%} ({severity})",
                )
            )

        # attach utilization back to snapshot for convenience
        snapshot.limit_utilization = utilization
        return breaches

    def status_summary(self, snapshot: RiskSnapshot) -> Dict[str, float]:
        """Return current utilization percentages."""
        _ = self.check(snapshot)  # populates utilization
        return snapshot.limit_utilization
