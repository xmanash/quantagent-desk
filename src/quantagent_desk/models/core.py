"""Core Pydantic data models for QuantAgent-Desk."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _id() -> str:
    return str(uuid4())


class PortfolioLevel(str, Enum):
    FIRM = "Firm"
    DESK = "Desk"
    BOOK = "Book"
    NETTING_SET = "NettingSet"


class Position(BaseModel):
    """Simplified trade / position representation."""
    position_id: str = Field(default_factory=_id)
    instrument: str
    asset_class: Literal["Rates", "FX", "Equity", "Credit", "Commodity", "Other"] = "Rates"
    notional: float
    currency: str = "USD"
    direction: Literal["Long", "Short", "Pay", "Receive"] = "Long"
    maturity: Optional[date] = None
    sensitivity: Optional[float] = None  # e.g. DV01 or delta
    counterparty_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Portfolio(BaseModel):
    """Hierarchical portfolio node."""
    portfolio_id: str = Field(default_factory=_id)
    name: str
    level: PortfolioLevel = PortfolioLevel.BOOK
    parent_id: Optional[str] = None
    as_of: date = Field(default_factory=date.today)
    positions: List[Position] = Field(default_factory=list)
    children_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_notional(self) -> float:
        return sum(abs(p.notional) for p in self.positions)


class RiskSnapshot(BaseModel):
    """Point-in-time risk metrics for a portfolio."""
    snapshot_id: str = Field(default_factory=_id)
    portfolio_id: str
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    var_95: float = 0.0
    var_99: float = 0.0
    es_97_5: float = 0.0
    peak_pfe: Optional[float] = None
    cva: Optional[float] = None
    limit_utilization: Dict[str, float] = Field(default_factory=dict)
    calculation_id: str = Field(default_factory=_id)
    methodology: str = "Historical"
    notes: Optional[str] = None


class RiskFactorShock(BaseModel):
    """Single factor shock definition."""
    factor: str  # e.g. "USD_10Y", "SPX", "EURUSD"
    shock_type: Literal["absolute", "relative", "vol"] = "relative"
    value: float  # e.g. 0.01 for +100bps or +1%
    unit: str = "decimal"


class StressScenario(BaseModel):
    """Stress scenario definition (historical, generative, or hybrid)."""
    scenario_id: str = Field(default_factory=_id)
    name: str
    description: str
    type: Literal["Historical", "Generative", "Hybrid", "UserDefined"] = "Generative"
    shocks: List[RiskFactorShock] = Field(default_factory=list)
    generated_by: str = "agent"
    grounded_sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)


class StressResult(BaseModel):
    """Impact of a stress scenario on a portfolio."""
    result_id: str = Field(default_factory=_id)
    scenario_id: str
    portfolio_id: str
    pnl_impact: float = 0.0
    var_impact: float = 0.0
    pfe_impact: Optional[float] = None
    cva_impact: Optional[float] = None
    capital_impact: Optional[float] = None
    breakdown: Dict[str, float] = Field(default_factory=dict)  # desk or factor level
    calculation_id: str = Field(default_factory=_id)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LimitBreach(BaseModel):
    """Limit warning or breach event."""
    breach_id: str = Field(default_factory=_id)
    portfolio_id: str
    limit_type: str  # "VaR", "PFE", "CVA", "Notional"
    current_value: float
    limit_value: float
    severity: Literal["Warning", "Breach"] = "Warning"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    message: Optional[str] = None


class QueryRequest(BaseModel):
    """Natural language query to the risk desk agent."""
    query: str
    session_id: str = Field(default_factory=_id)
    portfolio_id: Optional[str] = "FIRM"
    include_charts: bool = True
    max_tokens: Optional[int] = None


class QueryResponse(BaseModel):
    """Agent response with grounding and metadata."""
    answer: str
    session_id: str
    calculation_ids: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    risk_snapshot: Optional[RiskSnapshot] = None
    stress_result: Optional[StressResult] = None
    grounded: bool = True
    warnings: List[str] = Field(default_factory=list)
    research: List[str] = Field(default_factory=list)
    token_use: Dict[str, int] = Field(default_factory=dict)
