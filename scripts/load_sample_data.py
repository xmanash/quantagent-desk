#!/usr/bin/env python3
"""Load sample_positions.csv into a demo portfolio and print a quick risk snapshot."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent_desk.engines.portfolio import PortfolioEngine
from quantagent_desk.engines.market_risk import MarketRiskEngine
from quantagent_desk.engines.limits import LimitsEngine
from quantagent_desk.engines.stress import StressEngine
from quantagent_desk.models.core import Portfolio, Position, PortfolioLevel

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "synthetic" / "sample_positions.csv"


def main():
    pe = PortfolioEngine()
    mre = MarketRiskEngine()
    le = LimitsEngine()
    se = StressEngine()

    # Build firm + one book from CSV
    firm = Portfolio(
        portfolio_id="DEMO_FIRM",
        name="Demo Markets Firm",
        level=PortfolioLevel.FIRM,
    )
    book = Portfolio(
        portfolio_id="DEMO_BOOK",
        name="Demo Book",
        level=PortfolioLevel.BOOK,
        parent_id="DEMO_FIRM",
    )
    firm.children_ids.append("DEMO_BOOK")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book.positions.append(
                Position(
                    position_id=row["position_id"],
                    instrument=row["instrument"],
                    asset_class=row["asset_class"],  # type: ignore
                    notional=float(row["notional"]),
                    currency=row["currency"],
                    direction=row["direction"],  # type: ignore
                    sensitivity=float(row["sensitivity"]),
                    counterparty_id=row.get("counterparty_id"),
                )
            )

    pe.add_portfolio(firm)
    pe.add_portfolio(book)

    print(f"Loaded {len(book.positions)} positions into DEMO_FIRM / DEMO_BOOK\n")

    snap = mre.from_portfolio_positions(book)
    breaches = le.check(snap, "DEMO_FIRM")

    print("=== Risk Snapshot ===")
    print(f"  VaR 95%:  ${snap.var_95:,.0f}")
    print(f"  VaR 99%:  ${snap.var_99:,.0f}")
    print(f"  ES 97.5%: ${snap.es_97_5:,.0f}")
    if breaches:
        print("  Limits:", ", ".join(f"{b.severity}:{b.limit_type}" for b in breaches))
    else:
        print("  Limits: OK")

    scenario = se.get_historical("2008_GFC")
    result = se.apply_stress(book, scenario, base_var=snap.var_95)
    print("\n=== 2008 GFC Stress ===")
    print(f"  P&L Impact: ${result.pnl_impact:,.0f}")
    print(f"  VaR Impact: ${result.var_impact:,.0f}")
    print("\nSample data ready for demo.")


if __name__ == "__main__":
    main()
