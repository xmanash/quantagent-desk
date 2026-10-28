#!/usr/bin/env python3
"""Generate and print a synthetic firm portfolio for inspection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent_desk.engines.portfolio import PortfolioEngine
from quantagent_desk.engines.market_risk import MarketRiskEngine
from quantagent_desk.engines.limits import LimitsEngine


def main():
    pe = PortfolioEngine()
    firm = pe.create_synthetic_firm()
    print(f"Created firm: {firm.name} ({firm.portfolio_id})")
    print(f"Children desks: {firm.children_ids}")

    mre = MarketRiskEngine()
    le = LimitsEngine()

    for desk_id in firm.children_ids:
        desk = pe.get(desk_id)
        positions = pe.flatten_positions(desk_id)
        from quantagent_desk.models.core import Portfolio
        temp = Portfolio(portfolio_id=desk_id, name=desk.name, positions=positions)
        snap = mre.from_portfolio_positions(temp)
        breaches = le.check(snap)
        print(f"\n{desk.name}:")
        print(f"  Positions: {len(positions)}")
        print(f"  VaR 95%: ${snap.var_95:,.0f}")
        print(f"  ES 97.5%: ${snap.es_97_5:,.0f}")
        if breaches:
            for b in breaches:
                print(f"  {b.severity}: {b.limit_type}")


if __name__ == "__main__":
    main()
