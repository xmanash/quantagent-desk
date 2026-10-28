"""Portfolio hierarchy and synthetic data helpers."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from quantagent_desk.models.core import Portfolio, PortfolioLevel, Position


class PortfolioEngine:
    """Manage portfolio hierarchy and generate synthetic demo books."""

    def __init__(self):
        self.portfolios: Dict[str, Portfolio] = {}

    def create_synthetic_firm(self, seed: int = 42) -> Portfolio:
        """Create a simple Firm → Desk → Book hierarchy with positions."""
        rng = np.random.default_rng(seed)

        firm = Portfolio(
            portfolio_id="FIRM",
            name="Global Markets Firm",
            level=PortfolioLevel.FIRM,
            as_of=date.today(),
        )

        desks = [
            ("RATES_DESK", "Rates Trading"),
            ("FX_DESK", "FX Trading"),
            ("EQ_DESK", "Equity Derivatives"),
        ]

        for desk_id, desk_name in desks:
            desk = Portfolio(
                portfolio_id=desk_id,
                name=desk_name,
                level=PortfolioLevel.DESK,
                parent_id=firm.portfolio_id,
            )
            firm.children_ids.append(desk_id)
            self.portfolios[desk_id] = desk

            # two books per desk
            for i in range(1, 3):
                book_id = f"{desk_id}_BOOK_{i}"
                book = Portfolio(
                    portfolio_id=book_id,
                    name=f"{desk_name} Book {i}",
                    level=PortfolioLevel.BOOK,
                    parent_id=desk_id,
                )
                desk.children_ids.append(book_id)

                # synthetic positions
                for j in range(3):
                    asset = {
                        "RATES_DESK": "Rates",
                        "FX_DESK": "FX",
                        "EQ_DESK": "Equity",
                    }[desk_id]
                    pos = Position(
                        instrument=f"{asset}_INSTR_{j+1}",
                        asset_class=asset,  # type: ignore
                        notional=float(rng.integers(10_000_000, 100_000_000)),
                        direction="Long" if rng.random() > 0.4 else "Short",
                        sensitivity=float(rng.uniform(0.005, 0.05)),
                    )
                    book.positions.append(pos)

                self.portfolios[book_id] = book

        self.portfolios[firm.portfolio_id] = firm
        return firm

    def get(self, portfolio_id: str) -> Optional[Portfolio]:
        return self.portfolios.get(portfolio_id)

    def get_children(self, portfolio_id: str) -> List[Portfolio]:
        parent = self.get(portfolio_id)
        if not parent:
            return []
        return [self.portfolios[cid] for cid in parent.children_ids if cid in self.portfolios]

    def flatten_positions(self, portfolio_id: str) -> List[Position]:
        """Recursively collect all positions under a node."""
        node = self.get(portfolio_id)
        if not node:
            return []
        positions = list(node.positions)
        for child in self.get_children(portfolio_id):
            positions.extend(self.flatten_positions(child.portfolio_id))
        return positions

    def add_portfolio(self, portfolio: Portfolio) -> None:
        self.portfolios[portfolio.portfolio_id] = portfolio
