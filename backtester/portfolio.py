"""
portfolio.py
────────────
Central portfolio ledger.  Owns cash, open positions, and closed positions.
All mutations go through apply_executed_order() so the state is always
consistent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .order import ExecutedOrder, OrderSide
from .position import Position, PositionStatus


class Portfolio:
    """
    Manages capital allocation across multiple assets.

    Parameters
    ----------
    initial_capital : Starting cash balance in $.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self._initial_capital: float  = initial_capital
        self._cash:            float  = initial_capital
        self._open_positions:  Dict[str, Position] = {}
        self._closed_positions: List[Position]     = []
        self._realized_pnl:    float  = 0.0

    # ── read-only properties ──────────────────────────────────────────────

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def open_positions(self) -> Dict[str, Position]:
        return dict(self._open_positions)

    @property
    def closed_positions(self) -> List[Position]:
        return list(self._closed_positions)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    # ── computed values (require current prices) ──────────────────────────

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        return sum(
            pos.unrealized_pnl(prices.get(ticker, pos.avg_price))
            for ticker, pos in self._open_positions.items()
        )

    def positions_value(self, prices: Dict[str, float]) -> float:
        return sum(
            pos.market_value(prices.get(ticker, pos.avg_price))
            for ticker, pos in self._open_positions.items()
        )

    def total_equity(self, prices: Dict[str, float]) -> float:
        """Cash + mark-to-market value of all open positions."""
        return self._cash + self.positions_value(prices)

    def total_return(self, prices: Dict[str, float]) -> float:
        return (self.total_equity(prices) - self._initial_capital) / self._initial_capital

    # ── mutation ──────────────────────────────────────────────────────────

    def apply_executed_order(
        self,
        executed:  ExecutedOrder,
        regime:    str = "UNKNOWN",
    ) -> float:
        """
        Apply a filled order to portfolio state.

        Returns realised PnL of the trade (0 for buys, > or < 0 for sells).
        """
        ticker = executed.ticker
        pnl    = 0.0

        if executed.side == OrderSide.BUY:
            self._cash += executed.net_cash_impact   # negative value → reduces cash
            if ticker in self._open_positions:
                self._open_positions[ticker].add_fill(executed)
            else:
                self._open_positions[ticker] = Position(
                    ticker         = ticker,
                    quantity       = executed.quantity,
                    avg_price      = executed.fill_price,
                    opened_at      = executed.filled_at,
                    regime_at_open = regime,
                    _total_cost    = executed.commission,
                )

        elif executed.side == OrderSide.SELL:
            if ticker not in self._open_positions:
                return 0.0                            # nothing to sell

            pos = self._open_positions[ticker]
            pnl = pos.close_fill(executed)
            self._realized_pnl += pnl
            self._cash         += executed.net_cash_impact   # positive → increases cash

            if pos.status == PositionStatus.CLOSED:
                self._closed_positions.append(pos)
                del self._open_positions[ticker]

        return pnl

    def can_afford(self, cost: float) -> bool:
        """Check whether ``cost`` in $ can be covered by current cash."""
        return self._cash >= cost

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._open_positions.get(ticker)

    def has_position(self, ticker: str) -> bool:
        return ticker in self._open_positions

    # ── snapshots ─────────────────────────────────────────────────────────

    def snapshot(self, prices: Dict[str, float]) -> dict:
        """
        Light-weight read-only dict passed to agents on every bar.
        Agents must not mutate this.
        """
        return {
            "cash":          round(self._cash, 2),
            "equity":        round(self.total_equity(prices), 2),
            "unrealized_pnl": round(self.unrealized_pnl(prices), 2),
            "realized_pnl":  round(self._realized_pnl, 2),
            "positions": {
                t: {"qty": p.quantity, "avg_price": p.avg_price}
                for t, p in self._open_positions.items()
            },
        }

    def full_snapshot(self, prices: Dict[str, float]) -> dict:
        """Detailed snapshot for logging / reporting."""
        return {
            "initial_capital": self._initial_capital,
            "cash":            round(self._cash, 2),
            "positions_value": round(self.positions_value(prices), 2),
            "total_equity":    round(self.total_equity(prices), 2),
            "unrealized_pnl":  round(self.unrealized_pnl(prices), 2),
            "realized_pnl":    round(self._realized_pnl, 2),
            "total_return_pct": round(self.total_return(prices) * 100, 2),
            "num_open_positions": len(self._open_positions),
            "open_positions": {
                t: p.snapshot(prices.get(t, p.avg_price))
                for t, p in self._open_positions.items()
            },
        }
