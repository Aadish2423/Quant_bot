"""
position.py
───────────
Tracks a single open or closed position in one asset.
Portfolio holds a dict of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from .order import ExecutedOrder, OrderSide


class PositionStatus(Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """
    Represents a long position in a single asset.

    Attributes
    ----------
    ticker       : Asset symbol.
    quantity     : Number of units held (always > 0 for a long position).
    avg_price    : Volume-weighted average entry price across all fills.
    opened_at    : Timestamp of the first fill.
    regime_at_open : Market regime label when the position was opened.
    realized_pnl : Accumulated PnL from partial closes.
    closed_at    : Set when the position is fully closed.
    exit_price   : Average exit price across all closing fills.
    status       : OPEN or CLOSED.
    """

    ticker:          str
    quantity:        float
    avg_price:       float
    opened_at:       pd.Timestamp
    regime_at_open:  str            = "UNKNOWN"
    realized_pnl:    float          = 0.0
    closed_at:       Optional[pd.Timestamp] = None
    exit_price:      Optional[float]        = None
    status:          PositionStatus         = PositionStatus.OPEN

    # internal accounting
    _total_cost:     float = field(default=0.0, repr=False)   # cumulative commissions

    # ── mark-to-market ────────────────────────────────────────────────────

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealised PnL at current_price (does not include commissions)."""
        return (current_price - self.avg_price) * self.quantity

    def market_value(self, current_price: float) -> float:
        """Current market value of the position."""
        return current_price * self.quantity

    def return_pct(self, current_price: float) -> float:
        """Percentage return relative to average entry price."""
        if self.avg_price == 0:
            return 0.0
        return (current_price - self.avg_price) / self.avg_price

    # ── mutators ─────────────────────────────────────────────────────────

    def add_fill(self, executed: ExecutedOrder) -> None:
        """
        Increase position size (pyramid / scale-in).
        Recalculates the VWAP entry price.
        """
        assert executed.side == OrderSide.BUY
        new_qty   = self.quantity + executed.quantity
        self.avg_price = (
            (self.avg_price * self.quantity + executed.fill_price * executed.quantity)
            / new_qty
        )
        self.quantity = new_qty
        self._total_cost += executed.commission

    def close_fill(self, executed: ExecutedOrder) -> float:
        """
        Reduce or close position from a SELL fill.

        Returns the realised PnL for the closed quantity.
        """
        assert executed.side == OrderSide.SELL
        close_qty   = min(executed.quantity, self.quantity)
        pnl         = (executed.fill_price - self.avg_price) * close_qty - executed.commission
        self.realized_pnl += pnl
        self.quantity     -= close_qty
        self._total_cost  += executed.commission

        if self.quantity <= 1e-9:   # fully closed (float tolerance)
            self.quantity  = 0.0
            self.status    = PositionStatus.CLOSED
            self.closed_at = executed.filled_at
            self.exit_price = executed.fill_price

        return pnl

    # ── serialisation ─────────────────────────────────────────────────────

    def snapshot(self, current_price: float) -> dict:
        """Return a plain-dict view used by portfolio_snapshot()."""
        return {
            "ticker":         self.ticker,
            "quantity":       self.quantity,
            "avg_price":      round(self.avg_price, 4),
            "current_price":  round(current_price, 4),
            "unrealized_pnl": round(self.unrealized_pnl(current_price), 2),
            "market_value":   round(self.market_value(current_price), 2),
            "return_pct":     round(self.return_pct(current_price) * 100, 2),
            "status":         self.status.value,
            "opened_at":      str(self.opened_at.date()),
            "regime_at_open": self.regime_at_open,
        }
