"""
logger.py
─────────
Immutable trade record and the logger that accumulates them.
Separate from analytics so each concern is testable independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .order import ExecutedOrder, OrderSide


@dataclass(frozen=True)
class TradeRecord:
    """
    One completed trade (entry + exit pair or single-leg record).

    All monetary values are in $.
    """
    trade_id:      int
    ticker:        str
    side:          str          # "BUY" or "SELL"
    quantity:      float
    entry_price:   float
    exit_price:    Optional[float]    # None for open positions
    gross_pnl:     float              # before commissions
    net_pnl:       float              # after commissions
    commission:    float
    slippage_cost: float
    regime:        str
    opened_at:     pd.Timestamp
    closed_at:     Optional[pd.Timestamp]
    hold_bars:     Optional[int]      # bars held; None if still open
    return_pct:    float              # net_pnl / (entry_price * quantity)


class TradeLogger:
    """
    Accumulates TradeRecords throughout a backtest run.

    The engine calls log_open() on every BUY fill and log_close()
    on every SELL fill.  get_dataframe() returns the full log as a
    tidy DataFrame for analytics.
    """

    def __init__(self):
        self._records:    List[TradeRecord]  = []
        self._open_log:   dict               = {}   # ticker → {entry info}
        self._trade_counter: int             = 0
        self._rejected_orders: List[dict]    = []

    # ── logging interface ─────────────────────────────────────────────────

    def log_open(
        self,
        executed: ExecutedOrder,
        regime:   str,
        bar_idx:  int,
    ) -> None:
        """Record a BUY fill.  Stores entry info keyed by ticker."""
        self._open_log[executed.ticker] = {
            "entry_price":  executed.fill_price,
            "quantity":     executed.quantity,
            "commission":   executed.commission,
            "slippage":     executed.slippage_cost,
            "regime":       regime,
            "opened_at":    executed.filled_at,
            "bar_idx":      bar_idx,
        }

    def log_close(
        self,
        executed: ExecutedOrder,
        net_pnl:  float,
        regime:   str,
        bar_idx:  int,
    ) -> None:
        """Record a SELL fill and emit a completed TradeRecord."""
        entry = self._open_log.pop(executed.ticker, None)
        if entry is None:
            # Sell without a logged open — create a minimal record
            entry = {
                "entry_price": executed.fill_price,
                "quantity":    executed.quantity,
                "commission":  0.0,
                "slippage":    0.0,
                "regime":      regime,
                "opened_at":   executed.filled_at,
                "bar_idx":     bar_idx,
            }

        total_commission = entry["commission"] + executed.commission
        total_slippage   = entry["slippage"]   + executed.slippage_cost
        gross_pnl        = (executed.fill_price - entry["entry_price"]) * entry["quantity"]
        cost_basis       = entry["entry_price"] * entry["quantity"]
        return_pct       = net_pnl / cost_basis if cost_basis else 0.0

        self._trade_counter += 1
        record = TradeRecord(
            trade_id      = self._trade_counter,
            ticker        = executed.ticker,
            side          = "SELL",
            quantity      = entry["quantity"],
            entry_price   = entry["entry_price"],
            exit_price    = executed.fill_price,
            gross_pnl     = round(gross_pnl, 2),
            net_pnl       = round(net_pnl, 2),
            commission    = round(total_commission, 2),
            slippage_cost = round(total_slippage, 2),
            regime        = entry["regime"],
            opened_at     = entry["opened_at"],
            closed_at     = executed.filled_at,
            hold_bars     = bar_idx - entry["bar_idx"],
            return_pct    = round(return_pct * 100, 4),
        )
        self._records.append(record)

    def log_rejected(self, ticker: str, reason: str, timestamp: pd.Timestamp) -> None:
        self._rejected_orders.append({
            "ticker":    ticker,
            "reason":    reason,
            "timestamp": timestamp,
        })

    # ── retrieval ─────────────────────────────────────────────────────────

    def get_records(self) -> List[TradeRecord]:
        return list(self._records)

    def get_dataframe(self) -> pd.DataFrame:
        """Return all completed trade records as a tidy DataFrame."""
        if not self._records:
            return pd.DataFrame()
        rows = [r.__dict__ for r in self._records]
        return pd.DataFrame(rows).set_index("trade_id")

    def get_rejected_dataframe(self) -> pd.DataFrame:
        if not self._rejected_orders:
            return pd.DataFrame()
        return pd.DataFrame(self._rejected_orders)

    @property
    def num_trades(self) -> int:
        return len(self._records)
