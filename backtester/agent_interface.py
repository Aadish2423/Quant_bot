"""
agent_interface.py
──────────────────
Abstract base class every trading agent must implement.
The engine calls generate_signal() on each bar; agents never touch
portfolio or order internals directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Standardised signal returned by every agent on each bar.

    Attributes
    ----------
    signal_type : SignalType
        BUY / SELL / HOLD
    ticker : str
        Asset this signal applies to.
    strength : float
        Conviction in [0, 1].  1 = maximum conviction.
    target_weight : float | None
        Optional target portfolio weight the agent requests (0–1).
        When set the engine sizes the position as target_weight * equity.
    metadata : dict
        Arbitrary agent-specific debug/info payload — regime label,
        factor scores, etc.  Never used by the engine for logic.
    """

    signal_type:   SignalType
    ticker:        str
    strength:      float        = 1.0
    target_weight: float | None = None
    metadata:      dict         = field(default_factory=dict)

    def is_buy(self)  -> bool: return self.signal_type == SignalType.BUY
    def is_sell(self) -> bool: return self.signal_type == SignalType.SELL
    def is_hold(self) -> bool: return self.signal_type == SignalType.HOLD

    def __repr__(self) -> str:
        return (
            f"Signal({self.signal_type.value}, {self.ticker}, "
            f"strength={self.strength:.2f})"
        )


class StrategyAgent(ABC):
    """
    Abstract base class for all trading agents.

    Subclass this and implement:
        - get_agent_name()  → str
        - generate_signal() → Signal

    The engine guarantees that on each bar the agent receives:
        - A window of OHLCV data ending at the current bar (``data``)
        - The current detected regime label (``regime``)
        - The current portfolio snapshot (``portfolio_snapshot``)

    Agents must NOT mutate any passed objects.

    Example
    -------
    class MomentumAgent(StrategyAgent):
        def get_agent_name(self) -> str:
            return "MomentumAgent_v1"

        def generate_signal(self, ticker, data, regime, portfolio_snapshot) -> Signal:
            rsi = compute_rsi(data["Close"])
            if rsi.iloc[-1] < 30:
                return Signal(SignalType.BUY, ticker, strength=0.8)
            return Signal(SignalType.HOLD, ticker)
    """

    # ── lifecycle hooks (optional overrides) ──────────────────────────────

    def on_start(self, tickers: list[str], config: Any) -> None:
        """Called once before the backtest loop begins."""

    def on_end(self) -> None:
        """Called once after the backtest loop ends."""

    def on_bar_start(self, timestamp: pd.Timestamp) -> None:
        """Called at the start of every bar before generate_signal."""

    # ── required interface ────────────────────────────────────────────────

    @abstractmethod
    def get_agent_name(self) -> str:
        """Human-readable unique identifier for this agent."""

    @abstractmethod
    def generate_signal(
        self,
        ticker:             str,
        data:               pd.DataFrame,
        regime:             str,
        portfolio_snapshot: dict,
    ) -> Signal:
        """
        Generate a trading signal for ``ticker`` at the current bar.

        Parameters
        ----------
        ticker : str
            The asset being evaluated.
        data : pd.DataFrame
            OHLCV history up to and including the current bar.
            Columns: Open, High, Low, Close, Volume (and any pre-computed
            features added by data_pipeline.py).
        regime : str
            Current market regime from MarketRegimeAgent
            e.g. "BULLISH", "BEARISH", "HIGH VOLATILITY", "SIDEWAYS".
        portfolio_snapshot : dict
            Read-only view of current portfolio state::
                {
                    "cash": float,
                    "equity": float,
                    "positions": {ticker: {"qty": int, "avg_price": float}},
                    "unrealized_pnl": float,
                }

        Returns
        -------
        Signal
        """
