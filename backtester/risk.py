"""
risk.py
───────
Risk management layer.  Sits between the agent's signal and the
OrderExecutor.  Validates every order before it reaches the market.

Design note
───────────
RiskManager.validate() returns (approved: bool, reason: str).
If approved is False the engine drops the order and logs the reason.
Adding new rules means adding a private _check_*() method and calling
it inside validate() — no changes needed anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .order import Order, OrderSide
from .portfolio import Portfolio


@dataclass
class RiskConfig:
    """
    Configurable risk limits.

    max_position_pct   : Max single-position weight as % of total equity.
                         e.g. 0.20 = no position > 20 % of portfolio.
    max_portfolio_pct  : Max total invested (sum of positions) as % of equity.
                         e.g. 0.95 = keep at least 5 % cash.
    max_drawdown_halt  : Halt all new orders if portfolio drawdown exceeds
                         this fraction from peak equity.
                         e.g. 0.25 = halt trading at –25 % drawdown.
    max_orders_per_bar : Max number of orders allowed per single price bar.
    """
    max_position_pct:   float = 0.20
    max_portfolio_pct:  float = 0.95
    max_drawdown_halt:  float = 0.25
    max_orders_per_bar: int   = 5


class RiskManager:
    """
    Validates orders against portfolio state and configurable risk limits.

    Parameters
    ----------
    config    : RiskConfig instance.
    portfolio : Live Portfolio reference (read-only usage).
    """

    def __init__(self, config: RiskConfig, portfolio: Portfolio):
        self._cfg       = config
        self._portfolio = portfolio
        self._peak_equity: float = portfolio.initial_capital
        self._orders_this_bar: int = 0

    # ── public interface ──────────────────────────────────────────────────

    def validate(
        self,
        order:  Order,
        prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Run all risk checks against ``order``.

        Returns
        -------
        (True, "OK")                if all checks pass.
        (False, "<reason string>")  if any check fails.
        """
        checks = [
            self._check_order_limit(),
            self._check_drawdown(prices),
            self._check_cash(order, prices),
            self._check_position_size(order, prices),
            self._check_portfolio_exposure(order, prices),
        ]
        for approved, reason in checks:
            if not approved:
                return False, reason
        return True, "OK"

    def update_peak(self, current_equity: float) -> None:
        """Call at end of every bar to keep peak equity updated."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def reset_bar_counter(self) -> None:
        """Call at the start of every bar."""
        self._orders_this_bar = 0

    def record_order(self) -> None:
        """Call after each approved order."""
        self._orders_this_bar += 1

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (negative value)."""
        equity = self._portfolio.cash   # approximate; engine passes prices separately
        if self._peak_equity == 0:
            return 0.0
        return (equity - self._peak_equity) / self._peak_equity

    # ── private checks ────────────────────────────────────────────────────

    def _check_order_limit(self) -> Tuple[bool, str]:
        if self._orders_this_bar >= self._cfg.max_orders_per_bar:
            return False, f"Max orders per bar ({self._cfg.max_orders_per_bar}) reached"
        return True, "OK"

    def _check_drawdown(self, prices: Dict[str, float]) -> Tuple[bool, str]:
        equity = self._portfolio.total_equity(prices)
        self.update_peak(equity)
        if self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
            if dd <= -self._cfg.max_drawdown_halt:
                return False, f"Drawdown {dd*100:.1f}% exceeds halt threshold"
        return True, "OK"

    def _check_cash(self, order: Order, prices: Dict[str, float]) -> Tuple[bool, str]:
        if order.side == OrderSide.SELL:
            return True, "OK"
        price      = prices.get(order.ticker, 0.0)
        order_cost = price * order.quantity * 1.002   # rough cost with buffer
        if not self._portfolio.can_afford(order_cost):
            return False, f"Insufficient cash (need ${order_cost:.0f}, have ${self._portfolio.cash:.0f})"
        return True, "OK"

    def _check_position_size(self, order: Order, prices: Dict[str, float]) -> Tuple[bool, str]:
        if order.side == OrderSide.SELL:
            return True, "OK"
        equity         = self._portfolio.total_equity(prices)
        price          = prices.get(order.ticker, 1.0)
        position_value = price * order.quantity
        if equity > 0 and position_value / equity > self._cfg.max_position_pct:
            return False, (
                f"Position size {position_value/equity*100:.1f}% exceeds "
                f"limit {self._cfg.max_position_pct*100:.0f}%"
            )
        return True, "OK"

    def _check_portfolio_exposure(self, order: Order, prices: Dict[str, float]) -> Tuple[bool, str]:
        if order.side == OrderSide.SELL:
            return True, "OK"
        equity          = self._portfolio.total_equity(prices)
        invested        = self._portfolio.positions_value(prices)
        price           = prices.get(order.ticker, 1.0)
        new_invested    = invested + price * order.quantity
        if equity > 0 and new_invested / equity > self._cfg.max_portfolio_pct:
            return False, (
                f"Portfolio exposure {new_invested/equity*100:.1f}% would exceed "
                f"limit {self._cfg.max_portfolio_pct*100:.0f}%"
            )
        return True, "OK"

    # ── sizing helper ─────────────────────────────────────────────────────

    def size_order(
        self,
        ticker:        str,
        target_weight: float,
        prices:        Dict[str, float],
    ) -> float:
        """
        Convert a target portfolio weight (0–1) into a share quantity,
        clamped to max_position_pct.

        Used by the engine when Signal.target_weight is set.
        """
        equity   = self._portfolio.total_equity(prices)
        price    = prices.get(ticker, 1.0)
        weight   = min(target_weight, self._cfg.max_position_pct)
        qty      = (equity * weight) / price
        return max(0.0, qty)
