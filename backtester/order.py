"""
order.py
────────
Order types, execution model, and cost simulation.

Extensibility note
──────────────────
To add limit / stop orders: subclass Order, override the
``is_fillable(bar)`` method, then register the subclass in
OrderExecutor.execute().  The engine never imports concrete order
subclasses directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid

import pandas as pd


# ── enumerations ──────────────────────────────────────────────────────────────

class OrderType(Enum):
    MARKET     = "MARKET"
    LIMIT      = "LIMIT"      # placeholder — not yet executed
    STOP_LOSS  = "STOP_LOSS"  # placeholder
    TAKE_PROFIT = "TAKE_PROFIT"  # placeholder


class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    REJECTED  = "REJECTED"
    CANCELLED = "CANCELLED"


# ── order data model ──────────────────────────────────────────────────────────

@dataclass
class Order:
    """
    Represents an instruction to trade.  Created by the engine; never
    directly by an agent.

    Parameters
    ----------
    ticker      : Asset symbol.
    side        : BUY or SELL.
    order_type  : MARKET (only type currently executed).
    quantity    : Number of shares / units.  Must be > 0.
    limit_price : Required for LIMIT orders (future use).
    stop_price  : Required for STOP orders (future use).
    """

    ticker:      str
    side:        OrderSide
    quantity:    float
    order_type:  OrderType        = OrderType.MARKET
    limit_price: Optional[float]  = None
    stop_price:  Optional[float]  = None
    order_id:    str              = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status:      OrderStatus      = OrderStatus.PENDING
    created_at:  Optional[pd.Timestamp] = None

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")


@dataclass
class ExecutedOrder:
    """
    Immutable record of a filled order, returned by OrderExecutor.

    All downstream components (Portfolio, TradeLogger) consume this
    rather than the original Order.
    """

    order:           Order
    fill_price:      float        # price after slippage
    gross_price:     float        # raw bar price before slippage
    commission:      float        # absolute $ cost
    slippage_cost:   float        # absolute $ cost
    total_cost:      float        # commission + slippage
    filled_at:       pd.Timestamp
    net_cash_impact: float        # negative = cash spent, positive = cash received

    @property
    def ticker(self)   -> str:       return self.order.ticker
    @property
    def side(self)     -> OrderSide: return self.order.side
    @property
    def quantity(self) -> float:     return self.order.quantity


# ── cost configuration ────────────────────────────────────────────────────────

@dataclass
class CostConfig:
    """
    Trading cost parameters.  All values are optional; defaults model
    a low-cost retail broker.

    commission_rate : Fraction of trade value charged as commission.
                      e.g. 0.001 = 0.1 %.
    slippage_rate   : Fraction of price lost to market impact / spread.
                      e.g. 0.0005 = 0.05 %.
    min_commission  : Floor commission per order in $.
    """
    commission_rate: float = 0.001    # 0.10 %
    slippage_rate:   float = 0.0005   # 0.05 %
    min_commission:  float = 1.0      # $1 minimum


# ── executor ─────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Simulates order execution against a price bar.

    Only MARKET orders are executed today.  LIMIT, STOP_LOSS, and
    TAKE_PROFIT orders are rejected with status REJECTED so the engine
    stays consistent; implement ``is_fillable`` logic in a subclass when
    needed.
    """

    def __init__(self, cost_config: CostConfig | None = None):
        self._costs = cost_config or CostConfig()

    def execute(
        self,
        order:     Order,
        bar:       pd.Series,
        timestamp: pd.Timestamp,
    ) -> ExecutedOrder:
        """
        Attempt to fill ``order`` using data from ``bar``.

        For MARKET orders the fill price is the bar's Open price
        (next-bar open fill simulates realistic execution).

        Parameters
        ----------
        order     : The pending Order to execute.
        bar       : A row from the OHLCV DataFrame (Open/High/Low/Close/Volume).
        timestamp : Current bar timestamp.

        Returns
        -------
        ExecutedOrder with status FILLED or REJECTED.
        """
        if order.order_type != OrderType.MARKET:
            order.status = OrderStatus.REJECTED
            # Return a zero-cost rejected record so callers can inspect status
            return ExecutedOrder(
                order=order,
                fill_price=0.0,
                gross_price=0.0,
                commission=0.0,
                slippage_cost=0.0,
                total_cost=0.0,
                filled_at=timestamp,
                net_cash_impact=0.0,
            )

        gross_price  = float(bar["Open"])
        slippage_amt = gross_price * self._costs.slippage_rate
        slippage_cost = slippage_amt * order.quantity

        # Slippage increases cost of buys, decreases proceeds from sells
        if order.side == OrderSide.BUY:
            fill_price = gross_price + slippage_amt
        else:
            fill_price = gross_price - slippage_amt

        trade_value = fill_price * order.quantity
        commission  = max(trade_value * self._costs.commission_rate,
                          self._costs.min_commission)
        total_cost  = commission + slippage_cost

        if order.side == OrderSide.BUY:
            net_cash_impact = -(trade_value + commission)
        else:
            net_cash_impact = trade_value - commission

        order.status = OrderStatus.FILLED
        return ExecutedOrder(
            order=order,
            fill_price=fill_price,
            gross_price=gross_price,
            commission=commission,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            filled_at=timestamp,
            net_cash_impact=net_cash_impact,
        )
