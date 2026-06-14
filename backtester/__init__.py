"""
Backtester package — production-quality quantitative backtesting engine.

Import surface:
    from backtester import BacktestEngine, StrategyAgent, Signal
    from backtester import Order, OrderType, OrderSide
    from backtester import Portfolio, Position
    from backtester import RiskManager, PerformanceAnalytics
    from backtester import TradeLogger, RegimeEvaluator
"""

from .agent_interface import StrategyAgent, Signal, SignalType
from .order import Order, OrderType, OrderSide, OrderStatus, ExecutedOrder
from .position import Position, PositionStatus
from .portfolio import Portfolio
from .risk import RiskManager, RiskConfig
from .analytics import PerformanceAnalytics, PerformanceReport
from .logger import TradeLogger, TradeRecord
from .regime_evaluator import RegimeEvaluator, RegimeMetrics
from .engine import BacktestEngine, BacktestConfig

__all__ = [
    "BacktestEngine", "BacktestConfig",
    "StrategyAgent", "Signal", "SignalType",
    "Order", "OrderType", "OrderSide", "OrderStatus", "ExecutedOrder",
    "Position", "PositionStatus",
    "Portfolio",
    "RiskManager", "RiskConfig",
    "PerformanceAnalytics", "PerformanceReport",
    "TradeLogger", "TradeRecord",
    "RegimeEvaluator", "RegimeMetrics",
]
