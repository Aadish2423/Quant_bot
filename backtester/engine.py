"""
engine.py
─────────
Main backtesting loop.  Orchestrates all subsystems.

The engine is fully data-agnostic — it receives a pre-built DataFrame
(from data_pipeline.py) and a pre-built regime Series (from
regeim_detection.py) and drives the agent through every bar.

No agent logic lives here; agents plug in via the StrategyAgent ABC.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Allow running engine.py directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .agent_interface import StrategyAgent, Signal, SignalType
from .order import Order, OrderType, OrderSide, OrderStatus, CostConfig, OrderExecutor
from .portfolio import Portfolio
from .risk import RiskManager, RiskConfig
from .analytics import PerformanceAnalytics, PerformanceReport
from .logger import TradeLogger
from .regime_evaluator import RegimeEvaluator


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    All runtime parameters for one backtest run.

    Parameters
    ----------
    initial_capital     : Starting cash in $.
    tickers             : List of asset symbols to trade.
    default_qty         : Shares to trade when Signal.target_weight is None.
    cost_config         : Commission / slippage model.
    risk_config         : Position sizing / drawdown limits.
    warmup_bars         : Bars skipped at the start (for indicator warm-up).
    verbose             : Print bar-by-bar progress.
    """
    initial_capital: float        = 100_000.0
    tickers:         List[str]    = field(default_factory=lambda: ["SPY"])
    default_qty:     float        = 10.0
    cost_config:     CostConfig   = field(default_factory=CostConfig)
    risk_config:     RiskConfig   = field(default_factory=RiskConfig)
    warmup_bars:     int          = 200      # covers MA200 warm-up
    verbose:         bool         = False


# ── engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Usage
    -----
    engine = BacktestEngine(config)
    report = engine.run(
        agent        = MyAgent(),
        data         = {ticker: ohlcv_df},
        regime_map   = {ticker: regime_series},
    )
    PerformanceAnalytics.print_report(report)

    Parameters
    ----------
    config : BacktestConfig

    Notes
    -----
    - Bar resolution: daily (aligns with yfinance data from data_pipeline.py).
    - Fill model: next-bar open (order generated on bar N, filled at bar N+1 open).
    - Only long positions are supported today; short-selling is future work.
    - Multiple assets are supported; each bar all tickers receive a signal call.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self._cfg = config or BacktestConfig()

    # ── public run interface ──────────────────────────────────────────────

    def run(
        self,
        agent:       StrategyAgent,
        data:        Dict[str, pd.DataFrame],
        regime_map:  Optional[Dict[str, pd.Series]] = None,
    ) -> PerformanceReport:
        """
        Execute a full backtest and return a PerformanceReport.

        Parameters
        ----------
        agent      : Any StrategyAgent subclass.
        data       : Dict mapping ticker → OHLCV DataFrame (from data_pipeline.py).
                     All DataFrames must share the same DatetimeIndex.
        regime_map : Dict mapping ticker → pd.Series of regime labels indexed
                     by date (from regeim_detection.py).  If None, all bars
                     are labelled "UNKNOWN".

        Returns
        -------
        PerformanceReport
        """
        cfg       = self._cfg
        tickers   = list(data.keys())
        dates     = self._align_dates(data)

        # ── subsystem initialisation ──────────────────────────────────────
        portfolio = Portfolio(cfg.initial_capital)
        executor  = OrderExecutor(cfg.cost_config)
        risk_mgr  = RiskManager(cfg.risk_config, portfolio)
        logger    = TradeLogger()

        equity_curve: List[float] = []
        pending_orders: List[tuple] = []   # (Order, regime) filled next bar

        agent.on_start(tickers, cfg)

        # ── main loop ─────────────────────────────────────────────────────
        for bar_idx, timestamp in enumerate(dates):

            # -- current bar price lookup ---------------------------------
            prices = self._get_prices(data, tickers, timestamp)
            if not prices:
                continue

            # -- execute pending orders from previous bar (next-bar fill) -
            if pending_orders:
                self._fill_pending(
                    pending_orders, data, timestamp, bar_idx,
                    portfolio, executor, risk_mgr, logger,
                )
                pending_orders.clear()

            risk_mgr.reset_bar_counter()

            # -- skip warmup bars -----------------------------------------
            if bar_idx < cfg.warmup_bars:
                equity_curve.append(portfolio.total_equity(prices))
                continue

            # -- build portfolio snapshot for agents ----------------------
            snap = portfolio.snapshot(prices)

            agent.on_bar_start(timestamp)

            # -- signal generation for each ticker ------------------------
            for ticker in tickers:
                if ticker not in prices:
                    continue

                regime = self._get_regime(regime_map, ticker, timestamp)
                window = data[ticker].loc[:timestamp]   # history up to now

                signal: Signal = agent.generate_signal(
                    ticker             = ticker,
                    data               = window,
                    regime             = regime,
                    portfolio_snapshot = snap,
                )

                if cfg.verbose:
                    print(f"  {timestamp.date()} {ticker:8} {signal}")

                order = self._signal_to_order(
                    signal, ticker, prices, portfolio, risk_mgr
                )
                if order:
                    pending_orders.append((order, regime))

            # -- mark-to-market equity ------------------------------------
            equity_curve.append(portfolio.total_equity(prices))
            risk_mgr.update_peak(equity_curve[-1])

        agent.on_end()

        # ── close any remaining open positions at last price ──────────────
        self._close_all_open(
            portfolio, executor, risk_mgr, logger,
            data, dates[-1], len(dates) - 1, regime_map,
        )
        prices = self._get_prices(data, tickers, dates[-1])
        equity_curve.append(portfolio.total_equity(prices))

        # ── build analytics ───────────────────────────────────────────────
        trade_df      = logger.get_dataframe()
        regime_series = self._build_regime_series(regime_map, tickers, dates)

        regime_eval   = RegimeEvaluator(
            trade_log       = trade_df,
            equity_curve    = equity_curve,
            dates           = dates[:len(equity_curve)],
            regime_series   = regime_series,
            initial_capital = cfg.initial_capital,
        )
        regime_perf = regime_eval.evaluate()

        analytics = PerformanceAnalytics(
            equity_curve    = equity_curve,
            trade_log       = trade_df,
            dates           = dates[:len(equity_curve)],
            initial_capital = cfg.initial_capital,
            agent_name      = agent.get_agent_name(),
            ticker          = ", ".join(tickers),
        )
        return analytics.build_report(regime_performance=regime_perf)

    # ── private helpers ───────────────────────────────────────────────────

    def _align_dates(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Intersection of all ticker date indices, sorted ascending."""
        index = None
        for df in data.values():
            index = df.index if index is None else index.intersection(df.index)
        return index.sort_values()

    def _get_prices(
        self,
        data:      Dict[str, pd.DataFrame],
        tickers:   List[str],
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        prices = {}
        for t in tickers:
            try:
                prices[t] = float(data[t].loc[timestamp, "Close"])
            except (KeyError, TypeError):
                pass
        return prices

    def _get_regime(
        self,
        regime_map: Optional[Dict[str, pd.Series]],
        ticker:     str,
        timestamp:  pd.Timestamp,
    ) -> str:
        if not regime_map or ticker not in regime_map:
            return "UNKNOWN"
        series = regime_map[ticker]
        try:
            # nearest past label
            past = series[series.index <= timestamp]
            return str(past.iloc[-1]) if len(past) else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _signal_to_order(
        self,
        signal:    Signal,
        ticker:    str,
        prices:    Dict[str, float],
        portfolio: Portfolio,
        risk_mgr:  RiskManager,
    ) -> Optional[Order]:
        """Convert a Signal into an Order (or None if HOLD / no action needed)."""
        cfg = self._cfg

        if signal.is_hold():
            return None

        if signal.is_buy():
            if portfolio.has_position(ticker):
                return None   # already long; no pyramiding by default

            if signal.target_weight is not None:
                qty = risk_mgr.size_order(ticker, signal.target_weight, prices)
            else:
                qty = cfg.default_qty

            if qty <= 0:
                return None

            return Order(
                ticker     = ticker,
                side       = OrderSide.BUY,
                quantity   = qty,
                order_type = OrderType.MARKET,
            )

        if signal.is_sell():
            if not portfolio.has_position(ticker):
                return None   # nothing to sell

            pos = portfolio.get_position(ticker)
            return Order(
                ticker     = ticker,
                side       = OrderSide.SELL,
                quantity   = pos.quantity,
                order_type = OrderType.MARKET,
            )

        return None

    def _fill_pending(
        self,
        pending:   List[tuple],
        data:      Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        bar_idx:   int,
        portfolio: Portfolio,
        executor:  OrderExecutor,
        risk_mgr:  RiskManager,
        logger:    TradeLogger,
    ) -> None:
        prices = self._get_prices(data, list(data.keys()), timestamp)

        for order, regime in pending:
            # risk check
            approved, reason = risk_mgr.validate(order, prices)
            if not approved:
                logger.log_rejected(order.ticker, reason, timestamp)
                if self._cfg.verbose:
                    print(f"    REJECTED {order.ticker}: {reason}")
                continue

            # get bar for this ticker
            try:
                bar = data[order.ticker].loc[timestamp]
            except KeyError:
                continue

            executed = executor.execute(order, bar, timestamp)
            if executed.order.status != OrderStatus.FILLED:
                continue

            pnl = portfolio.apply_executed_order(executed, regime)
            risk_mgr.record_order()

            if order.side == OrderSide.BUY:
                logger.log_open(executed, regime, bar_idx)
            else:
                logger.log_close(executed, pnl, regime, bar_idx)

    def _close_all_open(
        self,
        portfolio: Portfolio,
        executor:  OrderExecutor,
        risk_mgr:  RiskManager,
        logger:    TradeLogger,
        data:      Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        bar_idx:   int,
        regime_map: Optional[Dict[str, pd.Series]],
    ) -> None:
        """Force-close all open positions at end of backtest."""
        for ticker, pos in list(portfolio.open_positions.items()):
            regime = self._get_regime(regime_map, ticker, timestamp)
            order  = Order(
                ticker     = ticker,
                side       = OrderSide.SELL,
                quantity   = pos.quantity,
                order_type = OrderType.MARKET,
            )
            try:
                bar = data[ticker].loc[timestamp]
            except KeyError:
                continue
            executed = executor.execute(order, bar, timestamp)
            if executed.order.status == OrderStatus.FILLED:
                pnl = portfolio.apply_executed_order(executed, regime)
                logger.log_close(executed, pnl, regime, bar_idx)

    def _build_regime_series(
        self,
        regime_map: Optional[Dict[str, pd.Series]],
        tickers:    List[str],
        dates:      pd.DatetimeIndex,
    ) -> pd.Series:
        """Collapse per-ticker regime maps into a single date-indexed series."""
        if not regime_map:
            return pd.Series("UNKNOWN", index=dates)
        # Use first ticker's regime as representative (single-asset case)
        ticker = tickers[0]
        if ticker in regime_map:
            return regime_map[ticker].reindex(dates, method="ffill").fillna("UNKNOWN")
        return pd.Series("UNKNOWN", index=dates)
