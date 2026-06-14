"""
analytics.py
────────────
Pure-function performance metrics + PerformanceReport dataclass.
No side effects; all inputs are plain lists / Series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── report container ──────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """
    Structured output of a completed backtest run.

    All monetary values are in $.
    All ratio / percentage fields are already in human-readable units
    (e.g. total_return = 0.28 means 28 %).
    """
    agent_name:       str
    ticker:           str
    initial_capital:  float
    final_capital:    float

    # returns
    total_return:     float    # e.g. 0.28  → 28 %
    cagr:             float    # annualised

    # risk-adjusted
    sharpe_ratio:     float
    sortino_ratio:    float
    volatility:       float    # annualised daily-return std

    # drawdown
    max_drawdown:     float    # e.g. −0.15 → –15 %
    max_drawdown_duration: int # bars in worst drawdown

    # trade statistics
    num_trades:       int
    win_rate:         float    # fraction of winning trades
    profit_factor:    float    # gross profit / gross loss
    avg_win:          float    # average winning trade $
    avg_loss:         float    # average losing trade $ (negative)
    expectancy:       float    # expected $ per trade
    avg_hold_bars:    float

    # regime breakdown
    regime_performance: Dict[str, dict] = field(default_factory=dict)

    # time series
    equity_curve:     List[float]        = field(default_factory=list)
    equity_dates:     List[str]          = field(default_factory=list)


# ── metrics functions ─────────────────────────────────────────────────────────

class PerformanceAnalytics:
    """
    Stateless calculator.  Call build_report() to get a PerformanceReport.

    Parameters
    ----------
    equity_curve : Portfolio equity sampled at each bar ($ value).
    trade_log    : DataFrame returned by TradeLogger.get_dataframe().
    dates        : DatetimeIndex aligned with equity_curve.
    initial_capital
    agent_name
    ticker
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        equity_curve:    List[float],
        trade_log:       pd.DataFrame,
        dates:           pd.DatetimeIndex,
        initial_capital: float,
        agent_name:      str,
        ticker:          str,
    ):
        self._equity   = np.array(equity_curve, dtype=float)
        self._trades   = trade_log
        self._dates    = dates
        self._cap      = initial_capital
        self._agent    = agent_name
        self._ticker   = ticker

    # ── public ───────────────────────────────────────────────────────────

    def build_report(self, regime_performance: Optional[Dict] = None) -> PerformanceReport:
        returns = self._daily_returns()

        return PerformanceReport(
            agent_name            = self._agent,
            ticker                = self._ticker,
            initial_capital       = round(self._cap, 2),
            final_capital         = round(float(self._equity[-1]), 2),
            total_return          = round(self._total_return(), 4),
            cagr                  = round(self._cagr(), 4),
            sharpe_ratio          = round(self._sharpe(returns), 4),
            sortino_ratio         = round(self._sortino(returns), 4),
            volatility            = round(self._volatility(returns), 4),
            max_drawdown          = round(self._max_drawdown(), 4),
            max_drawdown_duration = self._max_dd_duration(),
            num_trades            = len(self._trades),
            win_rate              = round(self._win_rate(), 4),
            profit_factor         = round(self._profit_factor(), 4),
            avg_win               = round(self._avg_win(), 2),
            avg_loss              = round(self._avg_loss(), 2),
            expectancy            = round(self._expectancy(), 2),
            avg_hold_bars         = round(self._avg_hold(), 2),
            regime_performance    = regime_performance or {},
            equity_curve          = [round(v, 2) for v in self._equity.tolist()],
            equity_dates          = [str(d.date()) for d in self._dates],
        )

    # ── return metrics ────────────────────────────────────────────────────

    def _daily_returns(self) -> np.ndarray:
        e = self._equity
        return np.diff(e) / e[:-1] if len(e) > 1 else np.array([0.0])

    def _total_return(self) -> float:
        if self._cap == 0:
            return 0.0
        return (self._equity[-1] - self._cap) / self._cap

    def _cagr(self) -> float:
        n_years = len(self._equity) / self.TRADING_DAYS
        if n_years <= 0 or self._cap <= 0:
            return 0.0
        return (self._equity[-1] / self._cap) ** (1 / n_years) - 1

    # ── risk metrics ──────────────────────────────────────────────────────

    def _volatility(self, returns: np.ndarray) -> float:
        return float(np.std(returns, ddof=1) * np.sqrt(self.TRADING_DAYS))

    def _sharpe(self, returns: np.ndarray, risk_free: float = 0.04) -> float:
        rf_daily = risk_free / self.TRADING_DAYS
        excess   = returns - rf_daily
        std      = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(self.TRADING_DAYS))

    def _sortino(self, returns: np.ndarray, risk_free: float = 0.04) -> float:
        rf_daily  = risk_free / self.TRADING_DAYS
        excess    = returns - rf_daily
        downside  = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        down_std  = np.std(downside, ddof=1)
        if down_std == 0:
            return 0.0
        return float(np.mean(excess) / down_std * np.sqrt(self.TRADING_DAYS))

    def _max_drawdown(self) -> float:
        e        = self._equity
        peak     = np.maximum.accumulate(e)
        dd       = (e - peak) / np.where(peak == 0, 1, peak)
        return float(np.min(dd))

    def _max_dd_duration(self) -> int:
        e    = self._equity
        peak = np.maximum.accumulate(e)
        in_dd = e < peak
        max_dur, cur_dur = 0, 0
        for v in in_dd:
            cur_dur = cur_dur + 1 if v else 0
            max_dur = max(max_dur, cur_dur)
        return max_dur

    # ── trade metrics ─────────────────────────────────────────────────────

    def _pnls(self):
        if self._trades.empty or "net_pnl" not in self._trades.columns:
            return np.array([]), np.array([])
        pnl  = self._trades["net_pnl"].values
        wins = pnl[pnl > 0]
        loss = pnl[pnl <= 0]
        return wins, loss

    def _win_rate(self) -> float:
        wins, loss = self._pnls()
        total = len(wins) + len(loss)
        return len(wins) / total if total > 0 else 0.0

    def _profit_factor(self) -> float:
        wins, loss = self._pnls()
        gross_profit = wins.sum() if len(wins) else 0.0
        gross_loss   = abs(loss.sum()) if len(loss) else 0.0
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def _avg_win(self) -> float:
        wins, _ = self._pnls()
        return float(wins.mean()) if len(wins) else 0.0

    def _avg_loss(self) -> float:
        _, loss = self._pnls()
        return float(loss.mean()) if len(loss) else 0.0

    def _expectancy(self) -> float:
        wr  = self._win_rate()
        aw  = self._avg_win()
        al  = abs(self._avg_loss())
        return wr * aw - (1 - wr) * al

    def _avg_hold(self) -> float:
        if self._trades.empty or "hold_bars" not in self._trades.columns:
            return 0.0
        return float(self._trades["hold_bars"].mean())

    # ── pretty print ──────────────────────────────────────────────────────

    @staticmethod
    def print_report(r: PerformanceReport) -> None:
        sep = "═" * 56
        print(f"\n{sep}")
        print(f"  BACKTEST REPORT — {r.agent_name}")
        print(f"  Ticker: {r.ticker}")
        print(sep)
        print(f"  Initial Capital   : ${r.initial_capital:>12,.2f}")
        print(f"  Final Capital     : ${r.final_capital:>12,.2f}")
        print(f"  Total Return      : {r.total_return*100:>+10.2f} %")
        print(f"  CAGR              : {r.cagr*100:>+10.2f} %")
        print(f"  Volatility (ann.) : {r.volatility*100:>10.2f} %")
        print(sep)
        print(f"  Sharpe Ratio      : {r.sharpe_ratio:>10.4f}")
        print(f"  Sortino Ratio     : {r.sortino_ratio:>10.4f}")
        print(f"  Max Drawdown      : {r.max_drawdown*100:>+10.2f} %")
        print(f"  Max DD Duration   : {r.max_drawdown_duration:>10} bars")
        print(sep)
        print(f"  Num Trades        : {r.num_trades:>10}")
        print(f"  Win Rate          : {r.win_rate*100:>10.1f} %")
        print(f"  Profit Factor     : {r.profit_factor:>10.4f}")
        print(f"  Avg Win           : ${r.avg_win:>11,.2f}")
        print(f"  Avg Loss          : ${r.avg_loss:>11,.2f}")
        print(f"  Expectancy        : ${r.expectancy:>11,.2f}")
        print(f"  Avg Hold (bars)   : {r.avg_hold_bars:>10.1f}")
        if r.regime_performance:
            print(sep)
            print("  REGIME BREAKDOWN")
            for regime, m in r.regime_performance.items():
                print(f"    {regime}")
                print(f"      Return     : {m.get('total_return',0)*100:>+8.2f} %")
                print(f"      Win Rate   : {m.get('win_rate',0)*100:>8.1f} %")
                print(f"      Sharpe     : {m.get('sharpe',0):>8.4f}")
                print(f"      Num Trades : {m.get('num_trades',0):>8}")
        print(f"{sep}\n")
