"""
regime_evaluator.py
───────────────────
Slices trade logs and equity curves by market regime and computes
per-regime performance metrics.

Integrates directly with the regime labels produced by
regeim_detection.py (BULLISH, BEARISH, HIGH VOLATILITY, SIDEWAYS, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RegimeMetrics:
    """Per-regime performance summary."""
    regime:       str
    num_trades:   int
    total_return: float    # sum of net PnL / starting capital (approx)
    win_rate:     float
    sharpe:       float
    avg_pnl:      float
    gross_profit: float
    gross_loss:   float
    profit_factor: float

    def to_dict(self) -> dict:
        return {
            "num_trades":    self.num_trades,
            "total_return":  round(self.total_return, 4),
            "win_rate":      round(self.win_rate, 4),
            "sharpe":        round(self.sharpe, 4),
            "avg_pnl":       round(self.avg_pnl, 2),
            "gross_profit":  round(self.gross_profit, 2),
            "gross_loss":    round(self.gross_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
        }


class RegimeEvaluator:
    """
    Breaks down backtest performance by market regime.

    Parameters
    ----------
    trade_log        : DataFrame from TradeLogger.get_dataframe().
    equity_curve     : List of portfolio values aligned with ``dates``.
    dates            : DatetimeIndex aligned with equity_curve.
    regime_series    : pd.Series[str] indexed by date — regime label per bar.
                       Produced by running MarketRegimeAgent across the full
                       date range.  Regimes: BULLISH, BEARISH, SIDEWAYS,
                       HIGH VOLATILITY, UNCERTAIN.
    initial_capital  : Used to normalise returns.
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        trade_log:       pd.DataFrame,
        equity_curve:    List[float],
        dates:           pd.DatetimeIndex,
        regime_series:   pd.Series,
        initial_capital: float,
    ):
        self._trades   = trade_log
        self._equity   = np.array(equity_curve, dtype=float)
        self._dates    = pd.DatetimeIndex(dates)
        self._regimes  = regime_series
        self._cap      = initial_capital

    # ── public ───────────────────────────────────────────────────────────

    def evaluate(self) -> Dict[str, dict]:
        """
        Return a dict keyed by regime label, each value being a metrics dict.

        Example
        -------
        {
          "BULLISH":        {"num_trades": 12, "total_return": 0.18, ...},
          "BEARISH":        {"num_trades":  3, "total_return":-0.02, ...},
          "HIGH VOLATILITY":{"num_trades":  5, "total_return": 0.04, ...},
          "SIDEWAYS":       {"num_trades":  2, "total_return": 0.01, ...},
        }
        """
        if self._trades.empty:
            return {}

        all_regimes = self._regimes.dropna().unique().tolist()
        results: Dict[str, dict] = {}

        for regime in all_regimes:
            metrics = self._metrics_for_regime(regime)
            if metrics.num_trades > 0:
                results[regime] = metrics.to_dict()

        return results

    # ── private ───────────────────────────────────────────────────────────

    def _metrics_for_regime(self, regime: str) -> RegimeMetrics:
        # trades opened in this regime
        regime_trades = self._trades[self._trades["regime"] == regime].copy() \
            if "regime" in self._trades.columns else pd.DataFrame()

        num_trades = len(regime_trades)

        if num_trades == 0:
            return RegimeMetrics(
                regime=regime, num_trades=0, total_return=0.0,
                win_rate=0.0, sharpe=0.0, avg_pnl=0.0,
                gross_profit=0.0, gross_loss=0.0, profit_factor=0.0,
            )

        pnl         = regime_trades["net_pnl"].values
        wins        = pnl[pnl > 0]
        losses      = pnl[pnl <= 0]
        gross_profit = float(wins.sum())  if len(wins)   else 0.0
        gross_loss   = float(losses.sum()) if len(losses) else 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

        # equity sub-series for regime bars
        eq_series = self._equity_for_regime(regime)
        sharpe    = self._regime_sharpe(eq_series)

        return RegimeMetrics(
            regime        = regime,
            num_trades    = num_trades,
            total_return  = float(pnl.sum()) / self._cap,
            win_rate      = len(wins) / num_trades,
            sharpe        = sharpe,
            avg_pnl       = float(pnl.mean()),
            gross_profit  = gross_profit,
            gross_loss    = gross_loss,
            profit_factor = profit_factor,
        )

    def _equity_for_regime(self, regime: str) -> np.ndarray:
        """Extract equity values for bars where the regime matches."""
        eq_df = pd.Series(self._equity, index=self._dates)
        aligned = self._regimes.reindex(self._dates)
        mask    = aligned == regime
        sub     = eq_df[mask].values
        return sub if len(sub) > 1 else np.array([self._cap, self._cap])

    def _regime_sharpe(self, eq: np.ndarray, risk_free: float = 0.04) -> float:
        if len(eq) < 2:
            return 0.0
        rets     = np.diff(eq) / eq[:-1]
        rf_daily = risk_free / self.TRADING_DAYS
        excess   = rets - rf_daily
        std      = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(self.TRADING_DAYS))
