"""
example_strategy.py
───────────────────
Demonstrates how to build a StrategyAgent and run it through the
BacktestEngine using data from data_pipeline.py and regime labels
from regeim_detection.py.

Run:
    & c:\Python\dl_env\Scripts\python.exe example_strategy.py
"""

from __future__ import annotations

import sys
import os

import pandas as pd
import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import (
    BacktestEngine, BacktestConfig,
    StrategyAgent, Signal, SignalType,
    PerformanceAnalytics,
    RiskConfig, CostConfig,
)
from data_pipeline import run as build_dataset
from regeim_detection import MarketRegimeAgent


# ─────────────────────────────────────────────────────────────────────────────
#  EXAMPLE AGENT: Dual-MA + RSI Confirmation
#  This is purely illustrative — not a recommendation.
# ─────────────────────────────────────────────────────────────────────────────

class DualMAWithRSIAgent(StrategyAgent):
    """
    Simple trend-following agent:
      - BUY  when MA50 crosses above MA200 AND RSI > 50 AND regime is BULLISH
      - SELL when MA50 crosses below MA200 OR  RSI < 40
      - HOLD otherwise

    This agent is intentionally straightforward to validate the engine,
    not to maximise returns.
    """

    def get_agent_name(self) -> str:
        return "DualMA_RSI_v1"

    def generate_signal(
        self,
        ticker:             str,
        data:               pd.DataFrame,
        regime:             str,
        portfolio_snapshot: dict,
    ) -> Signal:

        if len(data) < 201:
            return Signal(SignalType.HOLD, ticker)

        close = data["Close"]

        # ── compute indicators inline ─────────────────────────────────────
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        _ma50    = float(ma50.iloc[-1])
        _ma200   = float(ma200.iloc[-1])
        _ma50_p  = float(ma50.iloc[-2])    # previous bar
        _ma200_p = float(ma200.iloc[-2])
        _rsi     = float(rsi.iloc[-1])

        has_position = ticker in portfolio_snapshot.get("positions", {})

        # ── entry logic ───────────────────────────────────────────────────
        golden_cross = (_ma50 > _ma200) and (_ma50_p <= _ma200_p)
        bullish_bias = regime in ("BULLISH", "UNCERTAIN", "UNKNOWN")

        if golden_cross and _rsi > 50 and bullish_bias and not has_position:
            return Signal(
                signal_type   = SignalType.BUY,
                ticker        = ticker,
                strength      = min((_rsi - 50) / 50, 1.0),
                target_weight = 0.15,    # size to 15 % of equity
                metadata      = {"rsi": _rsi, "regime": regime},
            )

        # ── exit logic ────────────────────────────────────────────────────
        death_cross = (_ma50 < _ma200) and (_ma50_p >= _ma200_p)
        if has_position and (death_cross or _rsi < 40 or regime == "BEARISH"):
            return Signal(
                signal_type = SignalType.SELL,
                ticker      = ticker,
                strength    = 1.0,
                metadata    = {"rsi": _rsi, "regime": regime, "reason": "exit"},
            )

        return Signal(SignalType.HOLD, ticker)


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME SERIES BUILDER
#  Runs MarketRegimeAgent bar-by-bar to label every date in the dataset.
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Produce a pd.Series of regime labels indexed by date using the
    rolling detect_regime() logic from regeim_detection.py.

    Uses a 200-bar minimum window so labels are always computed on
    sufficient history (same constraint as MA200).
    """
    from regeim_detection import (
        compute_features,
        detect_regime,
    )

    df_feat = compute_features(df.copy())
    labels  = {}
    dates   = df_feat.index.tolist()

    print(f"[+] Computing regime labels for {ticker} ({len(dates)} bars)...")
    for i in range(200, len(dates)):
        window = df_feat.iloc[:i + 1]
        try:
            labels[dates[i]] = detect_regime(window)
        except Exception:
            labels[dates[i]] = "UNKNOWN"

    series = pd.Series(labels)
    series.index = pd.DatetimeIndex(series.index)
    return series


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    TICKER = "SPY"

    # 1. load data (uses data_pipeline.py — downloads if CSVs missing)
    csv_train = f"data/{TICKER}_train.csv"
    csv_test  = f"data/{TICKER}_test.csv"

    if os.path.exists(csv_train) and os.path.exists(csv_test):
        print("[+] Loading pre-built train/test CSVs ...")
        train = pd.read_csv(csv_train, index_col=0, parse_dates=True)
        test  = pd.read_csv(csv_test,  index_col=0, parse_dates=True)
    else:
        print("[+] CSVs not found — building dataset via data_pipeline.py ...")
        train, test = build_dataset(TICKER)

    # 2. build regime label series for test period
    regime_series = build_regime_series(test, TICKER)
    regime_map    = {TICKER: regime_series}

    # 3. configure and run backtest on TEST data only
    config = BacktestConfig(
        initial_capital = 100_000.0,
        tickers         = [TICKER],
        default_qty     = 10,
        warmup_bars     = 200,
        verbose         = False,
        cost_config     = CostConfig(
            commission_rate = 0.001,    # 0.1 %
            slippage_rate   = 0.0005,   # 0.05 %
            min_commission  = 1.0,
        ),
        risk_config     = RiskConfig(
            max_position_pct   = 0.20,
            max_portfolio_pct  = 0.95,
            max_drawdown_halt  = 0.25,
            max_orders_per_bar = 5,
        ),
    )

    engine = BacktestEngine(config)
    agent  = DualMAWithRSIAgent()

    print(f"[+] Running backtest: {agent.get_agent_name()} on {TICKER} ...")
    report = engine.run(
        agent      = agent,
        data       = {TICKER: test},
        regime_map = regime_map,
    )

    # 4. display results
    PerformanceAnalytics.print_report(report)

    # 5. save equity curve to CSV
    eq_df = pd.DataFrame({
        "date":   report.equity_dates,
        "equity": report.equity_curve,
    })
    eq_df.to_csv(f"data/{TICKER}_equity_curve.csv", index=False)
    print(f"[+] Equity curve saved → data/{TICKER}_equity_curve.csv")


if __name__ == "__main__":
    main()
