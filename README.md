# AdaptiQuant

**A regime-aware, multi-agent algorithmic trading & backtesting framework.**

AdaptiQuant downloads market data, classifies each trading day into a market
*regime* (bullish, bearish, sideways, etc.), and backtests multiple trading
strategies through a realistic event-driven engine. A **meta-agent** scores
every strategy and automatically selects the best performer.

> ⚠️ **Status: work in progress.** The framework runs end-to-end, but the
> current strategies are **not yet profitable** (see [Current Results](#current-results)).
> This is a research/simulation tool — it does **not** place live trades.

---

## Table of Contents
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Current Results](#current-results)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)

---

## Key Features

- **Event-driven backtesting engine** — bar-by-bar simulation with next-bar-open
  fills, so strategies can never trade on prices they haven't seen yet.
- **Realistic cost & risk modeling** — commission, slippage, position-size caps,
  portfolio exposure limits, and a max-drawdown trading halt.
- **Regime detection** — classifies markets into `BULLISH`, `BEARISH`,
  `SIDEWAYS`, `UNCERTAIN` (and `HIGH VOLATILITY` support) from 30+ technical
  indicators.
- **Multi-agent design** — strategies are interchangeable plug-ins implementing a
  single `StrategyAgent` interface.
- **Meta-agent selection** — runs the strategy pool and picks the winner using
  configurable, weighted scoring (return / Sharpe / win-rate / drawdown).
- **Per-regime analytics** — breaks performance down by regime so you can see
  exactly where a strategy makes or loses money.

---

## Architecture

```
 data_pipeline.py            regeim_detection.py
 (download + features)       (regime classification)
          \                        /
           \                      /
            v                    v
                 meta_agent.py  <-----  config.yaml
        (orchestrator + strategy selection)
                      |
                      v
              backtester/  (engine package)
   engine -> order -> portfolio -> position
          -> risk -> logger -> analytics -> regime_evaluator
                      |
                      v
       strategies/ (momentum, mean_reversion)
                      |
                      v
        PerformanceReport (returns, Sharpe,
        drawdown, per-regime breakdown)
```

**Design principle:** single responsibility per module. The engine knows nothing
about strategy logic; strategies know nothing about cash or order execution; risk
rules live in one place. Adding a new strategy means writing **one** file that
implements `StrategyAgent` — nothing else changes.

---

## Project Structure

| Path | Purpose |
|---|---|
| `meta_agent.py` | **Main entry point.** Builds regimes, runs the strategy pool, scores & selects the best strategy. |
| `data_pipeline.py` | Downloads OHLCV data and engineers ~30 technical features; saves train/test/full CSVs. |
| `regeim_detection.py` | Market-regime classifier (MA, RSI, MACD, Bollinger, volatility, ATR, OBV). |
| `config.yaml` | Central configuration — capital, costs, strategy params, scoring weights, strategy pool. |
| `example_strategy.py` | Standalone reference showing how to build an agent and run the engine. |
| **`backtester/`** | The engine package (see below). |
| `backtester/engine.py` | Event-driven backtest loop. |
| `backtester/agent_interface.py` | `StrategyAgent` ABC + `Signal` contract (the plug-in boundary). |
| `backtester/order.py` | Order model + execution with commission/slippage. |
| `backtester/portfolio.py` | Cash & positions ledger. |
| `backtester/position.py` | Single-position tracking (VWAP, PnL). |
| `backtester/risk.py` | Order validation & position sizing. |
| `backtester/logger.py` | Trade-by-trade record keeping. |
| `backtester/analytics.py` | Performance metrics (Sharpe, Sortino, drawdown, etc.). |
| `backtester/regime_evaluator.py` | Per-regime performance breakdown. |
| **`strategies/`** | Trading agents. |
| `strategies/momentum.py` | Momentum agent (trades BULLISH / SIDEWAYS / UNCERTAIN). |
| `strategies/mean_reversion_agent.py` | Mean-reversion agent (z-score entries/exits). |
| `data/` | Cached market data (`SPY_full.csv`, `SPY_train.csv`, `SPY_test.csv`) and generated regime labels (`regimes.csv`). |

---

## How It Works

1. **Data** — `data_pipeline.py` downloads ~10 years of SPY daily data and adds
   technical features (RSI, MACD, Bollinger Bands, ATR, OBV, moving averages).
2. **Regimes** — `meta_agent.build_regime_series()` labels every bar using
   MA50/MA200 trend logic and caches the result to `data/regimes.csv`.
3. **Backtest** — each strategy runs through the engine. On every bar the engine
   asks the strategy for a `Signal`, converts it to an order, applies risk checks,
   fills it at the next bar's open (with costs), and marks equity to market.
4. **Regime filtering** — each strategy only trades in regimes it's suited for
   (e.g. momentum avoids `BEARISH`).
5. **Selection** — the meta-agent scores every strategy with the weights in
   `config.yaml` and reports the winner plus a full performance breakdown.

---

## Installation

Requires **Python 3.11+** (tested on 3.14).

```bash
# clone / enter the project
cd "Quant BOt"

# (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline (data -> regimes -> backtest -> selection):

```bash
python meta_agent.py
```

Optionally point at a different config:

```bash
python meta_agent.py --config config.yaml
```

Other entry points:

```bash
python data_pipeline.py      # (re)download data and rebuild feature CSVs
python regeim_detection.py   # print the current market regime for SPY
python example_strategy.py   # run the standalone Dual-MA + RSI demo
```

---

## Current Results

Latest backtest on **SPY (~10 years, 2,317 bars)**. Meta-agent selected
**momentum**:

| Metric | Value |
|---|---|
| Total Return | **-2.08%** |
| CAGR | -0.23% |
| Sharpe Ratio | -2.71 |
| Max Drawdown | -5.59% |
| Trades | 80 |
| Win Rate | 41.2% |
| Profit Factor | 0.96 |

**Regime distribution:** BULLISH 1595 · BEARISH 281 · SIDEWAYS 149 · UNCERTAIN 92
(+199 warmup bars labeled UNKNOWN).

> **Interpretation:** the framework executes trades correctly and reproducibly,
> but the current strategies are marginally unprofitable at default parameters
> (profit factor just below 1.0). Finding profitable configurations — via
> parameter tuning and additional strategies — is the active area of work.

---

## Configuration

All tunable parameters live in `config.yaml`:

```yaml
paper_capital: 100000.0     # starting capital
commission: 0.001           # 0.1% per trade
slippage: 0.0005            # 0.05% per trade

momentum:
  lookback: 20
  threshold: 0.02

mean_reversion:
  lookback: 20
  zscore_threshold: 2.0

strategy_pool:              # which strategies the meta-agent runs
  - momentum
  - mean_reversion

scoring_weights:            # how the meta-agent ranks strategies
  return: 0.4
  sharpe: 0.3
  win_rate: 0.2
  drawdown: 0.1
```

---

## Roadmap

- [ ] **Parameter optimization** — sweep strategy params to find profitable configs.
- [ ] **ML strategy agent** — Random Forest classifier (scaffolded in config) with
      SHAP explainability.
- [ ] **Volatility regime labels** — emit `HIGH VOLATILITY` so mean-reversion can
      trade its intended regime.
- [ ] **Additional strategies** — sentiment and macro agents (params reserved in config).
- [ ] **Multi-asset support** — extend beyond single-ticker (SPY).
- [ ] **Cleanup** — remove superseded prototypes (`Data_extraction.py`,
      `strategies/mean_reversion.py`); consolidate duplicate regime-series builders.

---

## Tech Stack

**Language:** Python 3.11+
**Core:** pandas, NumPy
**Data:** yfinance
**ML / explainability (planned):** scikit-learn, SHAP
**Visualization:** matplotlib, seaborn

---

## Disclaimer

This project is for **educational and research purposes only**. It is a
backtesting simulation and does not execute real trades. Nothing here is
financial advice. Past (simulated) performance does not indicate future results.
