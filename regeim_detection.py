import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  MARKET DETECTION
# ─────────────────────────────────────────────

MARKET_SIGNATURES = {
    "CRYPTO": {
        "suffixes": ["-USD", "-USDT", "-BTC"],
        "keywords": ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX"],
    },
    "FOREX": {
        "suffixes": ["=X"],
        "keywords": ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY"],
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    },
    "COMMODITY": {
        "suffixes": ["=F"],
        "keywords": ["GC", "CL", "SI", "HG", "NG", "ZC", "ZW", "ZS"],
    },
    "EQUITY": {
        "suffixes": [],
        "keywords": [],  # fallback
    },
}


def detect_market(ticker: str) -> dict:
    """
    Returns confidence scores (0-1) for each market type.
    The highest score wins.
    """
    ticker = ticker.upper().strip()
    scores = {"EQUITY": 0.1, "CRYPTO": 0.0, "FOREX": 0.0, "COMMODITY": 0.0}

    for market, sig in MARKET_SIGNATURES.items():
        if market == "EQUITY":
            continue
        for suffix in sig.get("suffixes", []):
            if ticker.endswith(suffix):
                scores[market] += 0.6
        for kw in sig.get("keywords", []):
            if kw in ticker:
                scores[market] += 0.3
        for pair in sig.get("pairs", []):
            if pair in ticker:
                scores[market] += 0.1

    # Normalise so scores sum to 1
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}
    best_market = max(scores, key=scores.get)
    return {"market": best_market, "confidence": scores}


# ─────────────────────────────────────────────
#  STRATEGIES  (one function per strategy)
#  Each returns: {"regime": str, "signal": float (-1 to 1)}
# ─────────────────────────────────────────────

def strategy_moving_average(df: pd.DataFrame) -> dict:
    """Trend-following via MA50 / MA200 crossover."""
    close = df["Close"]
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    price   = float(close.iloc[-1])
    _ma50   = float(ma50.iloc[-1])
    _ma200  = float(ma200.iloc[-1])

    trend   = (price - _ma200) / _ma200
    signal  = np.clip(trend * 10, -1, 1)

    if price > _ma200 and _ma50 > _ma200:
        regime = "BULLISH"
    elif price < _ma200 and _ma50 < _ma200:
        regime = "BEARISH"
    elif abs(trend) < 0.03:
        regime = "SIDEWAYS"
    else:
        regime = "UNCERTAIN"

    return {"regime": regime, "signal": round(signal, 4)}


def strategy_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    """Momentum via Relative Strength Index."""
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))

    _rsi   = float(rsi.iloc[-1])
    signal = np.clip((_rsi - 50) / 50, -1, 1)   # +1 = overbought, -1 = oversold

    if _rsi >= 70:
        regime = "OVERBOUGHT"
    elif _rsi <= 30:
        regime = "OVERSOLD"
    elif _rsi > 50:
        regime = "BULLISH MOMENTUM"
    else:
        regime = "BEARISH MOMENTUM"

    return {"regime": regime, "signal": round(signal, 4)}


def strategy_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    """Mean-reversion via Bollinger Bands."""
    close  = df["Close"]
    mid    = close.rolling(period).mean()
    band   = close.rolling(period).std() * std
    upper  = mid + band
    lower  = mid - band

    price  = float(close.iloc[-1])
    _upper = float(upper.iloc[-1])
    _lower = float(lower.iloc[-1])
    _mid   = float(mid.iloc[-1])
    width  = (_upper - _lower) / _mid

    # position within bands: -1 (at lower) to +1 (at upper)
    signal = np.clip((price - _mid) / (band.iloc[-1] + 1e-9), -1, 1)

    if price > _upper:
        regime = "OVERBOUGHT / BREAKOUT"
    elif price < _lower:
        regime = "OVERSOLD / BREAKDOWN"
    elif width < 0.05:
        regime = "SQUEEZE (LOW VOLATILITY)"
    else:
        regime = "MEAN REVERTING"

    return {"regime": regime, "signal": round(float(signal), 4)}


def strategy_macd(df: pd.DataFrame) -> dict:
    """Trend & momentum via MACD."""
    close     = df["Close"]
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_ln = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_ln

    _hist  = float(histogram.iloc[-1])
    _macd  = float(macd_line.iloc[-1])
    _sig   = float(signal_ln.iloc[-1])
    signal = np.clip(_hist / (abs(_macd) + 1e-9), -1, 1)

    if _macd > _sig and _hist > 0:
        regime = "BULLISH MOMENTUM"
    elif _macd < _sig and _hist < 0:
        regime = "BEARISH MOMENTUM"
    elif _macd > 0:
        regime = "WEAKENING BULL"
    else:
        regime = "WEAKENING BEAR"

    return {"regime": regime, "signal": round(float(signal), 4)}


def strategy_volatility(df: pd.DataFrame) -> dict:
    """Volatility regime via annualised rolling std."""
    returns  = df["Close"].pct_change()
    vol      = returns.rolling(20).std() * np.sqrt(252)
    avg_vol  = float(vol.dropna().tail(252).mean())
    _vol     = float(vol.iloc[-1])
    ratio    = _vol / (avg_vol + 1e-9)
    signal   = np.clip(ratio - 1, -1, 1)          # positive = above-avg vol

    if ratio > 1.5:
        regime = "HIGH VOLATILITY"
    elif ratio < 0.75:
        regime = "LOW VOLATILITY / CALM"
    else:
        regime = "NORMAL VOLATILITY"

    return {"regime": regime, "signal": round(float(signal), 4)}


def strategy_atr_trend(df: pd.DataFrame, period: int = 14) -> dict:
    """Trend strength via Average True Range (good for commodities/forex)."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()

    _atr   = float(atr.iloc[-1])
    _close = float(close.iloc[-1])
    atr_pct = _atr / _close

    # direction from short EMA vs long EMA
    ema_s  = float(close.ewm(span=10).mean().iloc[-1])
    ema_l  = float(close.ewm(span=50).mean().iloc[-1])
    signal = np.clip((ema_s - ema_l) / (_close + 1e-9) * 20, -1, 1)

    if atr_pct > 0.03 and ema_s > ema_l:
        regime = "STRONG UPTREND"
    elif atr_pct > 0.03 and ema_s < ema_l:
        regime = "STRONG DOWNTREND"
    elif atr_pct < 0.01:
        regime = "CONSOLIDATION"
    else:
        regime = "CHOPPY"

    return {"regime": regime, "signal": round(float(signal), 4)}


def strategy_on_balance_volume(df: pd.DataFrame) -> dict:
    """Volume-based regime (equity / crypto)."""
    close  = df["Close"]
    volume = df["Volume"]
    direction = np.sign(close.diff()).fillna(0)
    obv    = (direction * volume).cumsum()
    obv_ma = obv.rolling(20).mean()

    _obv    = float(obv.iloc[-1])
    _obv_ma = float(obv_ma.iloc[-1])
    signal  = np.clip((_obv - _obv_ma) / (abs(_obv_ma) + 1e-9), -1, 1)

    if _obv > _obv_ma:
        regime = "ACCUMULATION (BUYING PRESSURE)"
    else:
        regime = "DISTRIBUTION (SELLING PRESSURE)"

    return {"regime": regime, "signal": round(float(signal), 4)}


# ─────────────────────────────────────────────
#  MARKET → STRATEGY MAPPING
# ─────────────────────────────────────────────

MARKET_STRATEGIES = {
    "EQUITY": [
        strategy_moving_average,
        strategy_rsi,
        strategy_macd,
        strategy_bollinger_bands,
        strategy_on_balance_volume,
        strategy_volatility,
    ],
    "CRYPTO": [
        strategy_rsi,
        strategy_bollinger_bands,
        strategy_volatility,
        strategy_macd,
        strategy_on_balance_volume,
    ],
    "FOREX": [
        strategy_macd,
        strategy_rsi,
        strategy_atr_trend,
        strategy_bollinger_bands,
        strategy_volatility,
    ],
    "COMMODITY": [
        strategy_atr_trend,
        strategy_moving_average,
        strategy_macd,
        strategy_volatility,
        strategy_bollinger_bands,
    ],
}


# ─────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────

class MarketRegimeAgent:

    def __init__(self, ticker: str = "SPY", strategy: str = "AUTO"):
        """
        ticker   : any Yahoo Finance ticker
        strategy : "AUTO" runs all market-appropriate strategies and picks
                   the one with the strongest signal magnitude, or pass
                   a specific name: "RSI", "MACD", "MA", "BB", "VOL", "ATR", "OBV"
        """
        self.ticker   = ticker.upper().strip()
        self.strategy = strategy.upper().strip()

    def fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, period="2y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(how="all", inplace=True)
        return df

    def _strategy_map(self) -> dict:
        return {
            "MA":   strategy_moving_average,
            "RSI":  strategy_rsi,
            "MACD": strategy_macd,
            "BB":   strategy_bollinger_bands,
            "VOL":  strategy_volatility,
            "ATR":  strategy_atr_trend,
            "OBV":  strategy_on_balance_volume,
        }

    def analyze(self) -> dict:
        df = self.fetch_data()

        # ── market detection ──────────────────────────────
        market_info   = detect_market(self.ticker)
        market        = market_info["market"]
        mkt_confidence = market_info["confidence"]

        # ── strategy selection ────────────────────────────
        if self.strategy != "AUTO":
            smap = self._strategy_map()
            if self.strategy not in smap:
                raise ValueError(
                    f"Unknown strategy '{self.strategy}'. "
                    f"Choose from: {list(smap.keys())} or 'AUTO'."
                )
            chosen_fn     = smap[self.strategy]
            result        = chosen_fn(df)
            chosen_name   = self.strategy
            all_results   = {chosen_name: result}
        else:
            candidates    = MARKET_STRATEGIES[market]
            all_results   = {}
            for fn in candidates:
                name = fn.__name__.replace("strategy_", "").upper()
                try:
                    all_results[name] = fn(df)
                except Exception:
                    pass

            # pick strategy with the strongest absolute signal
            chosen_name = max(all_results, key=lambda k: abs(all_results[k]["signal"]))
            result      = all_results[chosen_name]

        # ── signal → confidence ───────────────────────────
        strategy_confidence = round(abs(result["signal"]) * 100, 1)

        return {
            "ticker":              self.ticker,
            "market_detected":     market,
            "market_confidence":   mkt_confidence,
            "strategy_used":       chosen_name,
            "all_strategy_results": all_results,
            "regime":              result["regime"],
            "signal":              result["signal"],
            "strategy_confidence": f"{strategy_confidence}%",
        }


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────

def display(result: dict):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  TICKER          : {result['ticker']}")
    print(f"  MARKET DETECTED : {result['market_detected']}")
    print(f"{sep}")
    print("  Market confidence breakdown:")
    for mkt, conf in sorted(result["market_confidence"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(conf * 30)
        print(f"    {mkt:<12} {conf*100:5.1f}%  {bar}")
    print(f"{sep}")
    print("  Strategy signals (AUTO mode — all run):")
    for name, res in result["all_strategy_results"].items():
        marker = " ◄ CHOSEN" if name == result["strategy_used"] else ""
        sig    = res["signal"]
        bar    = ("█" * int(abs(sig) * 15)).ljust(15)
        direction = "+" if sig >= 0 else "-"
        print(f"    {name:<22} {direction}{bar}  {res['regime']}{marker}")
    print(f"{sep}")
    print(f"  REGIME          : {result['regime']}")
    print(f"  SIGNAL          : {result['signal']:+.4f}  (−1 bearish … +1 bullish)")
    print(f"  CONFIDENCE      : {result['strategy_confidence']}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────
#  MODULE-LEVEL HELPERS
#  Thin wrappers so backtester / example_strategy.py can import
#  compute_features() and detect_regime() without instantiating the class.
# ─────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA50, MA200, Returns, Volatility to df in-place and return it."""
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["MA200"]      = df["Close"].rolling(200).mean()
    df["Returns"]    = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    return df


def detect_regime(df: pd.DataFrame, strategy: str = "AUTO") -> str:
    """
    Detect regime for the last row of df using the chosen strategy.
    Returns a regime string e.g. "BULLISH", "BEARISH", "HIGH VOLATILITY".
    """
    market_info = detect_market("SPY")          # default equity
    market      = market_info["market"]

    if strategy == "AUTO":
        candidates  = MARKET_STRATEGIES[market]
        all_results = {}
        for fn in candidates:
            name = fn.__name__.replace("strategy_", "").upper()
            try:
                all_results[name] = fn(df)
            except Exception:
                pass
        if not all_results:
            return "UNKNOWN"
        chosen = max(all_results, key=lambda k: abs(all_results[k]["signal"]))
        return all_results[chosen]["regime"]
    else:
        smap = {
            "MA": strategy_moving_average, "RSI": strategy_rsi,
            "MACD": strategy_macd, "BB": strategy_bollinger_bands,
            "VOL": strategy_volatility, "ATR": strategy_atr_trend,
            "OBV": strategy_on_balance_volume,
        }
        fn = smap.get(strategy.upper())
        if fn is None:
            return "UNKNOWN"
        return fn(df)["regime"]


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Examples — swap ticker or strategy as needed
    # strategy options: "AUTO", "MA", "RSI", "MACD", "BB", "VOL", "ATR", "OBV"

    agent = MarketRegimeAgent(ticker="SPY", strategy="AUTO")
    result = agent.analyze()
    display(result)
