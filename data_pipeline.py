import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG  — change ticker here, mirrors regime_detection.py
# ─────────────────────────────────────────────

TICKER      = "SPY"
DATA_DIR    = Path("data")
PERIOD_YEARS = 10
SPLIT_YEARS  = 5        # first half = train, second half = test


# ─────────────────────────────────────────────
#  DOWNLOAD
# ─────────────────────────────────────────────

def download_data(ticker: str, years: int = 10) -> pd.DataFrame:
    print(f"[+] Downloading {years}y of data for {ticker} ...")
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(how="all", inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"    Rows fetched : {len(df)}")
    print(f"    Date range   : {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
#  (same logic as regime_detection so features are consistent)
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df.get("Volume", pd.Series(dtype=float))

    # ── trend ──────────────────────────────────
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # ── momentum ───────────────────────────────
    df["Returns"]   = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── volatility ─────────────────────────────
    df["Volatility"]  = df["Returns"].rolling(20).std() * np.sqrt(252)
    df["Vol_Ratio"]   = df["Volatility"] / df["Volatility"].rolling(252).mean()

    # Bollinger Bands
    bb_mid        = close.rolling(20).mean()
    bb_std        = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
    df["BB_Pos"]   = (close - bb_mid) / (bb_std + 1e-9)   # position within bands

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR"] / close

    # ── volume ─────────────────────────────────
    if not volume.empty:
        direction    = np.sign(close.diff()).fillna(0)
        df["OBV"]    = (direction * volume).cumsum()
        df["OBV_MA"] = df["OBV"].rolling(20).mean()
        df["Vol_MA20"] = volume.rolling(20).mean()
        df["Vol_Ratio_Vol"] = volume / df["Vol_MA20"]

    # ── price structure ────────────────────────
    df["High_52w"]   = high.rolling(252).max()
    df["Low_52w"]    = low.rolling(252).min()
    df["Pct_From_High"] = (close - df["High_52w"]) / df["High_52w"]
    df["Pct_From_Low"]  = (close - df["Low_52w"])  / df["Low_52w"]

    return df


# ─────────────────────────────────────────────
#  LABEL GENERATION
#  Simple forward-return label: +1 (up >1%), -1 (down >1%), 0 (flat)
#  Horizon can be changed (default 5 trading days)
# ─────────────────────────────────────────────

def add_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    fwd_return = df["Close"].pct_change(horizon).shift(-horizon)
    df["Label"] = np.where(
        fwd_return >  threshold,  1,
        np.where(fwd_return < -threshold, -1, 0)
    )
    df["Fwd_Return"] = fwd_return
    return df


# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT  (time-based, no shuffle)
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, split_years: int = 5):
    cutoff = df.index[-1] - pd.DateOffset(years=split_years)
    train  = df[df.index <= cutoff].copy()
    test   = df[df.index >  cutoff].copy()
    return train, test


# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────

def save(df: pd.DataFrame, name: str, ticker: str):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{ticker}_{name}.csv"
    df.to_csv(path)
    print(f"    Saved → {path}  ({len(df)} rows, {df.shape[1]} columns)")


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────

def print_summary(train: pd.DataFrame, test: pd.DataFrame, ticker: str):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  TICKER  : {ticker}")
    print(f"{sep}")
    print(f"  TRAIN   : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} rows)")
    print(f"  TEST    : {test.index[0].date()}  → {test.index[-1].date()}   ({len(test)} rows)")
    print(f"  FEATURES: {[c for c in train.columns if c not in ('Label','Fwd_Return')]}")
    print(f"{sep}")
    print("  Label distribution (train):")
    counts = train["Label"].value_counts().sort_index()
    labels = {-1: "BEARISH", 0: "FLAT", 1: "BULLISH"}
    for k, v in counts.items():
        pct = v / len(train) * 100
        print(f"    {labels.get(k, k):<10} {v:>5}  ({pct:.1f}%)")
    print(f"{sep}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run(ticker: str = TICKER):
    df = download_data(ticker, years=PERIOD_YEARS)
    df = add_features(df)
    df = add_labels(df)
    df.dropna(subset=["MA200", "RSI", "MACD", "Label"], inplace=True)

    train, test = split_data(df, split_years=SPLIT_YEARS)

    print_summary(train, test, ticker)

    save(df,    "full",  ticker)
    save(train, "train", ticker)
    save(test,  "test",  ticker)

    return train, test


if __name__ == "__main__":
    train, test = run(TICKER)
