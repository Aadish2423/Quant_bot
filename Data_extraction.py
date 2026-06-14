import yfinance as yf
import pandas as pd
import numpy as np

class MarketRegimeAgent:

    def __init__(self, ticker="SPY"):
        self.ticker = ticker

    def fetch_data(self):
        df = yf.download(
            self.ticker,
            period="5y",
            auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def compute_features(self, df):

        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        df["Returns"] = df["Close"].pct_change()

        df["Volatility"] = (
            df["Returns"]
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

        return df

    def detect_regime(self, df):

        latest = df.iloc[-1]

        price = latest["Close"]
        ma50 = latest["MA50"]
        ma200 = latest["MA200"]

        vol = latest["Volatility"]

        avg_vol = (
            df["Volatility"]
            .dropna()
            .tail(252)
            .mean()
        )

        trend_strength = (
            (price - ma200)
            / ma200
        )

        if vol > 1.5 * avg_vol:
            return "HIGH VOLATILITY"

        if price > ma200 and ma50 > ma200:
            return "BULLISH"

        if price < ma200 and ma50 < ma200:
            return "BEARISH"

        if abs(trend_strength) < 0.03:
            return "SIDEWAYS"

        return "UNCERTAIN"

    def analyze(self):

        df = self.fetch_data()

        df = self.compute_features(df)

        regime = self.detect_regime(df)

        return regime

agent = MarketRegimeAgent("SPY")

print(
    "Current Regime:",
    agent.analyze()
)