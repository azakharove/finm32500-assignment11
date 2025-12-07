import csv
from datetime import datetime
from pathlib import Path
from typing import List, Iterator
import os
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "Data"

def load_market_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "market_data_ml.csv", parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])
    return df

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=series.index).rolling(window).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(window).mean()

    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD, Signal Line, Histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line

    return macd, signal_line, hist

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily & Log Returns
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()
    df["log_return_1d"] = np.log1p(df["return_1d"])

    for w in [3, 5]:
        df[f"return_{w}d"] = (
            df.groupby("ticker")["return_1d"]
            .rolling(w)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # SMA
    for w in [5, 10]:
        df[f"sma_{w}"] = (
            df.groupby("ticker")["close"]
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # RSI
    df["rsi_14"] = (
        df.groupby("ticker")["close"]
        .apply(compute_rsi, window=14)
        .reset_index(level=0, drop=True)
    )

    # MACD
    # Initialize columns
    df["macd"] = np.nan
    df["macd_signal"] = np.nan
    df["macd_hist"] = np.nan

    # Compute MACD per ticker and assign back by index
    for ticker, grp in df.groupby("ticker"):
        macd, signal_line, hist = compute_macd(grp["close"])
        df.loc[grp.index, "macd"] = macd.values
        df.loc[grp.index, "macd_signal"] = signal_line.values
        df.loc[grp.index, "macd_hist"] = hist.values

    # Label for Next-Day Classification === #
    df["future_ret_1d"] = df.groupby("ticker")["return_1d"].shift(-1)
    df["label"] = (df["future_ret_1d"] > 0).astype(int)

    # Remove NaNs from rolling calc edges
    df = df.dropna().reset_index(drop=True)

    return df

def build_dataset() -> pd.DataFrame:
    df = load_market_data()
    df = add_features(df)
    return df

if __name__ == "__main__":
    df_raw = load_market_data()
    print(type(df_raw))
    print(df_raw.head())

    df_features = add_features(df_raw)
    print(df_features.head())


