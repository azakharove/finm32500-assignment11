# Simulates trading strategy and computes performance

# backtest.py
from typing import Tuple
import numpy as np
import pandas as pd


def compute_strategy_daily_pnl(
    df: pd.DataFrame,
    position_size: float,
    signal_col: str = "signal",
    future_ret_col: str = "future_ret_1d",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute daily PnL and position for the trading strategy.
    """
    df = df.copy()

    if signal_col not in df.columns:
        raise ValueError(f"{signal_col} column not found in DataFrame.")
    if future_ret_col not in df.columns:
        raise ValueError(f"{future_ret_col} column not found in DataFrame.")

    # notional position per row (per ticker)
    df["position_notional"] = df[signal_col] * position_size

    # PnL per row = position * percentage return
    df["pnl"] = df["position_notional"] * df[future_ret_col]

    daily = (
        df.groupby(date_col)
        .agg(
            total_position=("position_notional", "sum"),
            daily_pnl=("pnl", "sum"),
        )
        .sort_index()
    )
    return daily


def compute_buy_and_hold_daily_pnl(
    df: pd.DataFrame,
    position_size: float,
    future_ret_col: str = "future_ret_1d",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute daily PnL for a simple buy-and-hold baseline.
    """
    df = df.copy()

    if future_ret_col not in df.columns:
        raise ValueError(f"{future_ret_col} column not found in DataFrame.")

    df["bh_position_notional"] = position_size
    df["bh_pnl"] = df["bh_position_notional"] * df[future_ret_col]

    daily_bh = (
        df.groupby(date_col)
        .agg(bh_daily_pnl=("bh_pnl", "sum"))
        .sort_index()
    )
    return daily_bh


def build_equity_curve(
    daily_pnl: pd.Series,
    initial_capital: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    From daily PnL, build daily returns and equity curve.
    """
    equity_values = []
    return_values = []

    equity = initial_capital

    for pnl in daily_pnl:
        ret = pnl / equity if equity != 0 else 0.0
        return_values.append(ret)
        equity = equity + pnl
        equity_values.append(equity)

    returns_series = pd.Series(return_values, index=daily_pnl.index, name="return")
    equity_series = pd.Series(equity_values, index=daily_pnl.index, name="equity")

    return returns_series, equity_series


def run_backtest(
    df_with_signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    position_size: float = 1_000.0,
    signal_col: str = "signal",
    future_ret_col: str = "future_ret_1d",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Run a simple long-only backtest and compare to a buy-and-hold baseline.
    """
    # Strategy PnL per day
    strat_daily = compute_strategy_daily_pnl(
        df_with_signals,
        position_size=position_size,
        signal_col=signal_col,
        future_ret_col=future_ret_col,
        date_col=date_col,
    )

    # Buy & hold PnL per day
    bh_daily = compute_buy_and_hold_daily_pnl(
        df_with_signals,
        position_size=position_size,
        future_ret_col=future_ret_col,
        date_col=date_col,
    )

    # Equity curves
    strat_ret, strat_eq = build_equity_curve(
        strat_daily["daily_pnl"], initial_capital=initial_capital
    )
    bh_ret, bh_eq = build_equity_curve(
        bh_daily["bh_daily_pnl"], initial_capital=initial_capital
    )

    # Assemble results
    result = strat_daily.join(bh_daily, how="inner")

    result["strategy_return"] = strat_ret
    result["strategy_equity"] = strat_eq

    result["bh_return"] = bh_ret
    result["bh_equity"] = bh_eq

    return result
