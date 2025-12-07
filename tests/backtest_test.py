import numpy as np
import pandas as pd
from datetime import datetime

from backtest import (
    compute_strategy_daily_pnl,
    compute_buy_and_hold_daily_pnl,
    build_equity_curve,
    run_backtest,
)


def _make_single_ticker_df():
    """
    Create a tiny single-ticker DataFrame suitable for backtest tests.

    Dates: 3 days
    future_ret_1d: [0.01, 0.00, -0.02]
    """
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["TEST"] * len(dates),
            "future_ret_1d": [0.01, 0.0, -0.02],
        }
    )
    return df


def test_compute_strategy_daily_pnl_single_ticker():
    df = _make_single_ticker_df()
    df["signal"] = [1, 0, 1]  # long on day 1 and 3

    position_size = 1000.0

    daily = compute_strategy_daily_pnl(
        df,
        position_size=position_size,
        signal_col="signal",
        future_ret_col="future_ret_1d",
        date_col="date",
    )

    # We expect:
    # day1: pnl = 1000 * 0.01 = 10
    # day2: pnl = 0 * 0.00 = 0
    # day3: pnl = 1000 * (-0.02) = -20
    expected_pnls = [10.0, 0.0, -20.0]
    expected_positions = [1000.0, 0.0, 1000.0]

    assert daily["daily_pnl"].tolist() == expected_pnls
    assert daily["total_position"].tolist() == expected_positions


def test_compute_buy_and_hold_daily_pnl_single_ticker():
    df = _make_single_ticker_df()
    position_size = 1000.0

    daily_bh = compute_buy_and_hold_daily_pnl(
        df,
        position_size=position_size,
        future_ret_col="future_ret_1d",
        date_col="date",
    )

    # buy-and-hold is always long with same notional
    # so pnls should match strategy with all signals=1:
    # [10, 0, -20]
    expected_bh_pnls = [10.0, 0.0, -20.0]
    assert daily_bh["bh_daily_pnl"].tolist() == expected_bh_pnls


def test_build_equity_curve_consistency():
    # daily pnl as in previous tests
    index = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    daily_pnl = pd.Series([10.0, 0.0, -20.0], index=index, name="daily_pnl")
    initial_capital = 100_000.0

    returns, equity = build_equity_curve(daily_pnl, initial_capital=initial_capital)

    # length matches
    assert len(returns) == len(daily_pnl)
    assert len(equity) == len(daily_pnl)

    # equity progression: 100000 -> 100010 -> 100010 -> 99990
    assert equity.iloc[0] == initial_capital + 10.0
    assert equity.iloc[1] == equity.iloc[0] + 0.0
    assert equity.iloc[2] == equity.iloc[1] - 20.0

    # check returns are pnl / previous_equity
    prev_equity = initial_capital
    for t, pnl in zip(equity.index, daily_pnl):
        expected_ret = pnl / prev_equity if prev_equity != 0 else 0.0
        assert abs(returns.loc[t] - expected_ret) < 1e-9
        prev_equity = prev_equity + pnl


def test_run_backtest_strategy_vs_buy_and_hold():
    df = _make_single_ticker_df()
    # signals: long day 1 & 3, flat day 2
    df["signal"] = [1, 0, 1]

    initial_capital = 100_000.0
    position_size = 1000.0

    bt = run_backtest(
        df_with_signals=df,
        initial_capital=initial_capital,
        position_size=position_size,
        signal_col="signal",
        future_ret_col="future_ret_1d",
        date_col="date",
    )

    # required columns exist
    expected_cols = {
        "total_position",
        "daily_pnl",
        "strategy_return",
        "strategy_equity",
        "bh_daily_pnl",
        "bh_return",
        "bh_equity",
    }
    assert expected_cols.issubset(set(bt.columns))

    # final equity should be positive and finite
    final_strat = bt["strategy_equity"].iloc[-1]
    final_bh = bt["bh_equity"].iloc[-1]
    assert np.isfinite(final_strat)
    assert np.isfinite(final_bh)
    assert final_strat > 0
    assert final_bh > 0

    # sanity: when signals are all zero, strategy equity should remain flat
    df_all_flat = _make_single_ticker_df()
    df_all_flat["signal"] = [0, 0, 0]

    bt_flat = run_backtest(
        df_with_signals=df_all_flat,
        initial_capital=initial_capital,
        position_size=position_size,
        signal_col="signal",
        future_ret_col="future_ret_1d",
        date_col="date",
    )

    # strategy equity flat, buy-and-hold still moves with returns
    assert bt_flat["strategy_equity"].iloc[0] == initial_capital
    assert bt_flat["strategy_equity"].iloc[-1] == initial_capital
    assert bt_flat["bh_equity"].iloc[-1] != initial_capital
