import numpy as np
import pandas as pd
from datetime import datetime

from signal_generator import generate_long_only_signals, attach_signals


def _make_single_ticker_df():
    """
    Create a tiny single-ticker DataFrame suitable to test attach_signals.
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


def test_generate_long_only_signals_threshold_logic():
    proba = np.array([0.40, 0.60, 0.55])

    # default threshold 0.55: strictly > threshold
    sig_default = generate_long_only_signals(proba, threshold=0.55)
    assert sig_default.tolist() == [0, 1, 0]

    # lower threshold -> more 1s
    sig_lower = generate_long_only_signals(proba, threshold=0.50)
    assert sig_lower.tolist() == [0, 1, 1]

    # all signals must be 0 or 1
    assert set(sig_default.tolist()).issubset({0, 1})
    assert set(sig_lower.tolist()).issubset({0, 1})


def test_attach_signals_adds_column_and_preserves_length():
    df = _make_single_ticker_df()
    signals = np.array([1, 0, 1])

    df_with_sig = attach_signals(df, signals, signal_col="signal")

    # same number of rows
    assert len(df_with_sig) == len(df)

    # new column exists
    assert "signal" in df_with_sig.columns

    # original columns untouched
    for col in ["date", "ticker", "future_ret_1d"]:
        assert col in df_with_sig.columns

    # values match exactly
    assert df_with_sig["signal"].tolist() == signals.tolist()
