import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_engineering import add_features


def _make_dummy_data(num_days: int = 40) -> pd.DataFrame:
    """
    Create a simple multi-day, single-ticker OHLCV DataFrame
    that mimics market_data_ml.csv structure.
    """
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Make price drift upward then downward so we have both positive
    # and negative future returns.
    closes = np.linspace(100, 110, num_days // 2).tolist() + \
             np.linspace(110, 100, num_days - num_days // 2).tolist()

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["TEST"] * num_days,
            "open": closes,
            "high": np.array(closes) + 1.0,
            "low": np.array(closes) - 1.0,
            "close": closes,
            "volume": np.arange(num_days) + 1,
        }
    )
    return df


def test_add_features_creates_expected_columns():
    raw = _make_dummy_data()
    df = add_features(raw)

    # basic shape sanity
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_cols = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1d",
        "log_ret_1d",
        "ret_3d",
        "ret_5d",
        "sma_5",
        "sma_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "future_ret_1d",
        "label",
    }

    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"

    # ensure engineered numeric features do not contain NaNs
    feature_cols = [
        "ret_1d",
        "log_ret_1d",
        "ret_3d",
        "ret_5d",
        "sma_5",
        "sma_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "future_ret_1d",
    ]
    assert not df[feature_cols].isna().any().any(), "NaNs found in feature columns"


def test_label_matches_future_return_sign():
    raw = _make_dummy_data()
    df = add_features(raw)

    # labels should be only 0 or 1
    assert set(df["label"].unique()).issubset({0, 1})

    # by definition, label = (future_ret_1d > 0)
    expected_labels = (df["future_ret_1d"] > 0).astype(int)
    assert (df["label"] == expected_labels).all(), \
        "label column does not match sign of future_ret_1d"
