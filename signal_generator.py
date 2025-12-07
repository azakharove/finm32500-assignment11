# Converts predictions into trading signals
from typing import Union
import numpy as np
import pandas as pd


def generate_long_only_signals(
    predicted_proba: Union[np.ndarray, pd.Series],
    threshold: float = 0.55,
) -> np.ndarray:
    """
    Convert predicted probabilities of positive return into long-only signals.

    predicted_proba : Predicted probability that next-day return > 0 for each observation.
    threshold : float, optional
        Probability threshold above which we go long (1). Otherwise, flat (0).

    Returns: np.ndarray
        Array of signals: 1 for long, 0 for flat.
    """
    proba = np.asarray(predicted_proba)
    signals = (proba > threshold).astype(int)
    return signals


def attach_signals(
    df: pd.DataFrame,
    signals: Union[np.ndarray, pd.Series],
    signal_col: str = "signal",
) -> pd.DataFrame:

    out = df.copy()
    out[signal_col] = np.asarray(signals)
    return out
