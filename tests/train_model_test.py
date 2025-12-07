import numpy as np
import pandas as pd

from feature_engineering import build_dataset
from train_model import (
    load_configs,
    train_logistic_regression,
    train_random_forest,
)

from sklearn.model_selection import train_test_split


def _prepare_data():
    """
    Helper: build full dataset and produce a small train/test split
    consistent with train_model.py.
    """
    feature_cfg, model_cfg = load_configs()
    df = build_dataset()

    feature_cols = feature_cfg["features"]
    target_col = feature_cfg.get("target", "label")

    # Basic sanity
    assert isinstance(df, pd.DataFrame)
    for col in feature_cols + [target_col]:
        assert col in df.columns, f"Column {col} missing from dataset"

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    return X_train, X_test, y_train, y_test, feature_cfg, model_cfg


def test_train_logistic_regression_shapes_and_metrics():
    X_train, X_test, y_train, y_test, _, model_cfg = _prepare_data()

    params = model_cfg.get("logistic_regression", {"max_iter": 1000})

    res = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
    )

    # Basic keys exist
    for key in ["name", "model", "y_pred", "y_proba", "metrics"]:
        assert key in res

    y_pred = res["y_pred"]
    y_proba = res["y_proba"]
    metrics = res["metrics"]

    # Shape checks
    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)

    # Probabilities are between 0 and 1
    assert np.all((y_proba >= 0.0) & (y_proba <= 1.0))

    # Metric keys
    for m in ["accuracy", "precision", "recall", "mse", "r2", "cv_mean_accuracy"]:
        assert m in metrics

    # Classification metrics between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["cv_mean_accuracy"] <= 1.0

    # Regression-style metrics: MSE non-negative, R² finite
    assert metrics["mse"] >= 0.0
    assert np.isfinite(metrics["r2"])


def test_train_random_forest_shapes_and_metrics():
    X_train, X_test, y_train, y_test, feature_cfg, model_cfg = _prepare_data()

    params = model_cfg.get("random_forest", {"n_estimators": 200, "random_state": 42})

    res = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
    )

    for key in ["name", "model", "y_pred", "y_proba", "metrics"]:
        assert key in res

    y_pred = res["y_pred"]
    y_proba = res["y_proba"]
    metrics = res["metrics"]

    # Shape checks
    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)

    # Probabilities in [0, 1]
    assert np.all((y_proba >= 0.0) & (y_proba <= 1.0))

    # Metric keys
    for m in ["accuracy", "precision", "recall", "mse", "r2", "cv_mean_accuracy"]:
        assert m in metrics

    # Classification metrics between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["cv_mean_accuracy"] <= 1.0

    # Regression-style metrics: MSE non-negative, R² finite
    assert metrics["mse"] >= 0.0
    assert np.isfinite(metrics["r2"])

    if "Feature Importances" in res:
        fi = res["Feature Importances"]
        assert isinstance(fi, dict)
        assert len(fi) == len(feature_cfg["features"])
