import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_dataset


BASE_DIR = Path(__file__).resolve().parent

def load_configs():
    with open(BASE_DIR / "features_config.json", "r") as f:
        feature_cfg = json.load(f)
    with open(BASE_DIR / "model_params.json", "r") as f:
        model_cfg = json.load(f)
    return feature_cfg, model_cfg

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Create, save confusion matrix & return file path."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    filepath = PLOT_DIR / f"confusion_{model_name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    return str(filepath)


def plot_residuals(y_true, y_hat, model_name: str) -> str:
    """Create, save residual plot & return file path."""
    residuals = y_true - y_hat
    plt.figure()
    plt.scatter(y_hat, residuals, alpha=0.7, s=10)
    plt.axhline(0, linestyle="--", color="red")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Residual (Actual - Pred)")
    plt.title(f"{model_name} - Residual Plot")
    filepath = PLOT_DIR / f"residuals_{model_name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    return str(filepath)

def compute_metrics(y_true, y_pred, y_hat_for_reg):
    """
    Compute classification + regression-style metrics.

    y_pred: hard class predictions (0/1)
    y_hat_for_reg: continuous predictions (e.g., predicted probability of class 1)
    """
    metrics = {
        # classification
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        # regression-style metrics on probabilities vs {0,1} labels
        "mse": mean_squared_error(y_true, y_hat_for_reg),
        "r2": r2_score(y_true, y_hat_for_reg),
    }
    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test, params) -> dict:
    """
    Train Logistic Regression in a pipeline with StandardScaler.
    Returns dict with metrics, predictions, and CV score.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="accuracy"
    )
    metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))

    return {
        "name": "Logistic Regression",
        "model": pipe,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def train_random_forest(X_train, y_train, X_test, y_test, params) -> dict:
    """
    Train Random Forest classifier
    Returns dict with metrics, predictions, feature importances, and CV score.
    """
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    cv_scores = cross_val_score(
        rf, X_train, y_train, cv=5, scoring="accuracy"
    )
    metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))

    return {
        "name": "Random Forest",
        "model": rf,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def main():
    feature_cfg, model_cfg = load_configs()
    df = build_dataset()

    feature_cols = feature_cfg["features"]
    target_col = feature_cfg.get("target", "label")

    X = df[feature_cols].values
    y = df[target_col].values

    # time-ordered split (no shuffle to respect chronology)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # train models
    log_res = train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        model_cfg.get("logistic_regression", {"max_iter": 1000}),
    )

    rf_res = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        model_cfg.get("random_forest", {"n_estimators": 200, "random_state": 42}),
    )

    results = [log_res, rf_res]

    for res in results:
        name = res["name"]
        y_pred = res["y_pred"]
        y_proba = res["y_proba"]

        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        res_path = plot_residuals(y_test, y_proba, name)

        print(f"Saved plots for {name}:")
        print(f"  Confusion Matrix → {cm_path}")
        print(f"  Residual Plot → {res_path}")

    # print summary to console
    print("\n=== Model Evaluation Results ===")
    for res in results:
        name = res["name"]
        metrics = res["metrics"]
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
