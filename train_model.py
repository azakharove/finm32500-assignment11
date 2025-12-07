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

from backtest import run_backtest
from feature_engineering import build_dataset
from signal_generator import generate_long_only_signals, attach_signals

BASE_DIR = Path(__file__).resolve().parent

def load_configs():
    with open(BASE_DIR / "features_config.json", "r") as f:
        feature_cfg = json.load(f)
    with open(BASE_DIR / "model_params.json", "r") as f:
        model_cfg = json.load(f)
    return feature_cfg, model_cfg

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_").lower()


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Create and save a confusion matrix plot, return file path."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    filename = PLOT_DIR / f"confusion_{_sanitize_name(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return str(filename)


def plot_residuals(y_true, y_hat, model_name: str) -> str:
    """
    Plot residuals (y_true - y_hat) vs predicted values and save to plots/.
    """
    residuals = y_true - y_hat
    plt.figure()
    plt.scatter(y_hat, residuals, s=10, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted probability (class 1)")
    plt.ylabel("Residual (y_true - y_hat)")
    plt.title(f"{model_name} - Residuals")
    filename = PLOT_DIR / f"residuals_{_sanitize_name(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return str(filename)


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

def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_").lower()


def plot_prediction_distribution(y_proba, model_name: str) -> str:
    """Histogram of predicted probabilities."""
    plt.figure()
    plt.hist(y_proba, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Predicted probability (class 1)")
    plt.ylabel("Count")
    plt.title(f"{model_name} - Prediction Distribution")
    filename = PLOT_DIR / f"pred_dist_{_sanitize_name(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return str(filename)


def plot_equity_curves(backtest_df: pd.DataFrame, model_name: str) -> str:
    """Plot strategy vs buy-and-hold equity curves."""
    plt.figure()
    backtest_df["strategy_equity"].plot(label="Strategy")
    backtest_df["bh_equity"].plot(label="Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(f"{model_name} - Equity Curves")
    plt.legend()
    filename = PLOT_DIR / f"equity_{_sanitize_name(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return str(filename)


def plot_feature_importance_rf(rf_result: dict, feature_names, top_n: int = 10) -> str | None:
    """
    Plot Random Forest feature importances (top_n features).
    rf_result should be the dict returned by train_random_forest.
    """
    model = rf_result["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    series = pd.Series(importances, index=feature_names)
    series = series.sort_values(ascending=False).head(top_n)

    plt.figure()
    series[::-1].plot(kind="barh")  # reversed so most important is at top
    plt.xlabel("Importance")
    plt.title("Random Forest - Top Feature Importances")
    filename = PLOT_DIR / "feature_importance_random_forest.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return str(filename)

def write_comparison_markdown(
    model_results: list[dict],
    backtest_summaries: list[dict],
    feature_names: list[str],
    output_path: Path | None = None,
):
    """
    Write comparison.md summarizing model and strategy performance.

    model_results: list of dicts from train_logistic_regression / train_random_forest
    backtest_summaries: list of dicts from backtest_model_on_test_set
    feature_names: list of feature names from features_config.json
    """
    if output_path is None:
        output_path = BASE_DIR / "comparison.md"

    # index by name for easy lookup
    bt_by_name = {bt["name"]: bt for bt in backtest_summaries}

    # (optional) find RF result for feature importances
    rf_result = next((r for r in model_results if "Random Forest" in r["name"]), None)

    lines = []
    lines.append("# Model and Strategy Comparison\n")

    # ---------- Summary Table ---------- #
    lines.append("## Metric & Performance Summary\n")
    lines.append(
        "| Model | Accuracy | Precision | Recall | MSE | R² | CV Accuracy | Final Strategy Equity | Final Buy & Hold Equity |"
    )
    lines.append(
        "|-------|----------|-----------|--------|-----|----|-------------|------------------------|-------------------------|"
    )

    for res in model_results:
        name = res["name"]
        metrics = res["metrics"]
        bt = bt_by_name.get(name)
        strat_eq = bt["final_strategy_equity"] if bt is not None else float("nan")
        bh_eq = bt["final_bh_equity"] if bt is not None else float("nan")

        lines.append(
            f"| {name} | "
            f"{metrics['accuracy']:.3f} | "
            f"{metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | "
            f"{metrics['mse']:.4f} | "
            f"{metrics['r2']:.3f} | "
            f"{metrics['cv_mean_accuracy']:.3f} | "
            f"{strat_eq:,.2f} | "
            f"{bh_eq:,.2f} |"
        )

    # ---------- Visualizations ---------- #
    lines.append("\n## Visualizations\n")

    lines.append("### Confusion Matrices\n")
    for res in model_results:
        name = res["name"]
        san = _sanitize_name(name)
        cm_path = f"plots/confusion_{san}.png"
        lines.append(f"![Confusion Matrix - {name}]({cm_path})")

    lines.append("\n### Prediction Distributions\n")
    for res in model_results:
        name = res["name"]
        san = _sanitize_name(name)
        pd_path = f"plots/pred_dist_{san}.png"
        lines.append(f"![Prediction Distribution - {name}]({pd_path})")

    lines.append("\n### Equity Curves\n")
    for bt in backtest_summaries:
        name = bt["name"]
        san = _sanitize_name(name)
        eq_path = f"plots/equity_{san}.png"
        lines.append(f"![Equity Curves - {name}]({eq_path})")

    if rf_result is not None:
        lines.append("\n### Feature Importance (Random Forest)\n")
        lines.append("![Random Forest Feature Importance](plots/feature_importance_random_forest.png)")

    # ---------- Discussion ---------- #
    lines.append("\n## Discussion\n")

    # Which model performed best?
    lines.append("### Which Model Performed Best?\n")
    # pick by strategy equity primarily, fallback to accuracy
    best_by_equity = max(
        backtest_summaries,
        key=lambda bt: bt["final_strategy_equity"],
        default=None,
    )
    if best_by_equity is not None:
        lines.append(
            f"- **By financial performance (final strategy equity)**, "
            f"`{best_by_equity['name']}` performed best.\n"
        )
    best_by_accuracy = max(
        model_results,
        key=lambda r: r["metrics"]["accuracy"],
        default=None,
    )
    if best_by_accuracy is not None:
        lines.append(
            f"- **By predictive accuracy**, `{best_by_accuracy['name']}` achieved the highest accuracy "
            f"of {best_by_accuracy['metrics']['accuracy']:.3f}.\n"
        )

    # Which features most predictive? (use RF importances if available)
    lines.append("\n### Which Features Were Most Predictive?\n")
    if rf_result is not None and hasattr(rf_result["model"], "feature_importances_"):
        importances = rf_result["model"].feature_importances_
        series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        top = series.head(5)
        lines.append("Based on the Random Forest feature importances, the top predictive features were:\n")
        for feat, val in top.items():
            lines.append(f"- `{feat}` (importance ≈ {val:.3f})")
        lines.append("")
    else:
        lines.append(
            "- Random Forest feature importances were not available; feature-level interpretation is limited.\n"
        )

    # Limitations of ML in financial forecasting
    lines.append("\n### Limitations of ML in Financial Forecasting\n")
    lines.append(
        "- **Non-stationarity**: Market relationships change over time. A model trained on historical data may "
        "degrade when regimes, volatility, or liquidity conditions change.\n"
    )
    lines.append(
        "- **Overfitting risk**: With many features and limited history, models can fit noise rather than signal, "
        "leading to poor out-of-sample performance.\n"
    )
    lines.append(
        "- **Ignoring transaction costs and market impact**: Our backtest assumes zero transaction costs and "
        "frictionless execution, which overstates real-world profitability.\n"
    )
    lines.append(
        "- **Data leakage and alignment**: Care must be taken to ensure that only information available at time t "
        "is used to predict t+1. Using future data or misaligned labels can make performance look unrealistically good.\n"
    )
    lines.append(
        "- **Simplified position sizing and risk management**: We assume fixed position sizes and ignore portfolio "
        "constraints, risk limits, and drawdown controls that practitioners must consider.\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written comparison report to {output_path}")

def backtest_model_on_test_set(
    model_result: dict,
    df_test: pd.DataFrame,
    initial_capital: float = 100_000.0,
    position_size: float = 1_000.0,
    threshold: float = 0.55,
    signal_col: str = "signal",
    future_ret_col: str = "future_ret_1d",
    date_col: str = "date",
) -> dict:
    """
    Given a trained model result dict and the test DataFrame,
    generate trading signals, run backtest, and return a summary.
    """
    model_name = model_result["name"]
    y_proba = model_result["y_proba"]

    # 1) Generate signals from probabilities
    signals = generate_long_only_signals(y_proba, threshold=threshold)

    # 2) Attach signals to test DataFrame
    df_test_with_signals = attach_signals(df_test, signals, signal_col=signal_col)

    # 3) Run backtest
    bt = run_backtest(
        df_with_signals=df_test_with_signals,
        initial_capital=initial_capital,
        position_size=position_size,
        signal_col=signal_col,
        future_ret_col=future_ret_col,
        date_col=date_col,
    )

    final_strategy_equity = float(bt["strategy_equity"].iloc[-1])
    final_bh_equity = float(bt["bh_equity"].iloc[-1])

    return {
        "name": model_name,
        "backtest_df": bt,
        "final_strategy_equity": final_strategy_equity,
        "final_bh_equity": final_bh_equity,
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

    # align df_test with X_test
    df_train = df.iloc[: len(X_train)].copy()
    df_test = df.iloc[len(X_train):].copy()

    # Train models
    log_res = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=model_cfg.get("logistic_regression", {"max_iter": 1000}),
    )

    rf_res = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=model_cfg.get("random_forest", {"n_estimators": 200, "random_state": 42}),
    )

    model_results = [log_res, rf_res]

    # Plots: confusion matrices, residuals, prediction dists
    for res in model_results:
        name = res["name"]
        y_pred = res["y_pred"]
        y_proba = res["y_proba"]

        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        resid_path = plot_residuals(y_test, y_proba, name)
        pred_dist_path = plot_prediction_distribution(y_proba, name)

        plot_prediction_distribution(y_proba, name)

    # Backtests for each model
    backtest_summaries = []
    for res in model_results:
        bt_summary = backtest_model_on_test_set(
            model_result=res,
            df_test=df_test,
            initial_capital=100_000.0,
            position_size=1_000.0,
            threshold=0.55,
            signal_col="signal",
            future_ret_col="future_ret_1d",
            date_col="date",
        )
        backtest_summaries.append(bt_summary)
        # equity curve plots
        plot_equity_curves(bt_summary["backtest_df"], bt_summary["name"])

    # Feature importance for RF
    plot_feature_importance_rf(rf_res, feature_cols)

    # Print metrics and backtest summary to console
    print("\n=== Model Evaluation Results ===")
    for res in model_results:
        name = res["name"]
        metrics = res["metrics"]
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Backtest Results (Strategy vs Buy-and-Hold) ===")
    for bt in backtest_summaries:
        name = bt["name"]
        print(f"\n{name}:")
        print(f"  Final strategy equity:     {bt['final_strategy_equity']:.2f}")
        print(f"  Final buy-and-hold equity: {bt['final_bh_equity']:.2f}")

    # Write comparison.md
    write_comparison_markdown(
        model_results=model_results,
        backtest_summaries=backtest_summaries,
        feature_names=feature_cols,
        output_path=BASE_DIR / "comparison.md",
    )

if __name__ == "__main__":
    main()