# main.py
"""
Entry point for the ML trading assignment.

Running this file will:
  - Build the dataset (feature_engineering.build_dataset)
  - Train Logistic Regression and Random Forest models
  - Evaluate models (accuracy, precision, recall, MSE, R², CV accuracy)
  - Generate plots (confusion matrices, residuals, prediction distributions,
    equity curves, feature importance)
  - Run backtests for each model vs buy-and-hold
  - Write comparison.md summarizing results
"""

from train_model import main as run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()
