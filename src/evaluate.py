"""
Post-training evaluation: plots, metrics, forecast visualization.

Generates:
  reports/figures/forecast_comparison.png  — all models vs actual
  reports/figures/prediction_intervals.png — LightGBM + Chronos P10/P90
  reports/figures/residuals.png            — error distribution
  reports/figures/metrics_comparison.png   — bar chart of RMSLE per model

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FIGURES_DIR, MODELS_DIR, LGBM_MODEL_PATH,
    MODEL_META_PATH, HORIZON, ID_COL, DATE_COL, TARGET_COL,
)
from src.data_loader import load_m5
from src.metrics import evaluate_forecasts, print_metrics_table

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
sns.set_theme(style="whitegrid", palette="muted")


def plot_forecast_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    n_series: int = 3,
) -> None:
    """Plot actual vs predicted for n_series representative series."""
    series_ids = train_df[ID_COL].unique()[:n_series]
    n_models   = len(predictions)

    fig, axes = plt.subplots(n_series, 1, figsize=(14, 4 * n_series))
    if n_series == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    model_colors = dict(zip(predictions.keys(), colors))

    for ax, uid in zip(axes, series_ids):
        # Plot last 60 days of train + all of test
        train_s = train_df[train_df[ID_COL] == uid].tail(60)
        test_s  = test_df[test_df[ID_COL] == uid]

        ax.plot(train_s[DATE_COL], train_s[TARGET_COL],
                color="black", linewidth=1.5, label="Actual (train)", alpha=0.7)
        ax.plot(test_s[DATE_COL], test_s[TARGET_COL],
                color="black", linewidth=2.5, label="Actual (test)", linestyle="--")

        for model_name, preds in predictions.items():
            pred_s = preds[preds[ID_COL] == uid]
            if pred_s.empty:
                continue
            ax.plot(pred_s[DATE_COL], pred_s["y_pred"],
                    color=model_colors.get(model_name, "grey"),
                    linewidth=1.5, label=model_name, alpha=0.85)

        ax.set_title(f"Series: {uid}", fontweight="bold")
        ax.set_ylabel("Units Sold")
        ax.legend(fontsize=8, ncol=3)

    fig.suptitle(f"Forecast Comparison — {HORIZON}-day horizon",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "forecast_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_intervals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    model_name: str,
    n_series: int = 2,
) -> None:
    """Fan chart — median forecast + 80%/90% prediction intervals."""
    if "lo-90" not in preds_df.columns:
        return

    series_ids = train_df[ID_COL].unique()[:n_series]
    fig, axes  = plt.subplots(1, n_series, figsize=(14, 4))
    if n_series == 1:
        axes = [axes]

    for ax, uid in zip(axes, series_ids):
        train_s = train_df[train_df[ID_COL] == uid].tail(60)
        test_s  = test_df[test_df[ID_COL] == uid]
        pred_s  = preds_df[preds_df[ID_COL] == uid]

        ax.plot(train_s[DATE_COL], train_s[TARGET_COL],
                color="black", linewidth=1.5, label="Historical")
        ax.plot(test_s[DATE_COL], test_s[TARGET_COL],
                color="black", linewidth=2.5, linestyle="--", label="Actual")
        ax.plot(pred_s[DATE_COL], pred_s["y_pred"],
                color="#e74c3c", linewidth=2, label="Median forecast")

        if "lo-80" in pred_s.columns:
            ax.fill_between(pred_s[DATE_COL],
                            pred_s["lo-80"], pred_s["hi-80"],
                            alpha=0.3, color="#e74c3c", label="80% interval")
        ax.fill_between(pred_s[DATE_COL],
                        pred_s["lo-90"], pred_s["hi-90"],
                        alpha=0.15, color="#e74c3c", label="90% interval")

        ax.set_title(uid, fontweight="bold")
        ax.set_ylabel("Units Sold")
        ax.legend(fontsize=8)

    fig.suptitle(f"Prediction Intervals — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "prediction_intervals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metrics_comparison(results: dict[str, pd.DataFrame]) -> None:
    """Bar chart comparing RMSLE across all models."""
    rows = []
    for name, m in results.items():
        if m is None:
            continue
        agg = m[m[ID_COL] == "ALL (mean)"].iloc[0]
        for metric in ["rmse", "mae", "rmsle", "mase"]:
            if metric in agg and pd.notna(agg[metric]):
                rows.append({"Model": name, "Metric": metric.upper(),
                              "Value": float(agg[metric])})

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["RMSLE", "MASE"]):
        sub = df[df["Metric"] == metric].sort_values("Value")
        bars = ax.barh(sub["Model"], sub["Value"],
                       color=["#2ecc71" if i == 0 else "#3498db"
                              for i in range(len(sub))],
                       edgecolor="white")
        for bar, val in zip(bars, sub["Value"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel(metric)
        ax.set_title(f"Model Comparison — {metric}", fontweight="bold")
        ax.set_xlim(0, sub["Value"].max() * 1.2)

    plt.tight_layout()
    path = FIGURES_DIR / "metrics_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_evaluation(results: dict = None, predictions: dict = None):
    """
    Run full evaluation pipeline.
    If results/predictions not provided, loads from cache.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_m5()

    if results is None or predictions is None:
        print("Re-running LightGBM predictions for evaluation...")
        from src.train_lgbm import load as lgbm_load
        fcst  = lgbm_load()
        preds = fcst.predict(h=HORIZON).rename(columns={"LightGBM": "y_pred"})
        results     = {"LightGBM": evaluate_forecasts(test_df, preds, train_df)}
        predictions = {"LightGBM": preds}

    print("\nGenerating evaluation plots...")
    plot_forecast_comparison(train_df, test_df, predictions)

    # Prediction interval plot — use best available model
    for name in ["Chronos (zero-shot)", "Ensemble", "LightGBM"]:
        if name in predictions and predictions[name] is not None:
            if "lo-90" in predictions[name].columns:
                plot_prediction_intervals(train_df, test_df,
                                          predictions[name], name)
                break

    plot_metrics_comparison(results)
    print("\nAll evaluation artifacts saved to reports/figures/")


if __name__ == "__main__":
    run_evaluation()
