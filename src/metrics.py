"""
Forecasting evaluation metrics.

Standard metrics for the M5 competition and time series forecasting:
  - RMSE   : Root Mean Squared Error
  - MAE    : Mean Absolute Error
  - MASE   : Mean Absolute Scaled Error  (scale-independent, M5 official)
  - RMSLE  : Root Mean Squared Log Error (penalises under-forecasting)
  - sMAPE  : Symmetric MAPE
  - Coverage: fraction of actuals inside prediction interval (for probabilistic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE — clipped to avoid log(negative)."""
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe  = np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(safe) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, seasonality: int = 7) -> float:
    """
    Mean Absolute Scaled Error.
    Scale = MAE of seasonal naive forecast on training set.
    """
    naive_errors = np.abs(
        y_train[seasonality:] - y_train[:-seasonality]
    )
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def coverage(y_true: np.ndarray,
             lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of actuals inside [lo, hi] prediction interval."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def evaluate_forecasts(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    train: pd.DataFrame,
    id_col: str = "unique_id",
    date_col: str = "ds",
    target_col: str = "y",
    pred_col: str = "y_pred",
    lo_col: str | None = "lo-90",
    hi_col: str | None = "hi-90",
) -> pd.DataFrame:
    """
    Compute per-series metrics and return a summary DataFrame.

    Args:
        actuals   : long-format test set with target values
        forecasts : long-format predictions with pred_col
        train     : training set (for MASE denominator)

    Returns:
        DataFrame with one row per (model, unique_id) pair plus an 'All' aggregate.
    """
    merged = actuals[[id_col, date_col, target_col]].merge(
        forecasts[[id_col, date_col, pred_col,
                   *(c for c in [lo_col, hi_col] if c and c in forecasts.columns)]],
        on=[id_col, date_col],
        how="inner",
    )

    rows = []
    for uid, grp in merged.groupby(id_col):
        y_t = grp[target_col].values
        y_p = grp[pred_col].values
        y_train_series = train[train[id_col] == uid][target_col].values

        row = {
            id_col: uid,
            "rmse" : rmse(y_t, y_p),
            "mae"  : mae(y_t, y_p),
            "rmsle": rmsle(y_t, y_p),
            "smape": smape(y_t, y_p),
            "mase" : mase(y_t, y_p, y_train_series),
        }
        if lo_col and hi_col and lo_col in grp.columns and hi_col in grp.columns:
            row["coverage_90"] = coverage(y_t, grp[lo_col].values, grp[hi_col].values)
        rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Aggregate row
    agg = df_metrics.drop(columns=[id_col]).mean(numeric_only=True)
    agg[id_col] = "ALL (mean)"
    df_metrics = pd.concat(
        [df_metrics, pd.DataFrame([agg])], ignore_index=True
    )

    return df_metrics.round(4)


def print_metrics_table(df_metrics: pd.DataFrame, model_name: str = "") -> None:
    header = f"{'─'*55}\n  Metrics: {model_name}\n{'─'*55}"
    print(header)
    agg = df_metrics[df_metrics["unique_id"] == "ALL (mean)"]
    for col in ["rmse", "mae", "rmsle", "smape", "mase"]:
        if col in agg.columns:
            print(f"  {col.upper():10s}: {agg[col].values[0]:.4f}")
    if "coverage_90" in agg.columns:
        print(f"  {'COVERAGE90':10s}: {agg['coverage_90'].values[0]:.1%}")
    print()
