"""
Model comparison experiment:
  A. Baseline  : SeasonalNaive (benchmark floor)
  B. Statistical: AutoARIMA / AutoETS via statsforecast
  C. ML        : LightGBM with lag/rolling features via mlforecast
  D. Foundation : Amazon Chronos-2 (zero-shot, 2025 SOTA)
  E. Ensemble  : Weighted average of LightGBM + Chronos

All runs are logged to MLflow. Results saved to models/experiment_results.json.

Usage:
    python -m src.experiments
"""
from __future__ import annotations

import json
import mlflow
import numpy as np
import pandas as pd

from src.config import (
    MLFLOW_EXPERIMENT, MODELS_DIR, MODEL_META_PATH,
    HORIZON, FREQ, ID_COL, DATE_COL, TARGET_COL,
)
from src.data_loader import load_m5
from src.metrics import evaluate_forecasts, print_metrics_table


# ── A. Seasonal Naive baseline ────────────────────────────────────────────

def run_seasonal_naive(train_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonal Naive: forecast = value from same day last week (lag=7).
    The simplest benchmark — any real model should beat this.
    """
    print("\n[A] Seasonal Naive (baseline)")
    print("-" * 50)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    for uid, grp in train_df.groupby(ID_COL):
        grp = grp.sort_values(DATE_COL)
        last_week = grp.tail(7)[TARGET_COL].values   # last 7 days
        test_dates = test_df[test_df[ID_COL] == uid][DATE_COL].values

        for i, dt in enumerate(test_dates):
            results.append({
                ID_COL  : uid,
                DATE_COL: dt,
                "y_pred": float(last_week[i % 7]),
            })

    preds = pd.DataFrame(results)
    metrics = evaluate_forecasts(test_df, preds, train_df)
    print_metrics_table(metrics, "Seasonal Naive")

    with mlflow.start_run(run_name="SeasonalNaive"):
        agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})
        mlflow.log_param("model", "SeasonalNaive")

    return metrics, preds


# ── B. Statistical: AutoARIMA / AutoETS ──────────────────────────────────

def run_statistical(train_df: pd.DataFrame,
                    test_df: pd.DataFrame) -> pd.DataFrame:
    """
    AutoARIMA + AutoETS via Nixtla statsforecast.
    Uses the fastest/lightest configuration for portfolio purposes.
    """
    print("\n[B] Statistical (AutoARIMA + AutoETS)")
    print("-" * 50)
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS
    except ImportError:
        print("  [skip] statsforecast not installed")
        return None, None

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # statsforecast expects (unique_id, ds, y) long format
    sf = StatsForecast(
        models=[
            AutoARIMA(season_length=7, approximation=True),  # approximation=True = faster
        ],
        freq=FREQ,
        n_jobs=1,   # n_jobs=-1 causes paging-file OOM on Windows with many workers
    )

    sf.fit(train_df[[ID_COL, DATE_COL, TARGET_COL]])
    preds_wide = sf.predict(h=HORIZON, level=[80, 90])

    # statsforecast returns wide format — reshape to long
    # Choose AutoARIMA as the primary model
    best_col = "AutoARIMA"
    if best_col not in preds_wide.columns:
        best_col = [c for c in preds_wide.columns
                    if c not in [ID_COL, DATE_COL]][0]

    preds = preds_wide[[ID_COL, DATE_COL, best_col]].rename(
        columns={best_col: "y_pred"}
    )
    # Add prediction intervals if available
    lo_col = f"{best_col}-lo-90"
    hi_col = f"{best_col}-hi-90"
    if lo_col in preds_wide.columns:
        preds["lo-90"] = preds_wide[lo_col].values
        preds["hi-90"] = preds_wide[hi_col].values

    metrics = evaluate_forecasts(test_df, preds, train_df)
    print_metrics_table(metrics, "AutoARIMA")

    with mlflow.start_run(run_name="AutoARIMA"):
        agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})
        mlflow.log_param("model", "AutoARIMA")
        mlflow.log_param("season_length", 7)

    return metrics, preds


# ── C. LightGBM ──────────────────────────────────────────────────────────

def run_lgbm(train_df: pd.DataFrame,
             test_df: pd.DataFrame) -> tuple:
    print("\n[C] LightGBM (ML with lag features)")
    print("-" * 50)
    from src.train_lgbm import train as lgbm_train, save as lgbm_save
    fcst, metrics = lgbm_train(train_df, test_df, mlflow_run=True)
    lgbm_save(fcst)

    # Re-generate predictions for return
    preds = fcst.predict(h=HORIZON)
    preds = preds.rename(columns={"LightGBM": "y_pred"})
    return metrics, preds, fcst


# ── D. Chronos (foundation model) ────────────────────────────────────────

def run_chronos(train_df: pd.DataFrame,
                test_df: pd.DataFrame) -> tuple:
    print("\n[D] Chronos-2 (zero-shot foundation model)")
    print("-" * 50)
    try:
        from src.train_chronos import load_chronos, predict_all
    except ImportError:
        print("  [skip] chronos-forecasting not installed")
        return None, None

    pipeline = load_chronos()
    preds    = predict_all(pipeline, train_df, mlflow_run=True)
    metrics  = evaluate_forecasts(test_df, preds, train_df)
    print_metrics_table(metrics, "Chronos (zero-shot)")
    return metrics, preds


# ── E. Ensemble ───────────────────────────────────────────────────────────

def run_ensemble(preds_lgbm: pd.DataFrame,
                 preds_chronos: pd.DataFrame,
                 test_df: pd.DataFrame,
                 train_df: pd.DataFrame,
                 lgbm_weight: float = 0.6) -> tuple:
    """Simple weighted average: w*LightGBM + (1-w)*Chronos."""
    print(f"\n[E] Ensemble (LightGBM*{lgbm_weight} + Chronos*{1-lgbm_weight})")
    print("-" * 50)

    if preds_lgbm is None or preds_chronos is None:
        print("  [skip] missing component predictions")
        return None, None

    merged = preds_lgbm[[ID_COL, DATE_COL, "y_pred"]].merge(
        preds_chronos[[ID_COL, DATE_COL, "y_pred"]].rename(
            columns={"y_pred": "y_pred_ch"}),
        on=[ID_COL, DATE_COL],
    )
    merged["y_pred"] = (lgbm_weight * merged["y_pred"]
                        + (1 - lgbm_weight) * merged["y_pred_ch"])

    metrics = evaluate_forecasts(test_df, merged, train_df)
    print_metrics_table(metrics, "Ensemble")

    with mlflow.start_run(run_name=f"Ensemble_lgbm{lgbm_weight}"):
        agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})
        mlflow.log_params({"model": "Ensemble",
                           "lgbm_weight": lgbm_weight,
                           "chronos_weight": 1 - lgbm_weight})

    return metrics, merged


# ── Summary ───────────────────────────────────────────────────────────────

def compare_models(results: dict) -> pd.DataFrame:
    """Build a comparison table from a dict of {model_name: metrics_df}."""
    rows = []
    for name, m in results.items():
        if m is None:
            continue
        agg = m[m[ID_COL] == "ALL (mean)"].iloc[0]
        rows.append({
            "model" : name,
            "rmse"  : agg.get("rmse"),
            "mae"   : agg.get("mae"),
            "rmsle" : agg.get("rmsle"),
            "smape" : agg.get("smape"),
            "mase"  : agg.get("mase"),
        })
    df = pd.DataFrame(rows).set_index("model").round(4)
    return df


def run_all() -> dict:
    train_df, test_df = load_m5()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("=" * 55)
    print("  DEMAND FORECASTING - MODEL COMPARISON")
    print("=" * 55)

    results = {}
    preds   = {}

    # A. Baseline
    m, p = run_seasonal_naive(train_df, test_df)
    results["Seasonal Naive"] = m
    preds["naive"] = p

    # B. Statistical
    m, p = run_statistical(train_df, test_df)
    results["AutoARIMA"] = m
    preds["arima"] = p

    # C. LightGBM
    m, p, _ = run_lgbm(train_df, test_df)
    results["LightGBM"] = m
    preds["lgbm"] = p

    # D. Chronos
    m, p = run_chronos(train_df, test_df)
    results["Chronos (zero-shot)"] = m
    preds["chronos"] = p

    # E. Ensemble
    m, p = run_ensemble(preds.get("lgbm"), preds.get("chronos"),
                        test_df, train_df)
    results["Ensemble"] = m
    preds["ensemble"] = p

    # Summary
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    summary = compare_models(results)
    print(summary.to_string())

    # Save metadata
    best_model = summary["rmsle"].idxmin()
    best_rmsle = float(summary.loc[best_model, "rmsle"])
    print(f"\n  >> Best model: {best_model}  (RMSLE={best_rmsle:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    meta = {
        "best_model": best_model,
        "best_rmsle": best_rmsle,
        "results": summary.reset_index().to_dict(orient="records"),
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved: {MODEL_META_PATH}")

    return results, preds


if __name__ == "__main__":
    run_all()
