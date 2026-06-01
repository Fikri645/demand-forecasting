"""
Hyperparameter optimization for LightGBM forecaster using Optuna.

Strategy:
  - Objective: minimize RMSLE on a held-out validation window
  - 50 trials with TPE sampler (same approach as churn project)
  - HPO without prediction intervals (faster per trial)
  - Final model retrained with best params + prediction intervals

Usage:
    python -m src.tune_lgbm
"""
from __future__ import annotations

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from window_ops.rolling import rolling_mean, rolling_max, rolling_min

from src.config import (
    LGBM_MODEL_PATH, MLFLOW_EXPERIMENT, MODELS_DIR,
    HORIZON, FREQ, ID_COL, DATE_COL, TARGET_COL, RANDOM_SEED,
)
from src.data_loader import load_data
from src.metrics import evaluate_forecasts, print_metrics_table

OPTUNA_N_TRIALS  = 50
# Use last 2*HORIZON days as validation for HPO (faster than full eval)
HPO_VAL_SIZE     = HORIZON * 2


def build_fcst(params: dict) -> MLForecast:
    model = LGBMRegressor(
        n_estimators       = params["n_estimators"],
        learning_rate      = params["learning_rate"],
        num_leaves         = params["num_leaves"],
        min_child_samples  = params["min_child_samples"],
        subsample          = params["subsample"],
        colsample_bytree   = params["colsample_bytree"],
        reg_alpha          = params["reg_alpha"],
        reg_lambda         = params["reg_lambda"],
        random_state       = RANDOM_SEED,
        n_jobs             = -1,
        verbose            = -1,
    )
    return MLForecast(
        models       = {"LightGBM": model},
        freq         = FREQ,
        lags         = [7, 14, 21, 28, 35, 42, 56, 364],
        lag_transforms = {
            7:  [(rolling_mean, 7), (rolling_mean, 28),
                 (rolling_max, 7),  (rolling_min, 7)],
            28: [(rolling_mean, 28)],
        },
        date_features = ["dayofweek", "month", "year", "quarter", "day", "week"],
        num_threads   = 4,
    )


def objective(trial: optuna.Trial, train_df: pd.DataFrame) -> float:
    """Optuna objective: RMSLE on internal val window (no PI for speed)."""
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    # Internal val = last HPO_VAL_SIZE rows per series
    cutoff = train_df[DATE_COL].max() - pd.Timedelta(days=HPO_VAL_SIZE)
    hpo_train = train_df[train_df[DATE_COL] <= cutoff].copy()
    hpo_val   = train_df[train_df[DATE_COL] >  cutoff].copy()

    try:
        fcst = build_fcst(params)
        core = hpo_train[[ID_COL, DATE_COL, TARGET_COL]].copy()
        fcst.fit(core, id_col=ID_COL, time_col=DATE_COL,
                 target_col=TARGET_COL, static_features=[])
        preds = fcst.predict(h=HPO_VAL_SIZE)
        preds = preds.rename(columns={"LightGBM": "y_pred"})
        preds["y_pred"] = preds["y_pred"].clip(lower=0)

        metrics = evaluate_forecasts(hpo_val, preds, hpo_train)
        agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
        return float(agg["rmsle"])
    except Exception as e:
        return 9999.0


def run():
    train_df, test_df = load_data()

    print("=" * 55)
    print("  LightGBM Hyperparameter Optimization (Optuna)")
    print(f"  {OPTUNA_N_TRIALS} trials · TPE sampler")
    print("=" * 55)

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda t: objective(t, train_df),
        n_trials  = OPTUNA_N_TRIALS,
        n_jobs    = 1,   # n_jobs>1 causes multiprocessing OOM on Windows
        callbacks = [lambda s, t: print(
            f"  Trial {t.number+1:3d}/{OPTUNA_N_TRIALS}  "
            f"RMSLE={t.value:.4f}  "
            f"best={s.best_value:.4f}  "
            f"lr={t.params.get('learning_rate', 0):.4f}"
        )],
    )

    best = study.best_params
    best_rmsle = study.best_value
    print(f"\n  Best trial RMSLE (hpo val): {best_rmsle:.4f}")
    print("  Best params:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # ── Retrain final model with best params + prediction intervals ───────
    print("\n  Retraining final model with best params + PI...")
    fcst = build_fcst(best)
    core = train_df[[ID_COL, DATE_COL, TARGET_COL]].copy()
    fcst.fit(
        core,
        id_col=ID_COL, time_col=DATE_COL, target_col=TARGET_COL,
        static_features=[],
        prediction_intervals=PredictionIntervals(n_windows=3, h=HORIZON),
    )
    preds = fcst.predict(h=HORIZON, level=[80, 90])
    rename = {"LightGBM": "y_pred"}
    for lvl in [80, 90]:
        for side in ["lo", "hi"]:
            old = f"LightGBM-{side}-{lvl}"
            if old in preds.columns:
                rename[old] = f"{side}-{lvl}"
    preds = preds.rename(columns=rename)
    preds["y_pred"] = preds["y_pred"].clip(lower=0)

    metrics = evaluate_forecasts(test_df, preds, train_df)
    print_metrics_table(metrics, "LightGBM (Optuna-tuned)")
    agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]

    # ── Log to MLflow ─────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="LightGBM-Optuna"):
        mlflow.log_params({**best, "n_trials": OPTUNA_N_TRIALS})
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})

    # ── Save model ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    tuned_path = MODELS_DIR / "lgbm_tuned.pkl"
    with open(tuned_path, "wb") as f:
        pickle.dump(fcst, f)
    print(f"\n  Saved tuned model -> {tuned_path}")

    # ── Compare with baseline ─────────────────────────────────────────────
    print("\n=== LIGHTGBM COMPARISON ===")
    print(f"  Baseline   RMSLE=0.1672  MASE=0.877")
    print(f"  Optuna-ft  RMSLE={agg['rmsle']:.4f}  MASE={agg['mase']:.4f}")
    improvement = (0.1672 - float(agg['rmsle'])) / 0.1672 * 100
    print(f"  Improvement: {improvement:+.1f}%")

    # Update model_meta.json
    meta_path = MODELS_DIR / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["results"]["LightGBM-Optuna"] = {
            "rmsle": round(float(agg["rmsle"]), 4),
            "mase" : round(float(agg["mase"]),  4),
            "smape": round(float(agg.get("smape", 0)), 4),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return fcst, metrics


if __name__ == "__main__":
    run()
