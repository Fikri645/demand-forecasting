"""
LightGBM demand forecasting with recursive multi-step prediction.

Strategy: Direct multi-output — train one model per horizon step.
Actually we use the simpler "recursive" approach via mlforecast which
handles lag creation and recursive prediction automatically.

Usage:
    python -m src.train_lgbm
"""
from __future__ import annotations

import json
import pickle
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from window_ops.rolling import rolling_mean, rolling_max, rolling_min

from src.config import (
    LGBM_MODEL_PATH, MLFLOW_EXPERIMENT, MODEL_META_PATH,
    MODELS_DIR, HORIZON, FREQ, ID_COL, DATE_COL, TARGET_COL,
    RANDOM_SEED,
)
from src.data_loader import load_m5
from src.metrics import evaluate_forecasts, print_metrics_table


def build_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def build_mlforecast(model: LGBMRegressor) -> MLForecast:
    """Wrap LightGBM in mlforecast with lag + rolling features."""
    return MLForecast(
        models={"LightGBM": model},
        freq=FREQ,
        lags=[7, 14, 21, 28, 35, 42, 56, 364],
        lag_transforms={
            7:  [(rolling_mean, 7), (rolling_mean, 28),
                 (rolling_max, 7),  (rolling_min, 7)],
            28: [(rolling_mean, 28)],
        },
        date_features=["dayofweek", "month", "year", "quarter",
                       "day", "week"],
        num_threads=4,
    )


def train(train_df: pd.DataFrame, val_df: pd.DataFrame,
          mlflow_run: bool = True) -> tuple[MLForecast, pd.DataFrame]:
    """
    Fit LightGBM forecaster and evaluate on validation set.

    Returns:
        (fitted MLForecast, metrics DataFrame)
    """
    print("\n[LightGBM] Training...")
    model = build_lgbm()
    fcst  = build_mlforecast(model)

    if mlflow_run:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with (mlflow.start_run(run_name="LightGBM") if mlflow_run
          else __import__("contextlib").nullcontext()):

        # Keep only the columns mlforecast needs — extra exog columns (event_name_1,
        # snap_CA, etc.) are handled via lag/date features, not passed directly.
        train_core = train_df[[ID_COL, DATE_COL, TARGET_COL]].copy()

        # PredictionIntervals must be passed to fit(), then level= to predict()
        fcst.fit(
            train_core,
            id_col=ID_COL, time_col=DATE_COL, target_col=TARGET_COL,
            static_features=[],
            prediction_intervals=PredictionIntervals(n_windows=3, h=HORIZON),
        )

        preds = fcst.predict(h=HORIZON, level=[80, 90])

        # Rename prediction columns to standard names
        rename = {"LightGBM": "y_pred"}
        for lvl in [80, 90]:
            for side in ["lo", "hi"]:
                old = f"LightGBM-{side}-{lvl}"
                new = f"{side}-{lvl}"
                if old in preds.columns:
                    rename[old] = new
        preds = preds.rename(columns=rename)

        metrics_df = evaluate_forecasts(val_df, preds, train_df)
        print_metrics_table(metrics_df, "LightGBM")

        if mlflow_run:
            agg = metrics_df[metrics_df[ID_COL] == "ALL (mean)"].iloc[0]
            mlflow.log_metrics({k: float(v) for k, v in agg.items()
                                 if k != ID_COL and pd.notna(v)})
            mlflow.log_params({
                "model": "LightGBM",
                "n_estimators": model.n_estimators,
                "learning_rate": model.learning_rate,
                "horizon": HORIZON,
                "n_series": train_df[ID_COL].nunique(),
            })

    return fcst, metrics_df


def save(fcst: MLForecast) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open(LGBM_MODEL_PATH, "wb") as f:
        pickle.dump(fcst, f)
    print(f"  Saved LightGBM -> {LGBM_MODEL_PATH}")


def load() -> MLForecast:
    with open(LGBM_MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train_df, test_df = load_m5()
    fcst, metrics = train(train_df, test_df)
    save(fcst)
