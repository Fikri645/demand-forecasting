"""
Amazon Chronos-2 zero-shot forecasting.

Chronos is a family of pre-trained probabilistic time series models
(Amazon, Oct 2025). It requires NO training data — just load and predict.

We use chronos-t5-small (250M params) for the portfolio:
  - Fast enough to run on CPU
  - Better than chronos-t5-mini, close to chronos-t5-base
  - Open weights on HuggingFace: amazon/chronos-t5-small

For fine-tuning (Tier 3), see the chronos GitHub:
  https://github.com/amazon-science/chronos-forecasting

Usage:
    python -m src.train_chronos
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import torch

from src.config import (
    CHRONOS_MODEL_NAME, MLFLOW_EXPERIMENT,
    HORIZON, ID_COL, DATE_COL, TARGET_COL, RANDOM_SEED,
)
from src.data_loader import load_m5
from src.metrics import evaluate_forecasts, print_metrics_table

# Prediction interval levels (quantiles)
# Chronos returns full predictive distribution → we extract quantiles
QUANTILE_LEVELS = [0.1, 0.5, 0.9]   # P10, median, P90


def load_chronos(model_name: str = CHRONOS_MODEL_NAME):
    """Load Chronos pipeline from HuggingFace."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos not installed. Run: pip install chronos-forecasting"
        )

    print(f"  Loading {model_name} (may take a minute on first run)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Loaded on {device.upper()}")
    return pipeline


def forecast_series(
    pipeline,
    context: torch.Tensor,
    horizon: int = HORIZON,
    num_samples: int = 100,
) -> dict:
    """
    Run Chronos on a single context window.

    Returns:
        dict with 'median', 'lo_90', 'hi_90' arrays of length `horizon`.
    """
    # Chronos API: context is positional in newer versions
    forecast_samples = pipeline.predict(
        context,
        prediction_length=horizon,
        num_samples=num_samples,
    )  # shape: (1, num_samples, horizon)

    samples = forecast_samples[0].numpy()  # (num_samples, horizon)
    samples = np.clip(samples, 0, None)    # sales can't be negative

    return {
        "median": np.quantile(samples, 0.50, axis=0),
        "lo-80" : np.quantile(samples, 0.10, axis=0),
        "hi-80" : np.quantile(samples, 0.90, axis=0),
        "lo-90" : np.quantile(samples, 0.05, axis=0),
        "hi-90" : np.quantile(samples, 0.95, axis=0),
    }


def predict_all(
    pipeline,
    train_df: pd.DataFrame,
    horizon: int = HORIZON,
    context_length: int = 365,  # use last 1 year as context
    num_samples: int = 100,
    mlflow_run: bool = True,
) -> pd.DataFrame:
    """
    Run Chronos zero-shot forecasting for every series in train_df.

    Returns:
        Long-format DataFrame: (unique_id, ds, y_pred, lo-90, hi-90)
    """
    print(f"\n[Chronos] Zero-shot forecasting ({train_df[ID_COL].nunique()} series)...")

    if mlflow_run:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    series_ids = train_df[ID_COL].unique()

    ctx_mgr = (mlflow.start_run(run_name="Chronos-zero-shot")
               if mlflow_run else __import__("contextlib").nullcontext())

    with ctx_mgr:
        for i, uid in enumerate(series_ids):
            s = train_df[train_df[ID_COL] == uid].sort_values(DATE_COL)
            last_date = s[DATE_COL].max()

            # Build context tensor (last context_length observations)
            values = s[TARGET_COL].values[-context_length:]
            context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

            # Forecast
            fc = forecast_series(pipeline, context, horizon, num_samples)

            # Build date index for forecast
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon, freq="D"
            )

            for t, dt in enumerate(future_dates):
                results.append({
                    ID_COL  : uid,
                    DATE_COL: dt,
                    "y_pred": float(fc["median"][t]),
                    "lo-80" : float(fc["lo-80"][t]),
                    "hi-80" : float(fc["hi-80"][t]),
                    "lo-90" : float(fc["lo-90"][t]),
                    "hi-90" : float(fc["hi-90"][t]),
                })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(series_ids)} series done")

        preds_df = pd.DataFrame(results)

        if mlflow_run:
            mlflow.log_params({
                "model"         : "Chronos",
                "model_name"    : CHRONOS_MODEL_NAME,
                "zero_shot"     : True,
                "context_length": context_length,
                "num_samples"   : num_samples,
                "horizon"       : horizon,
            })

    return preds_df


if __name__ == "__main__":
    train_df, test_df = load_m5()
    pipeline = load_chronos()
    preds = predict_all(pipeline, train_df)

    metrics = evaluate_forecasts(test_df, preds, train_df)
    print_metrics_table(metrics, "Chronos (zero-shot)")
