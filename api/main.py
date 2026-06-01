"""
FastAPI serving layer for demand forecasting.

Endpoints:
  GET  /              — health check
  GET  /series        — list available series IDs
  POST /forecast      — single-series forecast
  POST /forecast/batch — multi-series forecast (CSV upload)

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from api.schemas import (
    ForecastRequest, ForecastResponse, DayForecast, HealthResponse,
)
from src.config import (
    LGBM_MODEL_PATH, MODEL_META_PATH, TRAIN_PARQUET,
    HORIZON, ID_COL, DATE_COL, TARGET_COL,
)

app = FastAPI(
    title="Demand Forecasting API",
    description="Retail demand forecasting — LightGBM + Amazon Chronos-2",
    version="1.0.0",
)

# ── Load artifacts on startup ─────────────────────────────────────────────

_lgbm_model = None
_train_df   = None
_meta       = {}


def _load_artifacts():
    global _lgbm_model, _train_df, _meta
    if LGBM_MODEL_PATH.exists():
        with open(LGBM_MODEL_PATH, "rb") as f:
            _lgbm_model = pickle.load(f)
    if TRAIN_PARQUET.exists():
        _train_df = pd.read_parquet(TRAIN_PARQUET)
    if MODEL_META_PATH.exists():
        _meta = json.loads(MODEL_META_PATH.read_text())


@app.on_event("startup")
async def startup_event():
    _load_artifacts()


# ── Helpers ───────────────────────────────────────────────────────────────

def _run_lgbm(series_id: str, horizon: int) -> pd.DataFrame:
    if _lgbm_model is None:
        raise HTTPException(503, "LightGBM model not loaded. Run src.train_lgbm first.")
    if _train_df is None or series_id not in _train_df[ID_COL].values:
        raise HTTPException(404, f"Series '{series_id}' not found in training data.")

    preds = _lgbm_model.predict(h=horizon)
    if ID_COL in preds.columns:
        preds = preds[preds[ID_COL] == series_id]
    return preds


def _forecast_to_response(
    series_id: str, model: str, preds: pd.DataFrame
) -> ForecastResponse:
    items = []
    for _, row in preds.iterrows():
        items.append(DayForecast(
            date   = str(row[DATE_COL])[:10],
            y_pred = round(max(0.0, float(row.get("y_pred", row.get("LightGBM", 0)))), 2),
            lo_90  = round(float(row["lo-90"]), 2) if "lo-90" in row else None,
            hi_90  = round(float(row["hi-90"]), 2) if "hi-90" in row else None,
        ))

    y_vals = [i.y_pred for i in items]
    return ForecastResponse(
        series_id = series_id,
        model     = model,
        horizon   = len(items),
        forecasts = items,
        summary   = {
            "total_units": round(sum(y_vals), 1),
            "avg_daily"  : round(sum(y_vals) / len(y_vals), 2) if y_vals else 0,
            "peak_day"   : items[int(np.argmax(y_vals))].date if y_vals else None,
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=_meta.get("best_model", "LightGBM"),
    )


@app.get("/series")
def list_series():
    """Return list of available series IDs."""
    if _train_df is None:
        raise HTTPException(503, "Training data not loaded.")
    ids = sorted(_train_df[ID_COL].unique().tolist())
    return {"series": ids, "count": len(ids)}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """Generate a demand forecast for a single series."""
    if req.model in ("lightgbm", "ensemble"):
        preds = _run_lgbm(req.series_id, req.horizon)
        return _forecast_to_response(req.series_id, req.model, preds)
    else:
        raise HTTPException(
            501,
            "Chronos endpoint not available in this deployment "
            "(requires GPU). Use model='lightgbm' instead."
        )
