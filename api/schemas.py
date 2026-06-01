"""Pydantic request/response schemas for the FastAPI forecast endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class ForecastRequest(BaseModel):
    """Request body for /forecast endpoint."""

    series_id : str = Field(..., description="Unique series identifier, e.g. 'FOODS_1_001_CA_1'")
    horizon   : int = Field(default=28, ge=1, le=90,
                             description="Forecast horizon in days (1–90)")
    model     : Literal["lightgbm", "chronos", "ensemble"] = Field(
        default="ensemble",
        description="Which model to use for forecasting",
    )

    model_config = {"json_schema_extra": {"example": {
        "series_id": "FOODS_1_001_CA_1",
        "horizon"  : 28,
        "model"    : "ensemble",
    }}}


class DayForecast(BaseModel):
    date    : str   = Field(..., description="Forecast date (YYYY-MM-DD)")
    y_pred  : float = Field(..., description="Point forecast (units sold)")
    lo_90   : float | None = Field(None, description="Lower 90% prediction interval")
    hi_90   : float | None = Field(None, description="Upper 90% prediction interval")


class ForecastResponse(BaseModel):
    series_id   : str
    model       : str
    horizon     : int
    forecasts   : list[DayForecast]
    summary     : dict = Field(description="Aggregate stats over the forecast window")

    model_config = {"json_schema_extra": {"example": {
        "series_id": "FOODS_1_001_CA_1",
        "model"    : "ensemble",
        "horizon"  : 28,
        "forecasts": [
            {"date": "2016-06-20", "y_pred": 3.5, "lo_90": 1.0, "hi_90": 6.0}
        ],
        "summary"  : {"total_units": 98.0, "avg_daily": 3.5},
    }}}


class HealthResponse(BaseModel):
    status  : Literal["ok"]
    model   : str
    version : str = "1.0"
