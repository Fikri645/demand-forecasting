"""Unit tests for api/schemas.py Pydantic validation."""
import pytest
from pydantic import ValidationError

from api.schemas import ForecastRequest, ForecastResponse, DayForecast, HealthResponse


class TestForecastRequest:
    def test_valid_default(self):
        r = ForecastRequest(series_id="FOODS_1_001_CA_1")
        assert r.horizon == 28
        assert r.model == "ensemble"

    def test_valid_custom(self):
        r = ForecastRequest(series_id="TEST", horizon=14, model="lightgbm")
        assert r.horizon == 14
        assert r.model == "lightgbm"

    def test_horizon_below_1_fails(self):
        with pytest.raises(ValidationError):
            ForecastRequest(series_id="TEST", horizon=0)

    def test_horizon_above_90_fails(self):
        with pytest.raises(ValidationError):
            ForecastRequest(series_id="TEST", horizon=91)

    def test_invalid_model_fails(self):
        with pytest.raises(ValidationError):
            ForecastRequest(series_id="TEST", model="xgboost")

    def test_all_valid_models_accepted(self):
        for m in ["lightgbm", "chronos", "ensemble"]:
            r = ForecastRequest(series_id="TEST", model=m)
            assert r.model == m


class TestDayForecast:
    def test_valid(self):
        d = DayForecast(date="2024-01-01", y_pred=5.0)
        assert d.y_pred == pytest.approx(5.0)
        assert d.lo_90 is None

    def test_with_intervals(self):
        d = DayForecast(date="2024-01-01", y_pred=5.0, lo_90=2.0, hi_90=8.0)
        assert d.lo_90 == pytest.approx(2.0)
        assert d.hi_90 == pytest.approx(8.0)


class TestForecastResponse:
    def test_valid(self):
        items = [DayForecast(date="2024-01-01", y_pred=3.0)]
        r = ForecastResponse(
            series_id="TEST",
            model="lightgbm",
            horizon=1,
            forecasts=items,
            summary={"total_units": 3.0},
        )
        assert len(r.forecasts) == 1


class TestHealthResponse:
    def test_valid(self):
        h = HealthResponse(status="ok", model="LightGBM")
        assert h.status == "ok"
        assert h.version == "1.0"
