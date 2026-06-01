"""Unit tests for src/metrics.py — no data or model needed."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import rmse, mae, rmsle, smape, mase, coverage, evaluate_forecasts


Y_TRUE = np.array([10.0, 5.0, 8.0, 12.0, 3.0])
Y_PRED = np.array([9.0,  6.0, 7.0, 13.0, 4.0])
Y_TRAIN = np.array([8.0, 9.0, 10.0, 8.0, 7.0, 9.0, 11.0, 10.0, 8.0, 9.0,
                    10.0, 8.0, 7.0, 9.0])


class TestRmse:
    def test_perfect_forecast(self):
        assert rmse(Y_TRUE, Y_TRUE) == pytest.approx(0.0)

    def test_known_value(self):
        y_t = np.array([0.0, 2.0])
        y_p = np.array([1.0, 1.0])
        assert rmse(y_t, y_p) == pytest.approx(1.0)

    def test_always_positive(self):
        assert rmse(Y_TRUE, Y_PRED) >= 0


class TestMae:
    def test_perfect_forecast(self):
        assert mae(Y_TRUE, Y_TRUE) == pytest.approx(0.0)

    def test_known_value(self):
        assert mae(np.array([0.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


class TestRmsle:
    def test_perfect_forecast(self):
        assert rmsle(Y_TRUE, Y_TRUE) == pytest.approx(0.0)

    def test_clips_negatives(self):
        """Negative predictions should be clipped to 0, not raise."""
        y_p = np.array([-1.0, 5.0, 8.0, 12.0, 3.0])
        assert rmsle(Y_TRUE, y_p) >= 0

    def test_under_forecast_penalised_more(self):
        """RMSLE should penalise under-forecasting (pred < actual) more than over."""
        y_t  = np.array([10.0])
        over  = rmsle(y_t, np.array([15.0]))   # over by 5
        under = rmsle(y_t, np.array([5.0]))    # under by 5
        assert under > over


class TestSmape:
    def test_perfect_forecast(self):
        assert smape(Y_TRUE, Y_TRUE) == pytest.approx(0.0)

    def test_range_0_to_200(self):
        val = smape(Y_TRUE, Y_PRED)
        assert 0 <= val <= 200


class TestMase:
    def test_perfect_forecast(self):
        assert mase(Y_TRUE, Y_TRUE, Y_TRAIN) == pytest.approx(0.0)

    def test_returns_float(self):
        result = mase(Y_TRUE, Y_PRED, Y_TRAIN)
        assert isinstance(result, float)

    def test_zero_scale_returns_nan(self):
        """If all training values are equal, scale=0 → nan."""
        constant_train = np.ones(20)
        result = mase(Y_TRUE, Y_PRED, constant_train)
        assert np.isnan(result)


class TestCoverage:
    def test_all_inside(self):
        lo = Y_TRUE - 1
        hi = Y_TRUE + 1
        assert coverage(Y_TRUE, lo, hi) == pytest.approx(1.0)

    def test_none_inside(self):
        lo = Y_TRUE + 10
        hi = Y_TRUE + 20
        assert coverage(Y_TRUE, lo, hi) == pytest.approx(0.0)

    def test_partial(self):
        lo = np.array([9.0, 4.0, 100.0, 11.0, 2.0])
        hi = np.array([11.0, 6.0, 200.0, 13.0, 4.0])
        cov = coverage(Y_TRUE, lo, hi)
        assert 0 < cov < 1


class TestEvaluateForecasts:
    def _make_dfs(self):
        ids   = ["A", "A", "B", "B"]
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"] * 2)
        actuals = pd.DataFrame({
            "unique_id": ids,
            "ds": dates,
            "y": [10.0, 8.0, 5.0, 6.0],
        })
        forecasts = pd.DataFrame({
            "unique_id": ids,
            "ds": dates,
            "y_pred": [9.0, 9.0, 4.0, 7.0],
        })
        train = pd.DataFrame({
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "ds": pd.date_range("2023-01-01", periods=10).tolist() * 2,
            "y": list(range(10)) * 2,
        })
        return actuals, forecasts, train

    def test_returns_dataframe(self):
        a, f, t = self._make_dfs()
        result = evaluate_forecasts(a, f, t)
        assert isinstance(result, pd.DataFrame)

    def test_has_aggregate_row(self):
        a, f, t = self._make_dfs()
        result = evaluate_forecasts(a, f, t)
        assert "ALL (mean)" in result["unique_id"].values

    def test_metric_columns_present(self):
        a, f, t = self._make_dfs()
        result = evaluate_forecasts(a, f, t)
        for col in ["rmse", "mae", "rmsle"]:
            assert col in result.columns
