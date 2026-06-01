"""Unit tests for src/features.py — no data download needed."""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_price_features,
    add_event_features,
    build_features,
)
from src.config import ID_COL, DATE_COL, TARGET_COL, HORIZON


def _make_series(n: int = 100, uid: str = "TEST") -> pd.DataFrame:
    """Synthetic single-series DataFrame in long format."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    np.random.seed(42)
    sales = np.random.poisson(lam=10, size=n).astype(float)
    return pd.DataFrame({
        ID_COL  : uid,
        DATE_COL: dates,
        TARGET_COL: sales,
    })


class TestCalendarFeatures:
    def test_adds_dayofweek(self):
        df = _make_series()
        out = add_calendar_features(df)
        assert "dayofweek" in out.columns

    def test_dayofweek_range(self):
        df = _make_series()
        out = add_calendar_features(df)
        assert out["dayofweek"].between(0, 6).all()

    def test_month_range(self):
        df = _make_series()
        out = add_calendar_features(df)
        assert out["month"].between(1, 12).all()

    def test_is_weekend_binary(self):
        df = _make_series()
        out = add_calendar_features(df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_does_not_modify_original(self):
        df = _make_series()
        cols_before = set(df.columns)
        add_calendar_features(df)
        assert set(df.columns) == cols_before


class TestLagFeatures:
    def test_lag_7_shifts_by_7(self):
        df = _make_series(50)
        out = add_lag_features(df)
        assert "lag_7" in out.columns
        # First 7 rows should be NaN (not enough history)
        assert out["lag_7"].iloc[:7].isna().all()

    def test_lag_values_match_original(self):
        df = _make_series(50)
        out = add_lag_features(df)
        # At row 7, lag_7 should equal row 0's TARGET_COL
        assert out["lag_7"].iloc[7] == df[TARGET_COL].iloc[0]

    def test_all_lag_columns_created(self):
        from src.features import LAG_DAYS
        df = _make_series(400)  # enough history for lag_364
        out = add_lag_features(df)
        for lag in LAG_DAYS:
            assert f"lag_{lag}" in out.columns


class TestRollingFeatures:
    def test_rolling_cols_created(self):
        df = _make_series(100)
        out = add_rolling_features(df)
        assert "rolling_mean_7" in out.columns
        assert "rolling_mean_28" in out.columns

    def test_no_nan_after_warmup(self):
        df = _make_series(100)
        out = add_rolling_features(df)
        # After HORIZON rows of warmup, rolling_mean_7 should not be NaN
        warmup = HORIZON + 7
        assert out["rolling_mean_7"].iloc[warmup:].notna().all()


class TestPriceFeatures:
    def test_no_op_without_price_column(self):
        df = _make_series()
        out = add_price_features(df)
        assert "price_norm" not in out.columns

    def test_adds_price_norm_when_price_present(self):
        df = _make_series()
        df["sell_price"] = 2.99
        out = add_price_features(df)
        assert "price_norm" in out.columns

    def test_price_norm_equals_1_when_constant(self):
        df = _make_series(30)
        df["sell_price"] = 5.0  # constant price
        out = add_price_features(df)
        # Normalised price = price / mean(price) = 5/5 = 1
        assert (out["price_norm"] - 1.0).abs().max() < 1e-6


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _make_series(400)
        out = build_features(df)
        assert isinstance(out, pd.DataFrame)

    def test_drops_nan_lag_rows(self):
        df = _make_series(400)
        out = build_features(df)
        # After build_features, no NaN in lag_7
        assert out["lag_7"].notna().all()

    def test_feature_count_increases(self):
        df = _make_series(400)
        out = build_features(df)
        assert out.shape[1] > df.shape[1]

    def test_target_col_preserved(self):
        df = _make_series(400)
        out = build_features(df)
        assert TARGET_COL in out.columns
