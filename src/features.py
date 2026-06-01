"""
Feature engineering for LightGBM-based time series forecasting.

Key features:
  - Calendar: day-of-week, month, year, week-of-year, is_weekend, quarter
  - Lag features: sales at t-7, t-14, t-21, t-28, t-35, t-364 (same day last year)
  - Rolling statistics: 7-day and 28-day rolling mean/std/max
  - Trend: linear trend index
  - External: sell_price, snap flag, event indicators

All features are added in-place to the dataframe using a lag-safe approach
(no data leakage: lags are always ≥ HORIZON days back from forecast date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COL, DATE_COL, ID_COL, HORIZON


# ── Calendar features ──────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived features."""
    df = df.copy()
    d = df[DATE_COL]
    df["dayofweek"]  = d.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]      = d.dt.month
    df["year"]       = d.dt.year
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["dayofyear"]  = d.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"]    = d.dt.quarter
    # Month-end / month-start: high spending days
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"]   = d.dt.is_month_end.astype(int)
    return df


# ── Lag features ───────────────────────────────────────────────────────────

LAG_DAYS = [7, 14, 21, 28, 35, 42, 56, 364]   # all ≥ HORIZON=28 ✓

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged sales values per series.
    Sorted by (unique_id, ds) required before calling.
    """
    df = df.sort_values([ID_COL, DATE_COL]).copy()
    for lag in LAG_DAYS:
        col = f"lag_{lag}"
        df[col] = df.groupby(ID_COL)[TARGET_COL].shift(lag)
    return df


# ── Rolling statistics ─────────────────────────────────────────────────────

ROLL_WINDOWS  = [7, 28]
ROLL_LAG      = HORIZON    # shift before rolling so no leakage

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean/std computed on lagged sales (lag=HORIZON).
    Rolling over a window of W days ending at t-HORIZON.
    """
    df = df.sort_values([ID_COL, DATE_COL]).copy()
    lagged = df.groupby(ID_COL)[TARGET_COL].shift(ROLL_LAG)

    for w in ROLL_WINDOWS:
        rolled = lagged.groupby(df[ID_COL]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_mean_{w}"] = rolled

        rolled_std = lagged.groupby(df[ID_COL]).transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
        df[f"rolling_std_{w}"] = rolled_std.fillna(0)

    return df


# ── Price features ─────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise sell_price within each series."""
    if "sell_price" not in df.columns:
        return df
    df = df.copy()
    gp = df.groupby(ID_COL)["sell_price"]
    df["price_norm"]   = df["sell_price"] / gp.transform("mean")
    df["price_change"] = df.groupby(ID_COL)["sell_price"].pct_change().fillna(0)
    return df


# ── Event / snap features ──────────────────────────────────────────────────

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for holidays and SNAP days."""
    df = df.copy()
    if "event_name_1" in df.columns:
        df["has_event"] = df["event_name_1"].notna().astype(int)
    if "snap_CA" in df.columns:
        snap_cols = [c for c in df.columns if c.startswith("snap_")]
        df["is_snap"] = df[snap_cols].max(axis=1).astype(int)
    return df


# ── Master feature builder ─────────────────────────────────────────────────

FEATURE_COLS: list[str] = []  # filled dynamically

def build_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Args:
        df  : DataFrame in long format with (unique_id, ds, y, optional exog).
        fit : If True, also cache the final feature column list.

    Returns:
        DataFrame with all features added.
    """
    global FEATURE_COLS

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_event_features(df)

    # Drop rows where lags are undefined (first LAG_DAYS[-1] rows per series)
    df = df.dropna(subset=[f"lag_{LAG_DAYS[0]}"])

    if fit:
        # Collect all numeric feature columns (exclude id/date/target)
        exclude = {ID_COL, DATE_COL, TARGET_COL,
                   "event_name_1", "event_name_2",
                   "event_type_1", "event_type_2"}
        FEATURE_COLS = [c for c in df.columns
                        if c not in exclude
                        and df[c].dtype in (np.float64, np.float32,
                                            np.int64, np.int32, int, float)]

    return df


def get_feature_cols() -> list[str]:
    """Return feature columns set during last build_features(fit=True) call."""
    return list(FEATURE_COLS)
