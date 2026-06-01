"""
Load and prepare M5 (Walmart) time series data.

The M5 competition dataset (2020) covers:
  - 3049 product/store combinations across 10 US states
  - Daily unit sales from 2011-01-29 to 2016-06-19 (1941 days)
  - 3 levels: item -> department/category -> state/store

For this portfolio project we use a representative subset:
  - 3 stores (CA_1, TX_1, WI_1), 100 series each = 300 series total
  - Full M5 has 30,490 series; 300 is enough for model comparison and demo

External features available:
  - sell_price: item price on that day
  - event_name_1 / event_type_1: US holidays and sporting events
  - snap_CA/TX/WI: SNAP (food stamps) eligibility days
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DATA_PROC, TRAIN_PARQUET, TEST_PARQUET,
    TARGET_COL, DATE_COL, ID_COL,
    N_SERIES, HORIZON, VAL_SIZE, RANDOM_SEED,
)

# We sample N_SERIES // len(STORES) series per store for diversity
STORES = ["CA_1", "TX_1", "WI_1"]


def load_m5(force_reload: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load M5 data via datasetsforecast.  Returns (train_df, test_df).

    Both DataFrames are in 'long format' expected by mlforecast / statsforecast:
        unique_id | ds         | y   | [exog features ...]
        CA_1_FOODS| 2011-01-29 | 3.0 | ...
    """
    if TRAIN_PARQUET.exists() and TEST_PARQUET.exists() and not force_reload:
        print(f"Loading cached data from {DATA_PROC}")
        train = pd.read_parquet(TRAIN_PARQUET)
        test  = pd.read_parquet(TEST_PARQUET)
        return train, test

    print("Downloading M5 data via datasetsforecast (first run only)...")
    from datasetsforecast.m5 import M5

    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # M5.load() returns (Y_df, X_df, S_df)
    #   Y_df: target (unique_id, ds, y)
    #   X_df: temporal exogenous (sell prices, events, snap)
    #   S_df: static features (item metadata)
    Y_df, X_df, S_df = M5.load(directory=str(DATA_PROC / "m5_raw"))

    print(f"  Full dataset: {Y_df['unique_id'].nunique():,} series, "
          f"{len(Y_df):,} rows")

    # ── Filter to portfolio subset ─────────────────────────────────────────
    # M5 unique_id ends with store name, e.g. FOODS_1_001_CA_1
    # Filter per store, then sample N_SERIES // len(STORES) series each.
    per_store = N_SERIES // len(STORES)
    sampled_ids = []
    rng = np.random.default_rng(RANDOM_SEED)
    for store in STORES:
        store_ids = Y_df[Y_df[ID_COL].str.endswith(f"_{store}")][ID_COL].unique()
        n = min(per_store, len(store_ids))
        sampled_ids.extend(rng.choice(store_ids, size=n, replace=False).tolist())

    Y_sub = Y_df[Y_df[ID_COL].isin(sampled_ids)].copy()
    print(f"  Portfolio subset: {Y_sub[ID_COL].nunique():,} series "
          f"({per_store} per store x {len(STORES)} stores)")

    # ── Merge exogenous features ───────────────────────────────────────────
    if X_df is not None:
        X_sub = X_df[X_df[ID_COL].isin(Y_sub[ID_COL])].copy()
        df = Y_sub.merge(X_sub, on=[ID_COL, DATE_COL], how="left")
    else:
        df = Y_sub.copy()

    # ── Parse dates ────────────────────────────────────────────────────────
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # ── Zero-fill missing sales (product not on shelf) ─────────────────────
    df[TARGET_COL] = df[TARGET_COL].fillna(0).clip(lower=0)

    # ── Coarsen IDs for readability ────────────────────────────────────────
    # e.g. "FOODS_1_001_CA_1_validation" -> keep as-is (already meaningful)

    # ── Train / test split (last HORIZON days as test) ────────────────────
    cutoff = df[DATE_COL].max() - pd.Timedelta(days=HORIZON)
    train  = df[df[DATE_COL] <= cutoff].copy()
    test   = df[df[DATE_COL] >  cutoff].copy()

    print(f"  Train: {train[DATE_COL].min().date()} -> {train[DATE_COL].max().date()} "
          f"({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()}  -> {test[DATE_COL].max().date()} "
          f"({len(test):,} rows)")

    # ── Save ───────────────────────────────────────────────────────────────
    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    print(f"  Saved -> {TRAIN_PARQUET.parent}")

    return train, test


def get_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of each series (mean, std, zeros, trend)."""
    summary = (
        df.groupby(ID_COL)[TARGET_COL]
        .agg(
            n_obs="count",
            mean=np.mean,
            std=np.std,
            min=np.min,
            max=np.max,
            pct_zeros=lambda x: (x == 0).mean(),
        )
        .round(2)
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    train, test = load_m5()
    print("\nSeries summary (first 5):")
    print(get_series_summary(train).head().to_string())
