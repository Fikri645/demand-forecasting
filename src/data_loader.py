"""
Load and prepare Store Sales - Time Series Forecasting dataset (Kaggle).

Dataset: Corporación Favorita (Ecuador grocery chain)
  - 54 stores, 33 product families
  - Daily unit sales + oil price + holidays + promotions
  - Date range: 2013-01-01 to 2017-08-15 (~4.5 years)

We use a representative subset for the portfolio:
  - Top N_SERIES series by total sales volume (well-established series)
  - Long format: (unique_id, ds, y, [exog features...])

Fallback: if Store Sales CSV not found, automatically uses M5 via datasetsforecast.

Directory layout expected:
    data/raw/
        train.csv           (from Kaggle zip)
        test.csv
        stores.csv
        oil.csv
        holidays_events.csv
        transactions.csv
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import (
    DATA_RAW, DATA_PROC, TRAIN_PARQUET, TEST_PARQUET,
    TARGET_COL, DATE_COL, ID_COL,
    N_SERIES, HORIZON, RANDOM_SEED,
)

KAGGLE_TRAIN_CSV = DATA_RAW / "train.csv"
OIL_CSV          = DATA_RAW / "oil.csv"
HOLIDAYS_CSV     = DATA_RAW / "holidays_events.csv"
STORES_CSV       = DATA_RAW / "stores.csv"


# ── Store Sales loader ─────────────────────────────────────────────────────

def _load_store_sales() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and reshape Kaggle Store Sales dataset into long format."""
    print("Loading Store Sales (Kaggle Favorita)...")

    train_raw = pd.read_csv(KAGGLE_TRAIN_CSV, parse_dates=["date"],
                            dtype={"store_nbr": str, "family": str})

    # ── Create unique_id = "store_{store_nbr}_{family}" ───────────────────
    train_raw[ID_COL]  = ("store_" + train_raw["store_nbr"] + "_"
                          + train_raw["family"].str.replace(" ", "_"))
    train_raw[DATE_COL] = train_raw["date"]
    train_raw[TARGET_COL] = train_raw["sales"].clip(lower=0)

    print(f"  Raw: {train_raw[ID_COL].nunique():,} series, "
          f"{len(train_raw):,} rows")

    # ── Select top N_SERIES by total sales ─────────────────────────────────
    top_ids = (
        train_raw.groupby(ID_COL)[TARGET_COL]
        .sum()
        .nlargest(N_SERIES)
        .index.tolist()
    )
    df = train_raw[train_raw[ID_COL].isin(top_ids)].copy()

    # ── Merge oil prices (external regressor) ──────────────────────────────
    if OIL_CSV.exists():
        oil = pd.read_csv(OIL_CSV, parse_dates=["date"])
        oil = oil.rename(columns={"date": DATE_COL, "dcoilwtico": "oil_price"})
        oil["oil_price"] = oil["oil_price"].interpolate()  # fill weekends
        df = df.merge(oil, on=DATE_COL, how="left")

    # ── Merge holiday flags ─────────────────────────────────────────────────
    if HOLIDAYS_CSV.exists():
        hol = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])
        hol = hol.rename(columns={"date": DATE_COL})
        hol["is_holiday"] = (~hol["transferred"]).astype(int)
        hol_agg = (hol.groupby(DATE_COL)["is_holiday"]
                      .max().reset_index())
        df = df.merge(hol_agg, on=DATE_COL, how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # ── Keep only essential columns ─────────────────────────────────────────
    keep = [ID_COL, DATE_COL, TARGET_COL, "onpromotion"]
    for col in ["oil_price", "is_holiday"]:
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()

    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # ── Fill date gaps (mlforecast requires complete daily sequences) ────────
    # Some product-store combos have missing days (store closed, out of stock).
    # Fill with y=0 and forward-fill external regressors.
    full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    filled_parts = []
    exog_cols = [c for c in df.columns
                 if c not in [ID_COL, DATE_COL, TARGET_COL]]

    for uid, grp in df.groupby(ID_COL, observed=True):
        grp = grp.set_index(DATE_COL).reindex(full_range)
        grp.index.name = DATE_COL
        grp[ID_COL] = uid
        grp[TARGET_COL] = grp[TARGET_COL].fillna(0)
        for col in exog_cols:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()
        filled_parts.append(grp.reset_index())

    df = pd.concat(filled_parts, ignore_index=True)
    df = df[[ID_COL, DATE_COL, TARGET_COL] + exog_cols]

    # ── Train / test split (last HORIZON days) ─────────────────────────────
    cutoff = df[DATE_COL].max() - pd.Timedelta(days=HORIZON)
    train  = df[df[DATE_COL] <= cutoff].copy()
    test   = df[df[DATE_COL] >  cutoff].copy()

    print(f"  Subset: {df[ID_COL].nunique()} series (gaps filled with 0)")
    print(f"  Train: {train[DATE_COL].min().date()} -> {train[DATE_COL].max().date()} "
          f"({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()}  -> {test[DATE_COL].max().date()} "
          f"({len(test):,} rows)")

    return train, test


# ── M5 fallback loader ─────────────────────────────────────────────────────

def _load_m5_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback: load M5 via datasetsforecast if Store Sales CSV not found."""
    print("Store Sales CSV not found - using M5 dataset (fallback).")
    from datasetsforecast.m5 import M5

    m5_dir = str(DATA_PROC / "m5_raw")
    Y_df, X_df, _ = M5.load(directory=m5_dir)

    print(f"  Full M5: {Y_df[ID_COL].nunique():,} series")

    # Sample N_SERIES series evenly across 3 stores
    STORES  = ["CA_1", "TX_1", "WI_1"]
    per_store = N_SERIES // len(STORES)
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = []
    for store in STORES:
        ids = Y_df[Y_df[ID_COL].str.endswith(f"_{store}")][ID_COL].unique()
        n   = min(per_store, len(ids))
        sampled.extend(rng.choice(ids, size=n, replace=False).tolist())

    Y_sub = Y_df[Y_df[ID_COL].isin(sampled)].copy()
    if X_df is not None:
        X_sub = X_df[X_df[ID_COL].isin(sampled)].copy()
        df = Y_sub.merge(X_sub, on=[ID_COL, DATE_COL], how="left")
    else:
        df = Y_sub.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].fillna(0).clip(lower=0)

    cutoff = df[DATE_COL].max() - pd.Timedelta(days=HORIZON)
    train  = df[df[DATE_COL] <= cutoff].copy()
    test   = df[df[DATE_COL] >  cutoff].copy()

    print(f"  Subset: {df[ID_COL].nunique()} series  ({per_store} per store)")
    print(f"  Train: {train[DATE_COL].min().date()} -> {train[DATE_COL].max().date()} "
          f"({len(train):,} rows)")
    print(f"  Test : {test[DATE_COL].min().date()} -> {test[DATE_COL].max().date()} "
          f"({len(test):,} rows)")

    return train, test


# ── Public API ─────────────────────────────────────────────────────────────

def load_data(force_reload: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset. Uses Store Sales (Kaggle) if CSV is present,
    otherwise falls back to M5 via datasetsforecast.

    Returns (train_df, test_df) in long format:
        unique_id | ds | y | [optional exog]
    """
    if TRAIN_PARQUET.exists() and TEST_PARQUET.exists() and not force_reload:
        print(f"Loading cached data from {DATA_PROC}")
        train = pd.read_parquet(TRAIN_PARQUET)
        test  = pd.read_parquet(TEST_PARQUET)
        print(f"  {train[ID_COL].nunique()} series, "
              f"{len(train):,} train rows, {len(test):,} test rows")
        return train, test

    DATA_PROC.mkdir(parents=True, exist_ok=True)

    if KAGGLE_TRAIN_CSV.exists():
        train, test = _load_store_sales()
    else:
        train, test = _load_m5_fallback()

    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    print(f"  Saved -> {DATA_PROC}")
    return train, test


# Alias for backward compat
load_m5 = load_data


def get_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per series."""
    return (
        df.groupby(ID_COL)[TARGET_COL]
        .agg(n_obs="count",
             mean=np.mean, std=np.std,
             min=np.min, max=np.max,
             pct_zeros=lambda x: (x == 0).mean())
        .round(2)
        .reset_index()
    )


if __name__ == "__main__":
    train, test = load_data()
    print("\nSeries summary (first 5):")
    print(get_series_summary(train).head().to_string())
