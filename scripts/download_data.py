"""
Download / prepare dataset.

Priority:
  1. Store Sales (Kaggle) — if data/raw/train.csv already downloaded
  2. M5 via datasetsforecast — automatic fallback (no credentials needed)

For Store Sales:
  1. Go to https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules
  2. Accept the competition rules
  3. Run: kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
  4. Unzip: cd data/raw && unzip store-sales-time-series-forecasting.zip
  5. Then run this script again

Usage:
    python scripts/download_data.py [--force]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_data, KAGGLE_TRAIN_CSV

if __name__ == "__main__":
    force = "--force" in sys.argv

    if KAGGLE_TRAIN_CSV.exists():
        print("Store Sales CSV found -> using Kaggle dataset")
    else:
        print("Store Sales CSV not found -> will use M5 fallback")
        print("To use Store Sales:")
        print("  1. Accept rules: https://www.kaggle.com/competitions/"
              "store-sales-time-series-forecasting/rules")
        print("  2. kaggle competitions download -c store-sales-time-series-forecasting"
              " -p data/raw/")
        print("  3. cd data/raw && unzip store-sales-time-series-forecasting.zip")
        print()

    train, test = load_data(force_reload=force)
    print(f"\nDone!")
    print(f"  Train: {len(train):,} rows, {train['unique_id'].nunique()} series")
    print(f"  Test : {len(test):,} rows")
    print(f"  Date range: {train['ds'].min().date()} -> {test['ds'].max().date()}")
