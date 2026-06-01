"""
Download M5 dataset via datasetsforecast (no Kaggle credentials needed).

Usage:
    python scripts/download_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_m5

if __name__ == "__main__":
    print("Downloading M5 dataset...")
    train, test = load_m5(force_reload=True)
    print(f"\nDone!")
    print(f"  Train: {len(train):,} rows, {train['unique_id'].nunique()} series")
    print(f"  Test : {len(test):,} rows")
    print(f"  Date range: {train['ds'].min().date()} -> {test['ds'].max().date()}")
