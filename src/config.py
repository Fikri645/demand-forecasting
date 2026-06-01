"""
Central config — paths, constants, feature definitions.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PARQUET = DATA_PROC / "train.parquet"
TEST_PARQUET  = DATA_PROC / "test.parquet"

# ── Dataset ────────────────────────────────────────────────────────────────
# Store Sales - Time Series Forecasting (Kaggle / Corporación Favorita)
# 54 stores × 33 product families × 4.5 years daily sales.
# We cap at N_SERIES series (top by total volume) for portfolio demo speed.
DATASET_NAME  = "store_sales"
N_SERIES      = 300
TARGET_COL    = "y"
DATE_COL      = "ds"
ID_COL        = "unique_id"

# ── Forecast horizon ───────────────────────────────────────────────────────
HORIZON       = 28      # 28-day ahead forecast
FREQ          = "D"     # Daily

# ── Train / validation split ───────────────────────────────────────────────
# Keep last HORIZON days as validation set (walk-forward style)
VAL_SIZE      = HORIZON

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "demand-forecasting"

# ── Random seed ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Model save paths ───────────────────────────────────────────────────────
LGBM_MODEL_PATH    = MODELS_DIR / "lgbm_model.pkl"
LGBM_TUNED_PATH    = MODELS_DIR / "lgbm_tuned.pkl"
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"  # ~250MB, good balance speed/accuracy
FORECAST_CACHE     = MODELS_DIR / "forecast_cache.parquet"
MODEL_META_PATH    = MODELS_DIR / "model_meta.json"
