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
# M5 competition (Walmart sales) loaded via datasetsforecast
# 42,840 series × 1941 daily obs. We use a representative subset.
DATASET_NAME  = "m5"
# We cap the portfolio subset at N_SERIES series (100 per store × 3 stores).
# Full M5 has 30,490 series — too large for a portfolio demo.
N_SERIES      = 300     # total series cap (evenly distributed across stores)
TARGET_COL    = "y"
DATE_COL      = "ds"
ID_COL        = "unique_id"

# ── Forecast horizon ───────────────────────────────────────────────────────
HORIZON       = 28      # 28-day ahead forecast (same as M5 competition)
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
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"  # ~250MB, good balance speed/accuracy
FORECAST_CACHE     = MODELS_DIR / "forecast_cache.parquet"
MODEL_META_PATH    = MODELS_DIR / "model_meta.json"
