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
N_STORES      = 3       # CA_1, TX_1, WI_1  — manageable for portfolio
N_CATEGORIES  = 3       # FOODS, HOBBIES, HOUSEHOLD
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
