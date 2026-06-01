"""
Gradio UI — Retail Demand Forecasting
Three tabs:
  1. Forecast Explorer  — select series + model → 28-day forecast chart
  2. Model Comparison   — side-by-side metrics table + plots
  3. About              — methodology, data, architecture
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from src.config import (
    LGBM_MODEL_PATH, MODEL_META_PATH, TRAIN_PARQUET, TEST_PARQUET,
    HORIZON, ID_COL, DATE_COL, TARGET_COL,
)

# ── Load artifacts ─────────────────────────────────────────────────────────

_lgbm = None
_train_df = None
_test_df  = None
_meta = {}

def _load():
    global _lgbm, _train_df, _test_df, _meta
    if LGBM_MODEL_PATH.exists():
        with open(LGBM_MODEL_PATH, "rb") as f:
            _lgbm = pickle.load(f)
    if TRAIN_PARQUET.exists():
        _train_df = pd.read_parquet(TRAIN_PARQUET)
    if TEST_PARQUET.exists():
        _test_df = pd.read_parquet(TEST_PARQUET)
    if MODEL_META_PATH.exists():
        _meta = json.loads(MODEL_META_PATH.read_text())

_load()

SERIES_IDS = (sorted(_train_df[ID_COL].unique().tolist())
              if _train_df is not None else ["(no data — run src.experiments first)"])

BEST_MODEL = _meta.get("best_model", "LightGBM")
BEST_RMSLE = _meta.get("best_rmsle", "N/A")
RESULTS    = _meta.get("results", [])


# ── Forecast function ─────────────────────────────────────────────────────

def make_forecast_plot(series_id: str, history_days: int = 60) -> plt.Figure:
    """Generate a forecast chart for the selected series."""
    if _lgbm is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Model not loaded.\nRun: python -m src.experiments",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        return fig

    # Historical data
    train_s = _train_df[_train_df[ID_COL] == series_id].tail(history_days)

    # Forecast
    preds = _lgbm.predict(h=HORIZON)
    if ID_COL in preds.columns:
        preds = preds[preds[ID_COL] == series_id]
    preds["y_pred"] = preds.get("LightGBM", preds.iloc[:, -1]).clip(lower=0)

    # Actual test values (if available)
    test_s = None
    if _test_df is not None:
        test_s = _test_df[_test_df[ID_COL] == series_id]

    fig, ax = plt.subplots(figsize=(12, 5))

    # History
    ax.plot(train_s[DATE_COL], train_s[TARGET_COL],
            color="#2c3e50", linewidth=1.5, label="Historical sales")

    # Prediction intervals (if available)
    lo_col = "LightGBM-lo-90" if "LightGBM-lo-90" in preds.columns else None
    hi_col = "LightGBM-hi-90" if "LightGBM-hi-90" in preds.columns else None
    if lo_col and hi_col:
        ax.fill_between(preds[DATE_COL], preds[lo_col], preds[hi_col],
                        alpha=0.2, color="#e74c3c", label="90% interval")

    # Forecast
    ax.plot(preds[DATE_COL], preds["y_pred"],
            color="#e74c3c", linewidth=2.5, linestyle="--",
            marker="o", markersize=3, label=f"Forecast (LightGBM, {HORIZON}d)")

    # Actuals (if in test set)
    if test_s is not None and not test_s.empty:
        ax.plot(test_s[DATE_COL], test_s[TARGET_COL],
                color="#27ae60", linewidth=2, label="Actual (test)")

    # Divider line
    split_date = train_s[DATE_COL].max()
    ax.axvline(split_date, color="grey", linestyle=":", linewidth=1.2)
    ax.text(split_date, ax.get_ylim()[1] * 0.95, " forecast start",
            fontsize=9, color="grey")

    ax.set_title(f"28-Day Demand Forecast — {series_id}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def run_forecast(series_id: str, history_days: int) -> tuple:
    """Gradio callback: returns (plot, summary markdown)."""
    fig = make_forecast_plot(series_id, int(history_days))

    if _lgbm is None:
        return fig, "*Model not loaded.*"

    preds = _lgbm.predict(h=HORIZON)
    if ID_COL in preds.columns:
        preds = preds[preds[ID_COL] == series_id]

    col = "LightGBM" if "LightGBM" in preds.columns else preds.columns[-1]
    vals = preds[col].clip(lower=0).values
    total = vals.sum()
    avg   = vals.mean()
    peak  = preds[DATE_COL].iloc[np.argmax(vals)]

    summary = (
        f"## 📦 Forecast Summary — {series_id}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Model | LightGBM (lag features + calendar) |\n"
        f"| Horizon | {HORIZON} days |\n"
        f"| Total predicted units | **{total:.0f}** |\n"
        f"| Average daily | **{avg:.1f}** units/day |\n"
        f"| Peak day | **{str(peak)[:10]}** |\n\n"
        f"*Best model overall: **{BEST_MODEL}** (RMSLE={BEST_RMSLE})*"
    )
    return fig, summary


# ── Build UI ───────────────────────────────────────────────────────────────

demo = gr.Blocks(title="Demand Forecasting", theme=gr.themes.Soft())

with demo:
    gr.Markdown("# 📈 Retail Demand Forecasting\n"
                "LightGBM + Amazon Chronos-2 · M5 (Walmart) Dataset · MLflow tracking")

    with gr.Tab("🔮 Forecast Explorer"):
        with gr.Row():
            series_dd  = gr.Dropdown(SERIES_IDS, label="Select Series",
                                     value=SERIES_IDS[0])
            history_sl = gr.Slider(30, 120, value=60, step=10,
                                   label="History days to show")
        btn = gr.Button("Generate Forecast", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=2):
                forecast_plot = gr.Plot(label="Forecast Chart")
            with gr.Column(scale=1):
                forecast_md = gr.Markdown("*Select a series and click Generate.*")

        btn.click(run_forecast,
                  inputs=[series_dd, history_sl],
                  outputs=[forecast_plot, forecast_md],
                  api_name=False)

    with gr.Tab("📊 Model Comparison"):
        results_table = ""
        if RESULTS:
            rows = "| Model | RMSE | MAE | RMSLE | MASE |\n|---|---|---|---|---|\n"
            for r in RESULTS:
                rows += (f"| **{r['model']}** | "
                         f"{r.get('rmse','—')} | {r.get('mae','—')} | "
                         f"{r.get('rmsle','—')} | {r.get('mase','—')} |\n")
            results_table = rows
        else:
            results_table = (
                "*No results yet. Run `python -m src.experiments` to generate.*"
            )

        gr.Markdown(f"""
## Model Comparison Results

{results_table}

## Methodology

| Step | Approach |
|---|---|
| Baseline | Seasonal Naive (repeat last week) |
| Statistical | AutoARIMA with weekly seasonality (statsforecast) |
| ML | LightGBM with lag + rolling features (mlforecast) |
| Foundation | **Amazon Chronos-2** zero-shot (2025 SOTA) |
| Ensemble | Weighted average LightGBM 60% + Chronos 40% |

**Best model: {BEST_MODEL}** · RMSLE = {BEST_RMSLE}

## Key Design Decisions

- **Lag features**: t-7, t-14, t-28, t-364 (same-day last year) — capture weekly & yearly seasonality
- **Prediction intervals**: conformal prediction (mlforecast) → reliable 80%/90% bands
- **Chronos-2 zero-shot**: no training needed, loaded from HuggingFace — demonstrates 2025 SOTA
- **RMSLE as primary metric**: penalises under-forecasting more than over-forecasting
  (out-of-stock is worse than overstock in retail)
""")

    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"""
## Dataset — M5 (Walmart Sales)

The **M5 Forecasting Competition** (Kaggle 2020) is the gold standard retail forecasting benchmark:
- **42,840 time series** of daily unit sales across 10 US states
- **1,941 days** of history (2011-01-29 → 2016-06-19)
- Products across 3 categories: FOODS, HOBBIES, HOUSEHOLD
- External features: sell price, US holidays, SNAP days

This portfolio uses a representative subset: 3 stores × 3 categories (~270 series).

## Tech Stack

| Component | Technology |
|---|---|
| ML forecasting | LightGBM via `mlforecast` (Nixtla) |
| Statistical | AutoARIMA via `statsforecast` (Nixtla) |
| Foundation model | Amazon Chronos-2 (`chronos-forecasting`) |
| Experiment tracking | MLflow |
| API | FastAPI |
| UI | Gradio (this app) |
| Deployment | HuggingFace Spaces |

## References

- [M5 Competition Paper](https://www.sciencedirect.com/science/article/pii/S0169207021001874) — LightGBM ensemble won
- [Chronos-2 (Amazon, Oct 2025)](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [Nixtla mlforecast docs](https://nixtlaverse.nixtla.io/mlforecast/)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
