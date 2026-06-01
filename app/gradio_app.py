"""
Gradio UI — Retail Demand Forecasting
Three tabs:
  1. Forecast Explorer  — select series → 28-day forecast chart + summary
  2. Model Comparison   — full results table + methodology
  3. About              — dataset, tech stack, references
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
    HORIZON, ID_COL, DATE_COL, TARGET_COL, MODELS_DIR,
)

# ── Load artifacts ─────────────────────────────────────────────────────────

_lgbm      = None
_train_df  = None
_test_df   = None
_meta      = {}

def _load():
    global _lgbm, _train_df, _test_df, _meta

    # Prefer tuned model if available
    tuned_path = MODELS_DIR / "lgbm_tuned.pkl"
    model_path = tuned_path if tuned_path.exists() else LGBM_MODEL_PATH
    if model_path.exists():
        with open(model_path, "rb") as f:
            _lgbm = pickle.load(f)

    if TRAIN_PARQUET.exists():
        _train_df = pd.read_parquet(TRAIN_PARQUET)
    if TEST_PARQUET.exists():
        _test_df = pd.read_parquet(TEST_PARQUET)
    if MODEL_META_PATH.exists():
        _meta = json.loads(MODEL_META_PATH.read_text())

_load()

SERIES_IDS = (sorted(_train_df[ID_COL].unique().tolist())
              if _train_df is not None
              else ["(no data loaded)"])

# Best model info
_results_dict = _meta.get("results", {})
BEST_MODEL    = _meta.get("best_model", "Ensemble")
BEST_RMSLE    = _meta.get("best_rmsle", "—")


# ── Forecast function ──────────────────────────────────────────────────────

def run_forecast(series_id: str, history_days: int) -> tuple:
    """Generate forecast chart and summary markdown."""
    if _lgbm is None or _train_df is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Model not loaded. Check Space logs.",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        return fig, "*Model not loaded.*"

    train_s = _train_df[_train_df[ID_COL] == series_id].tail(int(history_days))
    test_s  = (_test_df[_test_df[ID_COL] == series_id]
               if _test_df is not None else None)

    # Generate forecast
    preds = _lgbm.predict(h=HORIZON)
    if ID_COL in preds.columns:
        preds = preds[preds[ID_COL] == series_id]

    pred_col = "LightGBM" if "LightGBM" in preds.columns else preds.columns[-1]
    preds["y_pred"] = preds[pred_col].clip(lower=0)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(train_s[DATE_COL], train_s[TARGET_COL],
            color="#2c3e50", lw=1.5, label="Historical")

    # Prediction intervals
    lo = "LightGBM-lo-90" if "LightGBM-lo-90" in preds.columns else None
    hi = "LightGBM-hi-90" if "LightGBM-hi-90" in preds.columns else None
    if lo and hi:
        ax.fill_between(preds[DATE_COL], preds[lo], preds[hi],
                        alpha=0.2, color="#e74c3c", label="90% PI (conformal)")

    ax.plot(preds[DATE_COL], preds["y_pred"],
            color="#e74c3c", lw=2.5, ls="--", marker="o", markersize=3,
            label=f"LightGBM forecast ({HORIZON}d)")

    if test_s is not None and not test_s.empty:
        ax.plot(test_s[DATE_COL], test_s[TARGET_COL],
                color="#27ae60", lw=2, label="Actual (test)")

    ax.axvline(train_s[DATE_COL].max(), color="grey", ls=":", lw=1.2)
    ax.set_title(f"28-Day Demand Forecast — {series_id}",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel("Units Sold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── Summary ────────────────────────────────────────────────────────────
    vals  = preds["y_pred"].values
    peak  = preds[DATE_COL].iloc[int(np.argmax(vals))]
    summary = (
        f"## Forecast — {series_id}\n\n"
        f"| | Value |\n|---|---|\n"
        f"| Model | LightGBM (Optuna-tuned) |\n"
        f"| Horizon | {HORIZON} days |\n"
        f"| Total forecast | **{vals.sum():.0f} units** |\n"
        f"| Daily average | **{vals.mean():.1f}** units/day |\n"
        f"| Peak day | **{str(peak)[:10]}** |\n\n"
        f"Best overall model: **{BEST_MODEL}** (RMSLE = {BEST_RMSLE})"
    )
    return fig, summary


# ── Build results table ────────────────────────────────────────────────────

def _build_results_md() -> str:
    # model_meta.json results is a dict: {model_name: {rmsle, mase, smape}}
    full_results = {
        "Seasonal Naive":                   {"rmsle": 0.2145, "mase": 1.109,  "smape": 16.2},
        "AutoARIMA":                        {"rmsle": 0.2105, "mase": 1.121,  "smape": 16.3},
        "Chronos-2 (zero-shot)":            {"rmsle": 0.2040, "mase": 1.038,  "smape": 15.2},
        "LightGBM (default)":               {"rmsle": 0.1672, "mase": 0.877,  "smape": 12.8},
        "LightGBM (Optuna, 50 trials)":     {"rmsle": 0.1671, "mase": 0.880,  "smape": 12.8},
        "Chronos-2 (fine-tuned, 1000s)":    {"rmsle": 0.1690, "mase": 0.863,  "smape": 12.7},
        "Chronos-2 (extended, 3000s)":      {"rmsle": 0.1688, "mase": 0.863,  "smape": 12.7},
        "**Ensemble (LGB-Optuna + C-ft)**": {"rmsle": 0.1610, "mase": 0.835,  "smape": "—"},
    }

    rows = "| Model | RMSLE | MASE | SMAPE |\n|---|---|---|---|\n"
    for name, m in full_results.items():
        marker = " 🏆" if "Ensemble" in name else ""
        rows += f"| {name}{marker} | {m['rmsle']} | {m['mase']} | {m['smape']}% |\n"
    return rows


# ── Build UI ───────────────────────────────────────────────────────────────

demo = gr.Blocks(title="Retail Demand Forecasting", theme=gr.themes.Soft())

with demo:
    gr.Markdown(
        "# 📈 Retail Demand Forecasting\n"
        "**Store Sales (Corporación Favorita)** · LightGBM + Amazon Chronos-2 · MLflow\n\n"
        f"Best model: **{BEST_MODEL}** · RMSLE = **{BEST_RMSLE}** · "
        "[GitHub](https://github.com/Fikri645/demand-forecasting)"
    )

    with gr.Tab("🔮 Forecast Explorer"):
        with gr.Row():
            series_dd  = gr.Dropdown(SERIES_IDS, label="Select Series",
                                     value=SERIES_IDS[0] if SERIES_IDS else None)
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
        gr.Markdown(f"""
## Results — 28-Day Forecast (300 series, Store Sales / Favorita)

{_build_results_md()}

**MASE < 1.0** = beats seasonal naive. Only LightGBM, fine-tuned Chronos, and Ensemble achieve this.

## Methodology

| Step | Approach |
|---|---|
| Baseline | Seasonal Naive (repeat last week) |
| Statistical | AutoARIMA weekly seasonality (statsforecast) |
| ML | LightGBM + lag t-7..t-364 + oil price + holidays (mlforecast) |
| Foundation | **Amazon Chronos-2** zero-shot — no training needed |
| Fine-tuning | Chronos-2 fine-tuned on Store Sales (1000 steps, ~4 min GPU) |
| HPO | LightGBM Optuna 50 trials |
| Best ensemble | LightGBM-Optuna × 0.5 + Chronos-ft × 0.5 |

## Key Findings

- **Ensemble wins only when both components are strong.** Zero-shot Chronos dragged the ensemble down. Fine-tuned Chronos + LightGBM = new best.
- **LightGBM was near-optimal by default.** 50 Optuna trials only improved RMSLE by 0.0001.
- **Chronos converges fast.** 83% of the improvement happens in the first 1000 fine-tuning steps.
- **Foundation + feature engineering are complementary.** Chronos captures long-range patterns; LightGBM captures domain features (oil price, promotions).
""")

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## Dataset — Store Sales (Corporación Favorita)

Real grocery data from Ecuador's largest retail chain ([Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)):
- **54 stores**, 33 product families, daily unit sales
- **4.5 years**: 2013-01-01 to 2017-08-15
- External features: **oil price** (economic shock proxy), **holidays**, **promotions**
- Portfolio: top 300 series by volume

## Tech Stack

| Component | Technology |
|---|---|
| ML forecasting | LightGBM via `mlforecast` (Nixtla) |
| Statistical | AutoARIMA via `statsforecast` (Nixtla) |
| Foundation model | Amazon Chronos-2 (fine-tuned) |
| HPO | Optuna TPE, 50 trials |
| Experiment tracking | MLflow |
| API | FastAPI |
| UI | Gradio (this app) |

## References
- [M5 Competition (LightGBM won)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [Chronos-2 (Amazon, Oct 2025)](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [Nixtla mlforecast](https://nixtlaverse.nixtla.io/mlforecast/)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
