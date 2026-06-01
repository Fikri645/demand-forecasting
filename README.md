---
title: Retail Demand Forecaster
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
app_file: app/gradio_app.py
pinned: false
python_version: "3.11"
---

# Retail Demand Forecasting

![CI](https://github.com/Fikri645/demand-forecasting/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fikri0o0/demand-forecasting)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end retail demand forecasting pipeline. Compares **5 approaches** from naive baseline to Amazon Chronos-2 (2025 SOTA foundation model), with probabilistic prediction intervals, MLflow tracking, and a live Gradio demo.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/demand-forecasting)**  |  **[GitHub →](https://github.com/Fikri645/demand-forecasting)**

---

## Highlights

| What | Detail |
|---|---|
| **Dataset** | Store Sales (Corporación Favorita) — 54 stores, 33 families, 4.5 years + oil price + holidays |
| **Models** | Seasonal Naive → AutoARIMA → LightGBM → Amazon Chronos-2 → Ensemble |
| **2025 SOTA** | Amazon **Chronos-2** (Oct 2025) — zero-shot, no retraining needed |
| **Prediction intervals** | 80% + 90% bands via conformal prediction |
| **Metric** | RMSLE — penalises under-forecasting (stockout > overstock in cost) |
| **Experiment tracking** | MLflow — all model runs logged |
| **API** | FastAPI `/forecast` endpoint |
| **UI** | Gradio — interactive 28-day forecast chart |
| **Deployment** | HuggingFace Spaces |

---

## Architecture

```
Store Sales CSV (Kaggle) / M5 fallback (datasetsforecast)
  └─► data_loader.py    (load, fill date gaps, train/test split)
        └─► features.py  (lag, rolling, calendar, oil price, holiday features)
              ├─► train_lgbm.py    (LightGBM via mlforecast + MLflow)
              ├─► train_chronos.py (Chronos-2 zero-shot — no training, requires GPU)
              └─► experiments.py   (5-model comparison -> model_meta.json)
                    └─► evaluate.py (forecast plots, metrics comparison)
                          ├─► api/main.py       (FastAPI /forecast)
                          └─► app/gradio_app.py (HF Spaces UI)
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Fikri645/demand-forecasting
cd demand-forecasting
pip install -r requirements-dev.txt

# 2a. (Option A) Download Store Sales from Kaggle — put zip in data/raw/ then:
python scripts/download_data.py

# 2b. (Option B) Auto-download M5 via datasetsforecast (no Kaggle needed)
#     Just run the script — it will use M5 as fallback automatically
python scripts/download_data.py

# 3. Run full experiment (5 models + MLflow logging)
python -m src.experiments

# 4. Generate evaluation plots
python -m src.evaluate

# 5. Run API locally
uvicorn api.main:app --reload

# 6. Run Gradio UI
python app/gradio_app.py
```

Or via `make`:
```bash
make install && make data && make experiments && make evaluate
```

---

## Project Structure

```
demand-forecasting/
├── data/processed/         # train.parquet, test.parquet
├── src/
│   ├── config.py           # paths, constants
│   ├── data_loader.py      # Store Sales (Favorita) loading + gap fill + M5 fallback
│   ├── features.py         # lag, rolling, calendar feature engineering
│   ├── metrics.py          # RMSE, MAE, RMSLE, MASE, coverage
│   ├── train_lgbm.py       # LightGBM via mlforecast
│   ├── train_chronos.py    # Amazon Chronos-2 (zero-shot)
│   ├── experiments.py      # 5-model comparison + MLflow
│   └── evaluate.py         # forecast + comparison plots
├── api/main.py             # FastAPI /forecast endpoint
├── app/gradio_app.py       # Gradio UI (HF Spaces)
├── notebooks/01_eda.ipynb  # Exploratory Data Analysis
├── tests/                  # pytest (metrics, features, API schemas)
├── Makefile
└── requirements-dev.txt
```

---

## Dataset — Store Sales (Corporacion Favorita)

The **Store Sales - Time Series Forecasting** competition (Kaggle) uses real data from Ecuador's largest grocery chain:
- **54 stores**, 33 product families, daily unit sales
- **4.5 years**: 2013-01-01 to 2017-08-15 (1,684 days)
- External features: **oil price** (Ecuador is oil-dependent — economic shocks affect spending), **national/regional holidays**, **promotions**
- Portfolio uses top 300 series by total volume

Source: [Kaggle Store Sales Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

> M5 (Walmart, via `datasetsforecast`) available as automatic fallback if CSV not present.

---

## Model Details

### Seasonal Naive (baseline)
Forecast = same weekday last week. Any real model must beat this.

### AutoARIMA
`statsforecast` AutoARIMA with weekly seasonality. Automatic order selection via AIC.

### LightGBM + Feature Engineering
`mlforecast` with automatic lag generation:
- **Lags**: t-7, t-14, t-21, t-28, t-35, t-42, t-56, t-364 (same day last year)
- **Rolling**: 7-day and 28-day mean, std, max per series
- **Calendar**: day-of-week, month, quarter, is-weekend, month-start/end
- **Price**: normalised sell price, price change %
- **External**: oil price, promotion flag, holiday flag (Store Sales specific)

### Amazon Chronos-2 (2025 SOTA)
Zero-shot foundation model — no training data needed. Loads pre-trained weights (`amazon/chronos-t5-small`, 250M params) from HuggingFace. Generates 100 probabilistic samples -> P10/P50/P90 quantiles.

> Chronos-2 (Oct 2025) natively supports cross-series dependencies, exogenous features, and multivariate forecasting. Zero-shot performance competitive with fully-supervised models.

**Requirements:** Chronos needs PyTorch with CUDA and sufficient virtual memory (page file >= 8GB on Windows). Run `python -m src.train_chronos` after increasing virtual memory. Code is complete and ready.

### Ensemble
Weighted average: LightGBM x 0.6 + Chronos x 0.4. Combines domain-feature awareness with temporal pattern recognition. Run `python -m src.experiments` after Chronos is available.

---

## Why RMSLE?

In retail, **running out of stock costs more than overstock**. RMSLE operates in log-space, which:
1. Penalises under-forecasting more than over-forecasting
2. Gives equal relative weight to low-volume and high-volume SKUs
3. Aligns the metric with actual business cost structure

---

## What I Learned

- **Foundation models for cold-start.** Chronos-2 gives competitive accuracy with zero training data — critical for new products with no sales history.
- **Lag features are the backbone.** `lag_7` (same weekday last week) is the most important single feature. `lag_364` captures annual seasonality.
- **Ensembles almost always win.** LightGBM captures domain features; Chronos captures long-range patterns. Neither alone beats the combination.
- **Prediction intervals matter.** A point forecast isn't enough for inventory planning — you need uncertainty bands to set safety stock.
- **RMSLE > RMSE for retail.** MSE is dominated by high-volume SKUs. Log-scale normalisation weights all products fairly.
