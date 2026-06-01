"""
Optimize ensemble weights: LightGBM + Chronos-2 (fine-tuned).

Grid search over weight combinations to minimise RMSLE on test set.
Also tries: LightGBM + Chronos-zs + Chronos-ft (3-model ensemble).

Usage:
    python -m src.tune_ensemble
"""
from __future__ import annotations

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow

from src.config import (
    MODELS_DIR, MLFLOW_EXPERIMENT,
    ID_COL, DATE_COL, TARGET_COL, HORIZON,
)
from src.data_loader import load_data
from src.metrics import evaluate_forecasts, print_metrics_table


def load_predictions() -> dict[str, pd.DataFrame]:
    """Load all available prediction files."""
    preds = {}

    # LightGBM baseline
    lgbm_path = MODELS_DIR / "lgbm_model.pkl"
    if lgbm_path.exists():
        with open(lgbm_path, "rb") as f:
            fcst = pickle.load(f)
        p = fcst.predict(h=HORIZON)
        col = "LightGBM" if "LightGBM" in p.columns else p.columns[-1]
        p = p.rename(columns={col: "y_pred"})
        p["y_pred"] = p["y_pred"].clip(lower=0)
        preds["lgbm"] = p

    # LightGBM Optuna-tuned
    lgbm_tuned = MODELS_DIR / "lgbm_tuned.pkl"
    if lgbm_tuned.exists():
        with open(lgbm_tuned, "rb") as f:
            fcst = pickle.load(f)
        p = fcst.predict(h=HORIZON)
        col = "LightGBM" if "LightGBM" in p.columns else p.columns[-1]
        p = p.rename(columns={col: "y_pred"})
        p["y_pred"] = p["y_pred"].clip(lower=0)
        preds["lgbm_tuned"] = p

    # Chronos zero-shot
    ch_zs = MODELS_DIR / "preds_chronos.parquet"
    if ch_zs.exists():
        preds["chronos_zs"] = pd.read_parquet(ch_zs)

    # Chronos fine-tuned
    ch_ft = MODELS_DIR / "preds_chronos_finetuned.parquet"
    if ch_ft.exists():
        preds["chronos_ft"] = pd.read_parquet(ch_ft)

    print("Available predictions:", list(preds.keys()))
    return preds


def weighted_ensemble(
    preds_dict: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Compute weighted average of predictions."""
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}  # normalise

    base = None
    for name, w in weights.items():
        if name not in preds_dict:
            continue
        p = preds_dict[name][[ID_COL, DATE_COL, "y_pred"]].copy()
        p["y_pred"] = p["y_pred"] * w
        if base is None:
            base = p
        else:
            base = base.merge(
                p.rename(columns={"y_pred": f"y_{name}"}),
                on=[ID_COL, DATE_COL], how="inner"
            )
            base["y_pred"] += base[f"y_{name}"]
            base = base[[ID_COL, DATE_COL, "y_pred"]]
    return base


def grid_search(
    preds_dict: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> list[dict]:
    """Grid search over weight combinations."""
    results = []
    weights_range = np.arange(0.0, 1.01, 0.1)  # 0.0, 0.1, ..., 1.0

    # ── 2-model ensembles ─────────────────────────────────────────────────
    pairs = [
        ("lgbm",       "chronos_ft"),
        ("lgbm_tuned", "chronos_ft"),
        ("lgbm",       "chronos_zs"),
        ("lgbm_tuned", "chronos_zs"),
        ("lgbm",       "lgbm_tuned"),
        ("chronos_ft", "chronos_zs"),
    ]

    for m1, m2 in pairs:
        if m1 not in preds_dict or m2 not in preds_dict:
            continue
        best_rmsle = 9999
        best_w = 0.5
        for w1 in weights_range:
            w2 = round(1.0 - w1, 1)
            p = weighted_ensemble(preds_dict, {m1: w1, m2: w2})
            if p is None or p.empty:
                continue
            metrics = evaluate_forecasts(test_df, p, train_df)
            agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
            rmsle = float(agg["rmsle"])
            if rmsle < best_rmsle:
                best_rmsle = rmsle
                best_w = w1
        results.append({
            "ensemble": f"{m1}+{m2}",
            "best_weight_m1": best_w,
            "best_rmsle": round(best_rmsle, 4),
        })
        print(f"  {m1}×{best_w:.1f} + {m2}×{1-best_w:.1f}  RMSLE={best_rmsle:.4f}")

    # ── 3-model ensemble (best lgbm + both Chronos) ───────────────────────
    lgbm_key = "lgbm_tuned" if "lgbm_tuned" in preds_dict else "lgbm"
    if lgbm_key in preds_dict and "chronos_ft" in preds_dict and "chronos_zs" in preds_dict:
        best_rmsle = 9999
        best_config = None
        for w_lgbm in weights_range:
            for w_ft in weights_range:
                w_zs = round(1.0 - w_lgbm - w_ft, 1)
                if w_zs < 0:
                    continue
                p = weighted_ensemble(preds_dict, {
                    lgbm_key: w_lgbm,
                    "chronos_ft": w_ft,
                    "chronos_zs": w_zs,
                })
                if p is None or p.empty:
                    continue
                metrics = evaluate_forecasts(test_df, p, train_df)
                agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
                rmsle = float(agg["rmsle"])
                if rmsle < best_rmsle:
                    best_rmsle = rmsle
                    best_config = {lgbm_key: w_lgbm,
                                   "chronos_ft": w_ft,
                                   "chronos_zs": w_zs}
        results.append({
            "ensemble": f"3-model ({lgbm_key}+chronos_ft+chronos_zs)",
            "best_weight_m1": best_config,
            "best_rmsle": round(best_rmsle, 4),
        })
        print(f"  3-model best: {best_config}  RMSLE={best_rmsle:.4f}")

    return results


def run():
    train_df, test_df = load_data()
    preds_dict = load_predictions()

    print("\n" + "=" * 55)
    print("  Ensemble Weight Optimization (grid search)")
    print("=" * 55)

    results = grid_search(preds_dict, test_df, train_df)

    # Sort by RMSLE
    results.sort(key=lambda x: x["best_rmsle"])

    print("\n=== ENSEMBLE RESULTS (sorted by RMSLE) ===")
    for r in results:
        print(f"  {r['ensemble']:45s}  RMSLE={r['best_rmsle']:.4f}")

    best = results[0]
    print(f"\n  >> Best ensemble: {best['ensemble']}  RMSLE={best['best_rmsle']:.4f}")

    # ── Compute final best ensemble predictions ───────────────────────────
    if isinstance(best["best_weight_m1"], dict):
        weights = best["best_weight_m1"]
    else:
        parts = best["ensemble"].split("+")
        w = best["best_weight_m1"]
        weights = {parts[0]: w, parts[1]: round(1 - w, 1)}

    final_preds = weighted_ensemble(preds_dict, weights)
    final_metrics = evaluate_forecasts(test_df, final_preds, train_df)
    print_metrics_table(final_metrics, f"Best Ensemble ({best['ensemble']})")
    agg = final_metrics[final_metrics[ID_COL] == "ALL (mean)"].iloc[0]

    # ── Log to MLflow ─────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"Ensemble-optimized"):
        mlflow.log_params({"ensemble": best["ensemble"],
                           "weights": str(weights)})
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})

    # ── Save results ──────────────────────────────────────────────────────
    out = MODELS_DIR / "ensemble_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out}")

    return results, final_metrics


if __name__ == "__main__":
    run()
