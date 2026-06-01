"""
Data drift monitoring for retail demand forecasting.

Detects distribution shifts in:
  - Sales volume (target variable)
  - Oil price (key external regressor)
  - Promotion rate (feature drift)

Compares a recent window (last 30 days) against historical baseline.
Generates an HTML report saved to reports/drift_report.html.

Usage:
    python -m monitoring.drift_report
    python -m monitoring.drift_report --window 60   # use 60-day recent window
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DATA_PROC, REPORTS_DIR, TRAIN_PARQUET, TEST_PARQUET,
    TARGET_COL, DATE_COL, ID_COL,
)


def load_time_series() -> pd.DataFrame:
    """Load and aggregate to daily level for drift analysis."""
    dfs = []
    for path in [TRAIN_PARQUET, TEST_PARQUET]:
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError("No processed data found. Run scripts/download_data.py first.")
    df = pd.concat(dfs, ignore_index=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time series to daily feature matrix for drift detection."""
    agg = df.groupby(DATE_COL).agg(
        total_sales    = (TARGET_COL, "sum"),
        mean_sales     = (TARGET_COL, "mean"),
        pct_zeros      = (TARGET_COL, lambda x: (x == 0).mean()),
        n_series_active= (TARGET_COL, lambda x: (x > 0).sum()),
    )

    # Add oil price if present
    if "oil_price" in df.columns:
        agg["oil_price"] = df.groupby(DATE_COL)["oil_price"].mean()

    # Add promotion rate if present
    if "onpromotion" in df.columns:
        agg["promo_rate"] = df.groupby(DATE_COL)["onpromotion"].mean()

    agg["day_of_week"] = pd.to_datetime(agg.index).dayofweek
    agg["month"]       = pd.to_datetime(agg.index).month
    return agg.reset_index()


def compute_drift_stats(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute basic drift statistics (mean shift, PSI-like score).
    PSI > 0.2 = significant drift. 0.1-0.2 = moderate. <0.1 = stable.
    """
    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref = reference[feat].dropna()
        cur = current[feat].dropna()

        # Population Stability Index (simplified)
        n_bins = 10
        bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= 1e-6
        bins[-1] += 1e-6

        ref_counts = np.histogram(ref, bins=bins)[0]
        cur_counts = np.histogram(cur, bins=bins)[0]

        # Avoid zero
        ref_pct = np.maximum(ref_counts / len(ref), 1e-6)
        cur_pct = np.maximum(cur_counts / len(cur), 1e-6)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        rows.append({
            "feature"      : feat,
            "ref_mean"     : round(float(ref.mean()), 4),
            "cur_mean"     : round(float(cur.mean()), 4),
            "mean_shift_%" : round((float(cur.mean()) - float(ref.mean()))
                                   / (abs(float(ref.mean())) + 1e-9) * 100, 1),
            "ref_std"      : round(float(ref.std()), 4),
            "cur_std"      : round(float(cur.std()), 4),
            "psi"          : round(float(psi), 4),
            "drift_status" : ("DRIFT" if psi > 0.2
                              else "MODERATE" if psi > 0.1
                              else "STABLE"),
        })

    return pd.DataFrame(rows)


def generate_html_report(
    drift_df: pd.DataFrame,
    reference_period: str,
    current_period: str,
    output_path: Path,
) -> None:
    """Generate a minimal HTML drift report."""
    rows_html = ""
    for _, row in drift_df.iterrows():
        color = {
            "DRIFT"   : "#ffd7d7",
            "MODERATE": "#fff4cc",
            "STABLE"  : "#d4edda",
        }.get(row["drift_status"], "white")
        rows_html += (
            f"<tr style='background:{color}'>"
            f"<td><b>{row['feature']}</b></td>"
            f"<td>{row['ref_mean']}</td><td>{row['cur_mean']}</td>"
            f"<td>{row['mean_shift_%']:+.1f}%</td>"
            f"<td>{row['psi']:.4f}</td>"
            f"<td><b>{row['drift_status']}</b></td>"
            f"</tr>"
        )

    n_drift = (drift_df.drift_status == "DRIFT").sum()
    overall = "DRIFT DETECTED" if n_drift > 0 else "STABLE"
    overall_color = "#dc3545" if n_drift > 0 else "#28a745"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Demand Forecasting — Drift Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; }}
  .badge {{ display: inline-block; padding: 6px 14px; border-radius: 4px; color: white;
            background: {overall_color}; font-size: 16px; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #dee2e6; }}
  .legend {{ margin-top: 20px; font-size: 13px; color: #666; }}
</style></head><body>
<h1>Demand Forecasting — Data Drift Report</h1>
<p><b>Reference period:</b> {reference_period} &nbsp;|&nbsp;
   <b>Current period:</b> {current_period}</p>
<p>Overall status: <span class="badge">{overall}</span>
   &nbsp; ({n_drift}/{len(drift_df)} features drifted)</p>
<table>
<tr><th>Feature</th><th>Ref Mean</th><th>Curr Mean</th>
    <th>Mean Shift</th><th>PSI</th><th>Status</th></tr>
{rows_html}
</table>
<div class="legend">
  PSI interpretation: &lt;0.1 = STABLE &nbsp;|&nbsp; 0.1–0.2 = MODERATE &nbsp;|&nbsp; &gt;0.2 = DRIFT<br>
  Key indicator: <b>oil_price</b> drift precedes demand drops by 2–4 weeks (leading indicator).
</div>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Saved: {output_path}")


def run(recent_window: int = 30) -> pd.DataFrame:
    """
    Main drift detection pipeline.

    Args:
        recent_window: days to use as 'current' window (default 30)
    """
    print("=" * 55)
    print("  Demand Forecasting — Drift Report")
    print("=" * 55)

    df = load_time_series()
    daily = extract_features(df)
    daily = daily.sort_values(DATE_COL)

    cutoff     = daily[DATE_COL].max() - pd.Timedelta(days=recent_window)
    reference  = daily[daily[DATE_COL] <= cutoff]
    current    = daily[daily[DATE_COL] >  cutoff]

    ref_period = f"{reference[DATE_COL].min().date()} - {reference[DATE_COL].max().date()}"
    cur_period = f"{current[DATE_COL].min().date()} - {current[DATE_COL].max().date()}"

    print(f"\n  Reference: {ref_period} ({len(reference)} days)")
    print(f"  Current  : {cur_period} ({len(current)} days)")

    features = ["total_sales", "mean_sales", "pct_zeros",
                "oil_price", "promo_rate", "n_series_active"]
    drift_df = compute_drift_stats(reference, current, features)

    print(f"\n{'Feature':25s} {'RefMean':>10} {'CurMean':>10} {'Shift':>8} {'PSI':>8}  Status")
    print("-" * 75)
    for _, row in drift_df.iterrows():
        status_sym = "DRIFT!" if row["drift_status"] == "DRIFT" else row["drift_status"]
        print(f"  {row['feature']:23s} {row['ref_mean']:>10} {row['cur_mean']:>10} "
              f"{row['mean_shift_%']:>7.1f}% {row['psi']:>8.4f}  {status_sym}")

    output = REPORTS_DIR / "drift_report.html"
    generate_html_report(drift_df, ref_period, cur_period, output)

    n_drift = (drift_df.drift_status == "DRIFT").sum()
    if n_drift > 0:
        drifted = drift_df[drift_df.drift_status == "DRIFT"]["feature"].tolist()
        print(f"\n  WARNING: {n_drift} feature(s) drifted: {drifted}")
        print("  Consider retraining the LightGBM model.")
    else:
        print("\n  All features stable. No retraining needed.")

    return drift_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30,
                        help="Recent window size in days (default 30)")
    args = parser.parse_args()
    run(recent_window=args.window)
