"""Run both pipelines end-to-end and print a results table.

Usage:
    python benchmarks/run.py

Writes JSON to ``out/benchmark_results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data.loader import load_telco, split_telco
from churn.evaluation import evaluate
from churn.features import FeatureBuilder
from churn.models.baseline_lr import LRBaseline
from churn.models.xgb import XGBChurnClassifier
from mrr_forecast.evaluation import forecast_metrics
from mrr_forecast.forecaster import MRRForecaster
from mrr_forecast.generator import generate_mrr_series


def main() -> int:
    # --- Churn classification ---
    print("=" * 60)
    print("Churn classification on Telco Customer Churn")
    print("=" * 60)
    df = load_telco()
    split = split_telco(df)
    fb = FeatureBuilder().fit(split.X_train)
    Xtr, Xte = fb.transform(split.X_train), fb.transform(split.X_test)

    results = {}
    for name, clf in [("LR", LRBaseline()), ("XGBoost", XGBChurnClassifier())]:
        clf.fit(Xtr, split.y_train)
        m = evaluate(split.y_test, clf.predict_proba(Xte))
        results[name] = {
            "roc_auc": round(m.roc_auc, 3),
            "pr_auc": round(m.pr_auc, 3),
            "top_decile_precision": round(m.top_decile_precision, 3),
            "top_quintile_precision": round(m.top_quintile_precision, 3),
            "brier": round(m.brier, 3),
        }

    # --- MRR forecast ---
    print("\n" + "=" * 60)
    print("MRR forecast on synthetic 36-month series")
    print("=" * 60)
    s = generate_mrr_series(n_months=36)
    train, test = s.iloc[:-6], s.iloc[-6:]
    fc = MRRForecaster().fit(train).forecast(h=6)
    fm = forecast_metrics(test, fc.forecast)
    results["mrr_forecast_6m"] = {
        "mae": round(fm.mae, 1),
        "mape_pct": round(fm.mape, 2),
        "rmse": round(fm.rmse, 1),
    }

    # --- Output ---
    print("\nResults:")
    print(json.dumps(results, indent=2))

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir / 'benchmark_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
