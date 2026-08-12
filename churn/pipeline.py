"""End-to-end churn-classification pipeline.

Run:
    python -m churn.pipeline

Steps: load Telco → fit FeatureBuilder → train baseline + XGB → evaluate → print rankings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from churn.data.loader import load_telco, split_telco
from churn.evaluation import evaluate, format_metrics
from churn.features import FeatureBuilder
from churn.models.baseline_lr import LRBaseline
from churn.models.xgb import XGBChurnClassifier


def main() -> int:
    print("Loading Telco Customer Churn dataset ...")
    df = load_telco()
    print(f"  Loaded {len(df)} customers ({df['churn'].mean():.1%} churn rate)")
    split = split_telco(df)

    fb = FeatureBuilder().fit(split.X_train)
    Xtr = fb.transform(split.X_train)
    Xte = fb.transform(split.X_test)

    print("\nTraining LR baseline ...")
    lr = LRBaseline().fit(Xtr, split.y_train)
    lr_metrics = evaluate(split.y_test, lr.predict_proba(Xte))
    print("  " + format_metrics(lr_metrics, label="LR"))

    print("\nTraining XGBoost (with isotonic calibration) ...")
    xgb = XGBChurnClassifier().fit(Xtr, split.y_train)
    xgb_metrics = evaluate(split.y_test, xgb.predict_proba(Xte))
    print("  " + format_metrics(xgb_metrics, label="XGB"))

    # Save predictions
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"customer_idx": split.X_test.index, "churn_prob": xgb.predict_proba(Xte)}).to_csv(
        out_dir / "predictions.csv", index=False
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "lr": _metrics_as_dict(lr_metrics),
                "xgb": _metrics_as_dict(xgb_metrics),
            },
            indent=2,
        )
    )
    print(f"\nSaved predictions + metrics to {out_dir}/")
    return 0


def _metrics_as_dict(m) -> dict:
    return {
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "top_decile_precision": m.top_decile_precision,
        "top_quintile_precision": m.top_quintile_precision,
        "brier": m.brier,
    }


if __name__ == "__main__":
    sys.exit(main())
