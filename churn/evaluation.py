"""Evaluation metrics for churn classification.

The headline metric for retention teams is ``top_decile_precision`` — what fraction of
the top-10% of customers ranked by predicted churn probability actually churned. This is
the actionable number: it tells you what hit rate to expect when your CSM team works
through the high-risk list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class ChurnMetrics:
    roc_auc: float
    pr_auc: float
    top_decile_precision: float
    top_quintile_precision: float
    brier: float
    confusion_at_05: tuple[int, int, int, int]  # tn, fp, fn, tp


def top_decile_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision among the top 10% highest-scored customers."""
    return _top_k_precision(y_true, y_score, k_frac=0.10)


def top_quintile_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return _top_k_precision(y_true, y_score, k_frac=0.20)


def _top_k_precision(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(y_score)[::-1][:k]  # top-k indices
    if k == 0:
        return 0.0
    return float(np.mean(y_true[idx]))


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> ChurnMetrics:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ChurnMetrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        top_decile_precision=top_decile_precision(y_true, y_score),
        top_quintile_precision=top_quintile_precision(y_true, y_score),
        brier=float(brier_score_loss(y_true, y_score)),
        confusion_at_05=(int(tn), int(fp), int(fn), int(tp)),
    )


def format_metrics(m: ChurnMetrics, label: str = "") -> str:
    lbl = f"[{label}] " if label else ""
    return (
        f"{lbl}ROC-AUC={m.roc_auc:.3f}  PR-AUC={m.pr_auc:.3f}  "
        f"top-10%={m.top_decile_precision:.3f}  top-20%={m.top_quintile_precision:.3f}  "
        f"Brier={m.brier:.3f}"
    )
