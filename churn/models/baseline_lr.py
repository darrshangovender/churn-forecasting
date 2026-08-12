"""Logistic regression baseline for churn classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LRBaseline:
    """L2-regularised logistic regression with class-weighted training."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.model = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LRBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
