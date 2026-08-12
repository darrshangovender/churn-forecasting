"""XGBoost churn classifier with isotonic calibration.

In production retention workflows the absolute probability matters as much as the
ranking (revenue-saved estimates multiply by churn probability), so we wrap the raw
XGBoost classifier in ``CalibratedClassifierCV`` to give well-calibrated outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


class XGBChurnClassifier:
    """XGBoost + isotonic calibration.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate
        XGBoost hyperparameters. Defaults tuned for Telco-scale problems.
    calibrate
        If True, wrap in ``CalibratedClassifierCV`` with isotonic regression.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        self.base = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )
        self.calibrate = calibrate
        self.model: XGBClassifier | CalibratedClassifierCV = self.base

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBChurnClassifier":
        if self.calibrate:
            # 3-fold internal calibration; isotonic better than sigmoid for tree models
            self.model = CalibratedClassifierCV(self.base, method="isotonic", cv=3)
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
