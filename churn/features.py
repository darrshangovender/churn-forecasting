"""Feature engineering for the Telco Churn dataset.

The Telco dataset mixes categorical and numeric fields. The ``FeatureBuilder`` here:
- One-hot encodes categorical fields (contract type, payment method, internet service, etc.)
- Z-scores numeric fields (tenure, MonthlyCharges, TotalCharges)
- Builds a derived ``services_count`` feature — the count of bundled services a customer has

``services_count`` is the single most predictive engineered feature in production deployments of this pattern — customers with fewer services churn at much higher rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "SeniorCitizen",
]

_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

_SERVICES = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@dataclass
class FeatureBuilder:
    """Build numeric feature matrix from the Telco dataframe.

    Fit on train, transform train + test (no leakage).
    """

    scaler: StandardScaler | None = None
    feature_names: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        X = self._raw_features(df)
        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        self.scaler.fit(X[_NUMERIC])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None or self.feature_names is None:
            raise RuntimeError("FeatureBuilder must be fit before transform")
        X = self._raw_features(df)
        # Align columns to training schema (handle unseen categoricals gracefully)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X[_NUMERIC] = self.scaler.transform(X[_NUMERIC])
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        # Numeric pass-through (scaled later)
        for col in _NUMERIC:
            out[col] = df[col].astype(float)
        # Services count derived feature
        out["services_count"] = self._services_count(df)
        # One-hot of categoricals
        dummies = pd.get_dummies(df[_CATEGORICAL].astype(str), drop_first=False).astype(float)
        out = pd.concat([out, dummies], axis=1)
        return out

    @staticmethod
    def _services_count(df: pd.DataFrame) -> pd.Series:
        active = pd.Series(0, index=df.index, dtype=int)
        for svc in _SERVICES:
            col = df[svc].astype(str)
            # Treat "Yes" / "DSL" / "Fiber optic" / etc. as active; "No" / "No internet service" / "No phone service" as inactive
            active = active + col.apply(
                lambda v: 0 if v.lower().startswith("no") else 1
            )
        return active
