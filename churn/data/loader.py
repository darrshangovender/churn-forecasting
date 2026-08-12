"""Telco Customer Churn dataset loader.

Downloads the public Telco dataset on first use and caches it to ``~/.cache/churn-forecasting/telco.csv``.
The Telco dataset is a standard public churn benchmark with ~7,000 customer rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def _cache_path() -> Path:
    home = Path(os.environ.get("CHURN_CACHE_DIR", Path.home() / ".cache" / "churn-forecasting"))
    home.mkdir(parents=True, exist_ok=True)
    return home / "telco.csv"


def download_telco(force: bool = False) -> Path:
    """Download the Telco Churn CSV to the cache (idempotent)."""
    dest = _cache_path()
    if dest.exists() and not force:
        return dest
    resp = requests.get(TELCO_URL, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_telco() -> pd.DataFrame:
    """Return the cleaned Telco dataframe."""
    path = download_telco()
    df = pd.read_csv(path)
    # The 'TotalCharges' column has empty strings for new customers — coerce + impute with MonthlyCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def split_telco(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Split:
    """Stratified train/test split on the churn target."""
    y = df["churn"].values
    X = df.drop(columns=["churn", "Churn", "customerID"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return Split(X_tr.reset_index(drop=True), X_te.reset_index(drop=True), y_tr, y_te)
