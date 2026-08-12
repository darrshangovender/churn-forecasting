"""Forecast accuracy metrics (MAE + MAPE) on a held-out window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    mae: float
    mape: float
    rmse: float


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> ForecastMetrics:
    a = actual.astype(float).to_numpy()
    p = predicted.astype(float).to_numpy()
    if len(a) != len(p):
        raise ValueError(f"length mismatch: actual={len(a)} predicted={len(p)}")
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # Avoid /0
    safe_a = np.where(a == 0, np.nan, a)
    mape = float(np.nanmean(np.abs(err / safe_a)) * 100)
    return ForecastMetrics(mae=mae, mape=mape, rmse=rmse)
