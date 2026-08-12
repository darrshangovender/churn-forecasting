"""Generate a synthetic monthly MRR time series with realistic SaaS dynamics.

The series has:
- A compounding growth trend (~6% MoM early, decelerating to ~2% MoM)
- 12-month seasonality (Q4 expansion, Q1 dip)
- AR(1) noise for shock-and-recovery realism
- Optional churn shock at a configurable month (defaults to a 12% dip at month 18)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_mrr_series(
    n_months: int = 36,
    start_mrr: float = 50_000.0,
    seed: int = 42,
    churn_shock_at: int | None = 18,
    churn_shock_pct: float = 0.12,
) -> pd.Series:
    """Generate a monthly MRR series indexed by month-end dates."""
    rng = np.random.default_rng(seed)

    # Decelerating growth rate
    base_growth = np.linspace(0.06, 0.02, n_months)
    # Seasonality: Q4 (months 10-12) +3%, Q1 (months 1-3) -2%
    season = np.array([np.sin(2 * np.pi * (i % 12) / 12) * 0.025 for i in range(n_months)])
    growth = base_growth + season

    # AR(1) noise
    noise = np.zeros(n_months)
    for i in range(1, n_months):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 0.015)

    mrr = np.zeros(n_months)
    mrr[0] = start_mrr
    for i in range(1, n_months):
        mrr[i] = mrr[i - 1] * (1.0 + growth[i] + noise[i])
        if churn_shock_at is not None and i == churn_shock_at:
            mrr[i] = mrr[i] * (1.0 - churn_shock_pct)

    dates = pd.date_range(start="2023-01-31", periods=n_months, freq="ME")
    return pd.Series(mrr, index=dates, name="mrr")
