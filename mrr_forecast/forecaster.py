"""MRR forecaster using Holt-Winters exponential smoothing.

We use statsmodels' implementation instead of Prophet here:
- statsmodels installs cleanly on every platform (Prophet has C++ build steps on Windows)
- For series of 24-60 months Holt-Winters is competitive with Prophet on MAPE
- The interface is cleaner for testing

For production with multi-year history + holiday effects, the ``prophet`` extra
is available (`pip install "churn-forecasting[prophet]"`) and the README documents
the swap.

Run:
    python -m mrr_forecast.forecaster
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from mrr_forecast.generator import generate_mrr_series


@dataclass
class Forecast:
    fitted: pd.Series
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series


class MRRForecaster:
    """Holt-Winters seasonal additive forecaster.

    Parameters
    ----------
    seasonal_periods: months per season (12 for monthly MRR)
    trend: "add" or "mul"
    seasonal: "add" or "mul"
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: str = "add",
        seasonal: str = "add",
    ) -> None:
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.fitted_ = None

    def fit(self, series: pd.Series) -> "MRRForecaster":
        # Holt-Winters needs at least 2*seasonal_periods observations
        if len(series) < 2 * self.seasonal_periods:
            # Fall back to no seasonality on short series
            seasonal: str | None = None
            sp: int | None = None
        else:
            seasonal = self.seasonal
            sp = self.seasonal_periods
        self.model = ExponentialSmoothing(
            series.astype(float),
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=sp,
            initialization_method="estimated",
        )
        self.fitted_ = self.model.fit(optimized=True)
        return self

    def forecast(self, h: int = 12, alpha: float = 0.2) -> Forecast:
        """Forecast next ``h`` months with approximate ``(1-alpha)`` prediction intervals."""
        if self.fitted_ is None:
            raise RuntimeError("Must call fit() before forecast()")
        pred = self.fitted_.forecast(h)
        # Approximate intervals from residual std
        resid = self.fitted_.resid
        std = float(np.nanstd(resid))
        z = 1.28 if alpha == 0.2 else 1.96
        lower = pred - z * std
        upper = pred + z * std
        return Forecast(
            fitted=self.fitted_.fittedvalues,
            forecast=pred,
            lower=lower,
            upper=upper,
        )


def main() -> int:
    print("Generating synthetic 36-month MRR series ...")
    series = generate_mrr_series()
    print(f"  series: {len(series)} months, start={series.iloc[0]:,.0f}, end={series.iloc[-1]:,.0f}")
    print("\nFitting Holt-Winters ...")
    f = MRRForecaster().fit(series)
    fc = f.forecast(h=12)
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    fc_df = pd.DataFrame({"mrr": fc.forecast, "lower": fc.lower, "upper": fc.upper})
    fc_df.to_csv(out_dir / "mrr_forecast.csv")
    print(f"\nForecast (next 12 months):")
    print(fc_df.round(0).to_string())
    print(f"\nSaved to {out_dir}/mrr_forecast.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
