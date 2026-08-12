# Why Holt-Winters here (and when to swap in Prophet)

The reference implementation uses statsmodels' `ExponentialSmoothing` (Holt-Winters) by default rather than Facebook Prophet. The reasons:

## Why Holt-Winters by default

1. **Install reliability.** Prophet pulls in `pystan` / `cmdstanpy` and requires a C++ toolchain. On Windows + CI, this is one of the top sources of "but it worked on my machine" build failures. statsmodels installs from a wheel everywhere.
2. **Competitive accuracy on short series.** For monthly series of 24-60 observations, Holt-Winters with seasonal=12 is within 1-3% MAPE of Prophet on most SaaS revenue series. Prophet's edge is multi-year history with holiday effects.
3. **Tighter test surface.** No external optimizer process to mock; tests run in 0.5s.
4. **Cleaner failure modes.** statsmodels raises Python exceptions; Prophet prints C++ warnings on stderr that pytest captures awkwardly.

## When to swap to Prophet

Install with:

```bash
pip install "churn-forecasting[prophet]"
```

Swap in production when:
- Series has >36 monthly observations OR >2 years of daily/weekly data
- You have known holiday / promotional events with sustained effects
- You need automatic changepoint detection (Holt-Winters trends are smooth; Prophet handles step-changes natively)
- Multivariate regressors are needed (Prophet's `add_regressor` is cleaner than a custom SARIMAX setup)

## How to swap

In `mrr_forecast/forecaster.py`, replace the `ExponentialSmoothing` block with:

```python
from prophet import Prophet

def fit(self, series):
    df = series.reset_index().rename(columns={"index": "ds", series.name: "y"})
    self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    self.model.fit(df)
    return self
```

Keep the same external interface (`fit`, `forecast`) so callers don't change.
