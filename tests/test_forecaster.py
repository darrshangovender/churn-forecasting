"""Tests for the MRR forecaster (Holt-Winters)."""

import numpy as np
import pandas as pd
import pytest

from mrr_forecast.evaluation import forecast_metrics
from mrr_forecast.forecaster import MRRForecaster
from mrr_forecast.generator import generate_mrr_series


def test_generator_produces_expected_shape():
    s = generate_mrr_series(n_months=24)
    assert len(s) == 24
    # MRR must be strictly positive
    assert (s > 0).all()


def test_generator_is_seeded():
    s1 = generate_mrr_series(seed=42)
    s2 = generate_mrr_series(seed=42)
    pd.testing.assert_series_equal(s1, s2)


def test_forecaster_returns_correct_horizon():
    s = generate_mrr_series(n_months=36)
    fc = MRRForecaster().fit(s).forecast(h=12)
    assert len(fc.forecast) == 12
    assert len(fc.lower) == 12
    assert len(fc.upper) == 12
    # No NaN forecasts
    assert not fc.forecast.isna().any()


def test_forecaster_intervals_bracket_forecast():
    s = generate_mrr_series(n_months=36)
    fc = MRRForecaster().fit(s).forecast(h=6)
    assert (fc.lower <= fc.forecast).all()
    assert (fc.upper >= fc.forecast).all()


def test_forecast_metrics_on_perfect_forecast():
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = pd.Series([100.0, 200.0, 300.0])
    m = forecast_metrics(actual, predicted)
    assert m.mae == 0.0
    assert m.mape == 0.0
    assert m.rmse == 0.0


def test_forecaster_falls_back_on_short_series():
    """Series shorter than 2*seasonal_periods should fit without erroring (no seasonality)."""
    s = generate_mrr_series(n_months=18)
    fc = MRRForecaster(seasonal_periods=12).fit(s).forecast(h=6)
    assert len(fc.forecast) == 6
