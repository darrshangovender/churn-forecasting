"""Tests for the MRR forecaster (Holt-Winters)."""

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


def test_forecast_interval_width_tracks_alpha():
    """`z = 1.28 if alpha == 0.2 else 1.96` is an exact float comparison against
    a continuous parameter, so every level except 0.2 and 0.05 silently got the
    95% band — a requested 50% interval came back ~2.9x too wide."""
    f = MRRForecaster().fit(generate_mrr_series())

    def width(a: float) -> float:
        fc = f.forecast(h=6, alpha=a)
        return float((fc.upper - fc.lower).mean())

    widths = {a: width(a) for a in (0.5, 0.2, 0.1, 0.05, 0.01)}
    # Lower confidence (larger alpha) must give a strictly narrower band.
    assert widths[0.5] < widths[0.2] < widths[0.1] < widths[0.05] < widths[0.01]
    assert widths[0.5] < 0.6 * widths[0.05]


def test_alpha_reached_by_arithmetic_matches_the_literal():
    """1 - 0.8 == 0.19999999999999996, which missed the `== 0.2` branch entirely
    and silently widened an 80% interval to 95%."""
    f = MRRForecaster().fit(generate_mrr_series())
    exact = f.forecast(h=6, alpha=0.2)
    computed = f.forecast(h=6, alpha=1 - 0.8)
    assert float((computed.upper - computed.lower).mean()) == pytest.approx(
        float((exact.upper - exact.lower).mean()), rel=1e-9
    )


def test_invalid_alpha_rejected():
    f = MRRForecaster().fit(generate_mrr_series())
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            f.forecast(h=6, alpha=bad)
