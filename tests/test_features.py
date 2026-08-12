"""Tests for the FeatureBuilder.

These verify shape correctness and the services_count derivation on hand-crafted rows.
"""

import pandas as pd
import pytest

from churn.features import FeatureBuilder, _SERVICES


@pytest.fixture
def tiny_df():
    return pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["Yes", "No phone service"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "SeniorCitizen": [0, 1],
            "tenure": [3, 70],
            "MonthlyCharges": [89.5, 24.1],
            "TotalCharges": [268.5, 1687.0],
        }
    )


def test_fit_transform_returns_dataframe(tiny_df):
    fb = FeatureBuilder()
    out = fb.fit_transform(tiny_df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(tiny_df)


def test_services_count_includes_active_services(tiny_df):
    fb = FeatureBuilder()
    out = fb.fit_transform(tiny_df)
    # Row 0: PhoneService=Yes, MultipleLines=Yes, InternetService=Fiber optic, OnlineSecurity=Yes,
    # StreamingTV=Yes, StreamingMovies=Yes → 6 active. (OnlineBackup, DeviceProtection, TechSupport = No)
    assert out.loc[0, "services_count"] == 6
    # Row 1: InternetService=DSL, TechSupport=Yes → 2 active.
    assert out.loc[1, "services_count"] == 2


def test_transform_handles_unseen_categorical(tiny_df):
    """If a test row has a category unseen in training, transform should still align columns."""
    fb = FeatureBuilder().fit(tiny_df)
    new = tiny_df.copy()
    new["Contract"] = ["One year", "Two year"]  # "One year" unseen in fit
    out = fb.transform(new)
    # Columns must match the fit-time schema (the Contract_One year column gets backfilled to 0)
    assert list(out.columns) == fb.feature_names


def test_numeric_features_are_standardized(tiny_df):
    fb = FeatureBuilder().fit(tiny_df)
    out = fb.transform(tiny_df)
    # On a 2-row training set the scaled tenure values should have mean ~0 (exactly, since fit==transform here)
    assert abs(out["tenure"].mean()) < 1e-9
