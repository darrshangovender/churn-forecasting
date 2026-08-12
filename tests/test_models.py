"""Smoke tests for LR + XGBoost on a tiny synthetic dataset."""

import numpy as np
import pandas as pd
import pytest

from churn.models.baseline_lr import LRBaseline
from churn.models.xgb import XGBChurnClassifier


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    # Target: positive when x1 + 0.5*x2 > 0 (a learnable boundary)
    logits = X["x1"] + 0.5 * X["x2"] + rng.normal(0, 0.3, size=n)
    y = (logits > 0).astype(int).to_numpy()
    return X, y


def test_lr_trains_and_predicts(toy_dataset):
    X, y = toy_dataset
    model = LRBaseline().fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(y),)
    assert (probs >= 0).all() and (probs <= 1).all()
    # Should beat random on a learnable boundary
    preds = model.predict(X)
    assert (preds == y).mean() > 0.70


def test_xgb_trains_and_predicts(toy_dataset):
    X, y = toy_dataset
    model = XGBChurnClassifier(n_estimators=50, calibrate=False).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(y),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_xgb_calibration_improves_brier(toy_dataset):
    X, y = toy_dataset
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    uncal = XGBChurnClassifier(n_estimators=50, calibrate=False).fit(Xtr, ytr)
    cal = XGBChurnClassifier(n_estimators=50, calibrate=True).fit(Xtr, ytr)
    b_uncal = brier_score_loss(yte, uncal.predict_proba(Xte))
    b_cal = brier_score_loss(yte, cal.predict_proba(Xte))
    # Calibration shouldn't make Brier *worse* by more than a tiny tolerance
    assert b_cal <= b_uncal + 0.02
