"""Tests for evaluation metrics — most importantly top_decile_precision."""

import numpy as np
import pytest

from churn.evaluation import evaluate, top_decile_precision, top_quintile_precision


def test_top_decile_precision_perfect_ranker():
    # 100 customers, 10 churn — perfectly ranked → top decile precision = 1.0
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1.0, 0.0, 100)
    assert top_decile_precision(y_true, y_score) == 1.0


def test_top_decile_precision_random_ranker():
    np.random.seed(0)
    y_true = np.random.binomial(1, 0.2, size=1000)
    y_score = np.random.rand(1000)
    # Should be near base rate (~0.2), give a generous band
    assert 0.10 < top_decile_precision(y_true, y_score) < 0.30


def test_top_quintile_precision_perfect_ranker():
    y_true = np.array([1] * 20 + [0] * 80)
    y_score = np.linspace(1.0, 0.0, 100)
    assert top_quintile_precision(y_true, y_score) == 1.0


def test_evaluate_returns_all_metrics():
    np.random.seed(1)
    y_true = np.random.binomial(1, 0.3, size=200)
    y_score = np.random.rand(200) * 0.3 + y_true * 0.5  # imperfect-but-positive correlation
    m = evaluate(y_true, y_score)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0
    assert 0.0 <= m.brier <= 1.0
    tn, fp, fn, tp = m.confusion_at_05
    assert tn + fp + fn + tp == len(y_true)


def test_top_decile_handles_small_n():
    # n=5, top decile floored at 1 row
    y_true = np.array([1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    assert top_decile_precision(y_true, y_score) == 1.0
