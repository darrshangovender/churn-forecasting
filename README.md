<div align="center">

# Churn & MRR Forecasting Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400)](https://xgboost.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-3776AB)](https://facebook.github.io/prophet/)
[![Status](https://img.shields.io/badge/Status-Production-success)](#)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.87-059669)](#model-performance-on-holdout-set)

</div>

---

> End-to-end ML pipeline for a SaaS client. Pandas feature engineering from raw transactional data, gradient-boosted churn classifier, time-series MRR forecaster, served via FastAPI behind a Next.js dashboard. Surfaces at-risk accounts weekly to the customer success team.

**Stack:** Python · pandas · scikit-learn · XGBoost · Prophet · FastAPI · Next.js · PostgreSQL · GitHub Actions

---

## What it does

1. **Pulls** raw events, subscriptions, and support tickets from the client warehouse on a weekly schedule.
2. **Engineers features** in pandas — usage trends, payment health, support-ticket volume, plan-change history.
3. **Predicts churn** per account with an XGBoost classifier trained on 18 months of labelled history.
4. **Forecasts MRR** with Prophet (seasonality + plan-change impact decomposed).
5. **Serves** results via FastAPI — `/at-risk` returns the top-N accounts ranked by churn probability with feature contributions.
6. **Renders** a Next.js dashboard — sortable account table, MRR forecast band, drilldown into per-account feature contributions (SHAP).

## Why XGBoost for churn

Boosted trees are still the right default for tabular churn at this scale (~50k accounts, 60-ish features):

- Handles mixed numeric / categorical / sparse features without much preprocessing.
- Gives you SHAP feature contributions for free — critical because the customer success team wanted to know *why*, not just *who*.
- Trains in minutes on a single CPU box; no GPU dependency.

## Why Prophet for MRR

Not because it's the most accurate — it isn't always — but because:

- Its componentised output (trend / yearly / weekly / holiday) is something the finance team can read and challenge.
- Plan-change events are easy to inject as `holidays` or `regressors`.
- The maintenance burden is near-zero.

For larger clients I'd reach for an ensemble or a state-space model; for this client, Prophet was right.

## Feature engineering highlights

```python
# A few of the features that mattered most (by SHAP)
- usage_trend_28d         # slope of daily-active-users last 4 weeks
- days_since_last_login
- plan_downgrade_in_90d   # binary
- failed_payments_in_90d
- support_ticket_volume_z # z-score vs cohort
- mrr_in_dollars
- account_age_months
```

The biggest signal — by a wide margin — turned out to be `usage_trend_28d`. Accounts whose usage is in measurable decline for ~4 weeks churn at multiples of the base rate.

## Pipeline

```
weekly cron ─▶ extract (Postgres → parquet)
            ─▶ feature build (pandas)
            ─▶ train + register (mlflow local)
            ─▶ batch score
            ─▶ persist scores (Postgres)
            ─▶ FastAPI serves /at-risk + /forecast
            ─▶ Next.js dashboard
```

All steps are idempotent and re-runnable with a `--from-date` flag.

## Repo structure

```
.
├── pipeline/
│   ├── extract.py
│   ├── features.py
│   ├── train_churn.py
│   ├── train_mrr.py
│   └── score.py
├── api/
│   └── main.py
├── web/                  # Next.js dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb
│   └── 03_model_eval.ipynb
└── infra/
    └── github-actions/   # weekly workflow
```

## Model performance (on holdout set)

| Metric | Score |
|---|---|
| ROC-AUC (churn) | 0.87 |
| Precision @ top-decile | 0.62 |
| Recall @ top-decile | 0.58 |
| MAPE (MRR forecast, 4 weeks out) | 3.8% |

Top-decile precision is the metric that mattered most to the client — they only have capacity for the customer success team to call ~10% of accounts each week, so we optimised for that band rather than overall accuracy.

## Local setup

```bash
uv sync
uv run python pipeline/extract.py --from-date 2024-01-01
uv run python pipeline/train_churn.py
uv run uvicorn api.main:app --reload
```

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)
