# Churn Classification + MRR Forecasting

[![tests](https://github.com/darrshangovender/churn-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/darrshangovender/churn-forecasting/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6F00)](https://xgboost.readthedocs.io)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-3776AB)](https://www.statsmodels.org)

> **Public reference implementation** of churn classification + MRR forecasting on open data, structured the way I ship them for production retention teams. Two pipelines: a calibrated XGBoost churn classifier on the public Telco dataset, and a Holt-Winters MRR forecaster on a synthetic SaaS revenue series.

## Scope

This is a **public reference implementation**. The production version at the Agulhas Code client (under NDA) uses live billing data, real CSM-event signals, and pushes the ranked-customer list into their CRM. The reference impl here reproduces the same architecture and metrics on data anyone can re-run.

## Architecture

```
                       ┌─────────────────────────────┐
                       │   Telco Customer Churn      │
                       │   (public dataset, ~7k rows)│
                       └────────────┬────────────────┘
                                    ▼
                       ┌─────────────────────────────┐
                       │   FeatureBuilder            │
                       │   • z-score numerics        │
                       │   • one-hot categoricals    │
                       │   • derived services_count  │
                       └────────────┬────────────────┘
                                    ▼
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
       ┌──────────────────┐                ┌──────────────────────┐
       │ LR baseline      │                │ XGBoost + isotonic   │
       │ (class-balanced) │                │ calibration          │
       └────────┬─────────┘                └──────────┬───────────┘
                ▼                                     ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Evaluation:                                              │
       │   ROC-AUC · PR-AUC · top-decile precision · Brier        │
       └──────────────────────────────────────────────────────────┘

                       ┌─────────────────────────────┐
                       │   Synthetic MRR series      │
                       │   (36 months, growth +      │
                       │    seasonality + AR(1) noise)│
                       └────────────┬────────────────┘
                                    ▼
                       ┌─────────────────────────────┐
                       │   Holt-Winters (statsmodels)│
                       │   → 12-month forecast       │
                       │   → 80% prediction intervals│
                       └─────────────────────────────┘
```

## Quick start

```bash
# Install (uv recommended — handles the wheel fetches cleanly)
uv sync

# Run the churn pipeline (downloads Telco on first run, caches to ~/.cache/churn-forecasting/)
make train

# Run the MRR forecaster
make forecast

# Run both with a results table
make benchmark

# Run the tests
make test
```

## What's measured

### Churn classifier (Telco, 80/20 stratified split)

| Model | ROC-AUC | PR-AUC | Top-10% precision | Brier |
|---|---|---|---|---|
| Logistic regression baseline | run `make benchmark` to fill | | | |
| XGBoost + isotonic calibration | run `make benchmark` to fill | | | |

> The honest numbers depend on the run; the script prints them. On a typical run XGBoost lands around ROC-AUC ~0.84-0.86 and top-decile precision ~0.55-0.65 on Telco — a strong but unspectacular baseline that demonstrates the pattern.

### MRR forecast (synthetic 36-month series, 6-month hold-out)

| Metric | Value |
|---|---|
| MAE | run `make benchmark` |
| MAPE | run `make benchmark` |
| RMSE | run `make benchmark` |

Synthetic series is deterministic (`seed=42`), so results are reproducible.

## Why these choices

| Decision | Reasoning |
|---|---|
| **XGBoost + isotonic calibration** | XGBoost gives good ranking out-of-box; calibration matters because retention ROI calculations multiply by probability. Sigmoid calibration is too smooth for tree models; isotonic better. |
| **Top-decile precision as headline** | It's the metric the retention team actually acts on. ROC-AUC tells you "is the model better than random"; top-decile precision tells you "what hit rate will my CSMs see when they work this list." |
| **Holt-Winters over Prophet** | Installs reliably on every platform (Prophet's pystan/cmdstanpy install is fragile on Windows + CI). Competitive on series under 5 years. See [`docs/prophet-config.md`](docs/prophet-config.md) for the swap. |
| **`services_count` derived feature** | The bundling-depth signal is independent of price. In every production deployment of this pattern, customers with 1-2 services churn at 2-3× the rate of customers with 5+. |
| **Class-balanced LR / no class-balanced XGB** | LR needs `class_weight="balanced"` because logistic loss is sensitive to imbalance; gradient-boosted trees handle imbalance fine without it (and rebalancing hurts probabilistic calibration). |

## Repo structure

```
.
├── churn/                          # Churn classification pipeline
│   ├── data/loader.py              # Telco fetch + cache + stratified split
│   ├── features.py                 # FeatureBuilder (fit/transform on train)
│   ├── models/baseline_lr.py       # Logistic regression baseline
│   ├── models/xgb.py               # XGBoost + isotonic calibration
│   ├── evaluation.py               # ROC-AUC, PR-AUC, top-decile, Brier
│   └── pipeline.py                 # End-to-end pipeline (run with `make train`)
├── mrr_forecast/                   # MRR forecasting pipeline
│   ├── generator.py                # Synthetic 36-month series generator
│   ├── forecaster.py               # Holt-Winters seasonal forecaster
│   └── evaluation.py               # MAE, MAPE, RMSE
├── tests/                          # pytest tests for everything above
├── benchmarks/run.py               # Runs both pipelines, writes results.json
├── docs/
│   ├── churn-features.md           # Feature engineering rationale + leakage notes
│   └── prophet-config.md           # Why Holt-Winters here; how to swap in Prophet
├── Makefile                        # data / train / forecast / test / benchmark
└── pyproject.toml
```

## Run the tests

```bash
make test
```

CI runs `pytest tests/` on every push (see `.github/workflows/tests.yml`).

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za)
