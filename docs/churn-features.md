# Churn feature engineering — why each feature

This doc walks through the features the `FeatureBuilder` produces and why each carries signal on the Telco Customer Churn dataset.

## Numeric features (z-scored on the training split)

| Feature | Why it carries signal |
|---|---|
| `tenure` | Strongest single feature in nearly every SaaS-style churn dataset. Customers in months 1-3 churn at 3-5× the steady-state rate. |
| `MonthlyCharges` | Higher monthly bills correlate with both higher service tiers AND higher price-sensitivity. The relationship is non-linear — useful for tree models. |
| `TotalCharges` | Tenure × MonthlyCharges proxy. Coerce empty strings to NaN, then impute with MonthlyCharges (covers new customers). |

## Derived feature

| Feature | Construction | Why |
|---|---|---|
| `services_count` | Count of bundled services with a non-"No" status (Phone, Multiple lines, Internet, Online security, Online backup, Device protection, Tech support, Streaming TV, Streaming movies) | In every production deployment of this pattern I've seen, customers with 1-2 services churn at 2-3× the rate of customers with 5+. The bundling depth signal is independent of price. |

## Categorical features (one-hot)

All from the Telco schema: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `SeniorCitizen`.

The strongest categorical signal in this dataset is `Contract` — month-to-month customers churn at ~5× the rate of two-year-contract customers.

## Leakage gotchas

- The `TotalCharges` field has empty strings for very-new customers (the dataset has a few). Coerce-then-impute, never drop, otherwise the model "learns" that missing TotalCharges = new = high churn, which only works because of the leak.
- Don't include `Churn` or any direct churn-event field in the feature matrix — obvious but easy to miss when refactoring.
- The `customerID` field has no signal; drop it before fitting.
- When refitting the FeatureBuilder on production data, pass `df` from the same time window as training to avoid concept drift mistakenly being baked into the scaler statistics.
