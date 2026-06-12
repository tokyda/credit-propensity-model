# Revolut Overdraft Propensity Model — Design Spec

**Date:** 2026-06-11
**Author:** Tobias Surja
**Purpose:** Portfolio project for Revolut Data Scientist / Analyst internship application (2027)

---

## 1. Overview

### Business context

This project simulates the type of work a DS intern at Revolut would do to support a credit product launch. The fictional scenario: Revolut is launching an overdraft facility in a new market and needs to identify which existing current-account users are most likely to adopt it. A growth analyst will use the model output to decide who receives a pre-approved overdraft offer in-app.

The project is not affiliated with Revolut. It uses the Berka banking dataset as a proxy for neobank transactional data to demonstrate product thinking, SQL depth, and end-to-end ML delivery.

### Gaps addressed

| Gap | How this project fills it |
|---|---|
| SQL depth | dbt feature engineering layer with window functions, multi-table joins, layered models |
| Product analytics thinking | Business-framed output: a targeting decision, not a model accuracy metric |
| Fintech domain | Credit product adoption scenario, calibrated propensity scoring, SHAP explainability |

### Success criteria

- A deployed Streamlit dashboard accessible via a public URL
- A ranked targeting list of users with calibrated propensity scores and per-user SHAP explanations
- LightGBM model outperforms Logistic Regression baseline on AUC-ROC and PR-AUC
- dbt test suite passes with zero failures
- A coherent interview narrative: business question → data → features → model → decision

---

## 2. Business Question

**Scenario:** Revolut is launching an overdraft product in a new market.

**Question:** Which existing current-account users should receive a pre-approved overdraft offer in-app this week?

**End user:** Growth/marketing analyst

**Decision enabled:** The analyst opens the dashboard, filters to high-propensity users (score ≥ 0.70), downloads a targeting list, and loads it into the notification/CRM tool to send in-app offers.

**Proxy label rationale:** Historical loan uptake (Berka `loan` table, status A or C) is used as a proxy for healthy credit product adoption. Users who previously took and cleanly repaid a loan are treated as the positive class — they demonstrate willingness and ability to engage with credit products. Accounts with bad loan history (status B or D) are excluded from the training set entirely: they are neither positive nor negative, as their signal is ambiguous for propensity (adopted but failed).

---

## 3. Architecture

Four layers, each with a single responsibility:

```
[ Layer 1: Data ]
Berka CSVs → loaded into DuckDB (local, in-process analytical database)

[ Layer 2: Feature Engineering ]
dbt models: Raw → Staging → Intermediate → Features → Mart
Outputs: mart_loan_propensity, mart_funnel

[ Layer 3: ML ]
Python script consumes mart_loan_propensity
→ Logistic Regression (baseline)
→ LightGBM + SHAP + Platt calibration (primary)
Outputs: scored_users.csv, model_metrics.json

[ Layer 4: Dashboard ]
Streamlit app reads scored_users.csv + model_metrics.json + mart_funnel
→ Primary panel: ranked targeting list with SHAP explanations
→ Secondary panel: funnel drop-off visualisation
→ Model explainability panel: global SHAP feature importance
→ Deployed: Streamlit Community Cloud (public URL)
```

**Why DuckDB:** Embedded OLAP database — no server, no credentials, full SQL including window functions and CTEs, column-oriented for fast analytical aggregations, natively supported by dbt. Appropriate for local development against a ~4,500 account dataset.

---

## 4. Dataset

**Source:** Berka banking dataset (Czech bank, 1999). Available on Kaggle.

**Tables used:**

| Table | Contents | Role |
|---|---|---|
| `account` | account_id, district_id, statement frequency, opening date | Account-level features |
| `client` | client_id, birth_number, district_id | Demographics (age, gender) |
| `disp` | Links clients to accounts with role (OWNER / DISPONENT) | Joins clients to accounts |
| `trans` | 1M+ transaction records: date, amount, balance, type, operation | Behavioural features |
| `loan` | Loan records with status A/B/C/D | Label source |
| `card` | Card type (junior/classic/gold) per account | Product adoption feature |
| `district` | 15 demographic columns: salary, unemployment, crime, population | Socioeconomic context |
| `order` | Standing orders (regular automated payments) | Financial organisation features |

**Scale:** ~4,500 accounts, ~1 million transactions, 682 clean loan records (15% positive rate).

---

## 5. Feature Engineering Layer (dbt)

### Layer structure

```
models/
├── staging/        # One model per raw table. Rename columns, cast types. No logic.
├── intermediate/   # Join related tables. Build shapes needed for feature work.
├── features/       # Feature engineering: aggregations, window functions, ratios.
└── mart/           # Two final output tables: mart_loan_propensity, mart_funnel.
```

Each layer reads only from the layer above it. The ML script reads only from the mart.

### Staging models

One model per raw table: `stg_accounts`, `stg_clients`, `stg_transactions`, `stg_loans`, `stg_cards`, `stg_districts`, `stg_orders`, `stg_dispositions`.

### Intermediate models

| Model | Joins |
|---|---|
| `int_account_owner` | account + client (OWNER only via disp) + district |
| `int_account_transactions` | account + transactions |
| `int_account_cards` | account + disp + card |

### Feature models

**`feat_transaction_aggregates`** — behavioural signals from transaction history:

| Feature | Description |
|---|---|
| `avg_monthly_transaction_count` | Average transactions per month |
| `avg_transaction_amount` | Average transaction value |
| `total_credit_amount` | Total money credited |
| `total_debit_amount` | Total money debited |
| `credit_debit_ratio` | Ratio of credits to debits |
| `avg_balance` | Mean running balance |
| `balance_volatility` | Standard deviation of balance |
| `min_balance_ever` | Lowest balance ever reached |
| `days_since_last_transaction` | Recency signal |
| `account_age_days` | Days since account opening |
| `transaction_trend` | Avg amount last 90 days vs prior 90 days (window function) |

**`feat_account_demographics`** — account and owner characteristics:

| Feature | Description |
|---|---|
| `client_age` | Derived from birth_number |
| `client_gender` | Derived from birth_number (Czech encoding: even birth month = female) |
| `statement_frequency` | Monthly / weekly / after each transaction (encoded) |
| `district_avg_salary` | Average salary in client's district |
| `district_unemployment_rate` | Unemployment rate in district |
| `district_crime_rate` | Crime index in district |

**`feat_account_products`** — product adoption signals:

| Feature | Description |
|---|---|
| `has_card` | Binary — card ownership |
| `card_type` | junior / classic / gold (one-hot encoded downstream) |
| `has_standing_order` | Binary — automated regular payments set up |
| `standing_order_count` | Number of standing orders |
| `avg_standing_order_amount` | Average regular payment amount |

### Mart models

**`mart_loan_propensity`** — final ML-ready table. One row per eligible account.

Label construction:
```sql
CASE
    WHEN loan.status IN ('A', 'C') THEN 1   -- clean loan = adopted credit product
    WHEN loan.loan_id IS NULL      THEN 0   -- no loan = did not adopt
    ELSE NULL                               -- bad loan (B/D) = excluded from training
END AS adopted_credit_product
```

Rows where `adopted_credit_product IS NULL` are excluded from the final mart.

**`mart_funnel`** — four aggregated counts for the dashboard funnel panel:

| Stage | Definition |
|---|---|
| Accounts opened | All accounts in the dataset |
| Active accounts | Accounts with 3+ transactions within 90 days of opening |
| Eligible accounts | Active accounts with no bad loan history (no status B or D) |
| Adopted | Eligible accounts with clean loan record (status A or C) |

### SQL depth demonstration

`transaction_trend` is the primary window function in the feature layer:

```sql
-- Recent window: average amount in last 90 calendar days
AVG(CASE WHEN date >= max_date - INTERVAL '90 days' THEN amount END)
    OVER (PARTITION BY account_id) AS avg_amount_last_90d,

-- Prior window: average amount in the 90 days before that
AVG(CASE WHEN date BETWEEN max_date - INTERVAL '180 days'
                       AND max_date - INTERVAL '91 days'  THEN amount END)
    OVER (PARTITION BY account_id) AS avg_amount_prior_90d
```

`transaction_trend` is derived as `avg_amount_last_90d / NULLIF(avg_amount_prior_90d, 0)`. A ratio above 1 means increasing activity; below 1 means declining. Uses date arithmetic rather than `ROWS BETWEEN N PRECEDING`, which counts rows not days and produces inconsistent windows across accounts with different transaction frequencies.

---

## 6. ML Layer

### Data preparation

- **Train/test split:** 80/20, stratified on `adopted_credit_product` to maintain 15% positive rate in both splits
- **Class imbalance:** addressed via class weighting in both models (not oversampling). Without correction, models trained on 85% negatives learn to predict low scores for everyone, compressing calibrated probabilities near the 0.15 base rate.

### Model 1: Logistic Regression (baseline)

```
Pipeline:
  StandardScaler       → numerical features
  OneHotEncoder        → categorical features
  LogisticRegression(class_weight='balanced')
```

Purpose: establish a performance floor and make model selection explicit.

### Model 2: LightGBM (primary)

```python
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=neg_count / pos_count,
    categorical_feature=[...],
)
```

No exhaustive hyperparameter search. Three parameters tuned manually — appropriate for 4,500 rows.

### Probability calibration

Platt scaling applied post-training:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(lgbm_model, method='sigmoid', cv='prefit')
```

Output scores are genuine probabilities: a score of 0.70 corresponds to approximately 70% adoption likelihood in that score bucket. Required because the dashboard exposes scores as actionable thresholds, not just relative rankings.

### SHAP explanations

- **Global:** mean absolute SHAP values across all accounts → feature importance bar chart in dashboard
- **Per-user:** top 3 SHAP drivers stored alongside each account's score → shown in targeting list rows

### Evaluation metrics

| Metric | Purpose |
|---|---|
| AUC-ROC | Ranking ability: how well the model separates adopters from non-adopters |
| PR-AUC | Performance on minority class: better than AUC-ROC for imbalanced data |
| Calibration curve | Verifies predicted probabilities match actual adoption rates |

### Output files

**`scored_users.csv`**
```
account_id | score | segment | shap_1_name | shap_1_value | shap_2_name | shap_2_value | shap_3_name | shap_3_value
```

**`model_metrics.json`**
```json
{
  "lr_auc_roc": 0.XX,
  "lgbm_auc_roc": 0.XX,
  "lgbm_pr_auc": 0.XX
}
```

**Score segments:**

| Segment | Threshold |
|---|---|
| High propensity | score ≥ 0.70 |
| Medium propensity | 0.40 ≤ score < 0.70 |
| Low propensity | score < 0.40 |

---

## 7. Dashboard Layer

**Technology:** Streamlit, deployed to Streamlit Community Cloud (public URL).

**Inputs:** `scored_users.csv`, `model_metrics.json`, `mart_funnel.csv`.

`mart_funnel.csv` is exported from DuckDB to a flat file by a Python script that runs after `dbt run` completes, alongside the ML training script. All three files are committed to the repository so Streamlit Community Cloud can access them without a live database connection.

### Layout

```
┌──────────────────────────────────────────────────────┐
│  Overdraft Propensity — Growth Targeting Dashboard   │
│                                                      │
│  [Total scored] [High propensity] [AUC-ROC] [PR-AUC]│
├──────────────────────────────────────────────────────┤
│ SIDEBAR          │  PRIMARY PANEL                    │
│                  │  Ranked targeting list            │
│ Segment filter   │  account_id | score | segment     │
│ Score threshold  │  | driver 1 | driver 2 | driver 3 │
│ (slider 0→1)     │                                   │
│                  │  [Export targeting list as CSV]   │
│                  ├───────────────────────────────────┤
│                  │  SECONDARY PANEL                  │
│                  │  Funnel drop-off bar chart        │
│                  ├───────────────────────────────────┤
│                  │  MODEL EXPLAINABILITY             │
│                  │  Global SHAP importance chart     │
└──────────────────────────────────────────────────────┘
```

### KPI cards

Four metrics shown on load: total users scored, high propensity users (≥0.70), LightGBM AUC-ROC, LightGBM PR-AUC.

### Primary panel

Filterable table. Sidebar segment filter and score threshold slider narrow the table in real time. "Export targeting list as CSV" button is the primary action — the analyst downloads this and loads it into their notification tool.

### Secondary panel

Horizontal bar chart showing the four funnel stages with counts and percentage drop-off between each stage. Provides context: of all accounts, only 682 ever adopted a credit product — the model identifies who else looks like them.

### Model explainability panel

Horizontal bar chart of top 10 features by mean absolute SHAP value. Demonstrates that the model uses interpretable financial signals, not noise.

---

## 8. Testing

### Layer 1: dbt data quality tests (schema.yml)

- `not_null` on all primary keys across staging models
- `unique` on `account_id` in `mart_loan_propensity`
- `accepted_values` on `loan.status` (A, B, C, D only)
- `relationships` on all foreign keys (e.g., every `account_id` in transactions exists in accounts)
- Custom test: no accounts with loan status B or D appear as labelled rows in the mart
- Custom test: `adopted_credit_product` is strictly 0 or 1 — no nulls

Run with: `dbt test`

### Layer 2: ML sanity checks (Python assertions at end of training script)

- All predicted scores are in [0, 1]
- No NaN values in `scored_users.csv`
- LightGBM AUC-ROC > Logistic Regression AUC-ROC
- LightGBM AUC-ROC > 0.65 (meaningfully above random)
- Mean predicted probability on test set ≈ actual positive rate (calibration sanity check)

### Layer 3: Dashboard smoke test

`streamlit run app.py --server.headless true` runs without error against actual output files before deployment.

---

## 9. Stack Summary

| Component | Technology |
|---|---|
| Local database | DuckDB |
| Feature engineering | dbt |
| ML framework | scikit-learn + LightGBM |
| Explainability | SHAP (TreeSHAP) |
| Calibration | scikit-learn CalibratedClassifierCV (Platt) |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Testing | dbt built-in tests + Python assertions |
| Language | Python + SQL |

---

## 10. Out of Scope

- Credit risk / probability of default modelling (separate problem, separate model)
- Real-time scoring (batch scoring only)
- A/B test design or uplift modelling
- Hyperparameter optimisation beyond manual tuning of 3 parameters
- User authentication on the dashboard
- Automated pipeline orchestration (Airflow, Prefect, etc.)
