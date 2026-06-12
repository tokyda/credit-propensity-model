# Credit Product Propensity Model

End-to-end ML pipeline predicting which current-account users are most likely to adopt a credit product — from raw transaction data to a live targeting dashboard.

**[Live dashboard →](https://credit-propensity-model-av9cethd3x9cswins44etz.streamlit.app)**

---

## What it does

A growth analyst wants to know: *which users should receive a pre-approved overdraft offer in-app?* This project answers that question with a full production-style pipeline:

1. **Feature engineering in SQL** — dbt transforms raw banking data through four layers (staging → intermediate → features → mart), producing a model-ready feature table with 42 data quality tests
2. **ML model** — a calibrated LightGBM classifier (AUC-ROC **0.905** vs logistic regression baseline 0.862) with Platt scaling for reliable probability estimates
3. **Explainability** — SHAP values surface the top 3 drivers for every user
4. **Targeting dashboard** — Streamlit app with segment filters, per-user driver breakdown, funnel visualisation, and CSV export

---

## Stack

| Layer | Tool |
|---|---|
| Storage | DuckDB (embedded OLAP) |
| Feature pipeline | dbt-duckdb |
| ML | LightGBM, scikit-learn, SHAP |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

---

## Architecture

```
data/raw/          ← Berka banking CSVs (gitignored)
dbt_project/
  models/
    staging/       ← clean + cast raw tables (8 models, views)
    intermediate/  ← joins across entities (3 models, views)
    features/      ← aggregated ML features (3 models, tables)
    mart/          ← model-ready table + funnel aggregate (2 models, tables)
  tests/           ← 2 custom singular tests
ml/
  train.py         ← LR baseline → LightGBM → Platt calibration → SHAP
outputs/           ← scored_users.csv, model_metrics.json, global_shap.csv
dashboard/
  app.py           ← Streamlit targeting dashboard
```

---

## Dataset

[Berka banking dataset](https://www.kaggle.com/datasets/nitinpuri/berka-dataset) — Czech bank, 1999. 4,500 accounts, ~1M transactions, 682 loans across 8 relational tables.

**Label:** accounts with a successfully repaid loan (status A/C) = 1; accounts with no loan = 0; accounts with defaulted loans (B/D) excluded from training.

---

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. Download Berka CSVs into data/raw/ from Kaggle, then load into DuckDB
python3 scripts/load_data.py

# 3. Run dbt pipeline
cd dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir . && cd ..

# 4. Train model
python3 ml/train.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## Key results

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.862 | 0.441 |
| LightGBM + Platt calibration | **0.905** | **0.634** |

Top predictive features (global SHAP): `avg_monthly_transaction_count`, `avg_standing_order_amount`, `avg_transaction_amount`
