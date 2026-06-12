# Revolut Overdraft Propensity Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end overdraft propensity model — DuckDB + dbt feature engineering → LightGBM + SHAP → Streamlit dashboard deployed to Streamlit Community Cloud.

**Architecture:** Berka CSVs are loaded into a local DuckDB file. dbt transforms raw tables through four layers (staging → intermediate → features → mart) into an ML-ready feature table. A Python training script fits a calibrated LightGBM model with SHAP explanations and writes scored_users.csv and model_metrics.json. A Streamlit app reads those outputs and serves a targeting dashboard.

**Tech Stack:** Python 3.11, DuckDB 0.10, dbt-duckdb 1.7, LightGBM 4.3, scikit-learn 1.4, SHAP 0.45, Streamlit 1.33, Plotly 5.20

---

## File Map

```
revolut-propensity-model/
├── specs/
│   ├── model-design.md
│   └── implementation-plan.md
├── data/
│   ├── raw/                        ← Berka .asc files (gitignored)
│   └── berka.duckdb                ← generated (gitignored)
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── sources.yml
│   │   ├── staging/
│   │   │   ├── schema.yml
│   │   │   ├── stg_accounts.sql
│   │   │   ├── stg_clients.sql
│   │   │   ├── stg_transactions.sql
│   │   │   ├── stg_loans.sql
│   │   │   ├── stg_cards.sql
│   │   │   ├── stg_districts.sql
│   │   │   ├── stg_orders.sql
│   │   │   └── stg_dispositions.sql
│   │   ├── intermediate/
│   │   │   ├── schema.yml
│   │   │   ├── int_account_owner.sql
│   │   │   ├── int_account_transactions.sql
│   │   │   └── int_account_cards.sql
│   │   ├── features/
│   │   │   ├── schema.yml
│   │   │   ├── feat_transaction_aggregates.sql
│   │   │   ├── feat_account_demographics.sql
│   │   │   └── feat_account_products.sql
│   │   └── mart/
│   │       ├── schema.yml
│   │       ├── mart_loan_propensity.sql
│   │       └── mart_funnel.sql
│   └── tests/
│       ├── assert_no_bad_loans_in_mart.sql
│       └── assert_label_is_binary.sql
├── scripts/
│   └── load_data.py
├── ml/
│   └── train.py
├── dashboard/
│   └── app.py
├── outputs/                        ← committed to git for Streamlit Cloud
│   ├── scored_users.csv
│   ├── model_metrics.json
│   ├── global_shap.csv
│   └── mart_funnel.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Task 1: Project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: placeholder directories

- [ ] **Step 1: Create requirements.txt**

```
dbt-duckdb>=1.7.0
duckdb>=0.10.0
lightgbm>=4.3.0
shap>=0.45.0
scikit-learn>=1.4.0
streamlit>=1.33.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
```

- [ ] **Step 2: Create .gitignore**

```
data/raw/
data/berka.duckdb
dbt_project/target/
dbt_project/dbt_packages/
dbt_project/logs/
__pycache__/
*.pyc
.env
venv/
.venv/
.DS_Store
```

- [ ] **Step 3: Create directories and placeholders**

```bash
mkdir -p data/raw outputs scripts ml dashboard dbt_project/models/staging dbt_project/models/intermediate dbt_project/models/features dbt_project/models/mart dbt_project/tests
touch data/raw/.gitkeep outputs/.gitkeep
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore data/raw/.gitkeep outputs/.gitkeep
git commit -m "chore: project structure and dependencies"
```

---

## Task 2: Download Berka dataset

- [ ] **Step 1: Download from Kaggle**

Search Kaggle for "PKDD 1999 Berka dataset" or "berka banking dataset". Download and unzip into `data/raw/`. You need these 8 files:

```
data/raw/account.asc
data/raw/client.asc
data/raw/disp.asc
data/raw/trans.asc
data/raw/loan.asc
data/raw/card.asc
data/raw/district.asc
data/raw/order.asc
```

Files use semicolons as delimiters. Dates are stored as 6-digit integers (YYMMDD format, e.g. 930101 = 1993-01-01).

- [ ] **Step 2: Verify files exist**

```bash
ls data/raw/
```

Expected: 8 `.asc` files listed.

---

## Task 3: Data loading script

**Files:**
- Create: `scripts/load_data.py`

- [ ] **Step 1: Write load_data.py**

```python
# scripts/load_data.py
import duckdb
from pathlib import Path

DB_PATH = 'data/berka.duckdb'
RAW_PATH = 'data/raw'

TABLES = {
    'account':  'account.asc',
    'client':   'client.asc',
    'disp':     'disp.asc',
    'trans':    'trans.asc',
    'loan':     'loan.asc',
    'card':     'card.asc',
    'district': 'district.asc',
    'order':    'order.asc',
}


def load_raw_tables():
    conn = duckdb.connect(DB_PATH)
    for table_name, filename in TABLES.items():
        filepath = f"{RAW_PATH}/{filename}"
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{filepath}', delim=';', header=true)
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count:,} rows")
    conn.close()
    print("Done. Raw tables loaded into", DB_PATH)


def export_funnel(db_path=DB_PATH):
    """Run after dbt — exports mart_funnel to outputs/mart_funnel.csv."""
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM mart_funnel").df()
    conn.close()
    Path('outputs').mkdir(exist_ok=True)
    df.to_csv('outputs/mart_funnel.csv', index=False)
    print("Exported outputs/mart_funnel.csv")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'export-funnel':
        export_funnel()
    else:
        load_raw_tables()
```

- [ ] **Step 2: Run the loader**

```bash
python scripts/load_data.py
```

Expected output:
```
  account: 4,500 rows
  client: 5,369 rows
  disp: 5,369 rows
  trans: 1,056,320 rows
  loan: 682 rows
  card: 892 rows
  district: 77 rows
  order: 6,471 rows
Done. Raw tables loaded into data/berka.duckdb
```

- [ ] **Step 3: Verify in DuckDB**

```bash
python -c "
import duckdb
conn = duckdb.connect('data/berka.duckdb', read_only=True)
print(conn.execute('SHOW TABLES').df())
print(conn.execute('SELECT * FROM account LIMIT 3').df())
"
```

Expected: 8 tables shown, 3 account rows printed.

- [ ] **Step 4: Commit**

```bash
git add scripts/load_data.py
git commit -m "feat: load Berka raw tables into DuckDB"
```

---

## Task 4: dbt project initialisation

**Files:**
- Create: `dbt_project/dbt_project.yml`
- Create: `dbt_project/profiles.yml`
- Create: `dbt_project/models/sources.yml`

- [ ] **Step 1: Create dbt_project.yml**

```yaml
# dbt_project/dbt_project.yml
name: 'revolut_propensity'
version: '1.0.0'
config-version: 2

profile: 'revolut_propensity'

model-paths: ["models"]
test-paths:  ["tests"]
macro-paths: ["macros"]
target-path: "target"
clean-targets: ["target", "dbt_packages"]

models:
  revolut_propensity:
    staging:
      +materialized: view
    intermediate:
      +materialized: view
    features:
      +materialized: table
    mart:
      +materialized: table
```

- [ ] **Step 2: Create profiles.yml**

```yaml
# dbt_project/profiles.yml
revolut_propensity:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: '../data/berka.duckdb'
      threads: 4
```

- [ ] **Step 3: Create sources.yml**

```yaml
# dbt_project/models/sources.yml
version: 2

sources:
  - name: berka
    schema: main
    tables:
      - name: account
      - name: client
      - name: disp
      - name: trans
      - name: loan
      - name: card
      - name: district
      - name: order
```

- [ ] **Step 4: Verify dbt connection**

```bash
cd dbt_project && dbt debug --profiles-dir .
```

Expected: `All checks passed!` in output.

- [ ] **Step 5: Commit**

```bash
cd ..
git add dbt_project/dbt_project.yml dbt_project/profiles.yml dbt_project/models/sources.yml
git commit -m "feat: initialise dbt project with DuckDB connection"
```

---

## Task 5: Staging models

**Files:**
- Create: `dbt_project/models/staging/schema.yml`
- Create: `dbt_project/models/staging/stg_accounts.sql`
- Create: `dbt_project/models/staging/stg_clients.sql`
- Create: `dbt_project/models/staging/stg_transactions.sql`
- Create: `dbt_project/models/staging/stg_loans.sql`
- Create: `dbt_project/models/staging/stg_cards.sql`
- Create: `dbt_project/models/staging/stg_districts.sql`
- Create: `dbt_project/models/staging/stg_orders.sql`
- Create: `dbt_project/models/staging/stg_dispositions.sql`

- [ ] **Step 1: Write schema.yml (tests declared before models exist — TDD)**

```yaml
# dbt_project/models/staging/schema.yml
version: 2

models:
  - name: stg_accounts
    columns:
      - name: account_id
        tests: [not_null, unique]
      - name: district_id
        tests: [not_null]
      - name: opened_date
        tests: [not_null]

  - name: stg_clients
    columns:
      - name: client_id
        tests: [not_null, unique]

  - name: stg_transactions
    columns:
      - name: trans_id
        tests: [not_null, unique]
      - name: account_id
        tests: [not_null]
      - name: amount
        tests: [not_null]

  - name: stg_loans
    columns:
      - name: loan_id
        tests: [not_null, unique]
      - name: account_id
        tests: [not_null]
      - name: loan_status
        tests:
          - accepted_values:
              values: ['A', 'B', 'C', 'D']

  - name: stg_cards
    columns:
      - name: card_id
        tests: [not_null, unique]

  - name: stg_districts
    columns:
      - name: district_id
        tests: [not_null, unique]

  - name: stg_orders
    columns:
      - name: order_id
        tests: [not_null, unique]

  - name: stg_dispositions
    columns:
      - name: disp_id
        tests: [not_null, unique]
      - name: disp_type
        tests:
          - accepted_values:
              values: ['OWNER', 'DISPONENT']
```

- [ ] **Step 2: Run tests — confirm they fail (models don't exist yet)**

```bash
cd dbt_project && dbt test --select staging --profiles-dir .
```

Expected: errors saying relations do not exist. This is correct — tests are wired before code exists.

- [ ] **Step 3: Write stg_accounts.sql**

```sql
-- dbt_project/models/staging/stg_accounts.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'account') }}
)
SELECT
    account_id,
    district_id,
    frequency AS statement_frequency,
    STRPTIME(LPAD(CAST(date AS VARCHAR), 6, '0'), '%y%m%d') AS opened_date
FROM source
```

- [ ] **Step 4: Write stg_clients.sql**

```sql
-- dbt_project/models/staging/stg_clients.sql
-- birth_number encodes gender: month > 50 = female (subtract 50 for real month).
WITH source AS (
    SELECT * FROM {{ source('berka', 'client') }}
),
parsed AS (
    SELECT
        client_id,
        district_id,
        CASE
            WHEN CAST(SUBSTR(LPAD(CAST(birth_number AS VARCHAR), 6, '0'), 3, 2) AS INTEGER) > 50
            THEN 'F' ELSE 'M'
        END AS gender,
        1900 + CAST(SUBSTR(LPAD(CAST(birth_number AS VARCHAR), 6, '0'), 1, 2) AS INTEGER) AS birth_year
    FROM source
)
SELECT
    client_id,
    district_id,
    gender,
    1998 - birth_year AS client_age    -- age relative to dataset end ~1998
FROM parsed
```

- [ ] **Step 5: Write stg_transactions.sql**

```sql
-- dbt_project/models/staging/stg_transactions.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'trans') }}
)
SELECT
    trans_id,
    account_id,
    STRPTIME(LPAD(CAST(date AS VARCHAR), 6, '0'), '%y%m%d') AS trans_date,
    type    AS trans_type,           -- PRIJEM=credit, VYDAJ=debit, VYBER=cash withdrawal
    operation,
    amount,
    balance,
    k_symbol,
    bank    AS counterpart_bank,
    account AS counterpart_account   -- renamed: 'account' conflicts with table name
FROM source
```

- [ ] **Step 6: Write stg_loans.sql**

```sql
-- dbt_project/models/staging/stg_loans.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'loan') }}
)
SELECT
    loan_id,
    account_id,
    STRPTIME(LPAD(CAST(date AS VARCHAR), 6, '0'), '%y%m%d') AS loan_date,
    amount   AS loan_amount,
    duration AS loan_duration_months,
    payments AS monthly_payment,
    status   AS loan_status    -- A: finished OK, B: finished unpaid, C: running OK, D: in debt
FROM source
```

- [ ] **Step 7: Write stg_cards.sql**

```sql
-- dbt_project/models/staging/stg_cards.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'card') }}
)
SELECT
    card_id,
    disp_id,
    type AS card_type,
    -- issued stored as 'YYMMDD HH:MM:SS' — extract date portion only
    STRPTIME(SUBSTR(REPLACE(CAST(issued AS VARCHAR), ' ', ''), 1, 6), '%y%m%d') AS issued_date
FROM source
```

- [ ] **Step 8: Write stg_districts.sql**

```sql
-- dbt_project/models/staging/stg_districts.sql
-- Raw columns are A1-A16. Some values are '?' (missing) — TRY_CAST returns NULL for these.
WITH source AS (
    SELECT * FROM {{ source('berka', 'district') }}
)
SELECT
    A1                        AS district_id,
    A2                        AS district_name,
    A3                        AS region,
    TRY_CAST(A4  AS INTEGER)  AS population,
    TRY_CAST(A10 AS FLOAT)    AS urban_ratio,
    TRY_CAST(A11 AS FLOAT)    AS avg_salary,
    TRY_CAST(A13 AS FLOAT)    AS unemployment_rate,
    TRY_CAST(A14 AS INTEGER)  AS entrepreneurs_per_1000,
    TRY_CAST(A16 AS INTEGER)  AS crimes_count
FROM source
```

- [ ] **Step 9: Write stg_orders.sql**

```sql
-- dbt_project/models/staging/stg_orders.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'order') }}
)
SELECT
    order_id,
    account_id,
    bank_to,
    account_to,
    amount   AS order_amount,
    k_symbol AS order_type
FROM source
```

- [ ] **Step 10: Write stg_dispositions.sql**

```sql
-- dbt_project/models/staging/stg_dispositions.sql
WITH source AS (
    SELECT * FROM {{ source('berka', 'disp') }}
)
SELECT
    disp_id,
    client_id,
    account_id,
    type AS disp_type    -- OWNER or DISPONENT
FROM source
```

- [ ] **Step 11: Run staging models**

```bash
cd dbt_project && dbt run --select staging --profiles-dir .
```

Expected: `8 of 8 OK`

- [ ] **Step 12: Run staging tests**

```bash
dbt test --select staging --profiles-dir .
```

Expected: all tests pass. If `accepted_values` on `loan_status` fails, open `data/raw/loan.asc` and check whether status values are uppercase or lowercase — update the test values accordingly.

- [ ] **Step 13: Commit**

```bash
cd ..
git add dbt_project/models/staging/
git commit -m "feat: staging dbt models with schema tests"
```

---

## Task 6: Intermediate models

**Files:**
- Create: `dbt_project/models/intermediate/schema.yml`
- Create: `dbt_project/models/intermediate/int_account_owner.sql`
- Create: `dbt_project/models/intermediate/int_account_transactions.sql`
- Create: `dbt_project/models/intermediate/int_account_cards.sql`

- [ ] **Step 1: Write schema.yml**

```yaml
# dbt_project/models/intermediate/schema.yml
version: 2

models:
  - name: int_account_owner
    columns:
      - name: account_id
        tests: [not_null, unique]

  - name: int_account_transactions
    columns:
      - name: trans_id
        tests: [not_null]
      - name: account_id
        tests: [not_null]

  - name: int_account_cards
    columns:
      - name: account_id
        tests: [not_null, unique]
```

- [ ] **Step 2: Write int_account_owner.sql**

```sql
-- dbt_project/models/intermediate/int_account_owner.sql
-- One row per account. Joins account → OWNER client → district.
WITH accounts AS (
    SELECT * FROM {{ ref('stg_accounts') }}
),
owners AS (
    SELECT client_id, account_id
    FROM {{ ref('stg_dispositions') }}
    WHERE disp_type = 'OWNER'
),
clients  AS (SELECT * FROM {{ ref('stg_clients') }}),
districts AS (SELECT * FROM {{ ref('stg_districts') }})

SELECT
    a.account_id,
    a.district_id,
    a.statement_frequency,
    a.opened_date,
    c.client_id,
    c.gender,
    c.client_age,
    d.avg_salary         AS district_avg_salary,
    d.unemployment_rate  AS district_unemployment_rate,
    d.crimes_count       AS district_crimes_count,
    d.urban_ratio        AS district_urban_ratio
FROM accounts a
LEFT JOIN owners    o ON a.account_id  = o.account_id
LEFT JOIN clients   c ON o.client_id   = c.client_id
LEFT JOIN districts d ON a.district_id = d.district_id
```

- [ ] **Step 3: Write int_account_transactions.sql**

```sql
-- dbt_project/models/intermediate/int_account_transactions.sql
-- Transactions enriched with account opening date (needed for trend window).
SELECT
    t.trans_id,
    t.account_id,
    t.trans_date,
    t.trans_type,
    t.amount,
    t.balance,
    t.k_symbol,
    a.opened_date
FROM {{ ref('stg_transactions') }} t
LEFT JOIN {{ ref('stg_accounts') }} a ON t.account_id = a.account_id
```

- [ ] **Step 4: Write int_account_cards.sql**

```sql
-- dbt_project/models/intermediate/int_account_cards.sql
-- One row per account (most recently issued card for the OWNER).
WITH ranked AS (
    SELECT
        d.account_id,
        c.card_type,
        c.issued_date,
        ROW_NUMBER() OVER (
            PARTITION BY d.account_id
            ORDER BY c.issued_date DESC
        ) AS rn
    FROM {{ ref('stg_cards') }} c
    LEFT JOIN {{ ref('stg_dispositions') }} d ON c.disp_id = d.disp_id
    WHERE d.disp_type = 'OWNER'
)
SELECT
    account_id,
    card_type,
    issued_date,
    TRUE AS has_card
FROM ranked
WHERE rn = 1
```

- [ ] **Step 5: Run and test**

```bash
cd dbt_project
dbt run --select intermediate --profiles-dir .
dbt test --select intermediate --profiles-dir .
```

Expected: `3 of 3 OK`, all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ..
git add dbt_project/models/intermediate/
git commit -m "feat: intermediate dbt models"
```

---

## Task 7: Feature models

**Files:**
- Create: `dbt_project/models/features/schema.yml`
- Create: `dbt_project/models/features/feat_transaction_aggregates.sql`
- Create: `dbt_project/models/features/feat_account_demographics.sql`
- Create: `dbt_project/models/features/feat_account_products.sql`

- [ ] **Step 1: Write schema.yml**

```yaml
# dbt_project/models/features/schema.yml
version: 2

models:
  - name: feat_transaction_aggregates
    columns:
      - name: account_id
        tests: [not_null, unique]
      - name: avg_balance
        tests: [not_null]
      - name: account_age_days
        tests: [not_null]

  - name: feat_account_demographics
    columns:
      - name: account_id
        tests: [not_null, unique]

  - name: feat_account_products
    columns:
      - name: account_id
        tests: [not_null, unique]
```

- [ ] **Step 2: Write feat_transaction_aggregates.sql**

```sql
-- dbt_project/models/features/feat_transaction_aggregates.sql
-- Behavioural features from transaction history. One row per account.
WITH t AS (
    SELECT * FROM {{ ref('int_account_transactions') }}
),
max_dates AS (
    SELECT account_id, MAX(trans_date) AS max_trans_date
    FROM t GROUP BY account_id
),
base AS (
    SELECT
        t.account_id,
        COUNT(*)                                                                    AS total_transaction_count,
        DATE_DIFF('day', MIN(t.opened_date), MAX(t.trans_date))                    AS account_age_days,
        CAST(COUNT(*) AS FLOAT)
            / NULLIF(DATE_DIFF('month', MIN(t.opened_date), MAX(t.trans_date)), 0) AS avg_monthly_transaction_count,
        AVG(t.amount)                                                               AS avg_transaction_amount,
        SUM(CASE WHEN t.trans_type = 'PRIJEM' THEN t.amount ELSE 0 END)            AS total_credit_amount,
        SUM(CASE WHEN t.trans_type IN ('VYDAJ','VYBER') THEN t.amount ELSE 0 END)  AS total_debit_amount,
        AVG(t.balance)                                                              AS avg_balance,
        STDDEV(t.balance)                                                           AS balance_volatility,
        MIN(t.balance)                                                              AS min_balance_ever,
        DATE_DIFF('day', MAX(t.trans_date),
            (SELECT MAX(trans_date) FROM t))                                        AS days_since_last_transaction
    FROM t
    GROUP BY t.account_id
),
trend AS (
    -- Date-based windows (not row-based) to ensure consistent 90-day periods
    SELECT
        t.account_id,
        AVG(CASE
            WHEN t.trans_date >= m.max_trans_date - INTERVAL '90 days'
            THEN t.amount END)                                                      AS avg_amount_last_90d,
        AVG(CASE
            WHEN t.trans_date >= m.max_trans_date - INTERVAL '180 days'
             AND t.trans_date <  m.max_trans_date - INTERVAL '90 days'
            THEN t.amount END)                                                      AS avg_amount_prior_90d
    FROM t
    JOIN max_dates m ON t.account_id = m.account_id
    GROUP BY t.account_id
)

SELECT
    b.account_id,
    b.total_transaction_count,
    b.account_age_days,
    b.avg_monthly_transaction_count,
    b.avg_transaction_amount,
    b.total_credit_amount,
    b.total_debit_amount,
    b.total_credit_amount / NULLIF(b.total_debit_amount, 0)        AS credit_debit_ratio,
    b.avg_balance,
    b.balance_volatility,
    b.min_balance_ever,
    b.days_since_last_transaction,
    tr.avg_amount_last_90d / NULLIF(tr.avg_amount_prior_90d, 0)   AS transaction_trend
FROM base b
LEFT JOIN trend tr ON b.account_id = tr.account_id
```

- [ ] **Step 3: Write feat_account_demographics.sql**

```sql
-- dbt_project/models/features/feat_account_demographics.sql
SELECT
    account_id,
    gender,
    client_age,
    statement_frequency,
    district_avg_salary,
    district_unemployment_rate,
    district_crimes_count,
    district_urban_ratio,
    opened_date
FROM {{ ref('int_account_owner') }}
```

- [ ] **Step 4: Write feat_account_products.sql**

```sql
-- dbt_project/models/features/feat_account_products.sql
WITH all_accounts AS (
    SELECT DISTINCT account_id FROM {{ ref('stg_accounts') }}
),
cards AS (
    SELECT * FROM {{ ref('int_account_cards') }}
),
order_aggs AS (
    SELECT
        account_id,
        COUNT(*)            AS standing_order_count,
        AVG(order_amount)   AS avg_standing_order_amount
    FROM {{ ref('stg_orders') }}
    GROUP BY account_id
)

SELECT
    a.account_id,
    COALESCE(c.has_card, FALSE)                              AS has_card,
    COALESCE(c.card_type, 'none')                            AS card_type,
    COALESCE(o.standing_order_count, 0) > 0                  AS has_standing_order,
    COALESCE(o.standing_order_count, 0)                      AS standing_order_count,
    COALESCE(o.avg_standing_order_amount, 0)                 AS avg_standing_order_amount
FROM all_accounts a
LEFT JOIN cards      c ON a.account_id = c.account_id
LEFT JOIN order_aggs o ON a.account_id = o.account_id
```

- [ ] **Step 5: Run and test**

```bash
cd dbt_project
dbt run --select features --profiles-dir .
dbt test --select features --profiles-dir .
```

Expected: `3 of 3 OK`, all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ..
git add dbt_project/models/features/
git commit -m "feat: feature engineering dbt models with transaction trend window"
```

---

## Task 8: Mart models and custom tests

**Files:**
- Create: `dbt_project/models/mart/schema.yml`
- Create: `dbt_project/models/mart/mart_loan_propensity.sql`
- Create: `dbt_project/models/mart/mart_funnel.sql`
- Create: `dbt_project/tests/assert_no_bad_loans_in_mart.sql`
- Create: `dbt_project/tests/assert_label_is_binary.sql`

- [ ] **Step 1: Write custom tests (written before mart models exist)**

```sql
-- dbt_project/tests/assert_no_bad_loans_in_mart.sql
-- Returns rows if any bad-loan account (B/D) appears in the mart.
-- dbt passes this test when it returns 0 rows.
SELECT m.account_id
FROM {{ ref('mart_loan_propensity') }} m
JOIN {{ source('berka', 'loan') }} l ON m.account_id = l.account_id
WHERE l.status IN ('B', 'D')
```

```sql
-- dbt_project/tests/assert_label_is_binary.sql
-- Returns rows if label is anything other than 0 or 1.
SELECT account_id
FROM mart_loan_propensity
WHERE adopted_credit_product NOT IN (0, 1)
   OR adopted_credit_product IS NULL
```

- [ ] **Step 2: Write schema.yml**

```yaml
# dbt_project/models/mart/schema.yml
version: 2

models:
  - name: mart_loan_propensity
    columns:
      - name: account_id
        tests: [not_null, unique]
      - name: adopted_credit_product
        tests: [not_null]

  - name: mart_funnel
    description: "Single-row aggregate showing funnel stage counts."
```

- [ ] **Step 3: Write mart_loan_propensity.sql**

```sql
-- dbt_project/models/mart/mart_loan_propensity.sql
-- ML-ready table. One row per eligible account.
-- Accounts with bad loan history (B/D) are excluded entirely.
WITH loan_labels AS (
    SELECT
        account_id,
        CASE
            WHEN loan_status IN ('A', 'C') THEN 1
            WHEN loan_status IN ('B', 'D') THEN -1    -- sentinel for exclusion
        END AS label_raw
    FROM {{ ref('stg_loans') }}
)

SELECT
    ta.account_id,
    ta.total_transaction_count,
    ta.account_age_days,
    ta.avg_monthly_transaction_count,
    ta.avg_transaction_amount,
    ta.total_credit_amount,
    ta.total_debit_amount,
    ta.credit_debit_ratio,
    ta.avg_balance,
    ta.balance_volatility,
    ta.min_balance_ever,
    ta.days_since_last_transaction,
    ta.transaction_trend,
    dem.gender,
    dem.client_age,
    dem.statement_frequency,
    dem.district_avg_salary,
    dem.district_unemployment_rate,
    dem.district_crimes_count,
    dem.district_urban_ratio,
    prod.has_card,
    prod.card_type,
    prod.has_standing_order,
    prod.standing_order_count,
    prod.avg_standing_order_amount,
    COALESCE(l.label_raw, 0) AS adopted_credit_product
FROM {{ ref('feat_transaction_aggregates') }} ta
LEFT JOIN {{ ref('feat_account_demographics') }} dem  ON ta.account_id = dem.account_id
LEFT JOIN {{ ref('feat_account_products') }}     prod ON ta.account_id = prod.account_id
LEFT JOIN loan_labels l                               ON ta.account_id = l.account_id
WHERE COALESCE(l.label_raw, 0) != -1    -- drop accounts with bad loan history
```

- [ ] **Step 4: Write mart_funnel.sql**

```sql
-- dbt_project/models/mart/mart_funnel.sql
-- Single-row aggregate for the dashboard funnel panel.
WITH accounts AS (
    SELECT account_id, opened_date FROM {{ ref('stg_accounts') }}
),
early_activity AS (
    SELECT t.account_id
    FROM {{ ref('stg_transactions') }} t
    JOIN accounts a ON t.account_id = a.account_id
    WHERE t.trans_date <= a.opened_date + INTERVAL '90 days'
    GROUP BY t.account_id
    HAVING COUNT(*) >= 3
),
bad_loans AS (
    SELECT DISTINCT account_id FROM {{ ref('stg_loans') }}
    WHERE loan_status IN ('B', 'D')
),
adopters AS (
    SELECT DISTINCT account_id FROM {{ ref('stg_loans') }}
    WHERE loan_status IN ('A', 'C')
)

SELECT
    COUNT(DISTINCT a.account_id)                                             AS accounts_opened,
    COUNT(DISTINCT ea.account_id)                                            AS active_accounts,
    COUNT(DISTINCT CASE
        WHEN ea.account_id IS NOT NULL AND bl.account_id IS NULL
        THEN a.account_id END)                                               AS eligible_accounts,
    COUNT(DISTINCT adp.account_id)                                           AS adopted_accounts
FROM accounts a
LEFT JOIN early_activity ea  ON a.account_id = ea.account_id
LEFT JOIN bad_loans      bl  ON a.account_id = bl.account_id
LEFT JOIN adopters       adp ON a.account_id = adp.account_id
```

- [ ] **Step 5: Run mart models**

```bash
cd dbt_project && dbt run --select mart --profiles-dir .
```

Expected: `2 of 2 OK`

- [ ] **Step 6: Run all tests including custom tests**

```bash
dbt test --profiles-dir .
```

Expected: all tests pass including `assert_no_bad_loans_in_mart` and `assert_label_is_binary`.

- [ ] **Step 7: Export mart_funnel.csv**

```bash
cd .. && python scripts/load_data.py export-funnel
```

Expected: `Exported outputs/mart_funnel.csv`

- [ ] **Step 8: Commit**

```bash
git add dbt_project/models/mart/ dbt_project/tests/ outputs/mart_funnel.csv
git commit -m "feat: mart models, custom dbt tests, funnel export"
```

---

## Task 9: ML training script

**Files:**
- Create: `ml/train.py`

- [ ] **Step 1: Write train.py**

```python
# ml/train.py
import duckdb
import pandas as pd
import numpy as np
import json
import shap
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET      = 'adopted_credit_product'
CATEGORICAL = ['gender', 'statement_frequency', 'card_type']
NUMERICAL   = [
    'total_transaction_count', 'account_age_days', 'avg_monthly_transaction_count',
    'avg_transaction_amount', 'total_credit_amount', 'total_debit_amount',
    'credit_debit_ratio', 'avg_balance', 'balance_volatility', 'min_balance_ever',
    'days_since_last_transaction', 'transaction_trend', 'client_age',
    'district_avg_salary', 'district_unemployment_rate', 'district_crimes_count',
    'district_urban_ratio', 'standing_order_count', 'avg_standing_order_amount',
]
BINARY      = ['has_card', 'has_standing_order']
ALL_FEATURES = NUMERICAL + CATEGORICAL + BINARY

# ── 1. Load data ──────────────────────────────────────────────────────────────
conn = duckdb.connect('data/berka.duckdb', read_only=True)
df   = conn.execute("SELECT * FROM mart_loan_propensity").df()
conn.close()

print(f"Dataset: {len(df):,} rows | {df[TARGET].mean():.1%} positive rate")

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Fill nulls in numerical columns with median (transaction_trend can be null)
X[NUMERICAL] = X[NUMERICAL].fillna(X[NUMERICAL].median())

# ── 2. Three-way split: 60% train / 20% calibration / 20% test ───────────────
X_tmp,  X_test, y_tmp,  y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp
)
print(f"Split → train:{len(X_train)} cal:{len(X_cal)} test:{len(X_test)}")

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())

# ── 3. Logistic Regression baseline ──────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),
     NUMERICAL + BINARY),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     CATEGORICAL),
])
lr_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
])
lr_pipeline.fit(X_train, y_train)
lr_proba   = lr_pipeline.predict_proba(X_test)[:, 1]
lr_auc     = roc_auc_score(y_test, lr_proba)
lr_pr_auc  = average_precision_score(y_test, lr_proba)
print(f"LR    AUC-ROC={lr_auc:.4f}  PR-AUC={lr_pr_auc:.4f}")

# ── 4. LightGBM ───────────────────────────────────────────────────────────────
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=neg_count / pos_count,
    random_state=42,
    verbose=-1,
)
lgbm.fit(X_train, y_train, categorical_feature=CATEGORICAL)

# ── 5. Platt calibration (calibration set, not test set — avoids leakage) ────
calibrated = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
calibrated.fit(X_cal, y_cal)

lgbm_proba  = calibrated.predict_proba(X_test)[:, 1]
lgbm_auc    = roc_auc_score(y_test, lgbm_proba)
lgbm_pr_auc = average_precision_score(y_test, lgbm_proba)
print(f"LGBM  AUC-ROC={lgbm_auc:.4f}  PR-AUC={lgbm_pr_auc:.4f}")

# ── 6. SHAP on full dataset ───────────────────────────────────────────────────
explainer   = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X)    # shape: (n_accounts, n_features)

global_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=ALL_FEATURES,
).sort_values(ascending=False)

# ── 7. Score all accounts ─────────────────────────────────────────────────────
all_scores = calibrated.predict_proba(X)[:, 1]
top3_idx   = np.argsort(np.abs(shap_values), axis=1)[:, -3:][:, ::-1]

scored_users = pd.DataFrame({
    'account_id':   df['account_id'].values,
    'score':        all_scores.round(4),
    'segment':      pd.cut(
                        all_scores,
                        bins=[-0.001, 0.40, 0.70, 1.001],
                        labels=['Low', 'Medium', 'High'],
                    ),
    'shap_1_name':  [ALL_FEATURES[i[0]] for i in top3_idx],
    'shap_1_value': [round(float(shap_values[r, i[0]]), 4) for r, i in enumerate(top3_idx)],
    'shap_2_name':  [ALL_FEATURES[i[1]] for i in top3_idx],
    'shap_2_value': [round(float(shap_values[r, i[1]]), 4) for r, i in enumerate(top3_idx)],
    'shap_3_name':  [ALL_FEATURES[i[2]] for i in top3_idx],
    'shap_3_value': [round(float(shap_values[r, i[2]]), 4) for r, i in enumerate(top3_idx)],
})

# ── 8. Sanity checks ──────────────────────────────────────────────────────────
assert scored_users['score'].between(0, 1).all(), \
    "Scores outside [0, 1]"
assert not scored_users['score'].isna().any(), \
    "NaN scores found in output"
assert lgbm_auc > lr_auc, \
    f"LightGBM AUC ({lgbm_auc:.4f}) did not beat LR ({lr_auc:.4f})"
assert lgbm_auc > 0.65, \
    f"LightGBM AUC {lgbm_auc:.4f} below 0.65 minimum"
mean_pred   = float(lgbm_proba.mean())
actual_rate = float(y_test.mean())
assert abs(mean_pred - actual_rate) < 0.10, \
    f"Calibration gap: predicted {mean_pred:.3f} vs actual {actual_rate:.3f}"
print("All sanity checks passed.")

# ── 9. Save outputs ───────────────────────────────────────────────────────────
Path('outputs').mkdir(exist_ok=True)

scored_users.to_csv('outputs/scored_users.csv', index=False)

metrics = {
    'lr_auc_roc':   round(lr_auc,      4),
    'lr_pr_auc':    round(lr_pr_auc,   4),
    'lgbm_auc_roc': round(lgbm_auc,    4),
    'lgbm_pr_auc':  round(lgbm_pr_auc, 4),
}
with open('outputs/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

global_shap.to_csv('outputs/global_shap.csv', header=['mean_abs_shap'])

print(f"Saved outputs/scored_users.csv — {len(scored_users):,} rows")
print(f"Saved outputs/model_metrics.json — {metrics}")
print(f"Saved outputs/global_shap.csv — top feature: {global_shap.index[0]}")
```

- [ ] **Step 2: Run training script**

```bash
python ml/train.py
```

Expected (exact numbers vary):
```
Dataset: 4,182 rows | 16.3% positive rate
Split → train:2509 cal:837 test:836
LR    AUC-ROC=0.7XXX  PR-AUC=0.XXXX
LGBM  AUC-ROC=0.7XXX  PR-AUC=0.XXXX
All sanity checks passed.
Saved outputs/scored_users.csv — 4,182 rows
```

If a sanity check fires, read the assertion message and investigate before proceeding.

- [ ] **Step 3: Verify outputs**

```bash
python -c "
import pandas as pd, json
df = pd.read_csv('outputs/scored_users.csv')
print(df.head(3).to_string())
print(df['segment'].value_counts().to_string())
print(json.load(open('outputs/model_metrics.json')))
"
```

- [ ] **Step 4: Commit**

```bash
git add ml/train.py outputs/scored_users.csv outputs/model_metrics.json outputs/global_shap.csv
git commit -m "feat: LightGBM + Platt calibration + SHAP training script"
```

---

## Task 10: Streamlit dashboard

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: Write app.py**

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(
    page_title="Overdraft Propensity — Targeting Dashboard",
    layout="wide",
)

@st.cache_data
def load_data():
    scored   = pd.read_csv('outputs/scored_users.csv')
    funnel   = pd.read_csv('outputs/mart_funnel.csv')
    shap_imp = pd.read_csv('outputs/global_shap.csv',
                            names=['feature', 'mean_abs_shap'], header=0)
    with open('outputs/model_metrics.json') as f:
        metrics = json.load(f)
    return scored, funnel, shap_imp, metrics

scored, funnel, shap_imp, metrics = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Overdraft Propensity — Growth Targeting Dashboard")
st.caption(
    "Which users should receive a pre-approved overdraft offer in-app? "
    "Powered by a calibrated LightGBM model trained on historical banking behaviour."
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total users scored",       f"{len(scored):,}")
c2.metric("High propensity (≥ 0.70)", f"{(scored['score'] >= 0.70).sum():,}")
c3.metric("LightGBM AUC-ROC",         f"{metrics['lgbm_auc_roc']:.3f}")
c4.metric("LightGBM PR-AUC",          f"{metrics['lgbm_pr_auc']:.3f}")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    seg_filter = st.multiselect(
        "Segment", options=['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low'],
    )
    score_min = st.slider(
        "Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

# ── Primary panel: targeting list ─────────────────────────────────────────────
st.subheader("Targeting List")

filtered = (
    scored[scored['segment'].isin(seg_filter) & (scored['score'] >= score_min)]
    .sort_values('score', ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    filtered[[
        'account_id', 'score', 'segment',
        'shap_1_name', 'shap_1_value',
        'shap_2_name', 'shap_2_value',
        'shap_3_name', 'shap_3_value',
    ]].rename(columns={
        'account_id':   'Account ID',
        'score':        'Propensity Score',
        'segment':      'Segment',
        'shap_1_name':  'Driver 1',  'shap_1_value': 'Impact 1',
        'shap_2_name':  'Driver 2',  'shap_2_value': 'Impact 2',
        'shap_3_name':  'Driver 3',  'shap_3_value': 'Impact 3',
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Export targeting list as CSV",
    data=filtered.to_csv(index=False),
    file_name="overdraft_targets.csv",
    mime="text/csv",
)

st.divider()

# ── Secondary panel: funnel ───────────────────────────────────────────────────
st.subheader("User Funnel")

funnel_rows = pd.DataFrame({
    'Stage': ['Accounts opened', 'Active accounts',
              'Eligible accounts', 'Adopted credit product'],
    'Count': [
        int(funnel['accounts_opened'].iloc[0]),
        int(funnel['active_accounts'].iloc[0]),
        int(funnel['eligible_accounts'].iloc[0]),
        int(funnel['adopted_accounts'].iloc[0]),
    ],
})

fig_funnel = px.bar(
    funnel_rows, x='Count', y='Stage', orientation='h',
    text='Count', color='Count',
    color_continuous_scale='Blues', height=260,
)
fig_funnel.update_layout(showlegend=False, coloraxis_showscale=False)
fig_funnel.update_traces(textposition='outside')
st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# ── Model explainability ──────────────────────────────────────────────────────
st.subheader("Model Explainability — Global Feature Importance")

top10 = shap_imp.head(10)

fig_shap = px.bar(
    top10, x='mean_abs_shap', y='feature', orientation='h',
    color='mean_abs_shap', color_continuous_scale='Blues',
    labels={'mean_abs_shap': 'Mean |SHAP|', 'feature': 'Feature'},
    height=350,
)
fig_shap.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)
```

- [ ] **Step 2: Run dashboard locally**

```bash
streamlit run dashboard/app.py
```

Expected: browser opens at `http://localhost:8501`.

- [ ] **Step 3: Verify all panels in browser**

- KPI cards show non-zero values
- Targeting list populates and responds to sidebar filters
- Export button downloads `overdraft_targets.csv`
- Funnel chart renders with 4 bars
- SHAP chart renders with 10 features

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit targeting dashboard"
```

---

## Task 11: Deployment

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Revolut Overdraft Propensity Model

Portfolio project for Revolut DS/Analyst internship application.

**Business question:** Which existing current-account users should receive a
pre-approved overdraft offer in-app?

**Live dashboard:** [add URL after deployment]

## Stack
DuckDB + dbt (SQL feature engineering) · LightGBM + SHAP (calibrated propensity model) · Streamlit (targeting dashboard)

## Running locally

1. `pip install -r requirements.txt`
2. Place Berka `.asc` files in `data/raw/`
3. `python scripts/load_data.py`
4. `cd dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir . && cd ..`
5. `python scripts/load_data.py export-funnel`
6. `python ml/train.py`
7. `streamlit run dashboard/app.py`
```

- [ ] **Step 2: Push to GitHub**

Create a new public repo at github.com named `revolut-propensity-model`, then:

```bash
git remote add origin https://github.com/<your-username>/revolut-propensity-model.git
git push -u origin main
```

- [ ] **Step 3: Deploy to Streamlit Community Cloud**

1. Go to share.streamlit.io
2. Sign in with GitHub
3. New app → repository: `revolut-propensity-model` → branch: `main` → main file: `dashboard/app.py`
4. Click Deploy

Expected: live URL in the form `https://<username>-revolut-propensity-model-....streamlit.app`

- [ ] **Step 4: Update README with live URL and push**

```bash
# Edit README.md: replace [add URL after deployment] with the live Streamlit URL
git add README.md
git commit -m "docs: add live dashboard URL"
git push
```

---

## Full pipeline run order

```bash
python scripts/load_data.py                                  # load raw CSVs into DuckDB
cd dbt_project && dbt run --profiles-dir . && cd ..          # build all dbt models
cd dbt_project && dbt test --profiles-dir . && cd ..         # run all dbt tests
python scripts/load_data.py export-funnel                    # export mart_funnel.csv
python ml/train.py                                           # train model, write outputs
streamlit run dashboard/app.py                               # run dashboard locally
```
