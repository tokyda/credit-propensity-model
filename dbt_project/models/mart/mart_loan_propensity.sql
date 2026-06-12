-- ML-ready table. One row per eligible account.
-- Accounts with bad loan history (B/D) are excluded entirely.
WITH loan_labels AS (
    SELECT
        account_id,
        CASE
            WHEN loan_status IN ('A', 'C') THEN 1
            WHEN loan_status IN ('B', 'D') THEN -1
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
WHERE COALESCE(l.label_raw, 0) != -1
