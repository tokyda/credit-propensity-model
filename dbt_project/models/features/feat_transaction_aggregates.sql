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
    b.total_credit_amount / NULLIF(b.total_debit_amount, 0)       AS credit_debit_ratio,
    b.avg_balance,
    b.balance_volatility,
    b.min_balance_ever,
    b.days_since_last_transaction,
    tr.avg_amount_last_90d / NULLIF(tr.avg_amount_prior_90d, 0)   AS transaction_trend
FROM base b
LEFT JOIN trend tr ON b.account_id = tr.account_id
