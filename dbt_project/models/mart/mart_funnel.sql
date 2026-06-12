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
    COUNT(DISTINCT a.account_id)                                       AS accounts_opened,
    COUNT(DISTINCT ea.account_id)                                      AS active_accounts,
    COUNT(DISTINCT CASE
        WHEN ea.account_id IS NOT NULL AND bl.account_id IS NULL
        THEN a.account_id END)                                         AS eligible_accounts,
    COUNT(DISTINCT adp.account_id)                                     AS adopted_accounts
FROM accounts a
LEFT JOIN early_activity ea  ON a.account_id = ea.account_id
LEFT JOIN bad_loans      bl  ON a.account_id = bl.account_id
LEFT JOIN adopters       adp ON a.account_id = adp.account_id
