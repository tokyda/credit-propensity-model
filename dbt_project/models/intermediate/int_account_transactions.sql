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
