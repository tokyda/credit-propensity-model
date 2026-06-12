-- Returns rows if any bad-loan account (B/D) appears in the mart.
-- dbt passes this test when it returns 0 rows.
SELECT m.account_id
FROM {{ ref('mart_loan_propensity') }} m
JOIN {{ source('berka', 'loan') }} l ON m.account_id = l.account_id
WHERE l.status IN ('B', 'D')
