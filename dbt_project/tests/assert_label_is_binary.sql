-- Returns rows if label is anything other than 0 or 1.
SELECT account_id
FROM {{ ref('mart_loan_propensity') }}
WHERE adopted_credit_product NOT IN (0, 1)
   OR adopted_credit_product IS NULL
