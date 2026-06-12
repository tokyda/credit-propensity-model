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
    status   AS loan_status
FROM source
