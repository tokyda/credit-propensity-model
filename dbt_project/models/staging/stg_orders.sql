WITH source AS (
    SELECT * FROM {{ source('berka', 'order_') }}
)
SELECT
    order_id,
    account_id,
    bank_to,
    account_to,
    amount   AS order_amount,
    k_symbol AS order_type
FROM source
