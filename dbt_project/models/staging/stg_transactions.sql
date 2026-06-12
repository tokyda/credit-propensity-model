WITH source AS (
    SELECT * FROM {{ source('berka', 'trans') }}
)
SELECT
    trans_id,
    account_id,
    STRPTIME(LPAD(CAST(date AS VARCHAR), 6, '0'), '%y%m%d') AS trans_date,
    type      AS trans_type,
    operation,
    amount,
    balance,
    k_symbol,
    bank      AS counterpart_bank,
    account   AS counterpart_account
FROM source
