WITH source AS (
    SELECT * FROM {{ source('berka', 'account') }}
)
SELECT
    account_id,
    district_id,
    frequency AS statement_frequency,
    STRPTIME(LPAD(CAST(date AS VARCHAR), 6, '0'), '%y%m%d') AS opened_date
FROM source
