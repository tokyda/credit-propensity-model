WITH source AS (
    SELECT * FROM {{ source('berka', 'card') }}
)
SELECT
    card_id,
    disp_id,
    type AS card_type,
    STRPTIME(SUBSTR(REPLACE(CAST(issued AS VARCHAR), ' ', ''), 1, 6), '%y%m%d') AS issued_date
FROM source
