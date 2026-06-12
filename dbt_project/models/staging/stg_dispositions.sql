WITH source AS (
    SELECT * FROM {{ source('berka', 'disp') }}
)
SELECT
    disp_id,
    client_id,
    account_id,
    type AS disp_type
FROM source
