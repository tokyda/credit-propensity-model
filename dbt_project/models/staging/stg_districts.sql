-- Raw columns are A1-A16; some values are '?' (missing) — TRY_CAST returns NULL for those.
WITH source AS (
    SELECT * FROM {{ source('berka', 'district') }}
)
SELECT
    A1                        AS district_id,
    A2                        AS district_name,
    A3                        AS region,
    TRY_CAST(A4  AS INTEGER)  AS population,
    TRY_CAST(A10 AS FLOAT)    AS urban_ratio,
    TRY_CAST(A11 AS FLOAT)    AS avg_salary,
    TRY_CAST(A13 AS FLOAT)    AS unemployment_rate,
    TRY_CAST(A14 AS INTEGER)  AS entrepreneurs_per_1000,
    TRY_CAST(A16 AS INTEGER)  AS crimes_count
FROM source
