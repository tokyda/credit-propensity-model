-- birth_number encodes gender: month > 50 = female (subtract 50 for real month).
WITH source AS (
    SELECT * FROM {{ source('berka', 'client') }}
),
parsed AS (
    SELECT
        client_id,
        district_id,
        CASE
            WHEN CAST(SUBSTR(LPAD(CAST(birth_number AS VARCHAR), 6, '0'), 3, 2) AS INTEGER) > 50
            THEN 'F' ELSE 'M'
        END AS gender,
        1900 + CAST(SUBSTR(LPAD(CAST(birth_number AS VARCHAR), 6, '0'), 1, 2) AS INTEGER) AS birth_year
    FROM source
)
SELECT
    client_id,
    district_id,
    gender,
    1998 - birth_year AS client_age
FROM parsed
