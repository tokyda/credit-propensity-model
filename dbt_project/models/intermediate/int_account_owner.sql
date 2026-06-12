-- One row per account. Joins account → OWNER client → district.
WITH accounts AS (
    SELECT * FROM {{ ref('stg_accounts') }}
),
owners AS (
    SELECT client_id, account_id
    FROM {{ ref('stg_dispositions') }}
    WHERE disp_type = 'OWNER'
),
clients   AS (SELECT * FROM {{ ref('stg_clients') }}),
districts AS (SELECT * FROM {{ ref('stg_districts') }})

SELECT
    a.account_id,
    a.district_id,
    a.statement_frequency,
    a.opened_date,
    c.client_id,
    c.gender,
    c.client_age,
    d.avg_salary         AS district_avg_salary,
    d.unemployment_rate  AS district_unemployment_rate,
    d.crimes_count       AS district_crimes_count,
    d.urban_ratio        AS district_urban_ratio
FROM accounts a
LEFT JOIN owners    o ON a.account_id  = o.account_id
LEFT JOIN clients   c ON o.client_id   = c.client_id
LEFT JOIN districts d ON a.district_id = d.district_id
