SELECT
    account_id,
    gender,
    client_age,
    statement_frequency,
    district_avg_salary,
    district_unemployment_rate,
    district_crimes_count,
    district_urban_ratio,
    opened_date
FROM {{ ref('int_account_owner') }}
