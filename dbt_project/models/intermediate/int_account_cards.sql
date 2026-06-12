-- One row per account (most recently issued card for the OWNER).
WITH ranked AS (
    SELECT
        d.account_id,
        c.card_type,
        c.issued_date,
        ROW_NUMBER() OVER (
            PARTITION BY d.account_id
            ORDER BY c.issued_date DESC
        ) AS rn
    FROM {{ ref('stg_cards') }} c
    LEFT JOIN {{ ref('stg_dispositions') }} d ON c.disp_id = d.disp_id
    WHERE d.disp_type = 'OWNER'
)
SELECT
    account_id,
    card_type,
    issued_date,
    TRUE AS has_card
FROM ranked
WHERE rn = 1
