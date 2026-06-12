WITH all_accounts AS (
    SELECT DISTINCT account_id FROM {{ ref('stg_accounts') }}
),
cards AS (
    SELECT * FROM {{ ref('int_account_cards') }}
),
order_aggs AS (
    SELECT
        account_id,
        COUNT(*)            AS standing_order_count,
        AVG(order_amount)   AS avg_standing_order_amount
    FROM {{ ref('stg_orders') }}
    GROUP BY account_id
)

SELECT
    a.account_id,
    COALESCE(c.has_card, FALSE)               AS has_card,
    COALESCE(c.card_type, 'none')             AS card_type,
    COALESCE(o.standing_order_count, 0) > 0   AS has_standing_order,
    COALESCE(o.standing_order_count, 0)       AS standing_order_count,
    COALESCE(o.avg_standing_order_amount, 0)  AS avg_standing_order_amount
FROM all_accounts a
LEFT JOIN cards      c ON a.account_id = c.account_id
LEFT JOIN order_aggs o ON a.account_id = o.account_id
