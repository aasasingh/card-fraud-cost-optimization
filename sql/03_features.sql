-- 03: per-transaction velocity features over card_key_v2.
--
-- Grain: exactly one output row per transaction (the two LATERAL joins are
-- aggregates, so they return exactly one row each; txn_identity joins 1:1).
--
-- Leakage rules enforced here:
--   * isfraud is never referenced -- labels arrive 30-90 days late in
--     production, so no feature may be derived from them;
--   * every window frame ends at CURRENT ROW; both LATERALs bound their
--     scan at the current transaction's timestamp. Nothing reads forward.
--   * caveat, quantified in src/profile.py: RANGE frames include same-second
--     peer rows on the same key (a same-second later transaction is counted).
--
-- NULL policy: a feature is NULL exactly when it is undefined -- no previous
-- transaction (secs_since_prev), no prior amounts to take a median of
-- (amt_ratio_med). Counts over windows are never NULL: they include the
-- current row, so "1" is the true statement "nothing else in this window".
-- 0 is never used to mean "unknown".

DROP TABLE IF EXISTS features CASCADE;

CREATE TABLE features AS
SELECT
    t.transactionid,
    count(*) OVER w_1h  AS cnt_1h,
    count(*) OVER w_24h AS cnt_24h,
    count(*) OVER w_7d  AS cnt_7d,
    t.transactiondt - lag(t.transactiondt) OVER w_seq AS secs_since_prev,
    (t.transactionamt / nullif(m.med_prior_amt, 0))::double precision
        AS amt_ratio_med,
    (i.transactionid IS NOT NULL)::int AS has_identity
FROM txn t
LEFT JOIN txn_identity i USING (transactionid)
-- Trailing median of THIS key's strictly-prior amounts. Strict < excludes
-- the current row and same-second peers. percentile_cont cannot be used as
-- a window function in Postgres, hence LATERAL instead of a window frame.
LEFT JOIN LATERAL (
    SELECT percentile_cont(0.5) WITHIN GROUP
               (ORDER BY p.transactionamt::double precision) AS med_prior_amt
    FROM txn p
    WHERE p.card_key_v2 = t.card_key_v2
      AND p.transactiondt < t.transactiondt
) m ON true
-- Windowed diversity counts (distinct r_emaildomain / productcd in 7d) were
-- built here and then dropped: measured on the data, the first was 0 for 76%
-- of rows (r_emaildomain is mostly NULL, so it degenerated to a presence
-- flag the raw column already provides) and the second was 1 for 98.6%.
-- The measurement lives in src/profile.py (dropped-feature justification).
WINDOW
    w_seq AS (PARTITION BY t.card_key_v2
              ORDER BY t.transactiondt, t.transactionid),
    w_1h  AS (PARTITION BY t.card_key_v2 ORDER BY t.transactiondt
              RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW),
    w_24h AS (PARTITION BY t.card_key_v2 ORDER BY t.transactiondt
              RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW),
    w_7d  AS (PARTITION BY t.card_key_v2 ORDER BY t.transactiondt
              RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW);

ALTER TABLE features ADD PRIMARY KEY (transactionid);

ANALYZE features;
