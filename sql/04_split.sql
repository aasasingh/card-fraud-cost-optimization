-- 04: temporal split, materialized once as a column.
--
-- The cut is on a transactiondt VALUE, not a row index: percentile_disc
-- returns an actual timestamp from the data, and fold assignment compares
-- against it, so all rows sharing one second land in the same fold by
-- construction -- a row-index cut could slice through a timestamp tie.
-- Fold shares are therefore approximately (not exactly) 70/15/15; the exact
-- counts are reported by src/profile.py. Every downstream script reads this
-- column; nothing recomputes the boundary.
--
-- Intentional and correct, do not "fix": trailing-window features in
-- sql/03 may reach across fold boundaries (a test-window transaction sees
-- its card's train-window history). That is exactly what production looks
-- like at scoring time -- the past is available; only the future is not.

ALTER TABLE txn DROP COLUMN IF EXISTS fold;
ALTER TABLE txn ADD COLUMN fold text;

WITH cuts AS (
    SELECT percentile_disc(0.70) WITHIN GROUP (ORDER BY transactiondt) AS c_train,
           percentile_disc(0.85) WITHIN GROUP (ORDER BY transactiondt) AS c_val
    FROM txn
)
UPDATE txn SET fold = CASE
    WHEN transactiondt <= (SELECT c_train FROM cuts) THEN 'train'
    WHEN transactiondt <= (SELECT c_val FROM cuts) THEN 'val'
    ELSE 'test'
END;

ANALYZE txn;
