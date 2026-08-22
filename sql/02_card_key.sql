-- 02: pseudo-card identity.
--
-- There is no card ID in this dataset. We approximate one: two transactions
-- are "the same card" iff they agree on card1-card6 + addr1 + p_emaildomain.
--
-- Every field is coalesced to the sentinel 'NA' BEFORE concat_ws, because
-- concat_ws silently skips NULL arguments: without the coalesce,
-- (visa, NULL, 150) and (visa, 150, NULL) would collapse to the same key.
-- 'NA' cannot collide with real values (numeric fields and closed
-- categorical vocabularies on the text fields).
--
-- The key is kept human-readable (plain concatenation, not a hash) so a
-- suspicious key can be eyeballed directly.

ALTER TABLE txn DROP COLUMN IF EXISTS card_key;
ALTER TABLE txn ADD COLUMN card_key text;

UPDATE txn SET card_key = concat_ws('|',
    coalesce(card1::text, 'NA'),
    coalesce(card2::text, 'NA'),
    coalesce(card3::text, 'NA'),
    coalesce(card4,       'NA'),
    coalesce(card5::text, 'NA'),
    coalesce(card6,       'NA'),
    coalesce(addr1::text, 'NA'),
    coalesce(p_emaildomain, 'NA'));

-- v2 appends the D1 anchor: anchor_day = (transactiondt/86400) - d1.
-- We do NOT claim to know what D1 means. What we measured (src/profile.py):
-- within v1 keys, that expression is either exactly constant (one entity)
-- or spread over months (pooled entities) -- almost never off by one day.
-- A stable per-entity constant is exactly what a key component needs,
-- whatever the column "really" is. NULL d1 (0.2% of rows) falls back to
-- the sentinel, i.e. those rows keep v1 granularity.
ALTER TABLE txn DROP COLUMN IF EXISTS card_key_v2;
ALTER TABLE txn ADD COLUMN card_key_v2 text;

UPDATE txn SET card_key_v2 =
    card_key || '|' ||
    coalesce(((transactiondt / 86400) - d1::int)::text, 'NA');

-- Stage 2 window functions partition by the card key and order by
-- transactiondt; these indexes let Postgres feed them without re-sorting
-- 590k rows each time.
DROP INDEX IF EXISTS idx_txn_card_key_dt;
CREATE INDEX idx_txn_card_key_dt ON txn (card_key, transactiondt);
DROP INDEX IF EXISTS idx_txn_card_key_v2_dt;
CREATE INDEX idx_txn_card_key_v2_dt ON txn (card_key_v2, transactiondt);

ANALYZE txn;
