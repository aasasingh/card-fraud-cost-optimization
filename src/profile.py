"""Stage 1 profile: every number reported in Stage 1 comes from this script.

Sections: dataset volume, fraud rate, time span, null density by column
block, identity-join coverage, then the pseudo-card evidence -- v1
validation, the D1 anchor test that motivated v2, v2 validation on the
identical battery, what v2 did to the v1 head, and the addr1-null split
that diagnoses the pooling root cause.
"""

import os

import psycopg2

DB_URL = os.environ.get("DB_URL", "postgresql://fraud:fraud@localhost:5432/fraud")

DAY = 86400  # transactiondt is in seconds

SIZE_BUCKET = """CASE WHEN n = 1 THEN '1' WHEN n <= 4 THEN '2-4'
                      WHEN n <= 10 THEN '5-10' WHEN n <= 100 THEN '11-100'
                      WHEN n <= 1000 THEN '101-1000' ELSE '>1000' END"""


def block_null_density(cur, table, blocks):
    cur.execute(f"SELECT count(*) FROM {table}")
    nrows = cur.fetchone()[0]
    print(f"\n-- null density by block: {table} ({nrows:,} rows) --")
    for name, cols in blocks:
        exprs = ", ".join(f"count({c})" for c in cols)
        cur.execute(f"SELECT {exprs} FROM {table}")
        nonnull = cur.fetchone()
        fracs = sorted(1 - n / nrows for n in nonnull)
        avg = sum(fracs) / len(fracs)
        print(f"{name:<14} {len(cols):>3} cols   null% avg {100*avg:5.1f}   "
              f"min {100*fracs[0]:5.1f}   max {100*fracs[-1]:5.1f}")


def validate_key(cur, keycol, n, span_days):
    """The identical validation battery for any candidate card key."""
    print(f"\n== pseudo-card validation: {keycol} ==")
    cur.execute(f"""
        WITH k AS (
            SELECT {keycol}, count(*) AS n,
                   max(transactiondt) - min(transactiondt) AS span
            FROM txn GROUP BY {keycol}
        )
        SELECT count(*),
               min(n),
               percentile_disc(array[0.25, 0.5, 0.75, 0.9, 0.99])
                   WITHIN GROUP (ORDER BY n),
               max(n),
               count(*) FILTER (WHERE n = 1),
               sum(n) FILTER (WHERE n = 1),
               count(*) FILTER (WHERE span >= 150 * {DAY})
        FROM k
    """)
    nk, mn, pcts, mx, singles, single_tx, longspan = cur.fetchone()
    p25, p50, p75, p90, p99 = pcts
    print(f"distinct keys: {nk:,}")
    print(f"txns per key: min {mn}  p25 {p25}  median {p50}  p75 {p75}  "
          f"p90 {p90}  p99 {p99}  max {mx:,}")
    print(f"singleton keys: {singles:,} ({100*singles/nk:.1f}% of keys, "
          f"{100*single_tx/n:.1f}% of txns)")
    cur.execute(f"""
        WITH k AS (SELECT count(*) AS n FROM txn GROUP BY {keycol})
        SELECT sum(n) FROM (SELECT n FROM k ORDER BY n DESC LIMIT 100) top
    """)
    top100 = cur.fetchone()[0]
    print(f"largest 100 keys hold: {top100:,} txns ({100*top100/n:.1f}%)")
    cur.execute(f"""
        SELECT count(*) FROM (
            SELECT 1 FROM txn GROUP BY {keycol}
            HAVING max(transactiondt) - min(transactiondt) >= 150 * {DAY}
        ) s
    """)
    print(f"keys spanning >=150 of {span_days:.0f} days: {longspan:,} "
          f"({100*longspan/nk:.2f}%)")
    print("top 5 keys:")
    cur.execute(f"""
        SELECT {keycol}, count(*),
               (max(transactiondt) - min(transactiondt)) / {DAY}.0,
               avg(isfraud::float)
        FROM txn GROUP BY {keycol} ORDER BY count(*) DESC LIMIT 5
    """)
    for key, cnt, span, fr in cur.fetchall():
        print(f"  {cnt:>6,} txns  span {span:6.1f}d  fraud {100*fr:5.2f}%  {key}")


def d1_anchor_test(cur):
    """Is anchor_day = day - D1 a per-entity constant?

    If yes, a v1 key holding one entity shows exactly 1 distinct value and a
    pooled v1 key shows many, spread widely. The tell against it: ranges of
    exactly 1 day, which would mean rounding jitter splitting real cards.
    """
    print("\n== D1 anchor test (per v1 key: distinct values and range of "
          "day - d1) ==")
    cur.execute(f"""
        WITH per_key AS (
            SELECT card_key, count(*) AS n,
                   count(DISTINCT (transactiondt / {DAY}) - d1::int) AS ndv,
                   max((transactiondt / {DAY}) - d1::int)
                 - min((transactiondt / {DAY}) - d1::int) AS rng
            FROM txn
            WHERE d1 IS NOT NULL
            GROUP BY card_key
            HAVING count(*) >= 2
        )
        SELECT {SIZE_BUCKET} AS size_bucket,
               count(*),
               count(*) FILTER (WHERE ndv = 1),
               count(*) FILTER (WHERE ndv = 2),
               count(*) FILTER (WHERE ndv > 2),
               count(*) FILTER (WHERE rng = 1),
               count(*) FILTER (WHERE rng > 30)
        FROM per_key GROUP BY 1 ORDER BY min(n)
    """)
    print(f"{'key size':>9} {'keys':>7} {'ndv=1':>7} {'ndv=2':>7} "
          f"{'ndv>2':>7} {'rng=1':>7} {'rng>30':>7}")
    for r in cur.fetchall():
        print(f"{r[0]:>9} {r[1]:>7,} {r[2]:>7,} {r[3]:>7,} "
              f"{r[4]:>7,} {r[5]:>7,} {r[6]:>7,}")


def v1_head_fate(cur):
    print("\n== fate of the 5 largest v1 keys under v2 ==")
    cur.execute("""
        WITH top5 AS (
            SELECT card_key FROM txn
            GROUP BY card_key ORDER BY count(*) DESC LIMIT 5
        ),
        pieces AS (
            SELECT t.card_key, t.card_key_v2, count(*) AS n
            FROM txn t JOIN top5 USING (card_key)
            GROUP BY 1, 2
        )
        SELECT card_key, sum(n), count(*),
               max(n),
               percentile_disc(0.5) WITHIN GROUP (ORDER BY n),
               count(*) FILTER (WHERE n = 1)
        FROM pieces GROUP BY card_key ORDER BY sum(n) DESC
    """)
    for key, total, pieces, mx, med, singles in cur.fetchall():
        print(f"  {total:>6,} txns -> {pieces:>5,} v2 keys  "
              f"largest {mx:>4,}  median {med}  singletons {singles:,}  {key}")


def v2_head_composition(cur, n, base_rate):
    """The top v2 keys have 0% fraud, which at base rate is not chance.
    Characterize them: amounts, product, cadence, identity, test-window
    survival -- so the zero-fraud subpopulation is named, not shrugged at."""
    print("\n== v2 head composition ==")
    cur.execute("SELECT min(transactiondt) FROM txn WHERE fold = 'test'")
    test_dt = cur.fetchone()[0]
    print(f"test fold starts at transactiondt {test_dt:,}")
    cur.execute("""
        SELECT card_key_v2, count(*) FROM txn
        GROUP BY 1 ORDER BY count(*) DESC LIMIT 5""")
    for key, cnt in cur.fetchall():
        cur.execute("""
            SELECT (max(t.transactiondt) - min(t.transactiondt)) / 86400.0,
                   count(DISTINCT t.transactionamt),
                   avg((i.transactionid IS NOT NULL)::int),
                   max(t.transactiondt)
            FROM txn t LEFT JOIN txn_identity i USING (transactionid)
            WHERE t.card_key_v2 = %s""", (key,))
        span, namts, id_share, last_dt = cur.fetchone()
        cur.execute("""
            SELECT transactionamt, count(*) FROM txn
            WHERE card_key_v2 = %s GROUP BY 1
            ORDER BY count(*) DESC, transactionamt LIMIT 1""", (key,))
        amt, amtn = cur.fetchone()
        cur.execute("""
            SELECT productcd, count(*) FROM txn WHERE card_key_v2 = %s
            GROUP BY 1 ORDER BY count(*) DESC, productcd LIMIT 1""", (key,))
        prod, prodn = cur.fetchone()
        cur.execute("""
            WITH g AS (SELECT transactiondt - lag(transactiondt)
                           OVER (ORDER BY transactiondt, transactionid) AS gap
                       FROM txn WHERE card_key_v2 = %s)
            SELECT percentile_disc(array[0.25, 0.5, 0.75])
                   WITHIN GROUP (ORDER BY gap) FROM g WHERE gap IS NOT NULL""",
            (key,))
        q1, med, q3 = cur.fetchone()[0]
        print(f"  {cnt:>5,} txns  span {span:5.1f}d  modal amt {amt} x{amtn}  "
              f"distinct amts {namts}  productcd {prod} {prodn}/{cnt}  "
              f"identity {100*id_share:.0f}%  gap q1/med/q3 "
              f"{q1}/{med}/{q3}s  in test window: "
              f"{'yes' if last_dt >= test_dt else 'no'}")
    cur.execute("""
        WITH k AS (SELECT card_key_v2, count(*) AS nn, sum(isfraud) AS f,
                          max(transactiondt) AS last_dt
                   FROM txn GROUP BY 1 HAVING count(*) >= 100)
        SELECT count(*), sum(nn),
               count(*) FILTER (WHERE f = 0),
               coalesce(sum(nn) FILTER (WHERE f = 0), 0),
               count(*) FILTER (WHERE f = 0 AND last_dt >= %s)
        FROM k""", (test_dt,))
    nk, ntx, zk, ztx, zk_test = cur.fetchone()
    print(f"keys with >=100 txns: {nk:,} holding {ntx:,} txns; zero-fraud "
          f"among them: {zk:,} keys, {ztx:,} txns "
          f"(expected frauds at base rate: {base_rate*float(ztx):.0f})")
    print(f"zero-fraud >=100-txn keys still active in test window: {zk_test:,}")


def split_report(cur):
    print("\n== temporal split (fold column from sql/04) ==")
    cur.execute("""
        SELECT fold, count(*), min(transactiondt), max(transactiondt),
               avg(isfraud::float)
        FROM txn GROUP BY fold
        ORDER BY min(transactiondt)""")
    for fold, cnt, lo, hi, fr in cur.fetchall():
        print(f"{fold:>5}: {cnt:>7,} rows  dt [{lo:>10,} .. {hi:>10,}]  "
              f"fraud {100*fr:.3f}%")


def amounts_by_class(cur):
    """Input to the cost matrix: the FN cost is amount-driven."""
    print("\n== transactionamt by class ==")
    cur.execute("""
        SELECT isfraud, count(*), avg(transactionamt),
               percentile_cont(array[0.5, 0.9, 0.99])
                   WITHIN GROUP (ORDER BY transactionamt::double precision),
               max(transactionamt)
        FROM txn GROUP BY isfraud ORDER BY isfraud""")
    for isf, cnt, mean, pcts, mx in cur.fetchall():
        p50, p90, p99 = pcts
        lbl = "fraud" if isf else "legit"
        print(f"{lbl}: n {cnt:>7,}  mean {float(mean):8.2f}  median {p50:7.2f}  "
              f"p90 {p90:8.2f}  p99 {p99:8.2f}  max {float(mx):9.2f}")


def dropped_feature_justification(cur, n):
    """Two spec'd diversity features were dropped; this measures why, from
    txn directly (the columns no longer exist in features)."""
    print("\n== dropped diversity features (justification) ==")
    cur.execute("""
        SELECT least(d.nr, 3), least(d.np, 3), count(*)
        FROM txn t
        LEFT JOIN LATERAL (
            SELECT count(DISTINCT w.r_emaildomain) AS nr,
                   count(DISTINCT w.productcd)     AS np
            FROM txn w
            WHERE w.card_key_v2 = t.card_key_v2
              AND w.transactiondt >  t.transactiondt - 604800
              AND w.transactiondt <= t.transactiondt
        ) d ON true
        GROUP BY 1, 2""")
    nr_marg, np_marg = {}, {}
    for nr, np, c in cur.fetchall():
        nr_marg[nr] = nr_marg.get(nr, 0) + c
        np_marg[np] = np_marg.get(np, 0) + c
    nr = "  ".join(f"{'3+' if k == 3 else k}: {100*v/n:.1f}%"
                   for k, v in sorted(nr_marg.items()))
    np = "  ".join(f"{'3+' if k == 3 else k}: {100*v/n:.1f}%"
                   for k, v in sorted(np_marg.items()))
    print(f"distinct r_emaildomain in 7d window -> {nr}")
    print(f"distinct productcd in 7d window    -> {np}")


def feature_audit(cur, n):
    """Stage 2 verification: grain, NULL accounting, the same-second-peer
    caveat quantified, and fraud rate sliced by each feature family to show
    the features carry signal (slices use isfraud; the features never do)."""
    print("\n== feature audit ==")
    cur.execute("SELECT count(*) FROM features")
    nf = cur.fetchone()[0]
    print(f"features rows: {nf:,} (txn rows: {n:,}, grain "
          f"{'OK' if nf == n else 'BROKEN'})")
    cur.execute("""
        SELECT count(*) FILTER (WHERE secs_since_prev IS NULL),
               count(*) FILTER (WHERE amt_ratio_med IS NULL),
               max(cnt_1h), max(cnt_24h), max(cnt_7d)
        FROM features""")
    sp_null, ar_null, mx1, mx24, mx7 = cur.fetchone()
    cur.execute("SELECT count(DISTINCT card_key_v2) FROM txn")
    nkeys = cur.fetchone()[0]
    print(f"secs_since_prev NULL: {sp_null:,} ({100*sp_null/n:.1f}%)  "
          f"amt_ratio_med NULL: {ar_null:,}  first-txn rows (=#keys): {nkeys:,}")
    print(f"max cnt_1h {mx1}  max cnt_24h {mx24}  max cnt_7d {mx7}")
    cur.execute("""
        SELECT coalesce(sum(c), 0), count(*) FROM (
            SELECT count(*) AS c FROM txn
            GROUP BY card_key_v2, transactiondt HAVING count(*) > 1
        ) s""")
    peer_rows, peer_groups = cur.fetchone()
    print(f"same-key same-second rows: {peer_rows:,} in {peer_groups:,} "
          f"groups ({100*float(peer_rows)/n:.2f}% of txns)")

    print("fraud rate by cnt_1h:")
    cur.execute("""
        SELECT CASE WHEN cnt_1h = 1 THEN '1' WHEN cnt_1h = 2 THEN '2'
                    WHEN cnt_1h <= 5 THEN '3-5' ELSE '6+' END,
               count(*), avg(t.isfraud::float)
        FROM features f JOIN txn t USING (transactionid)
        GROUP BY 1 ORDER BY min(cnt_1h)""")
    for b, c, fr in cur.fetchall():
        print(f"  cnt_1h {b:>4}: {c:>7,} txns  fraud {100*fr:5.2f}%")
    print("fraud rate by secs_since_prev:")
    cur.execute("""
        SELECT CASE WHEN secs_since_prev IS NULL THEN 'NULL (first)'
                    WHEN secs_since_prev < 60 THEN '<60s'
                    WHEN secs_since_prev < 3600 THEN '<1h'
                    WHEN secs_since_prev < 86400 THEN '<24h'
                    ELSE '>=24h' END,
               count(*), avg(t.isfraud::float)
        FROM features f JOIN txn t USING (transactionid)
        GROUP BY 1 ORDER BY min(coalesce(secs_since_prev, 2147483647))""")
    for b, c, fr in cur.fetchall():
        print(f"  {b:>12}: {c:>7,} txns  fraud {100*fr:5.2f}%")
    print("fraud rate by amt_ratio_med:")
    cur.execute("""
        SELECT CASE WHEN amt_ratio_med IS NULL THEN 'NULL (first)'
                    WHEN amt_ratio_med < 0.5 THEN '<0.5x'
                    WHEN amt_ratio_med <= 2 THEN '0.5-2x'
                    WHEN amt_ratio_med <= 5 THEN '2-5x'
                    ELSE '>5x' END,
               count(*), avg(t.isfraud::float)
        FROM features f JOIN txn t USING (transactionid)
        GROUP BY 1 ORDER BY min(coalesce(amt_ratio_med, -1))""")
    for b, c, fr in cur.fetchall():
        print(f"  {b:>12}: {c:>7,} txns  fraud {100*fr:5.2f}%")


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    print("== volume ==")
    cur.execute("SELECT count(*), sum(isfraud), avg(isfraud::float) FROM txn")
    n, nf, rate = cur.fetchone()
    print(f"txn rows: {n:,}   fraud: {nf:,}   fraud rate: {100*rate:.3f}%")
    cur.execute("SELECT count(*) FROM txn_identity")
    print(f"txn_identity rows: {cur.fetchone()[0]:,}")

    print("\n== time ==")
    cur.execute("SELECT min(transactiondt), max(transactiondt) FROM txn")
    lo, hi = cur.fetchone()
    span_days = (hi - lo) / DAY
    print(f"transactiondt min {lo:,}  max {hi:,}  span {span_days:.1f} days")

    tx_blocks = [
        ("card1-6", [f"card{i}" for i in range(1, 7)]),
        ("addr1-2", ["addr1", "addr2"]),
        ("dist1-2", ["dist1", "dist2"]),
        ("emaildomain", ["p_emaildomain", "r_emaildomain"]),
        ("C1-14", [f"c{i}" for i in range(1, 15)]),
        ("D1-15", [f"d{i}" for i in range(1, 16)]),
        ("M1-9", [f"m{i}" for i in range(1, 10)]),
        ("V1-339", [f"v{i}" for i in range(1, 340)]),
    ]
    id_blocks = [
        ("id_01-38", [f"id_{i:02d}" for i in range(1, 39)]),
        ("device*", ["devicetype", "deviceinfo"]),
    ]
    block_null_density(cur, "txn", tx_blocks)
    block_null_density(cur, "txn_identity", id_blocks)

    print("\n== identity join coverage ==")
    cur.execute("""
        SELECT (i.transactionid IS NOT NULL) AS has_identity,
               count(*), avg(t.isfraud::float)
        FROM txn t LEFT JOIN txn_identity i USING (transactionid)
        GROUP BY 1 ORDER BY 1
    """)
    for has_id, cnt, fr in cur.fetchall():
        side = "with identity" if has_id else "no identity  "
        print(f"{side}: {cnt:>7,} txns ({100*cnt/n:5.1f}%)   fraud rate {100*fr:.3f}%")

    validate_key(cur, "card_key", n, span_days)
    d1_anchor_test(cur)
    validate_key(cur, "card_key_v2", n, span_days)
    v1_head_fate(cur)
    v2_head_composition(cur, n, rate)

    print("\n== d1-null rows under v2 ==")
    cur.execute("""
        SELECT count(*), count(DISTINCT card_key_v2) FROM txn WHERE d1 IS NULL
    """)
    rows, keys = cur.fetchone()
    print(f"{rows:,} rows ({100*rows/n:.2f}%) have d1 NULL; they form "
          f"{keys:,} v2 keys at v1 granularity (anchor = 'NA')")

    print("\n== v1 key size split by addr1 null (root cause) ==")
    cur.execute("""
        WITH k AS (
            SELECT card_key, bool_or(addr1 IS NULL) AS addr_null,
                   count(*) AS n
            FROM txn GROUP BY card_key
        )
        SELECT addr_null, count(*), sum(n),
               percentile_disc(array[0.5, 0.9, 0.99]) WITHIN GROUP (ORDER BY n),
               max(n),
               coalesce(sum(n) FILTER (WHERE n > 1000), 0)
        FROM k GROUP BY addr_null ORDER BY addr_null
    """)
    for addr_null, nk, ntx, pcts, mx, big in cur.fetchall():
        lbl = "addr1 NULL   " if addr_null else "addr1 present"
        print(f"{lbl}: {nk:>6,} keys  {ntx:>7,} txns  median {pcts[0]}  "
              f"p90 {pcts[1]}  p99 {pcts[2]}  max {mx:,}  "
              f"txns in keys>1000: {big:,} ({100*big/ntx:.1f}%)")

    split_report(cur)
    amounts_by_class(cur)
    feature_audit(cur, n)
    dropped_feature_justification(cur, n)

    conn.close()


if __name__ == "__main__":
    main()
