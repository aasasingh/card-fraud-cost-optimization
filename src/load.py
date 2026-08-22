"""Stage 1a: land raw CSVs in Postgres staging tables, every column text.

The staging schema is derived from each file's header row rather than
hardcoded: ~394 columns typed by hand is 394 chances for a silent typo,
and staging-as-text makes no claims about types anyway. The typed schema
(a separate SQL migration) is where we commit to types, explicitly.

Typing happens in a later cast step so that anything dirty (empty strings,
non-numeric junk) fails visibly there instead of being coerced here.
"""

import csv
import os
import time

import psycopg2

DB_URL = os.environ.get("DB_URL", "postgresql://fraud:fraud@localhost:5432/fraud")

FILES = {
    "stg_transaction": "data/raw/train_transaction.csv",
    "stg_identity": "data/raw/train_identity.csv",
}


def load_one(cur, table, path):
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    # Unquoted identifiers so Postgres folds them to lowercase and later SQL
    # needs no double quotes. Assert that folding doesn't create collisions.
    lowered = [c.lower() for c in header]
    assert len(set(lowered)) == len(lowered), f"case-insensitive column collision in {path}"

    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    cur.execute(f"CREATE TABLE {table} ({', '.join(c + ' text' for c in lowered)})")

    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        # CSV format: an unquoted empty field is already NULL; a quoted ""
        # survives as an empty string. The cast migration checks for the latter.
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)", f)
    elapsed = time.time() - t0

    cur.execute(f"SELECT count(*) FROM {table}")
    rows = cur.fetchone()[0]
    print(f"{table}: {rows:,} rows, {len(lowered)} cols, {elapsed:.1f}s COPY")


def main():
    for path in FILES.values():
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — download from Kaggle into data/raw/")

    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        for table, path in FILES.items():
            load_one(cur, table, path)
    conn.close()


if __name__ == "__main__":
    main()
