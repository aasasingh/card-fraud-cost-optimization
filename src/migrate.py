"""Apply sql/ migrations in filename order, one transaction each.

Numbered .sql files are the schema/feature contract; this runner exists so
the pipeline needs no psql on the host. A failed statement aborts the whole
file's transaction and crashes the run -- nothing is skipped or retried.
"""

import glob
import os
import time

import psycopg2

DB_URL = os.environ.get("DB_URL", "postgresql://fraud:fraud@localhost:5432/fraud")


def main():
    paths = sorted(glob.glob(os.path.join("sql", "*.sql")))
    if not paths:
        raise SystemExit("no sql/*.sql files found")
    conn = psycopg2.connect(DB_URL)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            sql = f.read()
        t0 = time.time()
        with conn, conn.cursor() as cur:
            cur.execute(sql)
        print(f"{os.path.basename(path)}: {time.time() - t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
