"""Read-only inspection of a simulation_ledger.db: schema, counts, SSE distribution."""
import sqlite3
import sys

DB = r"E:\Development_back_up_folder_2026\long_run data back up\simulation_ledger.db"

uri = f"file:{DB}?mode=ro"
con = sqlite3.connect(uri, uri=True)
cur = con.cursor()

print("=== TABLES ===")
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(tables)

for t in tables:
    print(f"\n=== SCHEMA: {t} ===")
    for row in cur.execute(f"PRAGMA table_info('{t}')").fetchall():
        # cid, name, type, notnull, dflt, pk
        print(f"  {row[0]:2d}  {row[1]:32s} {row[2]}")
    n = cur.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
    print(f"  rows: {n}")

# Try to find the SSE-like column(s) across tables
print("\n=== SAMPLE ROWS (first table with many cols) ===")
for t in tables:
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')").fetchall()]
    if len(cols) >= 5:
        print(f"\n[{t}] columns: {cols}")
        for row in cur.execute(f"SELECT * FROM '{t}' LIMIT 2").fetchall():
            print("  ", row)
        break

con.close()
