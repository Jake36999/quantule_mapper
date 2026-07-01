"""List configs with log_prime_sse < 6 and show the SSE distribution (read-only)."""
import sqlite3

DB = r"E:\Development_back_up_folder_2026\long_run data back up\simulation_ledger.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()

# Distribution
print("=== log_prime_sse distribution (metrics table) ===")
for lo, hi in [(None, 2), (2, 4), (4, 6), (6, 10), (10, 100), (100, None)]:
    if lo is None:
        q = "SELECT COUNT(*) FROM metrics WHERE log_prime_sse < ?"; args = (hi,)
        label = f"< {hi}"
    elif hi is None:
        q = "SELECT COUNT(*) FROM metrics WHERE log_prime_sse >= ?"; args = (lo,)
        label = f">= {lo}"
    else:
        q = "SELECT COUNT(*) FROM metrics WHERE log_prime_sse >= ? AND log_prime_sse < ?"; args = (lo, hi)
        label = f"[{lo}, {hi})"
    print(f"  {label:12s}: {cur.execute(q, args).fetchone()[0]}")

total = cur.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
nonnull = cur.execute("SELECT COUNT(*) FROM metrics WHERE log_prime_sse IS NOT NULL").fetchone()[0]
print(f"  total metrics rows: {total}  (non-null sse: {nonnull})")

# The < 6 set joined with params
print("\n=== configs with log_prime_sse < 6 (joined with parameters) ===")
rows = cur.execute("""
    SELECT m.config_hash, m.log_prime_sse,
           p.param_D, p.param_eta, p.param_rho_vac,
           p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction,
           m.max_amp_peak, m.clamp_fraction_mean, m.omega_sat_mean
    FROM metrics m JOIN parameters p ON m.config_hash = p.config_hash
    WHERE m.log_prime_sse < 6
    ORDER BY m.log_prime_sse ASC
""").fetchall()
print(f"count < 6: {len(rows)}\n")
hdr = ("hash8", "sse", "D", "eta", "rho_vac", "a_coup", "splash_c", "splash_f", "max_amp", "clampf", "om_sat")
print("  ".join(f"{h:>9}" for h in hdr))
for r in rows:
    h = r[0][:8]
    vals = [h] + [f"{x:.4g}" if isinstance(x, float) else str(x) for x in r[1:]]
    print("  ".join(f"{v:>9}" for v in vals))

con.close()
