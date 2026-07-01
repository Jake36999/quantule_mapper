"""Plan the low-SSE re-run: collect SSE<6 configs, verify input JSONs, dedupe by rounded params."""
import os
import json
import sqlite3
from collections import defaultdict

DB = r"E:\Development_back_up_folder_2026\long_run data back up\simulation_ledger.db"
CFG_DIR = r"E:\Development_back_up_folder_2026\long_run data back up\input_configs"
ROUND_DP = 3  # params are O(1); 3 dp collapses genetic micro-variants, keeps distinct configs

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute("""
    SELECT m.config_hash, m.log_prime_sse,
           p.param_D, p.param_eta, p.param_rho_vac,
           p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction
    FROM metrics m JOIN parameters p ON m.config_hash = p.config_hash
    WHERE m.log_prime_sse < 6
    ORDER BY m.log_prime_sse ASC
""").fetchall()
con.close()

print(f"configs with log_prime_sse < 6: {len(rows)}")

have, missing = [], []
for r in rows:
    h = r[0]
    cfg = os.path.join(CFG_DIR, f"config_{h}.json")
    (have if os.path.exists(cfg) else missing).append(h)
print(f"  with input_config JSON: {len(have)}   missing: {len(missing)}")
if missing:
    print("  MISSING:", [m[:8] for m in missing])

# Dedup by rounded 6-tuple; representative = lowest old SSE (rows already sorted asc)
groups = defaultdict(list)
for r in rows:
    key = tuple(round(float(x), ROUND_DP) for x in r[2:8])
    groups[key].append(r)

reps = [g[0] for g in groups.values()]
reps.sort(key=lambda r: r[1])
print(f"\nunique configs after dedup @ {ROUND_DP}dp: {len(reps)}  (from {len(rows)})")
print(f"estimated compute @ ~2.5 min/run: ~{len(reps)*2.5:.0f} min\n")

print(f"{'rep_hash8':>10} {'old_sse':>9} {'n_in_grp':>8}   params(D,eta,rho_vac,a_coup,spl_c,spl_f)")
for key, g in sorted(groups.items(), key=lambda kv: kv[1][0][1]):
    rep = g[0]
    print(f"{rep[0][:8]:>10} {rep[1]:9.4f} {len(g):8d}   {key}")
