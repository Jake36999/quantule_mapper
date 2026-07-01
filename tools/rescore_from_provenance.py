"""
Re-score low-SSE configs with CORRECTED prime-SSE, reconstructed exactly from the
ORIGINAL provenance (no re-running, so the original saturated fields are preserved).

The penalties are a closed-form function of n_peaks_found_main, so:
    missing_penalty = (7 - min(n_peaks,7)) * 1.0
    noise_penalty   = (n_peaks - min(n_peaks,7)) * 0.2
    corrected_sse   = log_prime_sse - missing_penalty - noise_penalty   (= sum matched errors)
Penalties are kept as columns (for the ASTE hunter); they are just not in the headline score.
"""
import os
import csv
import json
import sqlite3

DB = r"E:\Development_back_up_folder_2026\long_run data back up\simulation_ledger.db"
PROV = r"E:\Development_back_up_folder_2026\long_run data back up\provenance_reports"
OUT = r"E:\Development_back_up_folder_2026\lowsse_rerun_2026-06-18\rescore_corrected.csv"
N_TARGETS = 7  # len(TARGET_PRIMES = [2,3,5,7,11,13,17])

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute(
    "SELECT config_hash, log_prime_sse FROM metrics WHERE log_prime_sse < 6 ORDER BY log_prime_sse ASC"
).fetchall()
con.close()

out_rows, missing_prov, mismatch = [], [], []
for h, ledger_sse in rows:
    p = os.path.join(PROV, f"provenance_{h}.json")
    if not os.path.exists(p):
        missing_prov.append(h)
        continue
    sf = json.load(open(p)).get("spectral_fidelity", {})
    prov_sse = sf.get("log_prime_sse")
    n_peaks = sf.get("n_peaks_found_main")
    if prov_sse is None or n_peaks is None:
        missing_prov.append(h)
        continue
    if abs(float(prov_sse) - float(ledger_sse)) > 1e-6:
        mismatch.append((h, ledger_sse, prov_sse))

    n_peaks = int(n_peaks)
    min_len = min(n_peaks, N_TARGETS)
    missing_pen = (N_TARGETS - min_len) * 1.0
    noise_pen = (n_peaks - min_len) * 0.2
    corrected = float(prov_sse) - missing_pen - noise_pen
    out_rows.append({
        "config_hash": h, "old_sse": round(float(ledger_sse), 5),
        "corrected_sse": round(corrected, 5), "improvement": round(float(ledger_sse) - corrected, 5),
        "n_peaks": n_peaks, "missing_penalty": missing_pen, "noise_penalty": round(noise_pen, 4),
    })

out_rows.sort(key=lambda r: r["corrected_sse"])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"re-scored {len(out_rows)} configs (missing provenance: {len(missing_prov)}, "
      f"sse mismatches vs ledger: {len(mismatch)})")
print(f"written -> {OUT}\n")
print(f"{'hash8':>9} {'old_sse':>8} {'corrected':>10} {'improve':>8} {'n_pk':>5} {'miss':>5} {'noise':>6}")
print("-" * 60)
for r in out_rows[:25]:
    print(f"{r['config_hash'][:8]:>9} {r['old_sse']:8.3f} {r['corrected_sse']:10.4f} "
          f"{r['improvement']:8.3f} {r['n_peaks']:5d} {r['missing_penalty']:5.1f} {r['noise_penalty']:6.2f}")

print("\n-- biggest penalty-driven improvements (gems the old score buried) --")
for r in sorted(out_rows, key=lambda r: -r["improvement"])[:12]:
    print(f"{r['config_hash'][:8]:>9} old={r['old_sse']:.3f} -> corrected={r['corrected_sse']:.4f} "
          f"(improve {r['improvement']:.3f}, n_peaks={r['n_peaks']})")
