"""
Select a small high-value rerun panel from the SSE<6 set for Track 2, tagging each
config by why it was chosen. Dedupes near-identical params (3dp) so the panel spans
distinct regions rather than genetic micro-variants.
"""
import os
import csv
import json
import sqlite3
from collections import defaultdict, OrderedDict

DB = r"E:\Development_back_up_folder_2026\long_run data back up\simulation_ledger.db"
RESCORE = r"E:\Development_back_up_folder_2026\lowsse_rerun_2026-06-18\rescore_corrected.csv"
PANEL = r"E:\Development_back_up_folder_2026\lowsse_rerun_2026-06-18\panel.csv"

# corrected scores from Track 1
corr = {}
with open(RESCORE, newline="") as f:
    for r in csv.DictReader(f):
        corr[r["config_hash"]] = r

# saturation / peak diagnostics + params from ledger
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
meta = {}
for row in con.execute("""
    SELECT m.config_hash, m.log_prime_sse, m.max_amp_peak, m.clamp_fraction_mean,
           m.omega_sat_mean, p.param_D, p.param_eta, p.param_rho_vac,
           p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction
    FROM metrics m JOIN parameters p ON m.config_hash=p.config_hash
    WHERE m.log_prime_sse < 6
""").fetchall():
    h = row[0]
    meta[h] = {
        "old_sse": row[1], "max_amp_peak": row[2], "clamp_fraction_mean": row[3],
        "omega_sat_mean": row[4],
        "params": tuple(round(float(x), 3) for x in row[5:11]),
    }
con.close()

def get(h, k):
    return float(corr[h][k]) if h in corr else None

hashes = list(meta.keys())
tags = defaultdict(list)

# A) top corrected matched-SSE (strongest spectral hits historically)
for h in sorted(hashes, key=lambda h: get(h, "corrected_sse"))[:6]:
    tags[h].append("top_corrected")
# B) top old low-SSE
for h in sorted(hashes, key=lambda h: meta[h]["old_sse"])[:4]:
    tags[h].append("top_old_sse")
# C) multiple detected peaks
for h in sorted(hashes, key=lambda h: -int(corr[h]["n_peaks"]))[:4]:
    tags[h].append("multi_peak")
# D) saturated controls (highest amplitude / clamp)
for h in sorted(hashes, key=lambda h: -(meta[h]["max_amp_peak"] or 0))[:3]:
    tags[h].append("saturated_control")
# E) low-saturation candidates (the prize: hits WITHOUT the unstable saturation layer)
for h in hashes:
    if (meta[h]["clamp_fraction_mean"] or 1.0) < 0.5:
        tags[h].append("low_saturation")

# Dedupe by rounded params: keep best (lowest corrected) representative per cluster,
# but always keep low_saturation members (rare + high value).
by_param = defaultdict(list)
for h in tags:
    by_param[meta[h]["params"]].append(h)

panel = OrderedDict()
for cluster in by_param.values():
    cluster.sort(key=lambda h: get(h, "corrected_sse"))
    rep = cluster[0]
    panel[rep] = sorted(set(tags[rep]))
    for h in cluster:
        if "low_saturation" in tags[h] and h not in panel:
            panel[h] = sorted(set(tags[h]))

rows = []
for h, tg in panel.items():
    m = meta[h]
    rows.append({
        "config_hash": h, "tags": "|".join(tg),
        "old_sse": round(m["old_sse"], 4), "corrected_sse": float(corr[h]["corrected_sse"]),
        "n_peaks": int(corr[h]["n_peaks"]), "max_amp_peak": round(m["max_amp_peak"], 2),
        "clamp_fraction_mean": round(m["clamp_fraction_mean"], 4),
        "omega_sat_mean": round(m["omega_sat_mean"], 4),
    })
rows.sort(key=lambda r: r["corrected_sse"])

with open(PANEL, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"panel: {len(rows)} configs -> {PANEL}\n")
print(f"{'hash8':>9} {'old':>6} {'corr':>8} {'n_pk':>5} {'max_amp':>10} {'clampf':>7}  tags")
for r in rows:
    print(f"{r['config_hash'][:8]:>9} {r['old_sse']:6.3f} {r['corrected_sse']:8.4f} {r['n_peaks']:5d} "
          f"{r['max_amp_peak']:10.2f} {r['clamp_fraction_mean']:7.3f}  {r['tags']}")
