"""Re-classify saved stable-collapse sweeps with the CORRECTED coherence-based gate
(no re-running; all raw observables are in the CSVs). Reports tallies + promotable
candidates. Native .venv (numpy only)."""
import csv, glob, os, json
import numpy as np

AMP_BLOWUP, DISSIPATE_ER, CONC_SMOOTH, COH = 1e3, 0.5, 4.0, 0.1


def reclass(r):
    amp, er, conc = float(r["amp_max"]), float(r["energy_ratio"]), float(r["conc_final"])
    nm, nf, coh = int(r["nodes_mid"]), int(r["nodes_final"]), float(r["coherence"])
    if r["class"] == "unstable" or amp > AMP_BLOWUP:
        return "unstable"
    smooth = conc < CONC_SMOOTH
    if smooth and er < DISSIPATE_ER:
        return "dissipative"
    if nf >= 2 and nm >= 2:
        return "stable_multinode" if coh > COH else "incoherent_multinode"
    if nm >= 2 and nf < 2:
        return "transient"
    if smooth:
        return "dissipative"
    return "single_node"


def promotable(r):
    return (reclass(r) == "stable_multinode" and 0.3 <= float(r["energy_ratio"]) <= 5.0
            and float(r["coherence"]) > 0.2)


def label_of(d):
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        return json.load(open(mp)).get("ic", "gaussian")
    return "gaussian"


dirs = [d for d in sorted(glob.glob("sweep_runs/STABLE_COLLAPSE_*"))
        if os.path.exists(os.path.join(d, "stable_collapse_results.csv"))]
for d in dirs:
    rows = list(csv.DictReader(open(os.path.join(d, "stable_collapse_results.csv"))))
    if len(rows) < 100:
        continue  # skip tiny validation runs
    from collections import Counter
    tally = Counter(reclass(r) for r in rows)
    prom = [r for r in rows if promotable(r)]
    print(f"=== IC={label_of(d)}  n={len(rows)}  ({os.path.basename(d)}) ===")
    print("  corrected tally:", dict(tally))
    print(f"  PROMOTABLE (coherent persistent multi-node, energy retained 0.3-5x): {len(prom)}")
    for r in sorted(prom, key=lambda r: -float(r["coherence"]))[:6]:
        inc = r["incommens"]
        print(f"    idx={r['idx']:>4} nodes={r['nodes_final']} coh={float(r['coherence']):.3f} "
              f"e_ratio={float(r['energy_ratio']):.2f} modes={r['modes_final']} incommens={inc}")
    print()
