"""Phase D.6 — reduced node-network ("zoom-out") model. STANDALONE numpy; NO PDE solver, NO jax. Encodes ONLY the
measured Phase D dissipative-sector laws (merge<0.3, couple<0.5, pinned/no-drift) and validates against the harvested
PHASE_C_NODE_LIBRARY. This is a coarse-graining model, not a physics solver; no matter/macro claims.

  python jax_scout/reduced_node_model.py
"""
import os, sys, glob, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

R_MERGE = 0.3        # measured (D.5): settled pairs closer than this coalesce
R_COUPLE = 0.5       # measured (D.4): density-bridge coupling only within this
BOX = 1.0            # positions in box units (periodic)


def _mindist(a, b):
    d = a - b; d = d - BOX * np.round(d / BOX); return np.linalg.norm(d)


def merge_resolve(pos, mass=None, r=R_MERGE):
    """Iteratively coalesce the closest pair while separation < r (mass-weighted, minimal-image). Pinned: no drift,
    positions change ONLY by merging. Returns (positions, masses)."""
    pos = [np.array(p, float) for p in pos]
    mass = [1.0] * len(pos) if mass is None else list(map(float, mass))
    while len(pos) > 1:
        best, bi, bj = 1e9, -1, -1
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dd = _mindist(pos[i], pos[j])
                if dd < best: best, bi, bj = dd, i, j
        if best >= r:
            break
        # merge bj into bi at mass-weighted minimal-image midpoint
        d = pos[bj] - pos[bi]; d = d - BOX * np.round(d / BOX)
        w = mass[bj] / (mass[bi] + mass[bj])
        pos[bi] = (pos[bi] + w * d) % BOX; mass[bi] += mass[bj]
        del pos[bj]; del mass[bj]
    return pos, mass


def couple_edges(pos, r=R_COUPLE):
    return [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos)) if _mindist(np.array(pos[i]), np.array(pos[j])) < r]


def network(pos, r_merge=R_MERGE, r_couple=R_COUPLE):
    mpos, mmass = merge_resolve(pos, r=r_merge)
    edges = couple_edges(mpos, r_couple)
    deg = [0] * len(mpos)
    for i, j in edges: deg[i] += 1; deg[j] += 1
    isolated = sum(1 for d in deg if d == 0)
    return {"n_final": len(mpos), "n_edges": len(edges), "isolated": isolated,
            "mean_degree": float(np.mean(deg)) if deg else 0.0}


def _load_library():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "PHASE_C_NODE_LIBRARY_*")))[-1]
    lib = json.load(open(os.path.join(d, "PHASE_C_NODE_LIBRARY.json")))
    for r in lib:
        r["pos_box"] = [np.array(c, float) / 96.0 for c in r.get("centroids", [])]   # vox -> box units
    return lib


def _stable(k): return "TRUE" in str(k) or "SATUR" in str(k)


def main():
    rng = np.random.default_rng(20260704)
    lib = _load_library()
    stable = [r for r in lib if _stable(r.get("klass"))]
    print(f"=== Phase D.6 reduced node model | library {len(lib)} configs ({len(stable)} stable) ===")
    print(f"    laws: r_merge={R_MERGE}, r_couple={R_COUPLE}, pinned (no drift)\n")

    # 1) node-count tendency: random dense ICs -> merge-resolve; which r_merge matches the library median count?
    lib_counts = np.array([r["n_nodes"] for r in stable])
    print(f"[1] node-count tendency (library stable: median={int(np.median(lib_counts))}, "
          f"mean={lib_counts.mean():.1f}, hist={dict(zip(*np.unique(lib_counts, return_counts=True)))})")
    for rm in (0.3, 0.4, 0.5, 0.6):
        finals = []
        for _ in range(300):
            k0 = rng.integers(8, 16)
            pos = rng.uniform(0, BOX, (k0, 3))
            finals.append(len(merge_resolve(list(pos), r=rm)[0]))
        finals = np.array(finals)
        print(f"    r_merge={rm}: random dense ICs -> final count median={int(np.median(finals))} "
              f"mean={finals.mean():.1f}")

    # 2) cooperative-stability exceptions: STABLE configs whose min spacing < r_merge (pairwise law says 'merge')
    coop = [r for r in stable if len(r["pos_box"]) >= 2 and r["nn_spacing_min"] < R_MERGE]
    print(f"\n[2] cooperative-stability exceptions (stable but min-spacing < r_merge={R_MERGE}): {len(coop)}/{len(stable)}")
    for r in coop[:8]:
        print(f"    {r['library_key']:42s} n={r['n_nodes']} min_sp={r['nn_spacing_min']:.3f}")

    # 3) merge-resolution replay on the library: does applying r_merge change the observed node count? (over-merge)
    changed = 0; iso_configs = 0
    for r in stable:
        if len(r["pos_box"]) < 2: continue
        nf = network(r["pos_box"])["n_final"]
        if nf != r["n_nodes"]: changed += 1
        if all(_mindist(r["pos_box"][i], r["pos_box"][j]) > R_COUPLE
               for i in range(len(r["pos_box"])) for j in range(i + 1, len(r["pos_box"]))):
            iso_configs += 1
    print(f"\n[3] replay on library stable configs: {changed}/{len(stable)} would over-merge under the pairwise "
          f"r_merge={R_MERGE}; {iso_configs} are fully isolated (all pairs > r_couple={R_COUPLE})")

    # 4) coupling graph over the library
    degs = [network(r["pos_box"])["mean_degree"] for r in stable if len(r["pos_box"]) >= 1]
    print(f"\n[4] library coupling graph (r_couple={R_COUPLE}): mean node degree = {np.mean(degs):.2f} "
          f"(most stable configs are fully connected small cliques)")

    out = os.path.join(ROOT, "sweep_runs", "PHASE_D6_REDUCED_MODEL")
    os.makedirs(out, exist_ok=True)
    json.dump({"r_merge": R_MERGE, "r_couple": R_COUPLE, "n_stable": len(stable),
               "coop_exceptions": len(coop), "over_merge": changed, "isolated_configs": iso_configs,
               "mean_degree": float(np.mean(degs))}, open(os.path.join(out, "reduced_model_validation.json"), "w"), indent=2)
    print(f"\n=== wrote {out} ===")


if __name__ == "__main__":
    main()
