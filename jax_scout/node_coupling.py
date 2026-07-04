"""Phase D.4 / D.2-rerun — node-node coupling analysis (READ-ONLY) on the harvested Phase C library.

The library finding: stable nodes are near-current-free, so coupling (if any) is density/phase/geometry-mediated.
For every node PAIR across the stable configs we compute NON-current coupling proxies and test whether they are
SPACING-DEPENDENT and survive a density-only null:
  - conductance  = density-bridge (min inter-node rho / node rho)      [geometry/connectivity]
  - phase_diff   = |Δφ| between the two nodes                          [phase]
  - divT_axial   = (∇·T_dens)·û on the inter-node segment              [effective directional force from the
                   density-strain stress T_dens = ∂√ρ ∂√ρ]            (NOT the current term, which is ~0)
  - dstress_axial= T_dens projected along û                            [axial density strain]
No solver/gate/physics change; no active source term. Runs on saved fields (WSL jax env; maths is numpy).

  wsl:  python jax_scout/node_coupling.py
"""
import os, sys, csv, json, glob
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css
from jax_scout import transfer_diag as td

N = 96; L = css.L_; DX = L / N; FLOOR = 1e-7
DIRS = ["FEB_BASIN_20260625_122824", "FEB_BASIN_CONFIRM_20260625_154503",
        "FEB_JOINT_BASIN_20260626_224056", "FEB_JOINT_STAGE2_20260627_123235",
        "FEB_CORE_DELINEATION_T24000_20260627_175050", "FEB_PARAM_BASIN_20260626_004039",
        "FEB_ASTAR_CONFIRM_20260702_003055", "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708"]


def _stable(klass):
    return ("TRUE" in str(klass)) or ("SATUR" in str(klass))


def dens_stress_and_div(psi, dx):
    """T_dens_ij = di√rho dj√rho, and its divergence (∇·T)_i = dj T_ij (force-density-like)."""
    rho = np.maximum(np.abs(psi) ** 2, FLOOR)
    sq = np.sqrt(rho); g = td._grad(sq, dx)                 # (gx,gy,gz)
    comp = {}
    for (nm, i, j) in [("xx",0,0),("yy",1,1),("zz",2,2),("xy",0,1),("xz",0,2),("yz",1,2)]:
        comp[nm] = g[i] * g[j]
    dTx = td._grad(comp["xx"], dx)[0] + td._grad(comp["xy"], dx)[1] + td._grad(comp["xz"], dx)[2]
    dTy = td._grad(comp["xy"], dx)[0] + td._grad(comp["yy"], dx)[1] + td._grad(comp["yz"], dx)[2]
    dTz = td._grad(comp["xz"], dx)[0] + td._grad(comp["yz"], dx)[1] + td._grad(comp["zz"], dx)[2]
    return comp, (dTx, dTy, dTz), rho


def pair_rows(psi, klass, dx):
    comp, divT, rho = dens_stress_and_div(psi, dx)
    nodes = td.detect_nodes(psi, dx)
    rows = []
    names = ("xx","yy","zz","xy","xz","yz")
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            ci = np.asarray(nodes[a]["centroid"], float); cj = np.asarray(nodes[b]["centroid"], float)
            d = ci - cj; d = d - N * np.round(d / N); spacing = float(np.linalg.norm(d)) / N
            u = (-d) / (np.linalg.norm(d) + 1e-30)          # unit axis i->j
            dphi = float(np.abs(np.angle(np.exp(1j * (nodes[a]["phase"] - nodes[b]["phase"])))))
            # sample along ci->cj
            rho_p, disp = td._sample_line(rho, ci, cj, N, td.PATH_SAMPLES)
            rho_ref = max(1e-30, 0.5 * (rho[tuple(np.round(ci).astype(int) % N)] + rho[tuple(np.round(cj).astype(int) % N)]))
            inner = slice(2, -2) if len(rho_p) > 4 else slice(None)
            conductance = float(rho_p[inner].min() / rho_ref)
            samp = {nm: td._sample_line(comp[nm], ci, cj, N, td.PATH_SAMPLES)[0] for nm in names}
            Tuu = (samp["xx"]*u[0]**2 + samp["yy"]*u[1]**2 + samp["zz"]*u[2]**2
                   + 2*(samp["xy"]*u[0]*u[1] + samp["xz"]*u[0]*u[2] + samp["yz"]*u[1]*u[2]))
            dvx = td._sample_line(divT[0], ci, cj, N, td.PATH_SAMPLES)[0]
            dvy = td._sample_line(divT[1], ci, cj, N, td.PATH_SAMPLES)[0]
            dvz = td._sample_line(divT[2], ci, cj, N, td.PATH_SAMPLES)[0]
            divT_axial = float(np.mean((dvx*u[0] + dvy*u[1] + dvz*u[2])[inner]))
            rows.append({"stable": _stable(klass), "spacing": spacing, "phase_diff": dphi,
                         "conductance": conductance, "dstress_axial": float(np.mean(Tuu[inner])),
                         "divT_axial": divT_axial, "rho_ref": rho_ref})
    return rows


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0 or a.size < 4: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _partial_corr(y, x, z):
    """corr(y,x) controlling for z (density) — residual-based."""
    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    if y.size < 5: return float("nan")
    def resid(v):
        A = np.vstack([z, np.ones_like(z)]).T
        c = np.linalg.lstsq(A, v, rcond=None)[0]; return v - A @ c
    return _corr(resid(y), resid(x))


def main():
    base = os.path.join(ROOT, "sweep_runs")
    out = os.path.join(base, "PHASE_D_NODE_COUPLING_20260704"); os.makedirs(out, exist_ok=True)
    allrows = []
    for dname in DIRS:
        d = os.path.join(base, dname)
        if not os.path.isdir(d): continue
        csvf = glob.glob(os.path.join(d, "*.csv"))
        labels = {}
        if csvf:
            for r in csv.DictReader(open(csvf[0], newline="")):
                labels[(r.get("key") or r.get("name") or "")] = r.get("klass", "?")
        for pth in sorted(glob.glob(os.path.join(d, "*.npz"))):
            key = os.path.basename(pth).replace("_probe.npz", "").replace(".npz", "")
            klass = labels.get(key) or next((v for k, v in labels.items() if key.startswith(k) or k.startswith(key)), "?")
            try:
                z = np.load(pth)
                if "psi_fin" not in z: continue
                psi = z["psi_fin"].astype(np.complex128)
                if psi.shape != (N, N, N): continue
                allrows += pair_rows(psi, klass, DX)
            except Exception as exc:
                print(f"[err] {dname}/{key}: {str(exc)[:70]}", flush=True)
    json.dump(allrows, open(os.path.join(out, "node_coupling_pairs.json"), "w"), indent=1, default=float)
    st = [r for r in allrows if r["stable"]]
    fa = [r for r in allrows if not r["stable"]]
    print(f"=== node-coupling: {len(allrows)} pairs ({len(st)} stable, {len(fa)} failing) ===", flush=True)
    if len(st) < 10:
        print("too few stable pairs", flush=True); return

    def col(rows, k): return [r[k] for r in rows]
    sp = col(st, "spacing")
    print("\n-- STABLE node-pairs: proxy vs SPACING (raw corr | partial corr controlling for rho_ref) --", flush=True)
    for k in ("conductance", "phase_diff", "dstress_axial", "divT_axial"):
        raw = _corr(col(st, k), sp)
        par = _partial_corr(col(st, k), sp, col(st, "rho_ref"))
        print(f"   {k:14s}: raw r={raw:+.3f}   partial(|rho) r={par:+.3f}", flush=True)
    # binned law shape (conductance & divT_axial vs spacing)
    sp_a = np.array(sp); bins = np.quantile(sp_a, [0, .25, .5, .75, 1.0])
    print("\n-- STABLE: binned means by spacing quartile --", flush=True)
    print(f"   {'spacing_bin':22s} {'n':>4} {'conductance':>12} {'phase_diff':>11} {'divT_axial':>12}", flush=True)
    for q in range(4):
        m = (sp_a >= bins[q]) & (sp_a <= bins[q + 1] if q == 3 else sp_a < bins[q + 1])
        sub = [st[i] for i in range(len(st)) if m[i]]
        if not sub: continue
        print(f"   [{bins[q]:.3f},{bins[q+1]:.3f}]        {len(sub):>4} {np.mean(col(sub,'conductance')):>12.3f} "
              f"{np.mean(col(sub,'phase_diff')):>11.3f} {np.mean(col(sub,'divT_axial')):>12.3e}", flush=True)
    # stable vs failing contrast
    print("\n-- STABLE vs FAILING (mean proxy) --", flush=True)
    for k in ("conductance", "phase_diff", "dstress_axial", "divT_axial"):
        print(f"   {k:14s}: stable={np.mean(col(st,k)):+.3e}  failing={np.mean(col(fa,k)) if fa else float('nan'):+.3e}", flush=True)
    json.dump({"n_stable_pairs": len(st), "n_fail_pairs": len(fa)}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(f"\n=== wrote {out} ===", flush=True)


if __name__ == "__main__":
    main()
