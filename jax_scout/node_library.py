"""Phase D.3 — Phase C node-library cataloguer (READ-ONLY). Harvests node descriptors from already-saved fields.

Walks configured sweep_runs dirs, extracts per-config node descriptors from the saved psi (node count/centroids/
spacing/per-node mass+phase, current J=rho*grad(phi), vorticity, stress-tensor current-vs-density weight), joins the
stability label from the dir's CSV, and writes PHASE_C_NODE_LIBRARY.{csv,json}. No sims, no solver/gate change.

  wsl:  python jax_scout/node_library.py
"""
import os, sys, csv, json, glob
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css
from jax_scout import transfer_diag as td

N = 96
L = css.L_
DX = L / N
FLOOR = 1e-7

# dirs to harvest (diverse + stability-labelled): node-count families, a*/eta/rho grids, a* variants, controls
DIRS = [
    "FEB_BASIN_20260625_122824", "FEB_BASIN_CONFIRM_20260625_154503",
    "FEB_JOINT_BASIN_20260626_224056", "FEB_JOINT_STAGE2_20260627_123235",
    "FEB_CORE_DELINEATION_T24000_20260627_175050", "FEB_PARAM_BASIN_20260626_004039",
    "FEB_ASTAR_CONFIRM_20260702_003055", "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708",
]
CSV_KEYS = ("klass", "er_fin", "late_drift", "bounded_breathing", "n_fin", "T",
            "param_a", "param_eta", "param_rho_vac", "a_factor", "eta_factor", "rho_factor", "seed", "match")


def _curl(Fx, Fy, Fz, dx):
    gx = td._grad(Fx, dx); gy = td._grad(Fy, dx); gz = td._grad(Fz, dx)
    wx = gz[1] - gy[2]; wy = gx[2] - gz[0]; wz = gy[0] - gx[1]   # dFz/dy-dFy/dz, dFx/dz-dFz/dx, dFy/dx-dFx/dy
    return wx, wy, wz


def descriptors(psi, dx):
    rho = np.abs(psi) ** 2; rho_safe = np.maximum(rho, FLOOR)
    px, py, pz = td._grad(psi, dx)
    Jx = np.imag(np.conj(psi) * px); Jy = np.imag(np.conj(psi) * py); Jz = np.imag(np.conj(psi) * pz)
    Jmag = np.sqrt(Jx**2 + Jy**2 + Jz**2)
    wx, wy, wz = _curl(Jx, Jy, Jz, dx); wmag = np.sqrt(wx**2 + wy**2 + wz**2)
    sq = np.sqrt(rho_safe); gsx, gsy, gsz = td._grad(sq, dx)
    # stress-tensor terms (Frobenius) — current J(x)J/rho vs density-strain d√rho (x) d√rho
    Jc = (Jx, Jy, Jz); Gd = (gsx, gsy, gsz)
    curr = sum((Jc[i] * Jc[j] / rho_safe) ** 2 * (1 if i == j else 2)
               for i in range(3) for j in range(i, 3))
    dens = sum((Gd[i] * Gd[j]) ** 2 * (1 if i == j else 2)
               for i in range(3) for j in range(i, 3))
    Tcurr = float(np.sqrt(curr).mean()); Tdens = float(np.sqrt(dens).mean())
    nodes = td.detect_nodes(psi, dx)
    cens = [np.asarray(nd["centroid"], float) for nd in nodes]
    Ms = np.array([nd["M"] for nd in nodes]); phs = np.array([nd["phase"] for nd in nodes])
    # nearest-neighbour spacing (minimal image, box units)
    sp = []
    for a in range(len(cens)):
        dd = []
        for b in range(len(cens)):
            if a == b: continue
            d = cens[a] - cens[b]; d = d - N * np.round(d / N)
            dd.append(np.linalg.norm(d) / N)
        if dd: sp.append(min(dd))
    sp = np.array(sp) if sp else np.array([np.nan])
    # node-integrated current (mean |J| in each node's voxels)
    phase_spread = float(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phs)))))) if len(phs) else float("nan")
    return {
        "n_nodes": len(nodes),
        "nn_spacing_min": float(np.nanmin(sp)), "nn_spacing_mean": float(np.nanmean(sp)),
        "nn_spacing_max": float(np.nanmax(sp)),
        "mass_total": float(rho.sum()),
        "node_mass_mean": float(Ms.mean()) if len(Ms) else float("nan"),
        "node_mass_cv": float(Ms.std() / (Ms.mean() + 1e-30)) if len(Ms) else float("nan"),
        "phase_spread": phase_spread,
        "Jmag_mean": float(Jmag.mean()), "vort_mean": float(wmag.mean()),
        "rot_frac": float((wmag * dx).mean() / (Jmag.mean() + 1e-30)),
        "stress_frob_mean": float(np.sqrt(curr + dens).mean()),
        "stress_curr_frac": float(Tcurr / (Tcurr + Tdens + 1e-30)),
        "centroids": [c.tolist() for c in cens],
    }


def load_csv(d):
    files = glob.glob(os.path.join(d, "*.csv"))
    if not files:
        return {}
    rows = {}
    with open(files[0], newline="") as fh:
        for r in csv.DictReader(fh):
            k = r.get("key") or r.get("name") or ""
            rows[k] = {c: r.get(c) for c in CSV_KEYS if c in r}
    return rows


def main():
    base = os.path.join(ROOT, "sweep_runs")
    out = os.path.join(base, "PHASE_C_NODE_LIBRARY_20260704")
    os.makedirs(out, exist_ok=True)
    lib = []
    for dname in DIRS:
        d = os.path.join(base, dname)
        if not os.path.isdir(d):
            print(f"[skip] {dname} (missing)", flush=True); continue
        csv_rows = load_csv(d)
        npzs = sorted(glob.glob(os.path.join(d, "*.npz")))
        print(f"=== {dname}: {len(npzs)} states, {len(csv_rows)} csv rows ===", flush=True)
        for pth in npzs:
            key = os.path.basename(pth).replace("_probe.npz", "").replace(".npz", "")
            try:
                z = np.load(pth)
                if "psi_fin" not in z:
                    continue
                psi = z["psi_fin"].astype(np.complex128)
                if psi.shape != (N, N, N):
                    continue
                desc = descriptors(psi, DX)
            except Exception as exc:
                print(f"  [err] {key}: {str(exc)[:80]}", flush=True); continue
            meta = csv_rows.get(key, {})
            # loose CSV key match (strip suffixes) if exact miss
            if not meta:
                for ck, cv in csv_rows.items():
                    if key.startswith(ck) or ck.startswith(key):
                        meta = cv; break
            row = {"library_key": f"{dname}/{key}", "dir": dname, "key": key, **meta, **desc}
            lib.append(row)
            print(f"  {key:28s} n={desc['n_nodes']} sp_min={desc['nn_spacing_min']:.3f} "
                  f"mass_cv={desc['node_mass_cv']:.2f} rot={desc['rot_frac']:.2f} "
                  f"Tcurr_frac={desc['stress_curr_frac']:.2f} klass={meta.get('klass','?')}", flush=True)
    # write library
    flat_cols = [c for c in lib[0].keys() if c != "centroids"] if lib else []
    with open(os.path.join(out, "PHASE_C_NODE_LIBRARY.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_cols, extrasaction="ignore"); w.writeheader()
        for r in lib: w.writerow(r)
    json.dump(lib, open(os.path.join(out, "PHASE_C_NODE_LIBRARY.json"), "w"), indent=1, default=float)
    print(f"\n=== library: {len(lib)} configs -> {out}/PHASE_C_NODE_LIBRARY.{{csv,json}} ===", flush=True)
    # quick diversity summary
    if lib:
        import collections
        ncounts = collections.Counter(r["n_nodes"] for r in lib)
        klasses = collections.Counter(str(r.get("klass", "?")) for r in lib)
        print(f"node-count histogram: {dict(sorted(ncounts.items()))}", flush=True)
        print(f"stability klass histogram: {dict(klasses)}", flush=True)
        sp = np.array([r["nn_spacing_min"] for r in lib if np.isfinite(r["nn_spacing_min"])])
        print(f"NN spacing_min range: [{sp.min():.3f}, {sp.max():.3f}] box (n={sp.size})", flush=True)


if __name__ == "__main__":
    main()
