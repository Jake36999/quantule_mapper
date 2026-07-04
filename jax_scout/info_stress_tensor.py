"""Phase D.2 — informational stress-tensor bridge (READ-ONLY diagnostic). NOT a force/source term.

T_ij = kappa * rho * di_phi dj_phi  +  eta * di_sqrt(rho) dj_sqrt(rho)   (trace term optional; reported both ways)
with rho=|psi|^2, and rho*di_phi = J_i (the info current from transfer_diag.geometry_fields), so the phase term is
kappa * J_i J_j / rho. Reuses transfer_diag (detect_nodes, geometry_fields=Omega^2/R/J, corridor_pair_metrics) so
the tensor is the FIELD-level generalisation of the graph-level FMIA bridge diagnostic. Runs on saved Phase C/C1
fields (WSL jax env, because transfer_diag imports jax; the maths here is pure numpy).

  wsl:  python jax_scout/info_stress_tensor.py
"""
import os, sys, json
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css
from jax_scout import transfer_diag as td

N = 96
L = css.L_
DX = L / N
RHO_FLOOR = 1e-7


def stress_tensor(psi, geo, dx, kappa=1.0, eta=1.0):
    """6 unique components of the symmetric 3x3 T_info field + trace-free (deviatoric) version."""
    rho = geo["rho"]; rho_safe = np.maximum(rho, RHO_FLOOR)
    Jx, Jy, Jz = geo["J"]                              # J_i = rho * di_phi
    sq = np.sqrt(rho_safe)
    gx, gy, gz = td._grad(sq, dx)                      # di_sqrt(rho)
    J = (Jx, Jy, Jz); G = (gx, gy, gz)
    comp = {}
    idx = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    for name, (i, j) in idx.items():
        comp[name] = kappa * (J[i] * J[j]) / rho_safe + eta * (G[i] * G[j])
    return comp


def _tensor_at(comp, mask=None):
    """Stack the 6 components into an [...,3,3] array (optionally only where mask)."""
    def g(n): return comp[n][mask] if mask is not None else comp[n].ravel()
    xx, yy, zz, xy, xz, yz = (g(k) for k in ("xx", "yy", "zz", "xy", "xz", "yz"))
    T = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, zz], -1)], -2)
    return T


def pointwise_metrics(comp, rho):
    """||T|| (Frobenius), trace, deviatoric (shear) norm, anisotropy from eigenvalues; density-proxy correlation."""
    xx, yy, zz = comp["xx"], comp["yy"], comp["zz"]
    xy, xz, yz = comp["xy"], comp["xz"], comp["yz"]
    frob = np.sqrt(xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2))
    tr = xx + yy + zz
    # deviatoric (shear): remove isotropic part
    dxx, dyy, dzz = xx - tr / 3, yy - tr / 3, zz - tr / 3
    shear = np.sqrt(dxx**2 + dyy**2 + dzz**2 + 2 * (xy**2 + xz**2 + yz**2))
    # eigenvalue anisotropy only where the tensor is non-trivial (top-|T| voxels) to keep eigh cheap+meaningful
    thr = np.quantile(frob, 0.98)
    m = frob >= thr
    T = _tensor_at(comp, m)
    ev = np.linalg.eigvalsh(T)                          # ascending l3<=l2<=l1
    l1, l3 = ev[:, 2], ev[:, 0]
    aniso = (l1 - l3) / (np.abs(ev).sum(-1) + 1e-30)
    # density-proxy check: how much of ||T|| is explained by a monotone function of rho (rank corr)
    def rankcorr(a, b):
        a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
        return float(np.corrcoef(a, b)[0, 1])
    dproxy = rankcorr(frob.ravel(), rho.ravel())
    return {"frob_mean": float(frob.mean()), "frob_max": float(frob.max()),
            "shear_frac_mean": float((shear / (frob + 1e-30)).mean()),
            "aniso_top2pct_mean": float(aniso.mean()), "aniso_top2pct_p90": float(np.quantile(aniso, 0.9)),
            "frob_vs_rho_rankcorr": dproxy}, frob


def pair_metrics(comp, geo, nodes, N, dx):
    """Per node-pair: axial vs perpendicular projected stress on the inter-node segment, alongside the
    transfer_diag corridor metrics (conductance/path_align/J_flux). Endpoints (node interiors) excluded."""
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    out = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            ci, cj = np.asarray(nodes[a]["centroid"], float), np.asarray(nodes[b]["centroid"], float)
            dphi = float(np.abs(np.angle(np.exp(1j * (nodes[a]["phase"] - nodes[b]["phase"])))))  # |Δφ| in (0,pi]
            # sample each T component along ci->cj
            samp = {}
            for nm in names:
                s, disp = td._sample_line(comp[nm], ci, cj, N, td.PATH_SAMPLES)
                samp[nm] = s
            u = disp / (np.linalg.norm(disp) + 1e-30)
            # T_uu = u_i u_j T_ij along the path
            Tuu = (samp["xx"] * u[0]**2 + samp["yy"] * u[1]**2 + samp["zz"] * u[2]**2
                   + 2 * (samp["xy"] * u[0] * u[1] + samp["xz"] * u[0] * u[2] + samp["yz"] * u[1] * u[2]))
            frob = np.sqrt(samp["xx"]**2 + samp["yy"]**2 + samp["zz"]**2
                           + 2 * (samp["xy"]**2 + samp["xz"]**2 + samp["yz"]**2))
            inner = slice(2, -2) if len(Tuu) > 4 else slice(None)
            axial = float(np.mean(Tuu[inner]))
            axial_frac = float(np.mean(np.abs(Tuu[inner]) / (frob[inner] + 1e-30)))   # |axial|/||T|| on bridge
            corr = td.corridor_pair_metrics(geo, ci, cj, N, dx)
            spacing = float(np.linalg.norm(disp))
            out.append({"pair": (a, b), "spacing": spacing, "phase_diff": dphi, "axial_stress": axial,
                        "axial_frac": axial_frac, "bridge_frob_mean": float(frob[inner].mean()),
                        **corr})
    return out


def analyze(npz_path, params, label):
    psi = np.load(npz_path)["psi_fin"].astype(np.complex128)
    geo = td.geometry_fields(psi, params, DX)
    nodes = td.detect_nodes(psi, DX)
    comp = stress_tensor(psi, geo, DX)
    pm, frob = pointwise_metrics(comp, geo["rho"])
    pairs = pair_metrics(comp, geo, nodes, N, DX)
    # does axial bridge stress track corridor conductance across pairs? (the "stress sees bridges" test)
    r_cond = float("nan")
    if len(pairs) >= 3:
        af = np.array([p["axial_frac"] for p in pairs]); cd = np.array([p["conductance"] for p in pairs])
        if af.std() > 0 and cd.std() > 0:
            r_cond = float(np.corrcoef(af, cd)[0, 1])
    res = {"label": label, "n_nodes": len(nodes), "point": pm,
           "axialfrac_vs_conductance_corr": r_cond,
           "pairs": pairs, "mass": float(np.sum(np.abs(psi)**2))}
    print(f"[{label:16s}] nodes={len(nodes)} mass={res['mass']:.0f} | ||T||mean={pm['frob_mean']:.3e} "
          f"shear_frac={pm['shear_frac_mean']:.2f} aniso(top2%)={pm['aniso_top2pct_mean']:.2f} "
          f"||T||~rho rankcorr={pm['frob_vs_rho_rankcorr']:.2f} | axialfrac~conductance r={r_cond:+.2f}", flush=True)
    return res


def main():
    base = os.path.join(ROOT, "sweep_runs")
    feb = dict(css.FEB)
    def p(fac): d = dict(feb); d["param_a"] = float(feb["param_a"]) * fac; return d
    cases = [
        ("astar_4node", os.path.join(base, "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "a1.15_ladder_T72000_probe.npz"), p(1.15)),
        ("astar_seed620", os.path.join(base, "FEB_ASTAR_CONFIRM_20260702_003055", "a1.15_seed620_probe.npz"), p(1.15)),
        ("astar_seed621", os.path.join(base, "FEB_ASTAR_CONFIRM_20260702_003055", "a1.15_seed621_probe.npz"), p(1.15)),
        ("grower_a1.20", os.path.join(base, "FEB_ASTAR_CONFIRM_20260702_003055", "a1.20_probe.npz"), p(1.20)),
    ]
    out = os.path.join(base, "PHASE_D_STRESS_TENSOR_20260704")
    os.makedirs(out, exist_ok=True)
    results = []
    print("=== Phase D.2 informational stress-tensor diagnostic (READ-ONLY) ===", flush=True)
    for label, path, params in cases:
        if not os.path.exists(path):
            print(f"[{label}] MISSING {path}", flush=True); continue
        results.append(analyze(path, params, label))
    json.dump(results, open(os.path.join(out, "stress_tensor_results.json"), "w"), indent=2, default=float)
    print(f"=== wrote {out}/stress_tensor_results.json ===", flush=True)


if __name__ == "__main__":
    main()
