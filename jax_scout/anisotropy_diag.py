"""
STAGE 1 (passive) — anisotropy diagnostic. NO solver feedback.

Question (the Stage-2 gate): does a passive anisotropy tensor Q_ij derived from J_info /
stress-energy / density-gradient, computed on the A-ON settled field, LOCALIZE on the bridge,
ALIGN with the bridge axis, and DISTINGUISH the one web->wires case (gen18) from the failures
(gen29 anti-shift, gen20 no-shift)? If yes -> implement the minimal anisotropic proxy (Stage 2).
If no -> simple J/A tensors are insufficient (Payan-state coupling becomes the next RFC).

Candidate traceless Q_ij (per docs/ANISOTROPIC_METRIC_TENSOR_RFC.md, grounded in the ASTE brief's
"source geometry from T_munu" prescription):
  J-direction:  Jhat_i Jhat_j - d/3        (J = Im(psi* grad psi))
  stress:       J_i J_j / rho              (= rho dphi_i dphi_j, deviatoric)
  density-grad: d_i sqrt(rho) d_j sqrt(rho)  (deviatoric)
For each: in the BRIDGE region compute the mean 3x3 tensor, its fractional anisotropy (FA) and
principal axis; report FA and |principal-axis . bridge-axis|. Compare bridge vs a void region.

CAUTION: passive scout diagnostic. Not proof. No geometry change made here.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/anisotropy_diag.py
"""
import os, sys, json, glob, csv
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout.afield_current_coupled import capture_cc, multiseed_ic, L, order

SETTLE = 800
BASE_SEED = 20260619


def _grad(f, dx):
    return ((np.roll(f, -1, 0)-np.roll(f, 1, 0))/(2*dx),
            (np.roll(f, -1, 1)-np.roll(f, 1, 1))/(2*dx),
            (np.roll(f, -1, 2)-np.roll(f, 1, 2))/(2*dx))


def _bump_d2(N, c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def tensors(psi, dx):
    """Return per-voxel 3x3 tensor fields (as [3][3] lists of arrays) for J, stress, density."""
    rho = np.abs(psi)**2; rho_s = np.maximum(rho, 1e-7)
    gx, gy, gz = _grad(psi, dx); g = [gx, gy, gz]
    J = [np.imag(np.conj(psi)*g[i]) for i in range(3)]              # J_i
    Jmag = np.sqrt(sum(J[i]**2 for i in range(3))) + 1e-30
    Jhat = [J[i]/Jmag for i in range(3)]
    s = np.sqrt(rho_s); ds = _grad(s, dx)
    QJ = [[Jhat[i]*Jhat[j] for j in range(3)] for i in range(3)]    # direction tensor
    ST = [[J[i]*J[j]/rho_s for j in range(3)] for i in range(3)]    # stress (phase energy)
    QR = [[ds[i]*ds[j] for j in range(3)] for i in range(3)]        # density gradient
    return {"J_direction": QJ, "stress": ST, "density_grad": QR}


def region_tensor_stats(T, mask, u):
    """Mean 3x3 tensor over mask -> deviatoric -> FA + |principal axis . u|."""
    M = np.array([[float(T[i][j][mask].mean()) for j in range(3)] for i in range(3)])
    M = 0.5*(M+M.T)                                                 # symmetrize
    dev = M - np.trace(M)/3.0*np.eye(3)
    w, V = np.linalg.eigh(dev)
    lam = np.linalg.eigvalsh(M)
    denom = np.sqrt((lam**2).sum()) + 1e-30
    FA = float(np.sqrt(1.5*((lam-lam.mean())**2).sum())/denom)      # fractional anisotropy
    e = V[:, int(np.argmax(np.abs(w)))]                             # principal deviatoric axis
    align = float(abs(np.dot(e, u)))
    return FA, align


def analyze(par, g, kap, cA, label, verdict, N=48):
    dx = L/N
    snaps, _, fin = capture_cc(par, multiseed_ic(N, BASE_SEED), g, N, SETTLE, 20, kappa=kap, c_A=cA)
    if not fin:
        return {"label": label, "status": "nonfinite"}
    psi = snaps[-1]; nodes = td.detect_nodes(psi, dx)
    if len(nodes) < 2:
        return {"label": label, "status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"]); cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    geo = td.geometry_fields(psi, par, dx); best, bp = -1, (0, 1)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            if c > best:
                best, bp = c, (i, j)
    disp = (cents[bp[1]]-cents[bp[0]]).astype(float); disp = disp - N*np.round(disp/N)
    u = disp/(np.linalg.norm(disp)+1e-30)
    bpt = np.round(cents[bp[0]]+0.5*disp).astype(int) % N
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    bridge_mask = _bump_d2(N, bpt) <= node_r*node_r
    # void region for contrast
    rho = np.abs(psi)**2; far = np.ones((N, N, N), bool)
    for c in cents:
        far &= _bump_d2(N, c) > (2*node_r)**2
    void = np.array(np.unravel_index(np.argmin(np.where(far, rho, rho.max()+1)), rho.shape))
    void_mask = _bump_d2(N, void) <= node_r*node_r

    T = tensors(psi, dx)
    out = {"label": label, "status": "ok", "verdict": verdict, "bridge_conductance": float(best)}
    for nm, Tij in T.items():
        bFA, bAL = region_tensor_stats(Tij, bridge_mask, u)
        vFA, _ = region_tensor_stats(Tij, void_mask, u)
        out[nm] = {"bridge_FA": bFA, "bridge_align": bAL, "bridge_FA_x_align": bFA*bAL,
                   "void_FA": vFA}
    return out


def main():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    val = json.load(open(os.path.join(d, "afield_validation.json")))
    # map validation entries (gen18 web->wires, gen29/gen20 failures)
    targets = []
    for v in val:
        targets.append((v["params"], v["gamma_A"], v["kappa"], v["c_A"], v["label"], v["verdict"]))
    # add a no-corridor control (gamma_A=0 scalar web)
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def F(r, k):
        try: return float(r[k])
        except: return float("nan")
    nc = sorted([r for r in rows if r["klass"] == "A_BOUNDED_NO_BRIDGE" and 2 <= F(r, "nodes") <= 8
                 and 0.5 <= F(r, "er") <= 2.0], key=lambda r: F(r, "bridge"))
    if nc:
        r = nc[0]; targets.append(({k: F(r, k) for k in order}, 0.0, F(r, "kappa"), F(r, "c_A"),
                                   f"NEGCTRL_gen{r['gen']}", "scalar_web_control"))

    print("STAGE 1 passive anisotropy diagnostic (does bridge-aligned anisotropy distinguish web->wires?)\n")
    print(f"{'label':28} {'verdict':22} | tensor: bridge_FA  align  FA*align (void_FA)")
    report = []
    for par, g, kap, cA, label, verdict in targets:
        r = analyze(par, g, kap, cA, label, verdict)
        report.append(r)
        if r.get("status") != "ok":
            print(f"{label:28} {str(verdict)[:22]:22} | {r.get('status')}"); continue
        print(f"{label:28} {verdict[:22]:22} |")
        for nm in ("J_direction", "stress", "density_grad"):
            s = r[nm]
            print(f"    {nm:14} FA={s['bridge_FA']:.3f} align={s['bridge_align']:.3f} "
                  f"FAxAL={s['bridge_FA_x_align']:.3f} (void_FA={s['void_FA']:.3f})")
    od = os.path.join(d, "anisotropy_diag.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    # verdict: does the web->wires case (gen18) have higher bridge FA*align than the failures?
    ok = [r for r in report if r.get("status") == "ok"]
    shift = [r for r in ok if "WIRES" in str(r["verdict"]).upper()]
    noshift = [r for r in ok if r.get("verdict") and "WIRES" not in str(r["verdict"]).upper() and "control" not in r["verdict"]]
    print("\n=== does bridge-aligned anisotropy distinguish web->wires from failures? ===")
    for nm in ("J_direction", "stress", "density_grad"):
        sv = np.mean([r[nm]["bridge_FA_x_align"] for r in shift]) if shift else float("nan")
        nv = np.mean([r[nm]["bridge_FA_x_align"] for r in noshift]) if noshift else float("nan")
        print(f"  {nm:14}: web->wires FAxAL={sv:.3f}  vs  failures FAxAL={nv:.3f}  "
              f"{'-> DISTINGUISHES' if (sv==sv and nv==nv and sv>nv*1.3) else '-> no clear separation'}")
    print(f"\nwrote {od}")
    print("If a tensor distinguishes (web->wires FAxAL notably > failures): anisotropy is the missing "
          "directional DOF -> proceed to Stage 2 minimal anisotropic proxy. Else: J/A tensors "
          "insufficient -> Payan-state/phase-alignment coupling RFC next.")


if __name__ == "__main__":
    main()
