"""
STAGE 2 — minimal ANISOTROPIC diffusion proxy (experimental, default-off).
Implements docs/ANISOTROPIC_METRIC_TENSOR_RFC.md as a small falsifiable test.

Adds, on top of the current-coupled A-field, an anisotropic diffusion term
    D_diff * div( (lam * Q_ij) grad_j psi )       (physics.step(..., q_tensor))
where Q_ij is a BOUNDED, SYMMETRIC, TRACELESS direction tensor (so it cannot inflate diffusion
globally). lam = 0 -> q_tensor = 0 -> EXACT baseline (equivalence gate). Contract:
IRER-SNCGL-ANISO-METRIC-ETDRK4-v1. NOT for Hunter/CuPy; segregated from scalar/A leaderboards.

Q sources tested (panel): stress (J_i J_j direction), density (d_i sqrt(rho) d_j ...), A (A_i A_j).
The question: does lam-on DROP global_mode_fraction (web->wires), rise pairwise, become more
bridge-selective, stay bounded, and do so MORE for gen18 (web->wires) than gen29/gen20/controls?

Verdicts: ANISOTROPIC_PROXY_PROMISING / ANISOTROPIC_PROXY_NO_SUPPORT / ANISOTROPIC_DISTORTION_REJECT.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_anisotropic.py
"""
import os, sys, json, glob, csv, time
from functools import partial
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td
from jax_scout import afield_current_coupled as cc
from jax_scout.afield_current_coupled import multiseed_ic, L, dt, order, capture_cc

CONTRACT_KEY = "IRER-SNCGL-ANISO-METRIC-ETDRK4-v1"
BASE_SEED = 20260619


def _q_from_vec(vx, vy, vz, lam):
    """Bounded traceless direction tensor of a vector field, scaled by lam."""
    vsq = vx*vx + vy*vy + vz*vz + 1e-30
    return (lam*(vx*vx/vsq - 1.0/3), lam*(vy*vy/vsq - 1.0/3), lam*(vz*vz/vsq - 1.0/3),
            lam*(vx*vy/vsq), lam*(vx*vz/vsq), lam*(vy*vz/vsq))


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 12, 13, 14))
def _capture_aniso(pvec, psi0, gamma_A, lam, N, L_, dt_, n_snap, stride, kappa, c_A, damp, q_source_id, rd, cd):
    ops = physics._ops_from_vec(pvec, N, L_, dt_, rd, cd)
    kx, ky, kz, k_sq = cc._kvecs(ops)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask
    z = jnp.zeros((N, N, N), cd); A0 = (z, z, z); Ad0 = (z, z, z)

    def q_tensor_of(psi_k_, Areal):
        psi = jnp.fft.ifftn(psi_k_)
        gx = jnp.fft.ifftn(ops.ikx*psi_k_); gy = jnp.fft.ifftn(ops.iky*psi_k_); gz = jnp.fft.ifftn(ops.ikz*psi_k_)
        if q_source_id == 0:        # stress: J = Im(conj psi grad psi)
            vx = jnp.imag(jnp.conj(psi)*gx); vy = jnp.imag(jnp.conj(psi)*gy); vz = jnp.imag(jnp.conj(psi)*gz)
        elif q_source_id == 1:      # density: grad sqrt(rho)
            s = jnp.sqrt(jnp.maximum(jnp.abs(psi)**2, ops.rho_floor)); s_k = jnp.fft.fftn(s)
            vx = jnp.real(jnp.fft.ifftn(ops.ikx*s_k)); vy = jnp.real(jnp.fft.ifftn(ops.iky*s_k)); vz = jnp.real(jnp.fft.ifftn(ops.ikz*s_k))
        else:                       # A direction
            vx, vy, vz = Areal
        return _q_from_vec(vx, vy, vz, lam)

    def inner(carry, _):
        psi_k, Ak, Adk = carry
        Ak, Adk, Areal = cc._update_avec(Ak, Adk, psi_k, ops, dt_, kx, ky, kz, k_sq, kappa, damp, c_A)
        a_vec = (gamma_A*Areal[0], gamma_A*Areal[1], gamma_A*Areal[2])
        q = q_tensor_of(psi_k, Areal)
        psi_k = physics.step(psi_k, ops, None, None, a_vec, q)
        return (psi_k, Ak, Adk), None

    def outer(carry, _):
        carry, _ = lax.scan(inner, carry, None, length=stride)
        return carry, jnp.fft.ifftn(carry[0])

    carry, snaps = lax.scan(outer, (psi_k, A0, Ad0), None, length=n_snap)
    finite = jnp.all(jnp.isfinite(jnp.abs(snaps[-1])))
    return snaps, finite


_QSRC = {"stress": 0, "density": 1, "A": 2}


def capture_aniso(par, ic, gamma_A, lam, N, steps, n_snap, kappa=1.0, c_A=1.0, q_source="stress"):
    stride = max(1, steps//n_snap); pv = jnp.asarray([par[k] for k in order])
    snaps, fin = _capture_aniso(pv, jnp.asarray(ic), float(gamma_A), float(lam), N, L, dt, n_snap,
                                stride, float(kappa), float(c_A), 0.0, _QSRC[q_source], jnp.float64, jnp.complex128)
    snaps = np.concatenate([np.asarray(ic)[None], np.asarray(snaps)], axis=0)
    return snaps, bool(fin)


def equivalence_check():
    print("=== lam=0 EQUIVALENCE (aniso proxy vs current-coupled A-on) ===")
    par = {"param_D": 4.5, "param_eta": 0.0, "param_rho_vac": 0.2, "param_omega0": 1.0,
           "param_a_coupling": 0.5, "param_s": -0.5, "param_f": 0.1, "param_a": 0.2}
    ic = multiseed_ic(32, BASE_SEED)
    base, _, _ = capture_cc(par, ic, 0.15, 32, 200, 20, kappa=1.0, c_A=1.0)
    agg, fin = capture_aniso(par, ic, 0.15, 0.0, 32, 200, 20, kappa=1.0, c_A=1.0, q_source="stress")
    rel = float(np.linalg.norm(agg[-1]-base[-1])/(np.linalg.norm(base[-1])+1e-30))
    print(f"  rel_L2 lam=0 vs A-on = {rel:.2e} (finite={fin})  {'PASS' if rel < 1e-10 else 'FAIL'}")
    return rel < 1e-10


# ---- global_mode under aniso (response matrix) ----
def _bump_d2(N, c):
    G = np.meshgrid(*([np.arange(N)]*3), indexing="ij")
    return sum(np.minimum((G[a]-c[a]) % N, (c[a]-G[a]) % N).astype(float)**2 for a in range(3))


def _phase_kick(psi, c, N, r, th=0.4):
    return (psi*np.exp(1j*th*np.exp(-_bump_d2(N, c)/(2*(r/1.5)**2)))).astype(np.complex128)


def global_mode_aniso(par, g, kap, cA, lam, q_source, N=48, settle=800, cont=2000, csnap=50):
    dx = L/N
    s0, fin = capture_aniso(par, multiseed_ic(N, BASE_SEED), g, lam, N, settle, 20, kappa=kap, c_A=cA, q_source=q_source)
    if not fin:
        return {"status": "settle_nonfinite"}
    psi0 = s0[-1]; nodes = td.detect_nodes(psi0, dx)
    if len(nodes) < 2:
        return {"status": "too_few_nodes"}
    nodes = sorted(nodes, key=lambda n: -n["E"]); nn = len(nodes)
    cents = [np.round(n["centroid"]).astype(int) % N for n in nodes]
    node_r = max(2, int(round(np.mean([n["size"] for n in nodes])**(1/3))))
    masks = [(_bump_d2(N, c) <= node_r*node_r) for c in cents]
    er = float(np.sum(np.abs(psi0)**2)/(np.sum(np.abs(s0[0])**2)+1e-30))
    geo = td.geometry_fields(psi0, par, dx); conds = []
    for i in range(nn):
        for j in range(i+1, nn):
            conds.append(td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"])
    bridge = float(np.max(conds)) if conds else 0.0

    def contf(p0):
        s, f = capture_aniso(par, p0, g, lam, N, cont, csnap, kappa=kap, c_A=cA, q_source=q_source); return s, f
    sc, f0 = contf(psi0)
    if not f0:
        return {"status": "control_nonfinite"}
    T = sc.shape[0]; ctrlE = np.array([[float(np.sum(np.abs(sc[t][m])**2)) for t in range(T)] for m in masks])
    M = np.full((nn, nn), np.nan)
    for i in range(nn):
        sb, fb = contf(_phase_kick(psi0, cents[i], N, node_r))
        if not fb:
            continue
        bE = np.array([[float(np.sum(np.abs(sb[t][m])**2)) for t in range(T)] for m in masks])
        peaks = (np.abs(bE-ctrlE)/(ctrlE[:, :1]+1e-30)).max(1)
        for j in range(nn):
            if j != i:
                M[i, j] = peaks[j]
    Mf = np.nan_to_num(M, nan=0.0)
    if not np.any(Mf):
        return {"status": "no_response", "n_nodes": nn, "er": er, "bridge": bridge}
    sv = np.linalg.svd(Mf, compute_uv=False); tot = float(np.sum(sv**2))
    gmf = float(sv[0]**2/tot) if tot > 0 else float("nan")
    return {"status": "ok", "n_nodes": nn, "global_mode_fraction": gmf, "pairwise_fraction": 1.0-gmf,
            "er": er, "bridge": bridge, "structure_gain": float(np.mean(Mf[Mf > 0])) if np.any(Mf > 0) else 0.0}


def classify(gm0, gmA, lam):
    if gmA.get("status") != "ok" or gm0.get("status") != "ok":
        return "ANISOTROPIC_DISTORTION_REJECT" if (gmA.get("status") in
               ("settle_nonfinite", "control_nonfinite")) else "ANISOTROPIC_PROXY_NO_SUPPORT"
    if not (0.5 <= gmA["er"] <= 2.0) or gmA["bridge"] > 0.9 or gmA["n_nodes"] > 8:
        return "ANISOTROPIC_DISTORTION_REJECT"
    drop = gm0["global_mode_fraction"] - gmA["global_mode_fraction"]
    if drop > 0.05 and gmA["pairwise_fraction"] > gm0["pairwise_fraction"]:
        return "ANISOTROPIC_PROXY_PROMISING"
    return "ANISOTROPIC_PROXY_NO_SUPPORT"


def main():
    if not equivalence_check():
        print("Equivalence FAILED; abort."); return
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    val = json.load(open(os.path.join(d, "afield_validation.json")))
    panel = [(v["params"], v["gamma_A"], v["kappa"], v["c_A"], v["label"], v["verdict"]) for v in val]
    LAM = 0.1; QSRC = "stress"   # bounded conservative regime (lambda scan: er~0.94, nodes preserved)
    print(f"\n=== STAGE 2 anisotropic proxy: lam={LAM} q_source={QSRC} (global_mode lam=0 vs lam-on) ===\n")
    report = []
    for par, g, kap, cA, label, verdict in panel:
        t0 = time.time()
        gm0 = global_mode_aniso(par, g, kap, cA, 0.0, QSRC)
        gmA = global_mode_aniso(par, g, kap, cA, LAM, QSRC)
        kl = classify(gm0, gmA, LAM)
        f0 = gm0.get("global_mode_fraction"); fA = gmA.get("global_mode_fraction")
        drop = (f0-fA) if (f0 is not None and fA is not None) else None
        print(f"[{label}] {verdict}")
        print(f"   global_mode: lam0={f0}  lam{LAM}={fA}  drop={drop if isinstance(drop,float) else 'n/a'} "
              f"| er={gmA.get('er')} bridge={gmA.get('bridge')} -> {kl}  ({time.time()-t0:.0f}s)\n")
        report.append({"label": label, "verdict_Aonly": verdict, "gmf_lam0": f0, "gmf_lamon": fA,
                       "drop": drop, "er_lamon": gmA.get("er"), "bridge_lamon": gmA.get("bridge"),
                       "aniso_verdict": kl, "lam": LAM, "q_source": QSRC})
    od = os.path.join(d, "afield_anisotropic.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    promising = [r for r in report if r["aniso_verdict"] == "ANISOTROPIC_PROXY_PROMISING"]
    g18 = [r for r in report if "WIRES" in str(r["verdict_Aonly"]).upper()]
    print(f"=== {len(promising)}/{len(report)} ANISOTROPIC_PROXY_PROMISING ===")
    if g18 and g18[0]["drop"] is not None:
        print(f"web->wires case (gen18) global_mode drop under lam-on: {g18[0]['drop']:.3f}")
    print("Stronger drop on gen18 than gen29/gen20/controls + bounded => anisotropy is the missing DOF "
          "(justifies a true tensor-geometry branch). Else: record "
          "MINIMAL_ANISOTROPIC_PROXY_DID_NOT_CONVERT... -> pivot to Payan-state RFC.")
    print(f"wrote {od}")


if __name__ == "__main__":
    main()
