"""
CURRENT-COUPLED A-field (the theory-relevant "rate of interaction" / FMIA-wire mechanism).
Implements docs/AFIELD_CURRENT_COUPLED_RFC.md.

A is a 3-vector potential sourced by the TRANSVERSE part of the informational current
J_info = Im(conj(psi) grad psi), evolved with finite speed:
    d^2 A_i/dt^2 = -c_A^2 k^2 A_i - Gamma dA_i/dt + kappa * P_T[J_info]_i   (Coulomb gauge, k=0 pinned)
and couples to psi by MINIMAL COUPLING (grad -> grad - i gamma_A A) via physics.step(..., a_vec),
which adds D_diff*(-2i a.grad psi - |a|^2 psi) with a = gamma_A * A_real.

Unlike the scalar density-sourced prototype (afield_prototype.py), this couples to the CURRENT
(which carries direction), so it CAN, in principle, convert the holistic collective web into
directed selective routing. Falsifiable prediction (vs gamma_A=0): global_mode_fraction DROPS,
node_bridge_selectivity RISES, phase coupling persists to 1600, routing becomes resolved +
bridge-selective, all bounded.

CAUTION: ACTIVE experimental branch. Contract key distinct; NOT rank-compatible with gamma_A=0;
JAX-scout only (no CuPy current-coupled term yet); NOT for Hunter/CuPy promotion. gamma_A=0 MUST
reproduce the baseline to ~machine eps (checked in __main__).

WSL2 jax venv:
  equivalence + sweep:  python /mnt/f/quantule_mapper/jax_scout/afield_current_coupled.py
"""
import os, sys, json, glob, time
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
from jax_scout import physics, transfer_diag as td, geometry_diag as gd

CONTRACT_KEY = "IRER-SNCGL-CURRENT-COUPLED-AFFECT-ETDRK4-v1"
L, dt = 10.0, 0.005
order = physics.SWEEP_PARAM_ORDER
BASE_SEED = 20260619
FLOOR = td.THR_PHASECOUP
KAPPA, DAMP = 1.0, 0.0
GAMMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]


def _kvecs(ops):
    kx = jnp.imag(ops.ikx); ky = jnp.imag(ops.iky); kz = jnp.imag(ops.ikz)
    k_sq = -jnp.real(ops.minus_k_sq)
    return kx, ky, kz, k_sq


def _zero_dc(Ak):
    return Ak.at[0, 0, 0].set(0.0)


def _transverse(Vx, Vy, Vz, kx, ky, kz, k_sq):
    inv = jnp.where(k_sq > 0, 1.0 / jnp.maximum(k_sq, 1e-30), 0.0)
    kdotV = kx * Vx + ky * Vy + kz * Vz
    return (Vx - kx * kdotV * inv, Vy - ky * kdotV * inv, Vz - kz * kdotV * inv)


def _j_source_k(psi_k, ops, kx, ky, kz, k_sq):
    psi = jnp.fft.ifftn(psi_k)
    gx = jnp.fft.ifftn(ops.ikx * psi_k); gy = jnp.fft.ifftn(ops.iky * psi_k); gz = jnp.fft.ifftn(ops.ikz * psi_k)
    Jx = jnp.imag(jnp.conj(psi) * gx); Jy = jnp.imag(jnp.conj(psi) * gy); Jz = jnp.imag(jnp.conj(psi) * gz)
    Jx_k = jnp.fft.fftn(Jx) * ops.dealias_mask
    Jy_k = jnp.fft.fftn(Jy) * ops.dealias_mask
    Jz_k = jnp.fft.fftn(Jz) * ops.dealias_mask
    Jx_k, Jy_k, Jz_k = _transverse(Jx_k, Jy_k, Jz_k, kx, ky, kz, k_sq)
    return _zero_dc(Jx_k), _zero_dc(Jy_k), _zero_dc(Jz_k)


def _update_avec(Ak, Adotk, psi_k, ops, dt_, kx, ky, kz, k_sq, kappa, damp, c_A):
    Js = _j_source_k(psi_k, ops, kx, ky, kz, k_sq)
    c_sq_k_sq = (c_A * c_A) * k_sq                       # tunable propagation speed
    nA, nAd, Areal = [], [], []
    for i in range(3):
        accel = -c_sq_k_sq * Ak[i] - damp * Adotk[i] + kappa * Js[i]
        adot = Adotk[i] + accel * dt_
        a = _zero_dc((Ak[i] + adot * dt_) * ops.dealias_mask)
        adot = _zero_dc(adot)
        nA.append(a); nAd.append(adot); Areal.append(jnp.real(jnp.fft.ifftn(a)))
    return tuple(nA), tuple(nAd), tuple(Areal)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 11, 12))
def _capture_cc(pvec, psi0, gamma_A, N, L_, dt_, n_snap, stride, kappa, damp, c_A, rd, cd):
    ops = physics._ops_from_vec(pvec, N, L_, dt_, rd, cd)
    kx, ky, kz, k_sq = _kvecs(ops)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask
    z = jnp.zeros((N, N, N), cd)
    A0 = (z, z, z); Ad0 = (z, z, z)

    def inner(carry, _):
        psi_k, Ak, Adk = carry
        Ak, Adk, Areal = _update_avec(Ak, Adk, psi_k, ops, dt_, kx, ky, kz, k_sq, kappa, damp, c_A)
        a_vec = (gamma_A * Areal[0], gamma_A * Areal[1], gamma_A * Areal[2])
        psi_k = physics.step(psi_k, ops, None, None, a_vec)
        return (psi_k, Ak, Adk), None

    def outer(carry, _):
        carry, _ = lax.scan(inner, carry, None, length=stride)
        psi_k, Ak, _ = carry
        A_real = jnp.real(jnp.fft.ifftn(Ak[0]))**2 + jnp.real(jnp.fft.ifftn(Ak[1]))**2 + jnp.real(jnp.fft.ifftn(Ak[2]))**2
        return carry, (jnp.fft.ifftn(psi_k), A_real)   # |A|^2 snapshot for localization/energy

    carry, (snaps, Asnaps) = lax.scan(outer, (psi_k, A0, Ad0), None, length=n_snap)
    finite = jnp.all(jnp.isfinite(jnp.abs(snaps[-1]))) & jnp.all(jnp.isfinite(Asnaps[-1]))
    return snaps, Asnaps, finite


def capture_cc(par, ic, gamma_A, N, steps, n_snap, kappa=KAPPA, damp=DAMP, c_A=1.0):
    stride = max(1, steps // n_snap)
    pv = jnp.asarray([par[k] for k in order])
    snaps, Asnaps, fin = _capture_cc(pv, jnp.asarray(ic), float(gamma_A), N, L, dt, n_snap, stride,
                                     float(kappa), float(damp), float(c_A), jnp.float64, jnp.complex128)
    snaps = np.concatenate([np.asarray(ic)[None], np.asarray(snaps)], axis=0)
    return snaps, np.asarray(Asnaps), bool(fin)


def multiseed_ic(N, seed, K=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


# ---- analysis (reuse transfer_diag pipeline on captured snaps) ----
def analyze(snaps, par, N):
    dx = L/N
    snap_nodes = [td.detect_nodes(s, dx) for s in snaps]
    tracks = td.track_nodes(snap_nodes, N)
    out = {"n_persistent_nodes": len(tracks), "phase_coupling_score": 0.0, "energy_exchange_index": 0.0,
           "max_cond": 0.0}
    if len(tracks) >= 2:
        Tn = len(snaps); rng = np.random.default_rng(td.SURR_SEED)
        E_tot = np.array([float(np.sum(np.abs(s)**2)) for s in snaps])
        gph = np.unwrap(np.array([float(np.angle(np.sum(s*np.abs(s)**2))) for s in snaps]))
        gi = np.arange(len(gph)); gsl, gint = np.polyfit(gi, gph, 1); dphi_glob = gph-(gsl*gi+gint)
        prepped = [td.prep_track(t, E_tot, dphi_glob) for t in tracks]
        pcs, exs = [], []
        for i in range(len(prepped)):
            for j in range(i+1, len(prepped)):
                tm = td.temporal_pair_metrics(prepped[i], prepped[j], Tn, rng)
                if tm:
                    pcs.append(tm["phase_couple_excess"]); exs.append(tm["E_exchange_excess"])
        out["phase_coupling_score"] = float(np.mean(pcs)) if pcs else 0.0
        out["energy_exchange_index"] = float(np.mean(exs)) if exs else 0.0
    nodes = td.detect_nodes(snaps[-1], dx)
    if len(nodes) >= 2:
        geo = td.geometry_fields(snaps[-1], par, dx); conds = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                conds.append(td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"])
        out["max_cond"] = float(np.max(conds)) if conds else 0.0
    return out


def equivalence_check():
    print("=== gamma_A=0 EQUIVALENCE (current-coupled vs pure-physics baseline) ===")
    par = {"param_D": 4.964, "param_eta": 0.0, "param_rho_vac": 0.2, "param_omega0": 1.0,
           "param_a_coupling": 0.5, "param_s": -0.5, "param_f": 0.1, "param_a": 0.2}
    ic = multiseed_ic(32, BASE_SEED)
    base, _ = td.capture_trajectory([par[k] for k in order], ic, 32, L, dt, 200, 20)
    cc, _, fin = capture_cc(par, ic, 0.0, 32, 200, 20)
    rel = float(np.linalg.norm(cc[-1]-base[-1]) / (np.linalg.norm(base[-1])+1e-30))
    print(f"  rel_L2(psi_final) gamma_A=0 vs baseline = {rel:.2e}  (finite={fin})")
    print("  PASS (machine eps)" if rel < 1e-10 else "  FAIL — gamma_A=0 not equal to baseline!")
    return rel < 1e-10


def main():
    ok = equivalence_check()
    if not ok:
        print("Equivalence failed; aborting sweep."); return
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "BRIDGE_HUNT_2026*")))[-1]
    fz = json.load(open(os.path.join(d, "frozen_finalists.json")))
    STEPS, NSNAP = 1600, 40
    print(f"\n=== CURRENT-COUPLED A-FIELD SWEEP (J_info-sourced, minimal coupling) — top 3 finalists ===")
    print(f"contract={CONTRACT_KEY}  gamma_A sweep {GAMMAS}  (falsify: does phase coupling persist >{FLOOR} at 1600?)\n")
    report = []
    for fr in fz["finalists"][:3]:
        par = {k: float(fr["params"][k]) for k in order}; label = f"gen{fr['generation']}_{fr['config_hash']}"
        ic = multiseed_ic(48, BASE_SEED); print(f"[{label}]")
        for g in GAMMAS:
            t0 = time.time()
            snaps, Asnaps, fin = capture_cc(par, ic, g, 48, STEPS, NSNAP)
            amp = float(np.max(np.abs(snaps[-1]))) if fin else float("inf")
            er = float(np.sum(np.abs(snaps[-1])**2)/(np.sum(np.abs(snaps[0])**2)+1e-30))
            curv = gd.curvature_max_only(snaps[-1], par, L/48) if fin else float("inf")
            A_E = float(np.sum(Asnaps[-1])) if fin else float("nan")    # sum |A|^2
            a = analyze(snaps, par, 48) if fin else {"phase_coupling_score": 0.0, "n_persistent_nodes": 0, "energy_exchange_index": 0.0, "max_cond": 0.0}
            tag = "  >floor" if a["phase_coupling_score"] > FLOOR else ""
            print(f"   gA={g:<5} fin={int(bool(fin))} nP={a['n_persistent_nodes']} "
                  f"pcoup@1600={a['phase_coupling_score']:.3f}{tag} exch={a['energy_exchange_index']:.3f} "
                  f"bridge={a['max_cond']:.3f} er={er:.2f} curv={curv:.2f} A|^2={A_E:.2g} amp={amp:.1f}  ({time.time()-t0:.0f}s)")
            report.append({"label": label, "gamma_A": g, "finite": bool(fin), **a, "er": er,
                           "curv": float(curv), "A_sq_energy": A_E, "amp": amp})
        print()
    od = os.path.join(d, "afield_current_coupled.json")
    json.dump(report, open(od, "w"), indent=2, default=float)
    base = {r["label"]: None for r in report}
    print("=== summary: does current-coupled A raise phase coupling at 1600 vs gamma_A=0? ===")
    for lbl in dict.fromkeys(r["label"] for r in report):
        rs = [r for r in report if r["label"] == lbl and r["finite"]]
        if rs:
            g0 = [r for r in rs if r["gamma_A"] == 0.0]
            base_pc = g0[0]["phase_coupling_score"] if g0 else float("nan")
            best = max(rs, key=lambda r: r["phase_coupling_score"])
            print(f"  {lbl}: gamma0 pcoup={base_pc:.3f} -> best pcoup={best['phase_coupling_score']:.3f} "
                  f"at gamma_A={best['gamma_A']} (bridge={best['max_cond']:.3f}, er={best['er']:.2f})")
    print(f"\nwrote {od}")
    print("INTERPRETATION: pcoup rises above 0.73 at gamma_A>0 (bounded) => current coupling rescues "
          "long-lived transfer (run full global_mode/routing characterization next). Flat/no rise => "
          "current-coupled minimal form (this kappa/c_A) does not rescue at these gamma_A.")


if __name__ == "__main__":
    main()
