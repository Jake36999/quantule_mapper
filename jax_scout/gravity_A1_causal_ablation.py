"""Gravity Audit A.1 — causal-ordering / partner-ablation test.

Claim under test: the interaction-conditioned change (ΔF_int) PRECEDES and PREDICTS the subsequent divergence in
evolution — i.e. interaction forcing comes before the altered chronology, and removing the partner removes it.

Setup: identical C2 solitons L,R at sep S (in-phase). Evolve THREE fields with the frozen-baseline solver
(true-flat NLS): FULL (ψ_L+ψ_R), and the two ISOLATED counterfactuals (ψ_L alone, ψ_R alone). The linear ETDRK4
propagator is additive, so the divergence
    div(t) = ψ_full(t) − [ψ_L^iso(t) + ψ_R^iso(t)]
is driven ENTIRELY by the nonlinear interaction; div(0)=0.

Checks:
  (1) div(0)=0 and grows for t>0 — the divergence appears only after t=0 (caused, not present).
  (2) div(t) is ALIGNED with the t=0 interaction forcing ΔF_int(0)=nonlin(full)−nonlin(L)−nonlin(R): complex
      correlation |<div(t),ΔF_int(0)>|/(‖div‖‖ΔF_int‖) → 1 as t→0 (the forcing PREDICTS the divergence pattern).
  (3) PRECEDENCE: the relative field divergence ‖div‖/‖ψ‖ is non-negligible while the soliton COM has barely moved
      (interaction-conditioned change precedes macroscopic state change).
  Ablation is implicit: the isolated evolutions have no partner -> no divergence; div IS the partner's effect.

Mirror-only (jax_scout, param_geom_off true-flat); read-only re production; no matter/gravity claim.

  wsl:  ~/jax_irer/bin/python jax_scout/gravity_A1_causal_ablation.py [--sep 4.0 --T 2.0]
"""
import os, sys, json, argparse, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.phase_d_c2_8_two_node import build, axis, place, FAM


def nonlin(psi):
    rho = np.abs(psi) ** 2
    return (FAM["a"] * rho + FAM["s"] * rho ** 2 + FAM["f"] * rho ** 3) * psi


def evolve_snaps(psi0, ops, dt, ts):
    """Return {t: psi(t)} at requested times (ts sorted)."""
    pk = physics.initial_psi_k(jnp.asarray(psi0.astype(np.complex128)), ops)
    snaps = {0.0: psi0.astype(np.complex128)}
    tprev = 0.0
    for t in ts:
        nsteps = int(round((t - tprev) / dt))
        if nsteps > 0:
            pk = _evolve_chunk(pk, ops, nsteps)
        snaps[t] = np.asarray(jnp.fft.ifftn(pk))
        tprev = t
    return snaps


def core_com_x(rho_x, x, L, half="L"):
    """Circular COM of the density on the requested half of the box (periodic-safe)."""
    m = (x < 0) if half == "L" else (x >= 0)
    w = rho_x.copy(); w[~m] = 0.0
    th = 2 * np.pi * x / L
    C = float((w * np.cos(th)).sum()); S = float((w * np.sin(th)).sum())
    return L * np.arctan2(S, C) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--N", type=int, default=96); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--T", type=float, default=2.0); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    N, L, dt = A.N, A.L, A.dt
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A1_CAUSAL_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    ops = build(N, L, dt); x, _ = axis(N, L)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128)
    psiL0 = place(phi, -A.sep / 2, N, L); psiR0 = place(phi, +A.sep / 2, N, L)
    full0 = (psiL0 + psiR0)
    print(f"=== Gravity Audit A.1 — causal ablation | {FAM} | N={N} L={L} sep={A.sep} dt={dt} ===", flush=True)

    dF0 = nonlin(full0) - nonlin(psiL0) - nonlin(psiR0)        # interaction forcing at t=0 (i-stripped)
    nF0 = float(np.linalg.norm(dF0))
    comL0 = core_com_x((np.abs(full0) ** 2).sum(axis=(1, 2)), x, L, "L")

    ts = sorted(set([t for t in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, A.T] if t <= A.T] + [A.T]))
    sF = evolve_snaps(full0, ops, dt, ts)
    sL = evolve_snaps(psiL0, ops, dt, ts)
    sR = evolve_snaps(psiR0, ops, dt, ts)

    rows = []
    for t in ts:
        div = sF[t] - (sL[t] + sR[t])
        ndiv = float(np.linalg.norm(div))
        # complex alignment of div(t) with the t=0 interaction forcing dF0
        corr = abs(complex(np.vdot(dF0, div))) / (nF0 * ndiv + 1e-30)
        rel = ndiv / (float(np.linalg.norm(sF[t])) + 1e-30)   # relative field divergence
        comL = core_com_x((np.abs(sF[t]) ** 2).sum(axis=(1, 2)), x, L, "L")
        com_disp = abs(comL - comL0)
        rows.append({"t": t, "norm_div": ndiv, "align_with_dF0": corr, "rel_field_div": rel, "comL_disp": com_disp})
        print(f"[t={t:5.2f}] ||div||={ndiv:.4e}  align(dF0)={corr:.4f}  rel_field_div={rel:.3e}  COM_disp={com_disp:.4e}", flush=True)

    # checks
    grew = rows[-1]["norm_div"] > 10 * (rows[0]["norm_div"] + 1e-30) and rows[0]["norm_div"] > 0
    aligned_early = rows[0]["align_with_dF0"] > 0.95                     # forcing predicts the divergence at small t
    # precedence: at the earliest snapshot, field already diverging while COM ~unmoved
    early = rows[0]
    precedes = (early["rel_field_div"] > 10 * early["comL_disp"]) or (early["comL_disp"] < 1e-3 and early["rel_field_div"] > 1e-4)
    verdict = "CAUSAL_ORDERING_CONFIRMED" if (grew and aligned_early and precedes) else "CAUSAL_ORDERING_PARTIAL"
    res = {"N": N, "L": L, "sep": A.sep, "dt": dt, "dF0_norm": nF0, "rows": rows,
           "checks": {"divergence_grows_from_zero": bool(grew), "forcing_predicts_early_divergence": bool(aligned_early),
                      "interaction_precedes_COM_motion": bool(precedes)}, "verdict": verdict}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | grows={grew} early-align={aligned_early} precedes-COM={precedes} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
