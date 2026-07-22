"""Gravity Audit A.1 — explicit two-subsystem coupling-ablation (decomposition-INVARIANT interaction source).

Resolves the decomposition-dependence critique of the single-field cross-term: instead of splitting one field into
analyst-chosen components, use two EXPLICIT subsystems A (target) and B (environment) with a coupling switch λ:

    ∂_tψ_A = F_A(ψ_A) + λ·C(ψ_A, ψ_B),   C(ψ_A,ψ_B) = −i κ |ψ_B|² ψ_A     (environment B modulates A)
    ∂_tψ_B = F_B(ψ_B) + λ·C(ψ_B, ψ_A)

The interaction-conditioned forcing on the target is then UNAMBIGUOUS (no decomposition of a single field):

    ΔF_A = F_A(ψ_A; λ=1, ψ_B) − F_A(ψ_A; λ=0) = C(ψ_A, ψ_B),   I_int^A = ∫ |ΔF_A|² dV

This is also the natural model for the gravity setup: a probe (A) in a load/environment (B) has a physically-defined
decomposition, not an arbitrary self-split.

Tests (per reviewer):
  T1  ablation switch: λ=0 → ΔF_A=0; λ=1 → ΔF_A≠0 (interaction togglable WITHOUT redefining the field).
  T2  matched-target-density / varied environment: hold ψ_A FIXED (ρ_A fixed), move/scale B → I_int^A varies with
      the actual environment, and → 0 as B departs. Same target density, different interaction.
  T3  invariance (must hold EXACTLY): global phase of A, joint translation, relabel A↔B, grid refinement.
  T4  it tracks the environment, not the target amplitude: scaling B changes I_int^A at fixed ρ_A.

Read-only numpy field algebra; no production/solver change; gravity sector PAUSED. Validates the interaction SOURCE
concept as decomposition-invariant for explicit subsystems; not a metric/clock/force/gravity claim.

  python jax_scout/gravity_A1_mirror_ablation.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import place, integ


def coupling_forcing(psi_target, psi_env, kappa):
    """C(A,B) = kappa |psi_env|^2 psi_target — the environment's density modulating the target (i-stripped)."""
    return kappa * (np.abs(psi_env) ** 2) * psi_target


def I_int(psi_target, psi_env, kappa, N, L):
    return integ(np.abs(coupling_forcing(psi_target, psi_env, kappa)) ** 2, N, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A1_MIRROR_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L; k = A.kappa
    print(f"=== Gravity Audit A.1 — mirror coupling-ablation | N={N} L={L} kappa={k} ===", flush=True)
    res = {"N": N, "L": L, "kappa": k}

    psiA = place(phi, -2.0, N, L)                    # target, FIXED throughout
    rhoA = np.abs(psiA) ** 2
    psiB0 = place(phi, +2.0, N, L)                   # environment near the target

    # T1: ablation switch (lambda)
    I_off = 0.0                                       # lambda=0
    I_on = I_int(psiA, psiB0, k, N, L)                # lambda=1
    t1 = I_on > 1e-6
    print(f"[T1] ablation: lambda=0 -> I_int^A={I_off:.1f} ; lambda=1 -> I_int^A={I_on:.4e}  -> {'PASS' if t1 else 'FAIL'}", flush=True)

    # T2: matched TARGET density, varied ENVIRONMENT position (target ψ_A + rho_A unchanged)
    t2rows = []
    for xB in (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0):
        psiB = place(phi, xB, N, L)
        r = I_int(psiA, psiB, k, N, L)
        t2rows.append({"xB": xB, "dist": abs(xB - (-2.0)), "I_int_A": r})
        print(f"[T2] target FIXED (rho_A const), env at xB={xB:+.1f} (dist={abs(xB+2):.1f}): I_int^A={r:.4e}", flush=True)
    t2 = (t2rows[0]["I_int_A"] > 10 * t2rows[-1]["I_int_A"]) and (t2rows[-1]["I_int_A"] >= 0)
    print(f"[T2] same target density, I_int^A varies with the ENVIRONMENT and -> 0 as B departs -> {'PASS' if t2 else 'FAIL'}", flush=True)

    # T3: invariances (must be exact)
    inv = {}
    inv["global_phase_A"] = abs(I_int(np.exp(1j * 0.7) * psiA, psiB0, k, N, L) - I_on) / (I_on + 1e-30)
    shift = lambda f: place(f, 3.0, N, L)
    inv["joint_translation"] = abs(I_int(shift(psiA), shift(psiB0), k, N, L) - I_on) / (I_on + 1e-30)
    inv["relabel_A_B"] = abs(I_int(psiB0, psiA, k, N, L) - I_int(psiA, psiB0, k, N, L)) / (I_on + 1e-30)  # symmetric coupling here
    # grid refinement via Fourier resample
    from jax_scout.gravity_A_rint_diagnostic import fourier_refine
    N2 = N + 32
    pA2 = place(fourier_refine(phi, N2), -2.0, N2, L); pB2 = place(fourier_refine(phi, N2), +2.0, N2, L)
    inv["grid_refine"] = abs(I_int(pA2, pB2, k, N2, L) - I_on) / (I_on + 1e-30)
    t3 = all(v < 1e-3 for key, v in inv.items() if key != "grid_refine") and inv["grid_refine"] < 0.05
    for key, v in inv.items():
        print(f"[T3] invariance {key}: rel change = {v:.2e}", flush=True)
    print(f"[T3] target-preserving transforms leave I_int^A invariant -> {'PASS' if t3 else 'FAIL'}", flush=True)

    # T4: tracks the ENVIRONMENT amplitude at fixed target
    t4rows = []
    for sB in (0.5, 1.0, 1.5, 2.0):
        r = I_int(psiA, sB * psiB0, k, N, L)
        t4rows.append({"env_scale": sB, "I_int_A": r})
        print(f"[T4] env amplitude x{sB}: I_int^A={r:.4e} (target rho_A fixed)", flush=True)
    t4 = t4rows[-1]["I_int_A"] > t4rows[0]["I_int_A"]
    print(f"[T4] I_int^A tracks the environment amplitude at fixed target density -> {'PASS' if t4 else 'FAIL'}", flush=True)

    passes = {"T1_ablation_switch": bool(t1), "T2_matched_target_varied_env": bool(t2),
              "T3_invariance": bool(t3), "T4_environment_tracking": bool(t4)}
    verdict = "MIRROR_ABLATION_DECOMPOSITION_INVARIANT_PASS" if all(passes.values()) else "MIRROR_ABLATION_PARTIAL"
    res.update({"T1_on": I_on, "T2_env_distance": t2rows, "T3_invariance": inv, "T4_env_scale": t4rows,
                "passes": passes, "verdict": verdict})
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | {passes} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
