"""TG-B2 P1 charge audit — expose the effective source/receiver charges behind the F(mass)~M^0 result.

Per the three 2026-07-18 review audits (Gate 0 / Priority 1): before any new source branch, decide whether the
absent mass trend is a NORMALIZATION artifact by exposing, for each w-family node, the extensive load vs the
NORMALIZED source charge vs the receiver charge. Decisive test:

  ∫E_dens (raw extensive load)  should scale with mass M.
  ∫S_state (NORMALIZED source charge, ∝ what the loop actually sources)  — flat or scaling?
  ∫|∇φ|²   (receiver gradient-energy weight in F_R = -c²∫∂_xA|∇φ|²)      — flat or scaling?
  e_ref, q_ref (per-node normalization maxima, recomputed per-w in the run) — how do they scale?

  -> If raw load scales with M but ∫S_state is FLAT, and the force is flat: normalization removes the extensive
     source coupling => M^0 is a design artifact (SUPPORTED).
  -> If ∫S_state scales but the force does not: the flatness is receiver/observable normalization, not the source.

Matches the run exactly: per-w π = -i·w·φ, per-w recomputed e_ref/q_ref (b1s.ref_source_norms), frozen S0. Also
classifies what F_R_well is. CPU-only; frozen modules imported unchanged; no new simulation; no gravity/IRER claim.
  python jax_scout/gravity_TG_B2_charge_audit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jax_scout import gravity_TG_B2_static_force as sf  # noqa: E402 (frozen physics + static G solve)

BASE = dict(sf.BASE)
S0 = sf.S0
W_FAMILY = [0.945, 0.955, 0.964, 0.972, 0.980]   # the characterization masses (105.9 .. 38.7)


def solve(w):
    phi, prof = sf.solve_single(BASE["N"], BASE["L"], w)
    return phi, prof


def node_charges(phi, w, X, Y, Z, KX, KY, KZ, dV):
    """All intermediates, per-w, matching the run's source construction (per-w pi, per-node e_ref/q_ref, frozen S0)."""
    c, m, a, s, f = BASE["c"], BASE["m"], BASE["a"], BASE["s"], BASE["f"]
    pi = -1j * w * phi
    rho = np.abs(phi) ** 2
    g2 = sf.grad2(phi, KX, KY, KZ)                                   # |grad phi|^2 (receiver weight density)
    Gpot = a * rho ** 2 / 2 + s * rho ** 3 / 3 + f * rho ** 4 / 4
    E_dens = np.abs(pi) ** 2 + c ** 2 * g2 + m ** 2 * rho - Gpot     # KG energy density (the raw source before norm)
    charge_dens = np.abs(np.imag(np.conj(phi) * pi))
    e_ref = float(np.max(np.abs(E_dens))); q_ref = float(np.max(charge_dens))
    raw = 0.5 * np.maximum(E_dens, 0.0) / (e_ref + 1e-12) + 0.5 * charge_dens / (q_ref + 1e-12)
    S_state = raw / S0                                              # the NORMALIZED source field (as sourced into T)
    # geometric response the node produces (single node at origin), static screened solve -> A-well
    G = sf.static_G(S_state, KX, KY, KZ)
    A_well = np.exp(+BASE["epsilon_G"] * G)
    # extensive / effective charges
    M = float(np.sum(rho) * dV)                                     # node mass (int rho)
    Q = float(np.sum(np.imag(np.conj(phi) * pi)) * dV)             # U(1) charge (signed) = -w*M for ground Q-ball
    E_tot = float(np.sum(E_dens) * dV)                            # raw extensive load (energy)
    Cabs_tot = float(np.sum(charge_dens) * dV)
    Q_source = float(np.sum(S_state) * dV)                         # NORMALIZED source monopole / effective source charge
    Q_recv = float(np.sum(g2) * dV)                                # receiver gradient-energy weight (extensive)
    rr = np.sqrt(X * X + Y * Y + Z * Z)
    src_rms = float(np.sqrt(np.sum(S_state * rr ** 2) * dV / (Q_source + 1e-30)))   # source spatial width
    return {"w": w, "M": M, "Q": Q, "E_tot": E_tot, "Cabs_tot": Cabs_tot,
            "e_ref": e_ref, "q_ref": q_ref,
            "Q_source_int_Sstate": Q_source, "Q_recv_int_grad2": Q_recv,
            "source_rms_width": src_rms, "amp": float(np.max(np.abs(phi))),
            "A_well_min": float(np.min(A_well)), "G_min": float(np.min(G)),
            "well_integral": float(np.sum(1.0 - A_well) * dV)}


def measured_force(w):
    """Pull the measured in-phase <F_R> at sep=3.0 for this w from the characterization run (if present)."""
    runs = sorted(ROOT.glob("sweep_runs/TG_B2_CHARACTERIZATION_*"))
    for d in reversed(runs):
        for jf in d.glob("ROW_*_COMPLETE.json"):
            try:
                r = json.loads(jf.read_text())
            except Exception:
                continue
            if abs(r.get("sep", -1) - 3.0) < 1e-9 and abs(r.get("w", -1) - w) < 1e-9:
                return r.get("F_R_well_avg")
    return None


def powerfit(x, y):
    x = np.asarray(x, float); y = np.abs(np.asarray(y, float))
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def main():
    import time
    out = ROOT / "sweep_runs" / f"TG_B2_CHARGE_AUDIT_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B2 P1 charge audit | CPU | out={out} ===", flush=True)
    X, Y, Z, KX, KY, KZ, dV = sf.grids(BASE["N"], BASE["L"])
    rows = []
    for w in W_FAMILY:
        phi, prof = solve(w)
        r = node_charges(phi, w, X, Y, Z, KX, KY, KZ, dV)
        r["qball_resid"] = float(prof["residual"]); r["F_R_measured"] = measured_force(w)
        rows.append(r)
        print(f"[w={w}] M={r['M']:7.3f} Q={r['Q']:8.3f} E_tot={r['E_tot']:8.3f} | e_ref={r['e_ref']:.4f} q_ref={r['q_ref']:.4f} "
              f"| Q_source(intS)={r['Q_source_int_Sstate']:.4f} Q_recv(int|gradphi|^2)={r['Q_recv_int_grad2']:.4f} "
              f"| F_R={r['F_R_measured']}", flush=True)

    M = [r["M"] for r in rows]
    scaling = {
        "F_measured ~ M^p": powerfit(M, [r["F_R_measured"] for r in rows if r["F_R_measured"] is not None] or [np.nan]*len(rows)),
        "E_tot(raw load) ~ M^p": powerfit(M, [r["E_tot"] for r in rows]),
        "Q_charge ~ M^p": powerfit(M, [r["Q"] for r in rows]),
        "e_ref ~ M^p": powerfit(M, [r["e_ref"] for r in rows]),
        "q_ref ~ M^p": powerfit(M, [r["q_ref"] for r in rows]),
        "Q_source(intSstate) ~ M^p": powerfit(M, [r["Q_source_int_Sstate"] for r in rows]),
        "Q_recv(int|gradphi|2) ~ M^p": powerfit(M, [r["Q_recv_int_grad2"] for r in rows]),
        "well_integral ~ M^p": powerfit(M, [r["well_integral"] for r in rows]),
    }
    # verdict logic
    p_load = scaling["E_tot(raw load) ~ M^p"]; p_src = scaling["Q_source(intSstate) ~ M^p"]
    p_recv = scaling["Q_recv(int|gradphi|2) ~ M^p"]; p_F = scaling["F_measured ~ M^p"]
    load_extensive = abs(p_load - 1.0) < 0.4 or p_load > 0.5
    src_flat = abs(p_src) < 0.3
    if load_extensive and src_flat:
        verdict = "NORMALIZATION_REMOVES_SOURCE_SCALING__M0_IS_DESIGN_ARTIFACT"
    elif not src_flat and abs(p_F) < 0.3:
        verdict = "SOURCE_SCALES_BUT_FORCE_FLAT__RECEIVER_OR_OBSERVABLE_NORMALIZATION"
    else:
        verdict = "INCONCLUSIVE_SEE_EXPONENTS"

    classification = ("F_R_well = -c^2 * Integral_{x>0} (d_x A) |grad phi|^2 dV -> a TOTAL (half-space) body FORCE, "
                      "NOT an acceleration and NOT mass-normalized. So a fixed-source M^0 in FORCE would imply "
                      "acceleration ~ 1/M (not universal free fall). No mass/volume division is inside F_R.")
    summary = {"verdict": verdict, "rows": rows, "scaling_exponents_vs_M": scaling,
               "F_R_classification": classification,
               "e_ref_q_ref_provenance": "recomputed PER-W (per single-node maxima) in ref_source_norms; S0 frozen global. This is the run's actual construction.",
               "note": "Decisive: does raw extensive load (E_tot) scale with M while normalized source charge (int S_state) stays flat? Then M^0 in force is the peak-normalization. Caveat: also check receiver Q_recv and whether S_state broadens (source_rms_width).",
               "boundary": "CPU re-analysis of frozen Q-balls; no new sim; mirror-only; no gravity/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print("\n=== scaling exponents vs mass M (F ~ M^p) ===", flush=True)
    for k, v in scaling.items():
        print(f"  {k:32s} p = {v:+.3f}", flush=True)
    print(f"\nsource_rms_width across w: {[round(r['source_rms_width'],3) for r in rows]}", flush=True)
    print(f"\n=== {verdict} ===", flush=True)
    print(classification, flush=True)


if __name__ == "__main__":
    main()
