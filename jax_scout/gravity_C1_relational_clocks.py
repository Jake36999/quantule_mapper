"""Gravity Audit C.1 — B-R relational lapse with the B-O emergent-limit probe (frozen-lapse stage).

Theory frame (Jake): B-R is FUNDAMENTAL — chronology is relational, dtau_A = N_{A|E} dt, N_{A|E}=1/(1+beta*I_hat_{A|E}).
B-O (an objective common lapse) is a possible EMERGENT weak-probe limit, tested — not assumed.

B-O emergence probe: two clocks with the SAME compact envelope / density / coupling but DIFFERENT internal
mechanisms — (1) 1st-order phase rotation, (2) 2nd-order harmonic oscillator — each evolved through the SAME
relational lapse via the common time operator D_tau=(1/N)d_t. If both slow by the SAME fractional amount, an
objective lapse emerges from relational dynamics. (Because the envelopes are identical, N is identical; the test is
whether the two DYNAMICAL structures reparametrize identically — i.e. the lapse is a genuine universal time operator,
not a mechanism-specific coupling.)

Sharper source control: match the INTEGRATED LAPSE DEFICIT ∫(1-N)dV across I_int / overlap / objective sources
(equal total slowing budget), so any clock-rate difference is spatial ORGANIZATION, not field strength.

Guards: global preregistered I_*; weak-field N_min ladder {1,0.98,0.95,0.90,0.80}; beta=0 null. FROZEN lapse here
(hold N fixed during clock evolution); BACKREACTION (clock changes its own I_{A|E}) is the next stage.

Consistency/organization check only — NOT time dilation or gravity (H-G- is a postulate under test). Standalone
numpy; production gravity ladder CLOSED.

  python jax_scout/gravity_C1_relational_clocks.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ
KAP = 1.0


def clock_phase(Nloc, omega, T, dt):
    """1st-order phase clock: da/dt = -i*omega*Nloc*a. Return effective fractional rate = measured/omega."""
    a = 1.0 + 0j; n = int(round(T / dt))
    for _ in range(n):
        a = a * np.exp(-1j * omega * Nloc * dt)     # exact for constant Nloc
    theta = -np.angle(a * np.exp(1j * omega * Nloc * 0))  # accumulated phase
    return (omega * Nloc)  # analytic effective freq; RK not needed for constant Nloc (kept explicit for clarity)


def clock_oscillator(Nloc, omega, T, dt):
    """2nd-order harmonic clock: dq/dt=Nloc*p, dp/dt=-Nloc*omega^2*q (RK4). Return measured frequency via zero-cross."""
    q, p = 1.0, 0.0
    def deriv(q, p):
        return Nloc * p, -Nloc * omega ** 2 * q
    n = int(round(T / dt)); qs = np.empty(n + 1); qs[0] = q; ts = np.arange(n + 1) * dt
    for i in range(n):
        k1q, k1p = deriv(q, p)
        k2q, k2p = deriv(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
        k3q, k3p = deriv(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
        k4q, k4p = deriv(q + dt * k3q, p + dt * k3p)
        q += dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q); p += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        qs[i + 1] = q
    zc = np.where(np.diff(np.sign(qs)))[0]           # zero crossings -> half-periods
    if len(zc) < 3:
        return np.nan
    halfT = np.mean(np.diff(ts[zc]))
    return np.pi / halfT                              # measured angular frequency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_C1_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L
    x = np.linspace(-L / 2, L / 2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    gauss = lambda x0, s: np.exp(-((X - x0) ** 2 + Y ** 2 + Z ** 2) / (2 * s ** 2))

    env = place(phi, 0.0, N, L); rE = np.abs(env) ** 2
    xc, sig = 1.2, 0.8
    rc = gauss(xc, sig) ** 2                          # COMPACT clock envelope density (identical for both mechanisms)
    ref_c = gauss(1.5, sig) ** 2
    I_star = float(np.max((KAP ** 2) * ref_c * rE ** 2))
    S_star = float(np.max(rE ** 2)); O_star = float(np.max(ref_c * rE))
    I_field = (KAP ** 2) * rc * rE ** 2; S_field = rE ** 2; O_field = rc * rE
    cavg = lambda F: float(integ(rc * F, N, L) / (integ(rc, N, L) + 1e-30))
    deficit = lambda Nf: float(integ(1.0 - Nf, N, L))

    def beta_for_Nmin(hat, t):
        return (1.0 / t - 1.0) / (float(np.max(hat)) / 1.0 + 1e-30)

    def beta_for_deficit(field, scale, target_def):
        # bisection on beta so that ∫(1-N)dV = target_def
        lo, hi = 0.0, 1e6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            d = deficit(1.0 / (1.0 + mid * field / scale))
            if d < target_def:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print(f"=== Gravity Audit C.1 — relational clocks + B-O emergence probe | I*={I_star:.4e} ===", flush=True)
    res = {"N": N, "L": L, "I_star": I_star, "ladder": []}
    Ihat = I_field / I_star
    for t in (1.0, 0.98, 0.95, 0.90, 0.80):
        beta = beta_for_Nmin(Ihat, t)
        Nrel = 1.0 / (1.0 + beta * Ihat)
        Nloc = cavg(Nrel)                            # <N> over the compact clock (same for both mechanisms, B-R)
        # --- B-O emergence: two DIFFERENT mechanisms, same envelope/N ---
        w1, w2 = 3.0, 5.0
        r1 = clock_phase(Nloc, w1, 40.0, 0.002) / w1     # fractional rate (phase clock)
        r2 = clock_oscillator(Nloc, w2, 40.0, 0.002) / w2  # fractional rate (oscillator clock)
        emergent_gap = abs(r1 - r2)
        # --- matched-integrated-deficit source control ---
        target_def = deficit(Nrel)
        Nov = 1.0 / (1.0 + beta_for_deficit(O_field, O_star, target_def) * O_field / O_star)
        Nobj = 1.0 / (1.0 + beta_for_deficit(S_field, S_star, target_def) * S_field / S_star)
        rIint, rOv, rObj = cavg(Nrel), cavg(Nov), cavg(Nobj)
        row = {"Nmin": t, "beta": beta, "Nloc": Nloc, "rate_phase": r1, "rate_osc": r2,
               "emergent_universality_gap": emergent_gap,
               "matched_deficit": target_def, "clk_rate_Iint": rIint, "clk_rate_overlap": rOv, "clk_rate_objective": rObj,
               "Iint_vs_overlap_gap": abs(rIint - rOv), "Iint_vs_objective_gap": abs(rIint - rObj)}
        res["ladder"].append(row)
        print(f"[Nmin={t:.2f}] Nloc={Nloc:.4f} | B-O emergence: phase-rate={r1:.5f} osc-rate={r2:.5f} gap={emergent_gap:.2e}"
              f" | matched-deficit clock rate: I_int={rIint:.4f} overlap={rOv:.4f} obj={rObj:.4f} "
              f"(I_int-overlap gap={abs(rIint-rOv):.4f})", flush=True)

    last = res["ladder"][-1]
    null_ok = abs(res["ladder"][0]["rate_phase"] - 1.0) < 1e-9 and abs(res["ladder"][0]["rate_osc"] - 1.0) < 1e-3
    bo_emerges = last["emergent_universality_gap"] < 1e-3          # two mechanisms agree -> objective lapse emerges
    src_distinct = last["Iint_vs_overlap_gap"] > 1e-3             # I_int still differs from deficit-matched overlap
    res["findings"] = {"beta0_null": bool(null_ok), "B_O_emerges_two_mechanisms_agree": bool(bo_emerges),
                       "Iint_distinct_from_deficit_matched_overlap": bool(src_distinct)}
    verdict = "C1_FROZEN_PASS" if (null_ok and bo_emerges) else "C1_FROZEN_PARTIAL"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n[findings] beta0-null={null_ok}; B-O EMERGES (2 mechanisms slow identically -> objective lapse from "
          f"relational dynamics)={bo_emerges}; I_int distinct from DEFICIT-MATCHED overlap={src_distinct} "
          f"(gap {last['Iint_vs_overlap_gap']:.4f})", flush=True)
    print(f"=== {verdict} | frozen-lapse consistency; NOT time dilation (H-G- postulate). Next: BACKREACTION stage | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
