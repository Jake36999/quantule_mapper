"""Gravity Audit C.3 — gradual backreaction (recursive observer-environment loop).

The B-R loop: psi_A -> I_{A|E} -> N_{A|E} -> d_t psi_A -> I_{A|E}. Minimal self-consistent model: a breathing clock
(harmonic oscillator q,p) whose instantaneous amplitude q(t)^2 IS its relational load, so N modulates with q and q
evolves under N. Introduced GRADUALLY via
    N(t) = (1 - lam_br)*N_frozen + lam_br*N_dynamic(t),   N_dynamic = 1/(1+beta*Ihat*q(t)^2),
with lam_br in {0, 0.1, 0.25, 0.5, 1}. Frozen (lam=0) is the retained control. Tracks: measured frequency, energy
drift, max|q| (runaway?), and return-to-frozen as lam->0 and dt refinement (real recursion vs numerical artifact).

Consistency check only; NOT time dilation/gravity. Standalone numpy ODE; production gravity ladder CLOSED.

  python jax_scout/gravity_C3_backreaction.py
"""
import os, sys, json, time
import numpy as np


def run(lam, betaIhat, omega=3.0, T=60.0, dt=0.002):
    q, p = 1.0, 0.0
    Nfroz = 1.0 / (1.0 + betaIhat * 1.0)          # q0^2 = 1
    def Nof(q):
        Ndyn = 1.0 / (1.0 + betaIhat * q * q)
        return (1.0 - lam) * Nfroz + lam * Ndyn
    def deriv(q, p):
        Nl = Nof(q)
        return Nl * p, -Nl * omega ** 2 * q
    n = int(round(T / dt)); qs = np.empty(n + 1); qs[0] = q; E = np.empty(n + 1)
    ts = np.arange(n + 1) * dt
    E[0] = 0.5 * (p ** 2 + omega ** 2 * q ** 2)
    for i in range(n):
        k1q, k1p = deriv(q, p)
        k2q, k2p = deriv(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
        k3q, k3p = deriv(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
        k4q, k4p = deriv(q + dt * k3q, p + dt * k3p)
        q += dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q); p += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        qs[i + 1] = q; E[i + 1] = 0.5 * (p ** 2 + omega ** 2 * q ** 2)
    zc = np.where(np.diff(np.sign(qs)))[0]
    freq = np.pi / np.mean(np.diff(ts[zc])) if len(zc) >= 3 else np.nan
    return {"lam": lam, "freq": float(freq), "freq_frac": float(freq / omega),
            "max_abs_q": float(np.max(np.abs(qs))), "energy_drift": float(abs(E[-1] - E[0]) / (E[0] + 1e-30)),
            "N_frozen": float(Nfroz)}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep_runs",
                       f"GRAVITY_C3_BACKREACT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    betaIhat = 1.0 / 0.8 - 1.0                     # weak field: N_frozen = 0.8 at q^2=1
    print(f"=== Gravity Audit C.3 — backreaction | beta*Ihat={betaIhat:.3f} (N_frozen=0.80) ===", flush=True)
    res = {"betaIhat": betaIhat, "rows": [], "refine": []}
    for lam in (0.0, 0.1, 0.25, 0.5, 1.0):
        r = run(lam, betaIhat)
        res["rows"].append(r)
        print(f"[lam_br={lam:.2f}] freq/omega={r['freq_frac']:.5f}  max|q|={r['max_abs_q']:.4f}  "
              f"E_drift={r['energy_drift']:.2e}", flush=True)
    # return-to-frozen: lam=0 must equal N_frozen; dt refinement at lam=1 (real recursion vs artifact)
    r0 = res["rows"][0]
    returns = abs(r0["freq_frac"] - r0["N_frozen"]) < 1e-3
    for dt in (0.004, 0.002, 0.001):
        rr = run(1.0, betaIhat, dt=dt); res["refine"].append({"dt": dt, "freq_frac": rr["freq_frac"], "max_abs_q": rr["max_abs_q"]})
        print(f"[refine lam=1 dt={dt:.3f}] freq/omega={rr['freq_frac']:.5f} max|q|={rr['max_abs_q']:.4f}", flush=True)
    conv = abs(res["refine"][-1]["freq_frac"] - res["refine"][-2]["freq_frac"]) < 1e-3
    bounded = all(r["max_abs_q"] < 5.0 for r in res["rows"])           # no runaway
    monotone = all(res["rows"][i]["freq_frac"] >= res["rows"][i + 1]["freq_frac"] - 1e-6 for i in range(len(res["rows"]) - 1))
    res["findings"] = {"lam0_returns_to_frozen": bool(returns), "bounded_no_runaway": bool(bounded),
                       "dt_convergent": bool(conv), "freq_monotone_with_lam": bool(monotone)}
    verdict = "C3_BACKREACTION_STABLE" if (returns and bounded and conv) else "C3_BACKREACTION_UNSTABLE_OR_ARTIFACT"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n[findings] lam0->frozen={returns}; bounded(no runaway)={bounded}; dt-convergent={conv}; "
          f"freq monotone in lam={monotone}", flush=True)
    print(f"=== {verdict} | recursive loop is {'STABLE + convergent (real, not artifact)' if verdict=='C3_BACKREACTION_STABLE' else 'unstable/artifact'} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
