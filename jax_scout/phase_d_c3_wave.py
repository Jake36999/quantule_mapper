"""Phase D / C3 — second-order (wave) kinetic substrate: the INERTIA candidate. Prototype per
docs/PHASE_D_C3_WAVE_KINETIC_RFC.md. Complex nonlinear Klein-Gordon:  psi_tt = c^2 lap psi - m^2 psi + g(rho) psi,
g(rho) = a*rho + s*rho^2 + f*rho^3. Second-order in time => momentum lives in the field CONFIGURATION (P = -Re<pi|grad psi>),
so the C2 flow-through escape (momentum streaming through static density as phase flow) is structurally impossible.

Standalone module: does NOT touch physics.py Ops or any Phase C / first-order path. State = (psi_k, pi_k), pi = psi_t.
Stepper = Strang split: half nonlinear kick (pi += dt/2 g(rho)psi, local) -> exact per-mode linear rotation of the
harmonic oscillator omega_k^2 = c^2 k^2 + m^2 -> half kick. Time-reversible; E/Q drift = O(dt^2) splitting error only.
Native objects = Q-balls phi(r)e^{-i w t}, 0<w<m: stationary (m^2-w^2)phi = c^2 lap phi + g(phi^2)phi = C2 Petviashvili
with mu->m^2-w^2, D->c^2. Transport = configurational velocity kick pi0 = -v.grad phi - i w phi (moving Q-ball IC).

Gates: G1 linear parity (g=0 == analytic rotation), G2 invariant telemetry (E/Q, dt-ladder O(dt^2)),
G3 Q-ball existence scan (Petviashvili in w), G4 velocity-kick transport (density co-moves at v?), G5 unkicked null.

  wsl:  python jax_scout/phase_d_c3_wave.py [--out DIR]
"""
import os, sys, json, time, argparse
from functools import partial
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------- KG operator + Strang stepper ----------------
def build_kg(N, L, c, m, dt, dealias_frac=0.5):
    k1 = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k_sq = kx ** 2 + ky ** 2 + kz ** 2
    w = np.sqrt(c ** 2 * k_sq + m ** 2)                      # omega_k > 0 for m>0
    C = np.cos(w * dt)
    Sw = np.where(w > 1e-30, np.sin(w * dt) / w, dt)         # sin(w dt)/w  (-> dt as w->0)
    wS = w * np.sin(w * dt)
    kmag = np.sqrt(k_sq)
    mask = (kmag <= dealias_frac * kmag.max()).astype(np.float64)
    return {"N": N, "L": L, "c": c, "m": m, "dt": dt, "k_sq": jnp.asarray(k_sq),
            "ikx": jnp.asarray(1j * kx), "iky": jnp.asarray(1j * ky), "ikz": jnp.asarray(1j * kz),
            "C": jnp.asarray(C), "Sw": jnp.asarray(Sw), "wS": jnp.asarray(wS), "mask": jnp.asarray(mask),
            "w": jnp.asarray(w)}


def _g(rho, a, s, f):
    return a * rho + s * rho ** 2 + f * rho ** 3


@partial(jax.jit, static_argnames=("n_steps",))
def kg_evolve(psi_k, pi_k, op, a, s, f, n_steps):
    C, Sw, wS, mask = op["C"], op["Sw"], op["wS"], op["mask"]
    dt = op["dt"]

    def kick_half(psi_k, pi_k):
        psi = jnp.fft.ifftn(psi_k)
        rho = jnp.real(psi) ** 2 + jnp.imag(psi) ** 2
        return pi_k + (0.5 * dt) * jnp.fft.fftn(_g(rho, a, s, f) * psi) * mask

    def body(carry, _):
        psi_k, pi_k = carry
        pi_k = kick_half(psi_k, pi_k)                        # B(dt/2)
        psi_new = C * psi_k + Sw * pi_k                      # A(dt): exact per-mode oscillator rotation
        pi_new = -wS * psi_k + C * pi_k
        pi_new = kick_half(psi_new, pi_new)                  # B(dt/2)
        return (psi_new, pi_new), None

    (psi_k, pi_k), _ = jax.lax.scan(body, (psi_k, pi_k), None, length=n_steps)
    return psi_k, pi_k


def invariants(psi_k, pi_k, op, a, s, f):
    N, L, c, m = op["N"], op["L"], op["c"], op["m"]
    dV = (L / N) ** 3
    psi = np.asarray(jnp.fft.ifftn(psi_k)); pi = np.asarray(jnp.fft.ifftn(pi_k))
    rho = np.abs(psi) ** 2
    gx = np.asarray(jnp.fft.ifftn(op["ikx"] * psi_k)); gy = np.asarray(jnp.fft.ifftn(op["iky"] * psi_k))
    gz = np.asarray(jnp.fft.ifftn(op["ikz"] * psi_k))
    grad2 = np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2
    G = a * rho ** 2 / 2 + s * rho ** 3 / 3 + f * rho ** 4 / 4
    E = float(np.sum(np.abs(pi) ** 2 + c ** 2 * grad2 + m ** 2 * rho - G) * dV)
    Q = float(np.sum(np.imag(np.conj(psi) * pi)) * dV)
    Px = float(-np.sum(np.real(np.conj(pi) * gx)) * dV)
    mass = float(np.sum(rho) * dV)
    return {"E": E, "Q": Q, "Px": Px, "mass": mass, "amp": float(np.abs(psi).max())}


def centroid_x(rho, N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    th = 2 * np.pi * x / L
    Rx = rho.sum(axis=(1, 2))
    C_ = np.sum(Rx * np.cos(th)); S_ = np.sum(Rx * np.sin(th))
    return float(np.arctan2(S_, C_) * L / (2 * np.pi))


# ---------------- Q-ball existence (Petviashvili in omega) ----------------
def gaussian(A, sig, N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return (A * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))).astype(np.complex128)


def qball_petviashvili(op, a, s, f, mu, A=1.0, sig=1.2, iters=800, chunk=50, gamma=1.5, tol=1e-11):
    """Solve mu*phi = c^2 lap phi + g(phi^2) phi, mu = m^2 - w^2 > 0. Real profile from a Gaussian seed."""
    c = op["c"]
    denom = jnp.asarray(mu + c ** 2 * np.asarray(op["k_sq"]))     # > 0
    mask = op["mask"]
    pk = jnp.fft.fftn(jnp.asarray(gaussian(A, sig, op["N"], op["L"]))) * mask

    @partial(jax.jit, static_argnames=("n",))
    def run(pk, n):
        def body(pk, _):
            phi = jnp.fft.ifftn(pk)
            rho = jnp.real(phi) ** 2 + jnp.imag(phi) ** 2
            Nk = jnp.fft.fftn(_g(rho, a, s, f) * phi) * mask
            S = jnp.sum(jnp.real(jnp.conj(pk) * (denom * pk))) / jnp.sum(jnp.real(jnp.conj(pk) * Nk))
            return (jnp.abs(S) ** gamma) * Nk / denom, jnp.abs(S)
        pk, Ss = jax.lax.scan(body, pk, None, length=n)
        return pk, Ss[-1]

    prev = np.asarray(jnp.fft.ifftn(pk))
    for it in range(0, iters, chunk):
        pk, S = run(pk, chunk)
        cur = np.asarray(jnp.fft.ifftn(pk))
        if not np.isfinite(cur).all() or np.abs(cur).max() < 1e-6:
            return None, {"fail": "COLLAPSE"}
        drel = float(np.linalg.norm(cur - prev) / (np.linalg.norm(cur) + 1e-300)); prev = cur
        if drel < tol:
            break
    # residual of the stationary equation
    pkn = np.fft.fftn(cur); lap = np.fft.ifftn(-np.asarray(op["k_sq"]) * pkn); rho = np.abs(cur) ** 2
    Hphi = c ** 2 * lap + _g(rho, a, s, f) * cur
    mu_chk = float(np.real(np.vdot(cur, Hphi)) / np.real(np.vdot(cur, cur)))
    res = float(np.linalg.norm(Hphi - mu_chk * cur) / (abs(mu_chk) * np.linalg.norm(cur) + 1e-300))
    occ = (rho.sum() ** 2 / (rho ** 2).sum()) / rho.size
    return cur, {"residual": res, "mu": mu, "mu_check": mu_chk, "S": float(S),
                 "amp": float(np.abs(cur).max()), "occ": float(occ), "mass": float(rho.sum())}


# ---------------- gates ----------------
def g1_linear_parity(op):
    N = op["N"]
    rng = np.random.default_rng(0)
    psi_k = jnp.asarray(np.fft.fftn(rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))))
    pi_k = jnp.asarray(np.fft.fftn(rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))))
    psi1, pi1 = kg_evolve(psi_k, pi_k, op, 0.0, 0.0, 0.0, 1)          # g=0 -> pure linear
    C, Sw, wS = op["C"], op["Sw"], op["wS"]
    psi_a = C * psi_k + Sw * pi_k; pi_a = -wS * psi_k + C * pi_k      # analytic per-mode rotation
    e = float(jnp.max(jnp.abs(psi1 - psi_a)) + jnp.max(jnp.abs(pi1 - pi_a)))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=48); ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=1.0); ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8); ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1); ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C3_WAVE_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    op = build_kg(A.N, A.L, A.c, A.m, A.dt)
    a, s, f = A.a, A.s, A.f
    print(f"=== C3 WAVE-KINETIC | N={A.N} L={A.L} c={A.c} m={A.m} dt={A.dt} | g=({a},{s},{f}) | out={out} ===", flush=True)
    res = {"config": vars(A)}

    # G1 linear parity
    e1 = g1_linear_parity(op)
    res["G1_linear_parity_err"] = e1
    print(f"[G1] linear parity (g=0 == analytic rotation): max|Δ| = {e1:.2e} "
          f"({'PASS' if e1 < 1e-10 else 'FAIL'})", flush=True)

    # G3 Q-ball existence: scan omega (mu = m^2 - w^2)
    print("[G3] Q-ball existence scan (Petviashvili in omega):", flush=True)
    qball = None
    for wfrac in (0.9, 0.8, 0.6, 0.4):
        w = wfrac * A.m; mu = A.m ** 2 - w ** 2
        phi, prof = qball_petviashvili(op, a, s, f, mu)
        if phi is None:
            print(f"   w={w:.3f} mu={mu:.3f} -> FAIL {prof.get('fail')}", flush=True); continue
        loc = prof["occ"] < 0.5 and prof["amp"] > 0.05
        print(f"   w={w:.3f} mu={mu:.3f}: residual={prof['residual']:.2e} amp={prof['amp']:.3f} "
              f"occ={prof['occ']:.4f} {'Q-BALL' if loc and prof['residual'] < 1e-6 else '--'}", flush=True)
        if loc and prof["residual"] < 1e-6 and qball is None:
            qball = (phi, w, mu, prof)
    res["G3_qball_found"] = qball is not None

    if qball is None:
        res["verdict"] = "C3_NO_STABLE_QBALL_IN_SCAN"
        print(f"\n=== C3_NO_STABLE_QBALL_IN_SCAN ===", flush=True)
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
        print(f"C3_DONE {out}", flush=True); return
    phi, w, mu, prof = qball
    print(f"[G3] selected Q-ball: w={w:.3f} mu={mu:.3f} amp={prof['amp']:.3f} residual={prof['residual']:.2e}", flush=True)
    np.save(os.path.join(out, "qball.npy"), phi)

    # G2 invariant telemetry + G5 null: evolve the rest Q-ball, check E/Q drift + no drift of centroid
    psi_k = jnp.fft.fftn(jnp.asarray(phi)); pi_k = jnp.fft.fftn(jnp.asarray(-1j * w * phi))   # rest: psi_t = -i w phi
    inv0 = invariants(psi_k, pi_k, op, a, s, f)
    x0 = centroid_x(np.abs(phi) ** 2, A.N, A.L)
    steps = int(round(6.0 / A.dt)); chunk = min(500, steps)
    pk, qk = psi_k, pi_k
    for _ in range(steps // chunk):
        pk, qk = kg_evolve(pk, qk, op, a, s, f, chunk)
    inv1 = invariants(pk, qk, op, a, s, f)
    xN = centroid_x(np.abs(np.asarray(jnp.fft.ifftn(pk))) ** 2, A.N, A.L)
    dE = abs(inv1["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30)
    dQ = abs(inv1["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30)
    res["G2_dE_rel"] = dE; res["G2_dQ_rel"] = dQ; res["G5_null_drift"] = abs(xN - x0)
    print(f"[G2] rest Q-ball T=6: dE/E={dE:.2e} dQ/Q={dQ:.2e} mass {inv0['mass']:.2f}->{inv1['mass']:.2f}", flush=True)
    print(f"[G5] null control: centroid drift = {abs(xN-x0):.4f} (should be ~0)", flush=True)

    # G4 transport: moving Q-ball  psi=phi, pi = -v.grad phi - i w phi ; density should co-move at v
    print("[G4] velocity-kick transport:", flush=True)
    boosts = []
    gx = np.asarray(jnp.fft.ifftn(op["ikx"] * jnp.fft.fftn(jnp.asarray(phi))))
    for v in (0.1 * A.c, 0.25 * A.c):
        pi0 = (-v * gx - 1j * w * phi).astype(np.complex128)
        pk = jnp.fft.fftn(jnp.asarray(phi.astype(np.complex128))); qk = jnp.fft.fftn(jnp.asarray(pi0))
        xs, ts = [x0], [0.0]
        Tp = min(0.35 * A.L / v, 8.0); steps = int(round(Tp / A.dt)); chunk = min(500, steps)
        for c_ in range(steps // chunk):
            pk, qk = kg_evolve(pk, qk, op, a, s, f, chunk)
            rho = np.abs(np.asarray(jnp.fft.ifftn(pk))) ** 2
            xs.append(centroid_x(rho, A.N, A.L)); ts.append((c_ + 1) * chunk * A.dt)
        # unwrap + fit velocity
        xu = np.array(xs); tt = np.array(ts)
        for i in range(1, len(xu)):
            d = xu[i] - xu[i - 1]; xu[i] = xu[i - 1] + (d - A.L * round(d / A.L))
        vm = float(np.polyfit(tt, xu, 1)[0])
        invf = invariants(pk, qk, op, a, s, f)
        frac = vm / v
        boosts.append({"v": v, "v_measured": vm, "v_frac": frac, "mass_ret": invf["mass"] / inv0["mass"],
                       "amp": invf["amp"]})
        print(f"   v={v:.3f}: v_measured={vm:+.4f} (frac={frac:+.4f}) mass_ret={invf['mass']/inv0['mass']:.4f} "
              f"amp={invf['amp']:.3f}", flush=True)
    res["G4_boosts"] = boosts
    ok = all(abs(b["v_frac"] - 1.0) < 0.2 and b["mass_ret"] > 0.9 for b in boosts)
    verdict = "C3_INERTIAL_TRANSPORT_CONFIRMED" if (ok and dE < 1e-2 and res["G5_null_drift"] < 0.3) else \
              "C3_QBALL_TRANSPORT_DEGRADED"
    res["verdict"] = verdict
    print(f"\n=== {verdict} | G1={e1:.1e} dE={dE:.1e} null={res['G5_null_drift']:.3f} "
          f"vfracs={[round(b['v_frac'],3) for b in boosts]} ===", flush=True)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C3_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
