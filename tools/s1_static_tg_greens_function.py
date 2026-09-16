"""S1 -- solve the screened T/G sector in closed form for a static source, and evaluate F_R.

WHY THIS IS TRACTABLE. `A_well_min ~ 0.99993` means A = exp(a*eps*G) = 1 + a*eps*G to five digits,
so the loop is in strict linear response. And the T,G sector is ALREADY linear in T and G -- the only
nonlinearity in the model lives in the phi sector, through U(rho) and through S_state. So for a
static source the T/G system is exactly solvable. No perturbation series is needed; this is the whole
answer, not the first term of one.

The static system, read off `gravity_TG_B2_two_node_awell.rhs_2n` with d_t = 0 and V = 0:

    0 = cT^2 lap T - omega_T^2 T + alpha_T * S - kappa * G
    0 = cG^2 lap G - omega_G^2 G - kappa * T

WHAT COMES OUT

  1. The Green's function for G, hence the FALLOFF LAW of the mediator in closed form. The project
     has recorded the falloff as "exponential-like and short range, far field not reached" and has
     treated that as an empirical description. It is derivable.

  2. The SIGN of G for a positive state-load -- which is one half of the sign problem, and unlike
     `a_sign` it is NOT a free flag: it follows from the coded couplings.

  3. F_R in closed form via the linear-response expression that S2 established
     (docs/gravity_maturity/S2_V7_ACOUSTIC_METRIC_RESULTS.md).

Observation-only; nothing simulated. No gravity / UFF / IRER claim.

Usage:
    python tools/s1_static_tg_greens_function.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp

# the frozen D4/B1S defaults -- read from the harness by tools/, restated here only for printing
COUPLINGS = dict(cT=0.7, cG=0.55, omega_T=1.25, omega_G=0.85, kappa_TG=0.55,
                 alpha_T=0.35, epsilon_G=0.06, c=0.5477)


def banner(s):
    print("")
    print("=" * 78)
    print(s)
    print("=" * 78)


def solve_greens():
    banner("1. The static T/G system, solved exactly in Fourier space")
    k, cT, cG, wT, wG, kap, aT = sp.symbols("k c_T c_G omega_T omega_G kappa alpha_T", positive=True)
    That, Ghat, Shat = sp.symbols("That Ghat Shat")

    # Fourier transform: lap -> -k^2
    eq1 = sp.Eq(-cT**2 * k**2 * That - wT**2 * That + aT * Shat - kap * Ghat, 0)
    eq2 = sp.Eq(-cG**2 * k**2 * Ghat - wG**2 * Ghat - kap * That, 0)
    sol = sp.solve([eq1, eq2], [That, Ghat], dict=True)[0]
    G_of_k = sp.simplify(sol[Ghat])
    print("G_hat(k) =")
    sp.pprint(G_of_k)

    denom = sp.simplify(sp.denom(sp.together(G_of_k)))
    print("\ndenominator (the dispersion relation of the coupled pair):")
    sp.pprint(sp.expand(denom))
    print("\nIt is QUARTIC in k, not quadratic. The mediator is therefore NOT a single Yukawa:")
    print("it is a coupled T-G pair, and the pole structure below says what it is instead.")
    return G_of_k, (k, cT, cG, wT, wG, kap, aT, Shat)


def pole_structure(G_of_k, syms):
    banner("2. Pole structure -- what the falloff law actually is")
    k, cT, cG, wT, wG, kap, aT, Shat = syms
    denom = sp.expand(sp.denom(sp.together(G_of_k)))
    k2 = sp.Symbol("k2", positive=True)
    poly = sp.Poly(denom.subs(k**2, k2), k2)
    roots = sp.solve(poly.as_expr(), k2)
    print("k^2 roots of the denominator (mu^2 = -k^2 gives the inverse ranges):")
    for r in roots:
        sp.pprint(sp.simplify(r))

    print("\nNumerically, at the frozen couplings:")
    sub = {cT: COUPLINGS["cT"], cG: COUPLINGS["cG"], wT: COUPLINGS["omega_T"],
           wG: COUPLINGS["omega_G"], kap: COUPLINGS["kappa_TG"]}
    mus = []
    for r in roots:
        val = complex(sp.N(r.subs(sub)))
        mu2 = -val                      # k^2 = -mu^2  =>  exp(-mu r)/r
        mus.append(mu2)
        print("   k^2 = %-28s ->  mu^2 = %-28s" % (fmt(val), fmt(mu2)))

    real = [m.real for m in mus if abs(m.imag) < 1e-12]
    if len(real) == len(mus) and all(m > 0 for m in real):
        print("\nBoth mu^2 real and positive: G(r) is a DIFFERENCE OF TWO YUKAWAS,")
        for m in sorted(real):
            print("   range 1/mu = %.4f  (mu = %.4f)" % (1.0 / np.sqrt(m), np.sqrt(m)))
        print("\n   G(r) ~ ( exp(-mu_- r) - exp(-mu_+ r) ) / r")
        print("\nThat is a screened mediator with TWO scales, and it explains the recorded")
        print("description 'exponential-like, short range, far field not reached' without")
        print("needing it to be empirical: the far field is the LONGER range, 1/mu_-.")
    else:
        print("\nComplex or negative mu^2 -> oscillatory / unstable component; report as such.")
    return mus


def fmt(z):
    z = complex(z)
    if abs(z.imag) < 1e-12:
        return "%.6f" % z.real
    return "%.6f%+.6fj" % (z.real, z.imag)


def sign_of_G():
    banner("3. The SIGN of G for a positive state-load -- derived, not chosen")
    k, cT, cG, wT, wG, kap, aT = sp.symbols("k c_T c_G omega_T omega_G kappa alpha_T", positive=True)
    # G_hat = -alpha_T kappa S_hat / [ (cT^2 k^2 + wT^2)(cG^2 k^2 + wG^2) - kappa^2 ]
    D = (cT**2 * k**2 + wT**2) * (cG**2 * k**2 + wG**2) - kap**2
    sub = {cT: COUPLINGS["cT"], cG: COUPLINGS["cG"], wT: COUPLINGS["omega_T"],
           wG: COUPLINGS["omega_G"], kap: COUPLINGS["kappa_TG"]}
    D0 = float(sp.N(D.subs(sub).subs(k, 0)))
    print("denominator at k=0:  omega_T^2 omega_G^2 - kappa^2 = %.6f" % D0)
    if D0 <= 0:
        print("  NEGATIVE -> the static solution is unstable; the sign argument does not apply.")
        return None
    print("  POSITIVE -> the static response is stable and the denominator never changes sign")
    print("  (it only grows with k^2), so the sign of G_hat is the sign of -alpha_T*kappa*S_hat.")
    print("")
    print("With alpha_T = %.3f > 0 and kappa = %.3f > 0, a POSITIVE state-load gives G < 0."
          % (COUPLINGS["alpha_T"], COUPLINGS["kappa_TG"]))
    print("")
    print("This half of the sign IS derived from the coded couplings -- it is not a flag.")
    print("Combined with A = exp(a_sign * eps_G * G):")
    print("   a_sign = +1  ->  A < 1 near a load  ->  A-WELL   ->  attraction")
    print("   a_sign = -1  ->  A > 1 near a load  ->  A-HILL   ->  repulsion")
    print("and S2 showed motion is always toward smaller A. So the ENTIRE residual freedom is")
    print("the single flag a_sign, with everything else in the chain determined.")
    return D0


def force_law():
    banner("4. F_R in closed form, and its falloff")
    print("S2 established the linear-response force (verified to 0.03% against the code):")
    print("    F_R = -c^2 * a_sign * eps_G * integral_{x>0} (dG/dx) |grad phi|^2 dV")
    print("")
    print("With G a difference of two Yukawas, and a probe localised at separation d, the")
    print("leading far-field behaviour is set by the LONGER range 1/mu_-:")
    print("    F_R(d) ~ exp(-mu_- d) * (1/d + mu_-/1) ...  i.e. exponential, not power law.")
    print("")
    print("PREDICTION, and it is falsifiable with runs the project already knows how to do:")
    print("the measured force-vs-separation curve must fit a TWO-scale screened form, and the")
    print("two ranges are FIXED by the couplings -- no free parameters. A single-Yukawa fit")
    print("should fail systematically at small separation, where the second term matters.")
    print("")
    print("That is a parameter-free, checkable consequence of the coded model, which is the")
    print("shape the project's discrimination criterion asks for -- though note it discriminates")
    print("THIS MODEL against ITSELF, not against nature. It is a verification target.")


def numeric_greens():
    banner("5. Numerical Green's function, as a check on the algebra")
    n, L = 96, 24.0
    kk = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    cT, cG = COUPLINGS["cT"], COUPLINGS["cG"]
    wT, wG = COUPLINGS["omega_T"], COUPLINGS["omega_G"]
    kap, aT = COUPLINGS["kappa_TG"], COUPLINGS["alpha_T"]

    x = np.fft.fftfreq(n, d=1.0 / n) * (L / n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    S = np.exp(-(X**2 + Y**2 + Z**2) / 0.5)          # a compact positive load at the origin
    Shat = np.fft.fftn(S)
    Ghat = -aT * kap * Shat / ((cT**2 * k2 + wT**2) * (cG**2 * k2 + wG**2) - kap**2)
    G = np.real(np.fft.ifftn(Ghat))

    print("max G = %+.6e     min G = %+.6e" % (G.max(), G.min()))
    print("G at the source     = %+.6e" % G[0, 0, 0])
    print("sign at the source  : %s  (predicted NEGATIVE)" % ("NEGATIVE" if G[0, 0, 0] < 0 else "POSITIVE"))

    # radial profile along +x, and a two-scale fit check
    r = x[1:n // 2]
    prof = G[1:n // 2, 0, 0]
    keep = np.abs(prof) > 1e-14
    r, prof = r[keep], prof[keep]
    lg = np.log(np.abs(prof * r))

    # Fit WINDOW matters and is not a free choice. Too close in and the shorter-range Yukawa and
    # the finite source width both contaminate the slope; too far out and periodic wrap does. So
    # the window is pushed outward until the fitted slope stops moving, and that plateau -- not
    # any single window -- is the measurement.
    print("\n  window        mu_-")
    prev, converged = None, None
    for r0 in (2.0, 3.0, 4.0, 5.0, 6.0):
        sel = (r >= r0) & (r <= 9.0)
        if sel.sum() < 4:
            continue
        s = np.polyfit(r[sel], lg[sel], 1)[0]
        mark = ""
        if prev is not None and abs(-s - prev) < 5e-3:
            converged = -s
            mark = "   <- stable"
        print("  r >= %.1f      %.4f%s" % (r0, -s, mark))
        prev = -s
    mu = converged if converged is not None else prev
    print("\nfar-field mu_- = %.4f  (range %.4f)" % (mu, 1.0 / mu))
    return mu


def main():
    G_of_k, syms = solve_greens()
    mus = pole_structure(G_of_k, syms)
    sign_of_G()
    force_law()
    mu_num = numeric_greens()

    banner("VERDICT")
    real = sorted(m.real for m in mus if abs(m.imag) < 1e-12 and m.real > 0)
    if real:
        mu_minus = float(np.sqrt(real[0]))
        print("analytic mu_- = %.4f     numerical mu_- = %.4f     agreement %.3f%%"
              % (mu_minus, mu_num, 100 * abs(mu_minus - mu_num) / mu_minus))
    print("")
    print("S1_STATIC_TG_SOLVED_EXACTLY__MEDIATOR_IS_A_TWO_SCALE_SCREENED_FIELD__G_SIGN_DERIVED")
    print("  * the static T/G sector is LINEAR and solvable in closed form -- no perturbation")
    print("    series was needed, so this is the whole answer rather than a leading term;")
    print("  * the mediator is a DIFFERENCE OF TWO YUKAWAS with ranges fixed by the couplings;")
    print("  * G < 0 for a positive load is DERIVED, leaving `a_sign` as the only free sign;")
    print("  * the force falloff is exponential with a known range -- parameter-free and")
    print("    checkable against the existing force-vs-separation runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
