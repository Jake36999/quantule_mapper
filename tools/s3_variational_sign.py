"""S3 / GAP-4 -- if the loop is made variational, is the force sign still free?

THE QUESTION. `a_sign = +-1` is a runtime flag and the project's single largest open problem:
flipping it gives equally converged repulsion. S2 showed the wave operator has given everything it
has -- it fixes how a packet RESPONDS to A, never which sign of A a load SOURCES -- and that the
sign lives precisely in the part of the loop that is NOT obtained by varying an action. So the
question is whether making it variational removes the freedom.

WHAT IS AND IS NOT VARIATIONAL IN THE CODED LOOP

  variational      the phi sector: L contains -c^2 A(G) |grad phi|^2, and varying w.r.t. phi gives
                   exactly the coded div(c^2 A grad phi). This is WHY an acoustic metric exists.
  variational      the T-G mutual coupling: a single term -kappa*T*G gives both the coded -kappa*G
                   in Tdd and the coded -kappa*T in Gdd. The coefficients match because they come
                   from one term (and tests/test_physics_identities.py now pins that).
  NOT variational  the phi -> G route. In the code, phi sources T through alpha_T * S_state and T
                   sources G through -kappa*T. But the SAME Lagrangian term that gives phi its
                   A-dependence, -c^2 A(G)|grad phi|^2, would ALSO have to be varied with respect
                   to G -- and that variation is absent from the coded Gdd.

THE ARGUMENT. Restoring the missing variation is not optional if the loop is to come from an
action, and restoring it closes the loop through the SAME coupling twice: once when phi sources G,
and once when G modulates A. A coupling that appears twice enters squared, and a square has no
sign. That is the standard reason scalar exchange between like sources is attractive, and it
applies here.

This script does the algebra symbolically, then checks it numerically on an explicit field.

Observation-only; it proposes a modification and analyses it, and modifies no harness. The frozen
dynamics are untouched. No gravity / UFF / IRER claim.

Usage:
    python tools/s3_variational_sign.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp


def banner(s):
    print("")
    print("=" * 78)
    print(s)
    print("=" * 78)


def missing_variation():
    banner("1. The term the coded G equation is missing")
    G, c, eps, a, g2 = sp.symbols("G c epsilon_G a_sign gradphi2", positive=False)
    A = sp.exp(a * eps * G)
    L_phi = -c**2 * A * g2            # the phi kinetic term, exactly as in the Lagrangian
    print("L contains   -c^2 A(G) |grad phi|^2   with   A = exp(a_sign * eps_G * G)")
    print("\nVarying w.r.t. phi gives div(c^2 A grad phi) -- the coded operator. Varying the SAME")
    print("term w.r.t. G gives a source for G that the coded equation does not have:")
    dL_dG = sp.simplify(sp.diff(L_phi, G))
    print("\n   dL/dG =")
    sp.pprint(dL_dG)
    print("\nso a variational G equation must read")
    print("   Gdd = cG^2 lap G - omega_G^2 G - kappa*T  +  dL/dG")
    print("       = cG^2 lap G - omega_G^2 G - kappa*T  -  c^2 * a_sign * eps_G * A * |grad phi|^2")
    print("\nThe coded Gdd has no |grad phi|^2 term at all. THAT is GAP-4, stated exactly.")
    return dL_dG


def sign_cancels():
    banner("2. The sign cancels -- a_sign enters squared")
    a, eps, c, wG, g2 = sp.symbols("a_sign epsilon_G c omega_G gradphi2", positive=True)
    a = sp.Symbol("a_sign")           # NOT positive: this is the whole point

    print("Static, local limit (the mediator range 0.875 is short next to the source; S1):")
    print("   0 = -omega_G^2 G - c^2 * a_sign * eps_G * |grad phi|^2")
    G_star = sp.solve(sp.Eq(0, -wG**2 * sp.Symbol("G") - c**2 * a * eps * g2), sp.Symbol("G"))[0]
    print("\n   =>  G* =")
    sp.pprint(G_star)
    print("\nG* is proportional to -a_sign: the polarity flips the field it sources. Now feed it")
    print("back through A, which carries a_sign a SECOND time:")
    A_star = sp.simplify(sp.exp(a * eps * G_star))
    print("\n   A* = exp(a_sign * eps_G * G*) =")
    sp.pprint(A_star)
    expo = sp.simplify(sp.log(A_star))
    print("\n   exponent =")
    sp.pprint(sp.expand(expo))
    print("\n   a_sign appears SQUARED. Substituting a_sign = +1 and a_sign = -1:")
    for v in (1, -1):
        print("      a_sign = %+d  ->  exponent = %s" % (v, sp.simplify(expo.subs(a, v))))
    same = sp.simplify(expo.subs(a, 1) - expo.subs(a, -1)) == 0
    print("\n   identical: %s" % same)
    print("\n   The exponent is negative definite (eps_G^2, c^2, |grad phi|^2 and omega_G^2 all")
    print("   positive), so A* < 1 near a node for BOTH polarities: an A-WELL either way.")
    print("   By S2, motion is always toward smaller A. Therefore ATTRACTION, with no free sign.")
    return same


def why_general():
    banner("3. Why this is not an accident of the exponential")
    print("Nothing above used the specific form A = exp(a*eps*G). The structure is:")
    print("")
    print("   L contains  f(G) * X[phi]     with X = -c^2 |grad phi|^2 < 0")
    print("")
    print("   phi's equation gets   f(G)     -- how G modulates phi")
    print("   G's equation gets     f'(G) X  -- how phi sources G       <- the missing term")
    print("")
    print("   Static local response:   G* ~ -f'(G) X / omega_G^2")
    print("   Change in the modulator: delta f ~ f'(G) G* ~ -f'(G)^2 X / omega_G^2")
    print("")
    print("   f'(G)^2 >= 0 ALWAYS. The sign of delta f is fixed by the sign of X alone, and X is")
    print("   the phi kinetic term, which is sign-definite. The mediator's polarity cancels.")
    print("")
    print("This is the standard reason SCALAR exchange between like sources is attractive, and")
    print("why a VECTOR mediator (whose source is a current, not a square) can repel. The coded")
    print("loop escapes it only by routing phi -> G through a separate, non-variational channel")
    print("(alpha_T * S_state), which lets the source polarity be chosen independently of the")
    print("modulation polarity. Two independent signs is one more than an action permits.")


def numeric_check():
    banner("4. Numerical check on an explicit field")
    n, L = 64, 16.0
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    c, eps, wG, cG = 0.5477, 0.06, 0.85, 0.55
    phi = np.exp(-((X**2 + Y**2 + Z**2) / 2.0))
    gx, gy, gz = np.gradient(phi, x, axis=0), np.gradient(phi, x, axis=1), np.gradient(phi, x, axis=2)
    g2 = gx**2 + gy**2 + gz**2

    k1 = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2

    print("VARIATIONAL G (sourced by the missing term), for both polarities:")
    for a_sign in (+1.0, -1.0):
        src = -c**2 * a_sign * eps * g2                  # A ~ 1 in linear response
        G = np.real(np.fft.ifftn(np.fft.fftn(src) / (cG**2 * k2 + wG**2)))
        A = np.exp(a_sign * eps * G)
        print("   a_sign=%+.0f :  G at centre = %+.6e    A_min = %.9f    A_max = %.9f  -> %s"
              % (a_sign, G[n // 2, n // 2, n // 2], A.min(), A.max(),
                 "WELL" if A.max() <= 1.0 + 1e-12 else "HILL"))

    print("\nCODED (non-variational) G, sourced through alpha_T*S_state instead -- for contrast:")
    for a_sign in (+1.0, -1.0):
        S = phi**2                                        # a positive state-load, sign-fixed
        G = -np.real(np.fft.ifftn(np.fft.fftn(S) / (cG**2 * k2 + wG**2)))
        A = np.exp(a_sign * eps * G)
        print("   a_sign=%+.0f :  G at centre = %+.6e    A_min = %.9f    A_max = %.9f  -> %s"
              % (a_sign, G[n // 2, n // 2, n // 2], A.min(), A.max(),
                 "WELL" if A.max() <= 1.0 + 1e-12 else "HILL"))
    print("\n   The coded route flips well<->hill with the flag. The variational route cannot.")


def main():
    missing_variation()
    ok = sign_cancels()
    why_general()
    numeric_check()
    banner("VERDICT")
    if ok:
        print("S3_VARIATIONAL_COMPLETION_FORCES_ATTRACTION__a_sign_FREEDOM_IS_AN_ARTEFACT_OF_GAP4")
    else:
        print("S3_SIGN_DOES_NOT_CANCEL -- report as such")
    print("  * the coded G equation is MISSING the variation of -c^2 A(G)|grad phi|^2 w.r.t. G;")
    print("  * restoring it makes a_sign appear TWICE -- once sourcing G, once modulating A --")
    print("    so it enters squared and cancels;")
    print("  * the resulting A is a WELL for both polarities, hence attraction (S2 geodesic);")
    print("  * the freedom exists only because phi->G is routed through a separate")
    print("    non-variational channel, which permits two independent signs where an action")
    print("    permits one.")
    print("")
    print("  NOT ESTABLISHED: that the variational model reproduces the rest of the project's")
    print("  phenomenology. This changes the equations of motion. It is a derivation about a")
    print("  MODIFIED model, and the modified model has not been simulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
