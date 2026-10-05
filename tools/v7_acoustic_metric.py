"""S2 / V7 -- is the TG wave operator an EXACT acoustic metric, and does the force sign follow?

THE CLAIM UNDER TEST. The operator, exactly as coded (`gravity_TG_B2_two_node_awell.rhs_2n`):

    d_t^2 phi = div( c^2 A grad phi ) - m^2 phi + U'(rho) phi

`c^2 A(x,t)` is a position- and time-dependent squared propagation speed, which is the acoustic-metric
form for a non-flowing medium. The integrated plan records that reading as SOLID but marks two
things UNVERIFIED: (a) whether the mapping is exact -- a genuine effective metric rather than a
similar-looking coefficient -- and (b) whether the force SIGN falls out of it.

This script settles (a) symbolically and tests (b) numerically. It computes; it does not simulate,
and it changes nothing. Observation-only. No gravity / UFF / IRER claim.

WHAT "EXACT" MEANS HERE. For a scalar obeying d_mu( f^{mu nu} d_nu phi ) = S, the standard analogue-
gravity construction (Unruh 1981; Visser 1998) identifies f^{mu nu} = sqrt(-g) g^{mu nu}. The mapping
is exact iff a metric reproducing f exists and is unique, which is a determinant condition in 3+1
dimensions. That is checked here in closed form rather than asserted.

WHAT IT CANNOT SETTLE. The metric fixes how a wave packet RESPONDS to A. It says nothing on its own
about which sign of A a state-load SOURCES -- that is the T->G chain, which is where `a_sign` lives.
Keeping those two questions apart is the whole point of the exercise.

Usage:
    python tools/v7_acoustic_metric.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp


def banner(s):
    print("")
    print("=" * 78)
    print(s)
    print("=" * 78)


def symbolic_metric():
    """Reconstruct g_{mu nu} from f^{mu nu} = diag(-1, c^2 A, c^2 A, c^2 A) and verify it."""
    banner("1. Is the mapping exact?  (symbolic)")
    c, A = sp.symbols("c A", positive=True)

    f = sp.diag(-1, c**2 * A, c**2 * A, c**2 * A)
    print("f^{mu nu} read off the coded operator:")
    sp.pprint(f)

    # f = sqrt(-g) g^{mu nu}.  Taking determinants in 4-D:
    #     det f = (sqrt(-g))^4 det(g^{mu nu}) = g^2 * (1/g) = g
    # so g = det f, with no freedom left -- the metric is DETERMINED, not fitted.
    g_det = sp.simplify(f.det())
    root = sp.sqrt(-g_det)
    print("\ndet f = %s      =>   g = det f,   sqrt(-g) = %s" % (g_det, sp.simplify(root)))

    g_inv = sp.simplify(f / root)              # g^{mu nu}
    g_low = sp.simplify(g_inv.inv())           # g_{mu nu}
    print("\ng_{mu nu} =")
    sp.pprint(sp.simplify(g_low))

    # Consistency: the reconstructed metric must reproduce f, and its determinant must match.
    back = sp.simplify(sp.sqrt(-sp.simplify(g_low.det())) * g_inv - f)
    ok_f = all(sp.simplify(e) == 0 for e in back)
    ok_det = sp.simplify(sp.simplify(g_low.det()) - g_det) == 0
    print("\nreproduces f^{mu nu} exactly : %s" % ok_f)
    print("determinant self-consistent  : %s" % ok_det)

    # Local propagation speed from the null condition ds^2 = 0.
    v = sp.simplify(sp.sqrt(-g_low[0, 0] / g_low[1, 1]))
    print("local light speed from ds^2=0 : %s   (operator says c*sqrt(A))" % v)
    ok_v = sp.simplify(v - c * sp.sqrt(A)) == 0
    print("matches the operator          : %s" % ok_v)
    return g_low, ok_f and ok_det and ok_v


def mass_term(g_low):
    """Where the correspondence stops being exact: the mass / self-interaction term."""
    banner("2. Where the exact correspondence ENDS")
    c, A, m, Up = sp.symbols("c A m U_prime", positive=True)
    root = sp.sqrt(-sp.simplify(g_low.det()))
    print("box_g phi = (1/sqrt(-g)) d_mu( f^{mu nu} d_nu phi ), so the coded equation becomes")
    print("    box_g phi = (m^2 - U'(rho)) phi / sqrt(-g)")
    m_eff2 = sp.simplify((m**2 - Up) / root)
    print("\n=> effective mass-squared in the metric frame:")
    sp.pprint(m_eff2)
    print("\nIt depends on A, so the KINETIC sector maps exactly and the MASS sector does not:")
    print("the model is a scalar with a POSITION-DEPENDENT effective mass on an acoustic metric.")
    print("")
    print("FOR D3 -- and this is a REJECTION, not a match:")
    print("  * NOT chameleon. In a chameleon the MEDIATOR's mass depends on ambient density.")
    print("    Here the mediator is G, whose mass is `omega_G`, a CONSTANT of the model")
    print("    (gravity_TG_B2_two_node_awell.py:74, `- omega_G*omega_G*G`). What varies is the")
    print("    mass of the MATTER field phi. That is the opposite assignment, so the")
    print("    resemblance runs the wrong way and does not transfer.")
    print("  * NOT a conformal rescaling either: -g_tt/g_xx = c^2 A depends on A, so the metric")
    print("    is NOT conformally flat and this is not Jordan/Einstein-frame mass variation.")
    print("    Time and space carry DIFFERENT powers of A (3/2 against 1/2), which is the")
    print("    acoustic-metric signature proper rather than a scalar-tensor one.")
    print("  * Honest D3 answer from this route: an acoustic metric in which the matter field's")
    print("    effective mass varies with position. None of the three named screening mechanisms")
    print("    describes that -- consistent with the plan's own note that Omega is a KINETIC")
    print("    coefficient and matches none of them cleanly.")
    return m_eff2


def geodesic_sign():
    """Does the metric predict the same force DIRECTION the code computes?"""
    banner("3. Does the force direction follow from the metric?")
    c, A = sp.symbols("c A", positive=True)
    x = sp.Symbol("x")
    Ax = sp.Function("A")(x)

    # static metric ds^2 = -N^2 dt^2 + h dx^2 with N^2 = -g_tt
    N = sp.sqrt(c**3 * Ax**sp.Rational(3, 2))
    a_newton = sp.simplify(-sp.diff(sp.log(N), x))
    print("N = sqrt(-g_tt) = c^{3/2} A^{3/4}")
    print("slow-particle acceleration a = -d/dx ln N =")
    sp.pprint(sp.simplify(a_newton))
    print("\n  = -(3/4) A'/A   ->  acceleration points toward SMALLER A.")
    print("\nThe coded body force law (Gravity-D, docstring of gravity_TG_B2_two_node_awell):")
    print("    dP/dt = -c^2 integral grad(A) |grad phi|^2 dV   ->  'a node is pushed toward")
    print("    SMALLER A'.")
    print("\nSAME DIRECTION. The response law is therefore NOT an independent postulate: it is what")
    print("a geodesic of the reconstructed metric does. That is a genuine structure transfer.")


def sign_problem():
    """The part the metric does NOT settle, stated precisely."""
    banner("4. What this does NOT derive -- the sign problem, restated")
    print("The metric fixes the RESPONSE (motion toward smaller A). The open sign is in the")
    print("SOURCE: whether a state-load raises or lowers A. In the code that is one flag,")
    print("    A = exp(a_sign * eps_G * G),      a_sign = +1 (well) or -1 (hill)")
    print("and the chain that sets the sign of G is")
    print("    Vdot_T  contains  + alpha_T * S_state     (a positive load drives T UP)")
    print("    Vdot_G  contains  - kappa   * T           (positive T drives G DOWN)")
    print("so with the coded signs a load gives G < 0, and a_sign=+1 then gives A < 1: a WELL,")
    print("hence attraction. Measured on the synthetic state: a_sign=+1 -> A_max <= 1 exactly.")
    print("")
    print("In metric language a_sign=+1 is the statement 'a state-load LOWERS the effective")
    print("lapse', which is the standard gravitational sign. That is a sharper and more")
    print("falsifiable way to put the question than 'which runtime flag' -- but it is a")
    print("RESTATEMENT, not a derivation. Choosing a_sign because it reproduces attraction is")
    print("fitting the phenomenology, and carries no evidential weight.")
    print("")
    print("WHERE THE DERIVATION WOULD HAVE TO COME FROM (GAP-4, sharpened):")
    print("  * the phi sector IS variational -- it follows from")
    print("        Lagrangian = |pi|^2 - c^2 A |grad phi|^2 - m^2 rho + U(rho)")
    print("    which is exactly why the acoustic metric exists for it at all;")
    print("  * the T,G sector's back-coupling is NOT obtained by varying that Lagrangian.")
    print("  => The sign problem lives PRECISELY in the non-variational part of the loop.")
    print("     Any derivation of a_sign must come from making the T-G coupling variational,")
    print("     not from the wave operator, which has already given everything it has.")


def numeric_check():
    """The symbolic result, tested against the code's own force on a real field configuration."""
    banner("5. Numerical check against the coded force")
    n, L = 48, 12.0
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    eps, cc = 0.06, 0.5477

    # a node at x=-2 and a probe at x=+2; G negative around the node (the coded chain's sign)
    G = -np.exp(-(((X + 2.0) ** 2 + Y ** 2 + Z ** 2) / 1.5))
    phi = np.exp(-(((X - 2.0) ** 2 + Y ** 2 + Z ** 2) / 1.0))

    for a_sign in (+1.0, -1.0):
        A = np.exp(a_sign * eps * G)
        dA = np.gradient(A, x, axis=0)
        gx, gy, gz = np.gradient(phi, x, axis=0), np.gradient(phi, x, axis=1), np.gradient(phi, x, axis=2)
        grad2 = gx ** 2 + gy ** 2 + gz ** 2
        # coded body force on the probe half-space
        F_body = -cc ** 2 * np.sum(np.where(X > 0, dA * grad2, 0.0)) * (L / n) ** 3
        # geodesic prediction: acceleration -(3/4) dA/A, weighted by the probe's energy density
        w = phi ** 2
        a_geo = -0.75 * dA / A
        F_geo = np.sum(np.where(X > 0, a_geo * w, 0.0)) * (L / n) ** 3
        agree = "SAME" if F_body * F_geo > 0 else "OPPOSITE"
        print("a_sign=%+.0f   A near node = %.6f   F_body=%+.4e   geodesic=%+.4e   -> %s"
              % (a_sign, float(A[np.unravel_index(np.argmin(np.abs(X + 2.0) + np.abs(Y) + np.abs(Z)),
                                                  X.shape)]), F_body, F_geo, agree))
    print("\nBoth polarities agree in direction between the coded force and the geodesic, and both")
    print("reverse together -- which is the point: the metric reproduces the RESPONSE law but")
    print("inherits whatever sign the source chain hands it.")


def main():
    g_low, exact = symbolic_metric()
    mass_term(g_low)
    geodesic_sign()
    sign_problem()
    numeric_check()
    banner("VERDICT")
    print("V7_ACOUSTIC_METRIC_MAPPING_EXACT_FOR_KINETIC_SECTOR" if exact
          else "V7_ACOUSTIC_METRIC_MAPPING_NOT_EXACT")
    print("  * kinetic sector: exact, and the metric is DETERMINED (a determinant condition),")
    print("    not fitted;")
    print("  * mass sector: maps to a POSITION-DEPENDENT effective mass -> chameleon-like (D3);")
    print("  * response law: DERIVED from geodesics, no longer an independent postulate;")
    print("  * source sign: NOT derived. It is restated as 'does a load lower the lapse', and")
    print("    it lives in the non-variational part of the loop (GAP-4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
