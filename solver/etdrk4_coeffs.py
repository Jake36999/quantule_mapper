"""solver/etdrk4_coeffs.py - Kassam-Trefethen ETDRK4 coefficients, valid for COMPLEX linear operators.

The textbook Kassam-Trefethen shortcut (contour points on the upper half circle, then take real()) relies on
conjugate symmetry, which holds only when L is real. Every L_k in this project is complex
(-D k^2 - eta + i*omega0, or -i*D k^2 on the conservative branch), so the shortcut silently drops the
imaginary part of Q, f1, f2, f3 and degrades ETDRK4 from 4th to 1st order in time.
(Audit: F:/Maths_exploration/optimizer_audit/t1b_order.py.)

This routine averages over the FULL circle and keeps the complex result. For real L it reproduces the old
coefficients to ~1e-14 (the imaginary part is round-off), so purely real operators are unaffected.
"""

import cmath

KT_CONTOUR_POINTS = 128  # full circle; equivalent resolution to the old 64-point half circle


def etdrk4_coefficients(L_k, dt, xp, M=KT_CONTOUR_POINTS, r=1.0):
    """Return (E, E2, Q, f1, f2, f3) for ETDRK4 with linear operator L_k (complex array) and step dt.

    `xp` is the array module (numpy or cupy). All outputs are complex128.
    Accumulates point-by-point so peak memory stays at a few N^3 arrays (no M x N^3 tensor).
    """
    w = (L_k * dt).astype(xp.complex128, copy=False)
    Q = xp.zeros_like(w)
    f1 = xp.zeros_like(w)
    f2 = xp.zeros_like(w)
    f3 = xp.zeros_like(w)
    for j in range(M):
        z = r * cmath.exp(2j * cmath.pi * (j + 0.5) / M)
        we = w + z
        ew = xp.exp(we)
        we2 = we * we
        we3 = we2 * we
        Q += (xp.exp(we / 2.0) - 1.0) / we
        f1 += (-4.0 - we + ew * (4.0 - 3.0 * we + we2)) / we3
        f2 += (2.0 + we + ew * (we - 2.0)) / we3
        f3 += (-4.0 - 3.0 * we - we2 + ew * (4.0 - we)) / we3
    scale = dt / M
    return xp.exp(w), xp.exp(w / 2.0), Q * scale, f1 * scale, f2 * scale, f3 * scale
