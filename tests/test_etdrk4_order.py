"""ETDRK4 convergence-order gates (2026-10-02 fix).

Both integrator bugs found on 2026-10-02 passed every identity and parity test, because identity tests check
what the equations conserve and CuPy/JAX parity compared two copies of the same code. A dt-halving order test
catches both:
  1. contour coefficients took real() of a half-circle mean (valid only for real L; ours is complex);
  2. stage c used N(a) instead of N(u_n), making the scheme 2nd order.
"""
import numpy as np
import pytest

from solver.etdrk4_coeffs import etdrk4_coefficients


def _observed_orders(run, dts):
    ref = run(dts[-1] / 16)
    errs = [np.max(np.abs(run(dt) - ref)) for dt in dts]
    return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def _ode_run(L, T=2.0):
    c = 1.0 + 0.5j
    N = lambda u: -0.5 * np.abs(u) ** 2 * u + c

    def run(dt):
        E, E2, Q, f1, f2, f3 = etdrk4_coefficients(L, dt, np)
        u = np.ones_like(L)
        for _ in range(int(round(T / dt))):
            Nu = N(u); a = E2 * u + Q * Nu; Na = N(a); b = E2 * u + Q * Na; Nb = N(b)
            cc = E2 * a + Q * (2 * Nb - Nu); Nc = N(cc)
            u = E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        return u
    return run


@pytest.mark.parametrize("L", [
    np.array([-0.0704 + 1.1866j, -2.0 + 1.1866j]),   # Phase C: -D k^2 - eta + i*omega0
    np.array([-4.0j, -9.0j]),                          # conservative NLS branch: -i D k^2
    np.array([-0.0704 + 0j, -5.0 + 0j]),               # real L control
])
def test_coefficients_are_fourth_order_for_complex_L(L):
    orders = _observed_orders(_ode_run(L), [0.04, 0.02, 0.01])
    assert min(orders) > 3.7, orders


def test_coefficients_match_closed_form_away_from_cancellation():
    L = np.array([-3.0 + 2.0j, -0.5 - 4.0j]); dt = 0.5; w = L * dt
    _, _, Q, f1, f2, f3 = etdrk4_coefficients(L, dt, np)
    ew = np.exp(w)
    np.testing.assert_allclose(Q, dt * (np.exp(w / 2) - 1) / w, rtol=1e-12)
    np.testing.assert_allclose(f1, dt * (-4 - w + ew * (4 - 3 * w + w ** 2)) / w ** 3, rtol=1e-12)
    np.testing.assert_allclose(f2, dt * (2 + w + ew * (w - 2)) / w ** 3, rtol=1e-12)
    np.testing.assert_allclose(f3, dt * (-4 - 3 * w - w ** 2 + ew * (4 - w)) / w ** 3, rtol=1e-12)


def test_cupy_solver_step_is_fourth_order():
    """Full ETDRK4Solver.step with a smooth cubic nonlinearity (stage-c bug gave order 2.00 here)."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("no CUDA device")
    from solver.core import ETDRK4Solver

    params = {"param_D": 0.01, "param_eta": 0.0704, "param_rho_vac": 1.1866}
    n, box, T = 16, 10.0, 1.0
    x = cp.linspace(-box / 2, box / 2, n, endpoint=False)
    X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")
    psi0 = cp.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2).astype(cp.complex128)

    def run(dt):
        s = ETDRK4Solver(n, box, dt, params)
        s.N_op = lambda k: s.fft_single(-0.5 * cp.abs(s.ifft_single(k)) ** 2 * s.ifft_single(k)) * s.dealias_mask
        k = s.fft_single(psi0) * s.dealias_mask
        for _ in range(int(round(T / dt))):
            k = s.step(k)
        return cp.asnumpy(s.ifft_single(k))

    orders = _observed_orders(run, [0.1, 0.05, 0.025])
    assert min(orders) > 3.5, orders
