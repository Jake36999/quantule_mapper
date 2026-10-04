"""Basin mapping, stage 3: Newton-Krylov solves and pseudo-arclength continuation of relative equilibria
(IMPLEMENTATION_PLAN_2026-10 Phase F3). Reference design: BifurcationKit.jl's matrix-free shooting.

WHAT IS SOLVED. A state psi of the ETDRK4 substrate (jax_scout/physics.py) is a RELATIVE EQUILIBRIUM if
after time T it returns to itself up to the continuous symmetries of the periodic box -- a global phase
rotation (U(1)) and a translation:

    G(psi, theta, a) = S_a R_theta Phi_T(psi) - psi = 0,     R_theta = exp(-i theta),  S_a = shift by -a

Phi_T is T/dt steps of physics.step (the gated, fixed ETDRK4). The symmetries make G's Jacobian
singular along i*psi (phase) and d_j psi (translations), so the system is BORDERED with one condition
per symmetry, pinning psi against a reference: Im<psi_ref, psi> = 0 and Re<d_j psi_ref, psi - psi_ref> = 0.
Unknowns (psi, theta, a) and equations match: 2M + 4 each (M = N^3 complex values).

Newton steps are matrix-free: J.v comes from jax.jvp, and the linear solve is GMRES. Nothing forms or
stores the Jacobian, so N=48-96 fits on one GPU.

CONTINUATION. With one physics parameter p free (e.g. param_a), pseudo-arclength adds the equation
t . (X - X_prev) = ds along the secant tangent t, so the branch can be followed through folds where
p turns back.

STABILITY. The leading eigenvalues of the linearised return map (Floquet multipliers) come from scipy's
ARPACK on a jvp LinearOperator. Symmetry directions give multipliers ~1; any OTHER |mu| > 1 means the
state is unstable at that parameter.

LIMITS (honest). Breathing states (periodic orbits whose period is itself unknown) need T as an extra
unknown plus a time-phase condition; that is not implemented here. Choose T as a multiple of the
breathing period, or treat the result as a T-map fixed point only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.sparse.linalg import gmres  # noqa: E402

from jax_scout import physics  # noqa: E402

_DEFAULTS = {"param_D": 1.0, "param_eta": 0.1, "param_rho_vac": physics.DEFAULT_PARAM_RHO_VAC,
             "param_omega0": None, "param_a": 0.0, "param_s": 0.0, "param_f": 0.0, "param_a_coupling": 1.0}


def _get(params, key, cont_key, cont_val):
    if key == cont_key:
        return cont_val
    if key == "param_omega0" and params.get("param_omega0") is None:
        return float(params.get("param_rho_vac", _DEFAULTS["param_rho_vac"]))
    v = params.get(key)
    if v is None and key == "param_s":
        v = params.get("param_splash_coupling")
    if v is None and key == "param_f":
        v = params.get("param_splash_fraction")
    return float(v if v is not None else _DEFAULTS[key])


@dataclass
class RelEqProblem:
    N: int
    L: float
    dt: float
    T: float
    params: dict
    cont_key: str = None            # physics parameter to continue in (None = fixed-parameter solve)
    translations: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.n_steps = int(round(self.T / self.dt))
        self.M = self.N ** 3
        k = jnp.fft.fftfreq(self.N, d=self.L / self.N) * 2 * jnp.pi
        self.kx, self.ky, self.kz = jnp.meshgrid(k, k, k, indexing="ij")
        self.kinetic_mode = str(self.params.get("kinetic_mode", "dissipative"))
        self.geom_off = bool(self.params.get("param_geom_off", False))

    # -------------------------------------------------------------- map
    def ops(self, p):
        g = lambda key: _get(self.params, key, self.cont_key, p)  # noqa: E731
        return physics._construct_ops(
            self.N, self.L, self.dt, g("param_D"), g("param_eta"), g("param_rho_vac"), g("param_omega0"),
            g("param_a"), g("param_s"), g("param_f"), g("param_a_coupling"), 3.0,
            physics.OMEGA_SQ_MIN, physics.OMEGA_SQ_MAX, 1.0, 0.5, jnp.float64, jnp.complex128,
            kinetic_mode=self.kinetic_mode, geom_off=self.geom_off)

    def flow(self, psi, p):
        ops = self.ops(p)
        pk = jnp.fft.fftn(psi) * ops.dealias_mask
        pk = jax.lax.fori_loop(0, self.n_steps, lambda i, q: physics.step(q, ops), pk)
        return pk                                              # stays in k-space for the shift

    def return_map(self, psi, theta, a, p):
        pk = self.flow(psi, p)
        shift = jnp.exp(1j * (self.kx * a[0] + self.ky * a[1] + self.kz * a[2]))   # S_a: x -> x - a
        return jnp.fft.ifftn(pk * shift) * jnp.exp(-1j * theta)

    # -------------------------------------------------------------- packing
    def pack(self, psi, theta, a, p=None):
        parts = [jnp.real(psi).ravel(), jnp.imag(psi).ravel(), jnp.atleast_1d(theta), jnp.asarray(a)]
        if p is not None:
            parts.append(jnp.atleast_1d(p))
        return jnp.concatenate(parts)

    def unpack(self, x):
        M, n = self.M, self.N
        psi = (x[:M] + 1j * x[M:2 * M]).reshape(n, n, n)
        return psi, x[2 * M], x[2 * M + 1:2 * M + 4], (x[2 * M + 4] if x.shape[0] > 2 * self.M + 4 else None)

    def _grad(self, psi):
        pk = jnp.fft.fftn(psi)
        return [jnp.fft.ifftn(1j * kk * pk) for kk in (self.kx, self.ky, self.kz)]

    # -------------------------------------------------------------- bordered system
    def equations(self, x, psi_ref, p_fixed=None):
        psi, theta, a, p = self.unpack(x)
        p = p_fixed if p is None else p
        G = self.return_map(psi, theta, a, p) - psi
        phase = jnp.sum(jnp.imag(jnp.conj(psi_ref) * psi))
        if self.translations:
            tr = jnp.stack([jnp.sum(jnp.real(jnp.conj(d) * (psi - psi_ref))) for d in self._grad(psi_ref)])
        else:
            tr = a                                             # translations off: pin the shifts to 0
        return jnp.concatenate([jnp.real(G).ravel(), jnp.imag(G).ravel(), jnp.atleast_1d(phase), tr])


@dataclass
class NewtonResult:
    x: np.ndarray
    converged: bool
    residuals: list
    iters: int


def _make_solver(prob: RelEqProblem, psi_ref, p_fixed, arclength: bool, gmres_tol, restart, maxiter):
    """Compile ONCE: residual and Newton direction. The arclength data (tangent, previous point, ds) are
    jit ARGUMENTS, not closure constants, so a whole branch reuses one compilation. (The first version
    built a fresh jit closure per iteration and recompiled GMRES every Newton step.)"""
    def F(x, t, x_prev, ds):
        r = prob.equations(x, psi_ref, p_fixed)
        if arclength:
            r = jnp.concatenate([r, jnp.atleast_1d(jnp.dot(t, x - x_prev) - ds)])
        return r

    @jax.jit
    def residual(x, t, x_prev, ds):
        return F(x, t, x_prev, ds)

    @jax.jit
    def direction(x, t, x_prev, ds):
        r, lin = jax.linearize(lambda z: F(z, t, x_prev, ds), x)
        dx, _ = gmres(lin, -r, tol=gmres_tol, restart=restart, maxiter=maxiter, solve_method="incremental")
        return r, dx

    return residual, direction


def newton_krylov(prob: RelEqProblem, x0, psi_ref, p_fixed=None, tol=1e-9, max_iter=12, gmres_tol=1e-6,
                  restart=40, maxiter=4, arclength=None, log=None, _solver=None):
    """Matrix-free Newton on prob.equations. `arclength=(t, x_prev, ds)` appends the pseudo-arclength
    equation t.(x - x_prev) = ds (x then carries the free parameter as its last entry)."""
    x = jnp.asarray(x0)
    if arclength is None:
        t = xp = jnp.zeros_like(x)
        ds = 0.0
    else:
        t, xp, ds = arclength
    residual, direction = _solver or _make_solver(prob, psi_ref, p_fixed, arclength is not None,
                                                  gmres_tol, restart, maxiter)
    rms = lambda r: float(jnp.linalg.norm(r)) / np.sqrt(r.shape[0])  # noqa: E731
    res = []
    for it in range(max_iter):
        r, dx = direction(x, t, xp, ds)
        nr = rms(r)
        res.append(nr)
        if log:
            log("    newton %d: |F|_rms = %.3e" % (it, nr))
        if nr < tol:
            return NewtonResult(np.asarray(x), True, res, it)
        lam = 1.0
        while lam > 1e-3:                                   # backtracking on |F|
            xn = x + lam * dx
            if rms(residual(xn, t, xp, ds)) < nr:
                break
            lam *= 0.5
        x = xn
    res.append(rms(residual(x, t, xp, ds)))
    return NewtonResult(np.asarray(x), res[-1] < tol, res, max_iter)


def floquet_multipliers(prob: RelEqProblem, x, p=None, k=8):
    """Leading |mu| of the linearised return map at a converged relative equilibrium (ARPACK + jvp)."""
    from scipy.sparse.linalg import LinearOperator, eigs
    psi, theta, a, p_x = prob.unpack(jnp.asarray(x))
    p = p if p is not None else p_x
    M = prob.M

    def phi(v):
        ps = (v[:M] + 1j * v[M:]).reshape(psi.shape)
        out = prob.return_map(ps, theta, a, p)
        return jnp.concatenate([jnp.real(out).ravel(), jnp.imag(out).ravel()])

    base = jnp.concatenate([jnp.real(psi).ravel(), jnp.imag(psi).ravel()])
    jv = jax.jit(lambda v: jax.jvp(phi, (base,), (v,))[1])
    op = LinearOperator((2 * M, 2 * M), matvec=lambda v: np.asarray(jv(jnp.asarray(v))), dtype=np.float64)
    vals = eigs(op, k=k, which="LM", return_eigenvectors=False, tol=1e-8)
    return np.sort(np.abs(vals))[::-1]


def continue_branch(prob: RelEqProblem, x_a, x_b, psi_ref, n_points=10, ds=None, log=print, tol=1e-9, **kw):
    """Pseudo-arclength continuation from two converged points x_a, x_b (each packed WITH p as last entry).
    Returns the converged packed states along the branch (including x_a, x_b). One compilation serves
    every branch point."""
    pts = [jnp.asarray(x_a), jnp.asarray(x_b)]
    ds = ds if ds is not None else float(jnp.linalg.norm(pts[1] - pts[0]))
    solver = _make_solver(prob, psi_ref, None, True, kw.get("gmres_tol", 1e-6), kw.get("restart", 40),
                          kw.get("maxiter", 4))
    for _ in range(n_points):
        t = pts[-1] - pts[-2]
        t = t / jnp.linalg.norm(t)
        guess = pts[-1] + ds * t
        r = newton_krylov(prob, guess, psi_ref, tol=tol, arclength=(t, pts[-1], ds), _solver=solver,
                          max_iter=kw.get("max_iter", 12))
        p = float(r.x[-1])
        log("  branch point %d: p = %.6g  converged=%s  |F|=%.2e" % (len(pts), p, r.converged, r.residuals[-1]))
        if not r.converged:
            ds *= 0.5
            if ds < 1e-6:
                break
            continue
        pts.append(jnp.asarray(r.x))
    return [np.asarray(x) for x in pts]
