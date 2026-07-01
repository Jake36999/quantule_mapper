"""
jax_scout/physics.py

Jittable JAX port of the ASTE ETDRK4 SNCGL solver. This is an *exact mirror* of:
  - solver/core.py            (ETDRK4Solver: L_k, Kassam-Trefethen coeffs, N_op, step)
  - solver/kernels.py         (fused elementwise kernels)
  - gravity/unified_omega.py  (conformal geometry Omega^2 + analytic gradient)

Role in the asymmetric hybrid: CuPy/ASTE is the FP64 source of truth; this module is
the high-speed FP32 *scout* for vmap parameter sweeps and autodiff sensitivity. It is
proven numerically equivalent to the CuPy backbone in FP64 (see jax_scout/equivalence/).

Design notes
------------
* dtype-parametric: build_operators(..., real_dtype, complex_dtype) serves both the
  FP64 equivalence proof (float64/complex128) and the FP32 scout (float32/complex64).
* For FP64 parity the caller MUST enable x64 BEFORE creating arrays:
      import jax; jax.config.update("jax_enable_x64", True)
  This module never creates arrays at import time, so that ordering is respected.
* The causal "field of affect" (A) in solver/core.py is decoupled from psi evolution
  (it is integrated alongside but never fed back into N_op), so it is intentionally
  omitted here: the psi trajectory is self-contained. It can be added if ever coupled.
"""
from functools import partial
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
from jax import lax

# Mirrors orchestrator/contracts.py
DEFAULT_PARAM_RHO_VAC = 1.0
DEFAULT_PARAM_OMEGA0 = 1.0

# Solver-hardcoded constants from solver/core.py ETDRK4Solver.__init__
RHO_FLOOR = 1e-7
OMEGA_SQ_MIN = 1e-9      # self.omega_sq_min
OMEGA_SQ_MAX = 1e6       # self.omega_sq_max
D_SPATIAL = 3.0
KT_CONTOUR_M = 64        # M >= 64 (ALETHEIA V4.4 Directive 4)
GEOM_EPSILON = 1e-12     # derive_stable_conformal_factor default epsilon


class Ops(NamedTuple):
    """Precomputed operators + scalar params (a pytree; jit/vmap friendly)."""
    # k-space arrays
    ikx: jnp.ndarray
    iky: jnp.ndarray
    ikz: jnp.ndarray
    minus_k_sq: jnp.ndarray
    L_k: jnp.ndarray
    E: jnp.ndarray
    E2: jnp.ndarray
    Q: jnp.ndarray
    f1: jnp.ndarray
    f2: jnp.ndarray
    f3: jnp.ndarray
    dealias_mask: jnp.ndarray
    c_sq_k_sq: jnp.ndarray
    # scalar params (0-d arrays)
    D_diff: jnp.ndarray
    eta: jnp.ndarray
    a: jnp.ndarray
    s: jnp.ndarray
    f: jnp.ndarray
    rho_vac: jnp.ndarray         # geometry conformal reference density
    a_coupling: jnp.ndarray      # geometry conformal exponent (param_a_coupling)
    soft_clip_beta: jnp.ndarray
    omega_min_geo: jnp.ndarray   # geometry soft-clip bounds (param_omega_sq_min/max)
    omega_max_geo: jnp.ndarray
    omega_sq_min: jnp.ndarray    # solver fused_process_omega / fused_scale_derivative bounds
    omega_sq_max: jnp.ndarray
    rho_floor: jnp.ndarray
    D_spatial: jnp.ndarray
    geom_eps: jnp.ndarray


# ---------------------------------------------------------------------------
# Geometry — mirror of gravity/unified_omega.py (param_skip_topology_cap=True path)
# ---------------------------------------------------------------------------

def _soft_clip_log_with_derivative(value, low, high, beta):
    """Mirror unified_omega._soft_clip_log_with_derivative."""
    tiny = 1e-30
    value_safe = jnp.maximum(value, tiny)
    low_safe = jnp.maximum(low, tiny)
    high_safe = jnp.maximum(high, low_safe + tiny)

    log_v = jnp.log(value_safe)
    log_lo = jnp.log(low_safe)
    log_hi = jnp.log(high_safe)

    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    denom = half + 1e-12
    z = beta * ((log_v - center) / denom)
    tanh_z = jnp.tanh(z)

    soft_value = jnp.exp(center + half * tanh_z)
    dlogsoft_dlogv = beta * (1.0 - tanh_z * tanh_z)
    dsoft_dvalue = (soft_value / value_safe) * dlogsoft_dlogv
    return soft_value, dsoft_dvalue


def _geometry_with_gradient(rho, ops, rho_vac_eff=None):
    """Mirror derive_stable_conformal_factor_with_gradient with skip_topology_cap=True.
    rho_vac_eff (optional field) overrides the scalar ops.rho_vac for the A-field
    prototype: rho_vac_eff = max(rho_vac + gamma_A*A, eps). None -> scalar ops.rho_vac
    (no-op; gamma_A=0 path)."""
    rho_safe = jnp.maximum(rho, ops.geom_eps)
    rho_capped = rho_safe  # skip cap: d_rho_capped_d_rho = 1

    rv = ops.rho_vac if rho_vac_eff is None else rho_vac_eff
    omega_sq = (rv / rho_capped) ** ops.a_coupling
    ratio = rv / rho_capped
    d_omega_sq_d_rho = ops.a_coupling * (ratio ** (ops.a_coupling - 1.0)) * (-ratio / rho_capped)

    omega_sq, d_soft = _soft_clip_log_with_derivative(
        omega_sq, ops.omega_min_geo, ops.omega_max_geo, ops.soft_clip_beta
    )
    d_omega_sq_d_rho = d_soft * d_omega_sq_d_rho
    return omega_sq, d_omega_sq_d_rho


# ---------------------------------------------------------------------------
# Fused-kernel equivalents — mirror of solver/kernels.py
# ---------------------------------------------------------------------------

def _fused_process_omega(omega_sq_tmp, min_val, max_val):
    """Mirror kernels.fused_process_omega (log-space tanh soft clip, no beta)."""
    tiny = 1e-30
    omega_safe = jnp.maximum(omega_sq_tmp, tiny)
    log_omega = jnp.log(omega_safe)
    log_lo = jnp.log(jnp.maximum(min_val, tiny))
    log_hi = jnp.log(jnp.maximum(max_val, jnp.maximum(min_val, tiny) + tiny))
    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    z = (log_omega - center) / (half + 1e-12)
    omega_sq_soft = jnp.exp(center + half * jnp.tanh(z))
    omega = jnp.sqrt(omega_sq_soft)
    return omega_sq_soft, omega


def _fused_scale_derivative(omega_sq, d_omega_d_rho, min_val, max_val):
    """Mirror kernels.fused_scale_derivative (boundary soft-decay weighting)."""
    tiny = 1e-12
    log_lo = jnp.log(jnp.maximum(min_val, tiny))
    log_hi = jnp.log(jnp.maximum(max_val, jnp.maximum(min_val, tiny) + tiny))
    log_omega = jnp.log(jnp.maximum(omega_sq, tiny))
    span = (log_hi - log_lo) + tiny
    xi = (log_omega - log_lo) / span
    d_low = xi + 1e-6
    d_high = (1.0 - xi) + 1e-6
    soft_decay = 1.0 / (1.0 + 1.0 / (d_low * d_low) + 1.0 / (d_high * d_high))
    return d_omega_d_rho * soft_decay


def _cov_laplacian(psi, dx, dy, dz, lap_flat, omega, omega_sq, d_omega_d_rho, D_spatial):
    """Mirror kernels.calculate_cov_laplacian_fused."""
    psi_conj = jnp.conj(psi)
    gx = 2.0 * jnp.real(psi_conj * dx)
    gy = 2.0 * jnp.real(psi_conj * dy)
    gz = 2.0 * jnp.real(psi_conj * dz)

    g_om_x = d_omega_d_rho * gx
    g_om_y = d_omega_d_rho * gy
    g_om_z = d_omega_d_rho * gz

    grad_omega_dot_grad_psi = g_om_x * dx + g_om_y * dy + g_om_z * dz
    cov_term = (D_spatial - 2.0) * grad_omega_dot_grad_psi / omega
    return (lap_flat + cov_term) / omega_sq


def _nonlinear_rhs(psi, rho, lap_cov, lap_flat, D_diff, a, s, f):
    """Mirror kernels.calculate_nonlinear_rhs (cubic-quintic-septic GL)."""
    nonlin = a * psi * rho + s * psi * (rho ** 2) + f * psi * (rho ** 3)
    return D_diff * (lap_cov - lap_flat) + nonlin


# ---------------------------------------------------------------------------
# N_op + ETDRK4 step — mirror of solver/core.py
# ---------------------------------------------------------------------------

def n_op(psi_k, ops, rho_vac_eff=None, omega_sq_mult=None, a_vec=None, q_tensor=None):
    """Mirror ETDRK4Solver.N_op (geometry-corrected nonlinear operator in k-space).
    rho_vac_eff: optional A-modulated vacuum reference field (vacuum_ref topology).
    omega_sq_mult: optional multiplier on Omega^2 (additive_potential topology:
        log Omega^2_eff = log Omega^2 + gamma_A*A  ->  mult = exp(gamma_A*A)).
    a_vec: optional CURRENT-COUPLED vector potential (ax,ay,az) ALREADY scaled by gamma_A
        (i.e. a_vec = gamma_A * A_real). Adds minimal coupling (grad -> grad - i*a_vec) to the
        kinetic term: D_diff*( -2i (a.grad psi) - |a|^2 psi ). See docs/AFIELD_CURRENT_COUPLED_RFC.md.
    All None = scalar local geometry (gamma_A=0, exact baseline)."""
    # 1. Derivative spectra (ikx_filtered == ikx since damping is disabled).
    psi = jnp.fft.ifftn(psi_k)
    grad_x = jnp.fft.ifftn(ops.ikx * psi_k)
    grad_y = jnp.fft.ifftn(ops.iky * psi_k)
    grad_z = jnp.fft.ifftn(ops.ikz * psi_k)
    lap_flat = jnp.fft.ifftn(ops.minus_k_sq * psi_k)

    # 2. Local density (rho floor)
    rho = jnp.maximum(jnp.real(psi) ** 2 + jnp.imag(psi) ** 2, ops.rho_floor)

    # 3. Geometry from local rho only (skip topology cap)
    omega_sq_tmp, d_omega_sq_d_rho_tmp = _geometry_with_gradient(rho, ops, rho_vac_eff)
    if omega_sq_mult is not None:
        # additive-potential A-coupling: A added to the conformal (Weyl) potential log Omega^2.
        # A is independent of rho, so the rho-derivative scales by the same multiplier.
        omega_sq_tmp = omega_sq_tmp * omega_sq_mult
        d_omega_sq_d_rho_tmp = d_omega_sq_d_rho_tmp * omega_sq_mult
    omega_sq, omega = _fused_process_omega(omega_sq_tmp, ops.omega_sq_min, ops.omega_sq_max)
    d_omega_sq_d_rho = _fused_scale_derivative(
        omega_sq, d_omega_sq_d_rho_tmp, ops.omega_sq_min, ops.omega_sq_max
    )
    d_omega_d_rho = d_omega_sq_d_rho / (2.0 * jnp.maximum(omega, 1e-15))

    # 4. Covariant Laplacian
    lap_cov = _cov_laplacian(
        psi, grad_x, grad_y, grad_z, lap_flat, omega, omega_sq, d_omega_d_rho, ops.D_spatial
    )

    # 5. Nonlinear field operator
    n_real = _nonlinear_rhs(psi, rho, lap_cov, lap_flat, ops.D_diff, ops.a, ops.s, ops.f)

    # 5b. Current-coupled minimal-coupling term (grad -> grad - i*a_vec), a_vec = gamma_A*A_real.
    #     (grad_x/y/z are already d psi/d{x,y,z}.) a_vec=None -> exact baseline (no-op).
    if a_vec is not None:
        ax, ay, az = a_vec
        a_dot_grad = ax * grad_x + ay * grad_y + az * grad_z
        a_sq = ax * ax + ay * ay + az * az
        n_real = n_real + ops.D_diff * (-2j * a_dot_grad - a_sq * psi)

    # 5c. ANISOTROPIC metric proxy: CONSERVATIVE (dispersive) anisotropic propagation
    #     i * D_diff * div( (lam*Q_ij) grad_j psi ), Q symmetric traceless. The imaginary factor
    #     makes it metric-like (Schrodinger i*D*lap, |psi|^2-conserving) — REDIRECTS wave flow
    #     anisotropically rather than dissipating it (a real D*div(Q grad) just smooths/dissipates).
    #     q_tensor ALREADY scaled by lam; None -> exact baseline (no-op).
    #     See docs/ANISOTROPIC_METRIC_TENSOR_RFC.md (resonance-weighted g_ij = Omega^2(d + lam Q)).
    if q_tensor is not None:
        qxx, qyy, qzz, qxy, qxz, qyz = q_tensor
        Fx = qxx * grad_x + qxy * grad_y + qxz * grad_z
        Fy = qxy * grad_x + qyy * grad_y + qyz * grad_z
        Fz = qxz * grad_x + qyz * grad_y + qzz * grad_z
        divF = (jnp.fft.ifftn(ops.ikx * jnp.fft.fftn(Fx))
                + jnp.fft.ifftn(ops.iky * jnp.fft.fftn(Fy))
                + jnp.fft.ifftn(ops.ikz * jnp.fft.fftn(Fz)))
        n_real = n_real + 1j * ops.D_diff * divF

    # 6. Single spectral transform + dealias
    n_k = jnp.fft.fftn(n_real) * ops.dealias_mask
    return n_k


def step(psi_k, ops, rho_vac_eff=None, omega_sq_mult=None, a_vec=None, q_tensor=None):
    """Mirror ETDRK4Solver.step (Kassam-Trefethen ETDRK4 in spectral space).
    rho_vac_eff / omega_sq_mult / a_vec (optional A-coupling fields) are held FIXED across the 4
    ETDRK4 substages (A updated once per outer step, like solver/run.py). All None = baseline."""
    n_n = n_op(psi_k, ops, rho_vac_eff, omega_sq_mult, a_vec, q_tensor)
    a_k = ops.E2 * psi_k + ops.Q * n_n
    n_a = n_op(a_k, ops, rho_vac_eff, omega_sq_mult, a_vec, q_tensor)
    b_k = ops.E2 * psi_k + ops.Q * n_a
    n_b = n_op(b_k, ops, rho_vac_eff, omega_sq_mult, a_vec, q_tensor)
    c_k = ops.E2 * a_k + ops.Q * (2.0 * n_b - n_a)
    n_c = n_op(c_k, ops, rho_vac_eff, omega_sq_mult, a_vec, q_tensor)
    psi_next_k = ops.E * psi_k + ops.f1 * n_n + 2.0 * ops.f2 * (n_a + n_b) + ops.f3 * n_c
    psi_next_k = psi_next_k * ops.dealias_mask
    return psi_next_k


# ---------------------------------------------------------------------------
# Operator construction — mirror of ETDRK4Solver.__init__
# ---------------------------------------------------------------------------

def _construct_ops(N, L, dt, D_diff, eta, rho_vac, omega0, a, s, f,
                   a_coupling, soft_clip_beta, omega_min_geo, omega_max_geo,
                   c_affect, dealias_frac, rd, cd) -> Ops:
    """Build Ops from scalar values. Scalars may be Python floats OR traced jnp
    scalars, so this is safe under jax.vmap (the sweep path)."""
    # --- k grid (computed in float64 for determinism, then cast) ---
    k = jnp.fft.fftfreq(N, d=L / N).astype(jnp.float64) * (2.0 * jnp.pi)
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij')
    k_sq = (kx ** 2 + ky ** 2 + kz ** 2)

    ikx = (1j * kx).astype(cd)
    iky = (1j * ky).astype(cd)
    ikz = (1j * kz).astype(cd)
    minus_k_sq = (-k_sq).astype(cd)
    c_sq_k_sq = ((c_affect ** 2) * k_sq).astype(rd)

    L_k = (-D_diff * k_sq + (-eta + 1j * omega0)).astype(jnp.complex128)

    # --- Kassam-Trefethen contour-integral coefficients (M points) ---
    M = KT_CONTOUR_M
    theta = jnp.exp(1j * jnp.pi * (jnp.arange(1, M + 1, dtype=jnp.float64) - 0.5) / M).astype(jnp.complex128)
    r = 1.0
    w = (L_k * dt).astype(jnp.complex128)

    def body(i, acc):
        Q_acc, f1_acc, f2_acc, f3_acc = acc
        z = r * theta[i]
        we = w + z
        ew = jnp.exp(we)
        we3 = we ** 3
        Q_acc = Q_acc + (jnp.exp(we / 2.0) - 1.0) / we
        f1_acc = f1_acc + (-4.0 - we + ew * (4.0 - 3.0 * we + we ** 2)) / we3
        f2_acc = f2_acc + (2.0 + we + ew * (we - 2.0)) / we3
        f3_acc = f3_acc + (-4.0 - 3.0 * we - we ** 2 + ew * (4.0 - we)) / we3
        return (Q_acc, f1_acc, f2_acc, f3_acc)

    z0 = jnp.zeros_like(w)
    Q_acc, f1_acc, f2_acc, f3_acc = lax.fori_loop(0, M, body, (z0, z0, z0, z0))

    Q = (dt * jnp.real(Q_acc / M)).astype(rd)
    f1 = (dt * jnp.real(f1_acc / M)).astype(rd)
    f2 = (dt * jnp.real(f2_acc / M)).astype(rd)
    f3 = (dt * jnp.real(f3_acc / M)).astype(rd)
    E = jnp.exp(w).astype(cd)
    E2 = jnp.exp(w / 2.0).astype(cd)

    k_mag = jnp.sqrt(k_sq)
    dealias_mask = (k_mag <= (dealias_frac * jnp.max(k_mag))).astype(rd)

    sc = lambda v: jnp.asarray(v, dtype=rd)
    return Ops(
        ikx=ikx, iky=iky, ikz=ikz, minus_k_sq=minus_k_sq, L_k=L_k.astype(cd),
        E=E, E2=E2, Q=Q, f1=f1, f2=f2, f3=f3,
        dealias_mask=dealias_mask, c_sq_k_sq=c_sq_k_sq,
        D_diff=sc(D_diff), eta=sc(eta), a=sc(a), s=sc(s), f=sc(f),
        rho_vac=sc(rho_vac), a_coupling=sc(a_coupling), soft_clip_beta=sc(soft_clip_beta),
        omega_min_geo=sc(omega_min_geo), omega_max_geo=sc(omega_max_geo),
        omega_sq_min=sc(OMEGA_SQ_MIN), omega_sq_max=sc(OMEGA_SQ_MAX),
        rho_floor=sc(RHO_FLOOR), D_spatial=sc(D_SPATIAL), geom_eps=sc(GEOM_EPSILON),
    )


def build_operators(N, L, dt, params: Dict[str, float],
                    real_dtype=jnp.float64, complex_dtype=jnp.complex128) -> Ops:
    def fget(*keys_and_default):
        *keys, default = keys_and_default
        for k in keys:
            v = params.get(k, None)
            if v is not None:
                return float(v)
        return float(default)
    return _construct_ops(
        N, L, dt,
        fget('param_D', 1.0), fget('param_eta', 0.1),
        fget('param_rho_vac', DEFAULT_PARAM_RHO_VAC),
        fget('param_omega0', 'param_rho_vac', DEFAULT_PARAM_OMEGA0),
        fget('param_a', 0.0),
        fget('param_s', 'param_splash_coupling', 0.0),
        fget('param_f', 'param_splash_fraction', 0.0),
        fget('param_a_coupling', 1.0),
        fget('param_conformal_softclip_beta', 3.0),
        fget('param_omega_sq_min', OMEGA_SQ_MIN),
        fget('param_omega_sq_max', OMEGA_SQ_MAX),
        fget('param_c_affect', 1.0),
        fget('param_dealias_fraction', 0.5),
        real_dtype, complex_dtype,
    )


# === vmap sweep API ========================================================
# Swept parameter order. param_a (cubic gain) is included so the edge-of-stability
# search can supply a bounded energy source; c_affect=1, dealias=0.5,
# soft_clip_beta=3 remain at the corrected-solver-contract defaults.
SWEEP_PARAM_ORDER = ["param_D", "param_eta", "param_rho_vac", "param_omega0",
                     "param_a_coupling", "param_s", "param_f", "param_a"]


def _ops_from_vec(pvec, N, L, dt, rd, cd):
    D, eta, rho_vac, omega0, a_coupling, s, f, a = (pvec[i] for i in range(8))
    return _construct_ops(N, L, dt, D, eta, rho_vac, omega0, a, s, f,
                          a_coupling, 3.0, OMEGA_SQ_MIN, OMEGA_SQ_MAX, 1.0, 0.5, rd, cd)


def _simulate_one(pvec, psi0, N, L, dt, n_steps, rd, cd):
    ops = _ops_from_vec(pvec, N, L, dt, rd, cd)
    psi_k0 = jnp.fft.fftn(psi0) * ops.dealias_mask

    def step_body(psi_k, _):
        return step(psi_k, ops), None

    psi_k_final, _ = lax.scan(step_body, psi_k0, None, length=n_steps)
    psi_final = jnp.fft.ifftn(psi_k_final)
    rho = jnp.abs(psi_final) ** 2
    max_amp = jnp.max(jnp.abs(psi_final))
    energy0 = jnp.sum(jnp.abs(jnp.fft.ifftn(psi_k0)) ** 2)
    energy1 = jnp.sum(rho)
    finite = jnp.all(jnp.isfinite(jnp.abs(psi_final)))
    return psi_final, max_amp, energy0, energy1, finite


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def sweep(param_matrix, psi0, N, L, dt, n_steps,
          real_dtype=jnp.float64, complex_dtype=jnp.complex128):
    """vmap _simulate_one over a [B,8] parameter matrix.
    Returns (fields[B,N,N,N], max_amp[B], energy0[B], energy1[B], finite[B])."""
    fn = lambda pv: _simulate_one(pv, psi0, N, L, dt, n_steps, real_dtype, complex_dtype)
    return jax.vmap(fn)(param_matrix)


def _simulate_probe(pvec, psi0, N, L, dt, n_steps, rd, cd):
    """Trajectory probe for stable-collapse classification. Returns
    (psi_mid, psi_final, energy[n_steps], max_amp[n_steps], finite). Per-step energy and
    max-amplitude time series are cheap scalars; mid + final fields support node/mode
    persistence + transient detection on the host."""
    ops = _ops_from_vec(pvec, N, L, dt, rd, cd)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask

    def body(psi_k, _):
        psi_k = step(psi_k, ops)
        psi = jnp.fft.ifftn(psi_k)
        return psi_k, (jnp.sum(jnp.abs(psi) ** 2), jnp.max(jnp.abs(psi)))

    half = n_steps // 2
    psi_k, (e1, a1) = lax.scan(body, psi_k, None, length=half)
    psi_mid = jnp.fft.ifftn(psi_k)
    psi_k, (e2, a2) = lax.scan(body, psi_k, None, length=n_steps - half)
    psi_fin = jnp.fft.ifftn(psi_k)
    return (psi_mid, psi_fin, jnp.concatenate([e1, e2]), jnp.concatenate([a1, a2]),
            jnp.all(jnp.isfinite(jnp.abs(psi_fin))))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def sweep_probe(param_matrix, psi0, N, L, dt, n_steps,
                real_dtype=jnp.float64, complex_dtype=jnp.complex128):
    """vmap _simulate_probe over [B,8] params -> trajectory observables for classification.
    Returns (psi_mid[B,...], psi_final[B,...], energy[B,n_steps], max_amp[B,n_steps], finite[B])."""
    fn = lambda pv: _simulate_probe(pv, psi0, N, L, dt, n_steps, real_dtype, complex_dtype)
    return jax.vmap(fn)(param_matrix)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def probe_one(pvec, psi0, N, L, dt, n_steps,
              real_dtype=jnp.float64, complex_dtype=jnp.complex128):
    """Single-config trajectory probe (own psi0) for the candidate deep-dive
    (intact / ablation / isolated-baseline runs use different ICs)."""
    return _simulate_probe(pvec, psi0, N, L, dt, n_steps, real_dtype, complex_dtype)


def initial_psi_k(psi0, ops):
    """Mirror run.py: psi_k = fft(psi) * dealias_mask."""
    return jnp.fft.fftn(psi0) * ops.dealias_mask


@partial(jax.jit, static_argnums=(2,))
def simulate(psi0, ops, n_steps):
    """Run n_steps and return (psi_k0, psi_k_final, trajectory[n_steps] in k-space)."""
    psi_k0 = initial_psi_k(psi0, ops)

    def scan_body(psi_k, _):
        psi_k = step(psi_k, ops)
        return psi_k, psi_k

    psi_k_final, traj = lax.scan(scan_body, psi_k0, None, length=n_steps)
    return psi_k0, psi_k_final, traj
