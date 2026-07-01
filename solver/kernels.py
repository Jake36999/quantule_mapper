"""solver/kernels.py - stateless GPU fused kernels (verbatim from worker_cupy.py)."""
import cupy as cp


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

@cp.fuse()
def calculate_cov_laplacian_fused(psi, dx, dy, dz, lap_flat, omega, omega_sq, d_omega_d_rho, D_spatial):
    # Dynamically resolve gradient of density from complex fields
    psi_conj = cp.conj(psi)
    gx = 2.0 * cp.real(psi_conj * dx)
    gy = 2.0 * cp.real(psi_conj * dy)
    gz = 2.0 * cp.real(psi_conj * dz)
    
    # Resolve gradient of conformal factor
    g_om_x = d_omega_d_rho * gx
    g_om_y = d_omega_d_rho * gy
    g_om_z = d_omega_d_rho * gz
    
    grad_omega_dot_grad_psi = g_om_x*dx + g_om_y*dy + g_om_z*dz
    cov_term = (D_spatial - 2.0) * grad_omega_dot_grad_psi / omega
    return (lap_flat + cov_term) / omega_sq

@cp.fuse()
def calculate_nonlinear_rhs(psi, rho, lap_cov, lap_flat, D_diff, a, s, f):
    nonlin = a * psi * rho + s * psi * (rho**2) + f * psi * (rho**3)
    # -Dk² is now in L_k (exponentiated by ETDRK4), so N carries only the geometry correction.
    return D_diff * (lap_cov - lap_flat) + nonlin

@cp.fuse()
def fused_compute_rho(psi, epsilon):
    re = cp.real(psi)
    im = cp.imag(psi)
    return cp.maximum((re * re) + (im * im), epsilon)

@cp.fuse()
def fused_process_omega(omega_sq_tmp, min_val, max_val):
    tiny = cp.float64(1e-30)
    omega_safe = cp.maximum(omega_sq_tmp, tiny)
    log_omega = cp.log(omega_safe)
    log_lo = cp.log(cp.maximum(min_val, tiny))
    log_hi = cp.log(cp.maximum(max_val, cp.maximum(min_val, tiny) + tiny))
    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    z = (log_omega - center) / (half + cp.float64(1e-12))
    omega_sq_soft = cp.exp(center + half * cp.tanh(z))
    omega = cp.sqrt(omega_sq_soft)
    return omega_sq_soft, omega

@cp.fuse()
def fused_scale_derivative(omega_sq, d_omega_d_rho, min_val, max_val):
    # Preserve nonzero geometric feedback at conformal boundaries with smooth inverse-square decay.
    tiny = cp.float64(1e-12)
    log_lo = cp.log(cp.maximum(min_val, tiny))
    log_hi = cp.log(cp.maximum(max_val, cp.maximum(min_val, tiny) + tiny))
    log_omega = cp.log(cp.maximum(omega_sq, tiny))
    span = (log_hi - log_lo) + tiny
    xi = (log_omega - log_lo) / span
    d_low = xi + cp.float64(1e-6)
    d_high = (cp.float64(1.0) - xi) + cp.float64(1e-6)
    soft_decay = cp.float64(1.0) / (
        cp.float64(1.0)
        + cp.float64(1.0) / (d_low * d_low)
        + cp.float64(1.0) / (d_high * d_high)
    )
    return d_omega_d_rho * soft_decay

@cp.fuse()
def compute_kt_stage_base(E2, psi_k, Q, N_k):
    return E2 * psi_k + Q * N_k

@cp.fuse()
def compute_kt_stage_c(E2, a_k, Q, N_b, N_a):
    return E2 * a_k + Q * (2.0 * N_b - N_a)

@cp.fuse()
def combine_kt_etdrk4(psi_k, N_n, N_a, N_b, N_c, E, f1, f2, f3):
    return E * psi_k + f1 * N_n + 2.0 * f2 * (N_a + N_b) + f3 * N_c

# =============================================================================
# ETDRK4 Solver Architecture (Rules 3, 4, 8, 9, 13)
# =============================================================================

