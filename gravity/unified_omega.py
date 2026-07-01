"""
gravity/unified_omega.py

Single source of truth for the IRER conformal geometry mapping.

This module maps informational density ρ = |ψ|² to the spatial
conformal metric Ω² used by the covariant Laplacian.

The mapping is:

    Ω² = (ρ_vac / ρ_capped)^a

Stability protections ensure safe operation in FP32 spectral solvers.
"""

import numpy as np
from typing import Any, Dict, cast

try:
    import cupy as cp
except ImportError:
    cp = None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_mu_sigma(rho_safe, xp, params: Dict[str, float]):
    distributed_active = _as_bool(params.get("distributed_enabled", False)) or _as_bool(
        params.get("domain_decomposed", False)
    )
    use_global_stats = _as_bool(params.get("global_stats_enabled", False)) or _as_bool(
        params.get("use_global_stats", False)
    )

    if not use_global_stats:
        if distributed_active:
            raise RuntimeError(
                "Distributed/decomposed execution requires global mu/sigma stats. "
                "Enable global_stats_enabled and provide synchronized global_mu/global_sigma."
            )
        return xp.mean(rho_safe), xp.std(rho_safe)

    if "global_mu" in params and "global_sigma" in params:
        global_mu = xp.asarray(params["global_mu"], dtype=rho_safe.dtype)
        global_sigma = xp.maximum(
            xp.asarray(params["global_sigma"], dtype=rho_safe.dtype),
            xp.asarray(1e-16, dtype=rho_safe.dtype),
        )
        return global_mu, global_sigma

    if distributed_active:
        raise RuntimeError(
            "Global stats feature is enabled for distributed mode, but global_mu/global_sigma were not provided. "
            "Inject all-reduced global statistics from the distributed launcher."
        )

    return xp.mean(rho_safe), xp.std(rho_safe)


def _as_xp_float64(value, xp):
    return xp.asarray(value, dtype=xp.float64)


def _softplus(x, xp):
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, xp.asarray(0.0, dtype=x.dtype))


def _smooth_min_with_derivative(x, cap, beta, xp):
    z = beta * (x - cap)
    soft = _softplus(z, xp)
    x_capped = x - (soft / beta)
    d_x_capped_dx = 1.0 / (1.0 + xp.exp(z))
    return x_capped, d_x_capped_dx


def _soft_clip_log_with_derivative(value, low, high, beta, xp):
    tiny = xp.asarray(1e-30, dtype=xp.float64)
    value_safe = xp.maximum(value, tiny)
    low_safe = xp.maximum(low, tiny)
    high_safe = xp.maximum(high, low_safe + tiny)

    log_v = xp.log(value_safe)
    log_lo = xp.log(low_safe)
    log_hi = xp.log(high_safe)

    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    denom = half + xp.asarray(1e-12, dtype=xp.float64)
    z = beta * ((log_v - center) / denom)
    tanh_z = xp.tanh(z)

    log_soft = center + half * tanh_z
    soft_value = xp.exp(log_soft)

    dlogsoft_dlogv = beta * (1.0 - tanh_z * tanh_z)
    dsoft_dvalue = (soft_value / value_safe) * dlogsoft_dlogv
    return soft_value, dsoft_dvalue


def derive_stable_conformal_factor(
    rho,
    params: Dict[str, float],
    epsilon: float = 1e-12,
    debug: bool = False
) -> Any:
    """
    Compute conformal metric factor Ω² from density field ρ.

    Stability protections:

    • Density floor: prevents division by zero
    • Topological cap: limits extreme density spikes
    • Conformal horizon: ensures FP32 spectral stability

    Guarantees:

        1e-9 ≤ Ω² ≤ 1e6
    """

    # Device detection
    xp = cp.get_array_module(rho) if (cp is not None) else np
    rho64 = _as_xp_float64(rho, xp)

    # Parameter extraction
    # param_rho_vac here is the CONFORMAL REFERENCE DENSITY (geometry only); the
    # vacuum oscillator frequency is the separate param_omega0 used in the solver's
    # L_k.  Canonical default (1.0) matches orchestrator.contracts.DEFAULT_PARAM_RHO_VAC.
    rho_vac = _as_xp_float64(float(params.get("param_rho_vac", 1.0)), xp)
    a = _as_xp_float64(float(params.get("param_a_coupling", 1.0)), xp)
    soft_min_beta = _as_xp_float64(float(params.get("param_conformal_softmin_beta", 8.0)), xp)
    soft_clip_beta = _as_xp_float64(float(params.get("param_conformal_softclip_beta", 3.0)), xp)
    omega_min = _as_xp_float64(float(params.get("param_omega_sq_min", 1e-9)), xp)
    omega_max = _as_xp_float64(float(params.get("param_omega_sq_max", 1e6)), xp)

    # Density floor
    rho_safe = xp.maximum(rho64, _as_xp_float64(epsilon, xp))

    # Topological cap (μ + 3σ).
    # Bypassed in the simulation path (param_skip_topology_cap=True) so that geometry
    # depends only on local pointwise density, not on field-wide statistics.
    if _as_bool(params.get("param_skip_topology_cap", False)):
        rho_capped = rho_safe
    else:
        mu, sigma = _resolve_mu_sigma(rho_safe, xp, params)
        rho_cap = mu + 3.0 * sigma
        rho_capped, _ = _smooth_min_with_derivative(rho_safe, rho_cap, soft_min_beta, xp)

    # Conformal law
    omega_sq = (rho_vac / rho_capped) ** a

    # Conformal horizon with smooth asymptotic saturation in log-space
    omega_sq, _ = _soft_clip_log_with_derivative(omega_sq, omega_min, omega_max, soft_clip_beta, xp)

    if debug:
        if not xp.isfinite(omega_sq).all():
            raise RuntimeError("Non-finite Ω² detected")

    return cast(Any, omega_sq)


def derive_stable_conformal_factor_with_gradient(
    rho,
    params: Dict[str, float],
    epsilon: float = 1e-12,
    debug: bool = False
) -> tuple[Any, Any]:
    """
    Compute conformal metric factor Ω² and its gradient ∂Ω²/∂ρ from density field ρ.

    Returns:
        omega_sq: The conformal factor Ω²
        d_omega_sq_d_rho: Gradient ∂Ω²/∂ρ
    """

    # Device detection
    xp = cp.get_array_module(rho) if (cp is not None) else np
    rho64 = _as_xp_float64(rho, xp)

    # Parameter extraction
    # param_rho_vac here is the CONFORMAL REFERENCE DENSITY (geometry only); the
    # vacuum oscillator frequency is the separate param_omega0 used in the solver's
    # L_k.  Canonical default (1.0) matches orchestrator.contracts.DEFAULT_PARAM_RHO_VAC.
    rho_vac = _as_xp_float64(float(params.get("param_rho_vac", 1.0)), xp)
    a = _as_xp_float64(float(params.get("param_a_coupling", 1.0)), xp)
    soft_min_beta = _as_xp_float64(float(params.get("param_conformal_softmin_beta", 8.0)), xp)
    soft_clip_beta = _as_xp_float64(float(params.get("param_conformal_softclip_beta", 3.0)), xp)
    omega_min = _as_xp_float64(float(params.get("param_omega_sq_min", 1e-9)), xp)
    omega_max = _as_xp_float64(float(params.get("param_omega_sq_max", 1e6)), xp)

    # Density floor
    rho_safe = xp.maximum(rho64, _as_xp_float64(epsilon, xp))

    # Topological cap (μ + 3σ).
    # Bypassed in the simulation path (param_skip_topology_cap=True) so that geometry
    # depends only on local pointwise density, not on field-wide statistics.
    if _as_bool(params.get("param_skip_topology_cap", False)):
        rho_capped = rho_safe
        d_rho_capped_d_rho = xp.ones_like(rho_safe)
    else:
        mu, sigma = _resolve_mu_sigma(rho_safe, xp, params)
        rho_cap = mu + 3.0 * sigma
        rho_capped, d_rho_capped_d_rho = _smooth_min_with_derivative(rho_safe, rho_cap, soft_min_beta, xp)

    # Conformal law
    omega_sq = (rho_vac / rho_capped) ** a

    # Compute gradient ∂Ω²/∂ρ
    ratio = rho_vac / rho_capped
    d_omega_sq_d_rho = a * (ratio ** (a - 1)) * (-ratio / rho_capped) * d_rho_capped_d_rho

    # Smooth conformal horizon and chain-rule derivative propagation.
    omega_sq, d_soft_d_raw = _soft_clip_log_with_derivative(omega_sq, omega_min, omega_max, soft_clip_beta, xp)
    d_omega_sq_d_rho = d_soft_d_raw * d_omega_sq_d_rho

    if debug:
        if not xp.isfinite(omega_sq).all() or not xp.isfinite(d_omega_sq_d_rho).all():
            raise RuntimeError("Non-finite Ω² or gradient detected")

    return cast(Any, omega_sq), cast(Any, d_omega_sq_d_rho)