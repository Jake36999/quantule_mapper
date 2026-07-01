"""
mcp_server.guards — pure validation guards for the write/GPU tools.

No MCP SDK, no cupy, no GPU.  Each guard returns plain data so it is fully
unit-testable.  Implements the safety checks in MCP_TOOLS_SPEC.md §3.8 / §4.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

# A run with param_rho_vac below this tests a degenerate geometry (Omega^2 -> 0).
DEGENERATE_RHO_VAC = 0.05
# Smoke-test hard caps (MCP_TOOLS_SPEC.md §3.7).
SMOKE_MAX_N_GRID = 32
SMOKE_MAX_T_STEPS = 100
# Explicit symplectic-Euler stability bound for the A integrator (conservative).
CFL_LIMIT = 1.0


def is_power_of_two(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def k_cut(N_grid: int, L_domain: float, dealias_fraction: float = 0.5) -> float:
    """Max retained wavenumber magnitude after dealiasing.  k_nyquist = pi*N/L."""
    k_nyquist = math.pi * float(N_grid) / float(L_domain)
    return float(dealias_fraction) * k_nyquist


def cfl_number(c_affect: float, dt: float, N_grid: int, L_domain: float, dealias_fraction: float = 0.5) -> float:
    """c_affect * dt * k_cut — the A integrator becomes unstable when this exceeds CFL_LIMIT."""
    return float(c_affect) * float(dt) * k_cut(N_grid, L_domain, dealias_fraction)


def validate_manifest(
    params: Dict[str, Any],
    N_grid: int,
    T_steps: int,
    dt: float,
    L_domain: float,
) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).  Errors block staging; warnings are surfaced for
    review but do not block (e.g. degenerate geometry, which may still be run but
    cannot enter the main leaderboard).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not is_power_of_two(int(N_grid)):
        errors.append(f"N_grid={N_grid} must be a positive power of 2")
    if int(T_steps) <= 0:
        errors.append(f"T_steps={T_steps} must be positive")
    if float(dt) <= 0:
        errors.append(f"dt={dt} must be positive")
    if float(L_domain) <= 0:
        errors.append(f"L_domain={L_domain} must be positive")

    try:
        rho_vac = float(params.get("param_rho_vac", 1.0))
    except (TypeError, ValueError):
        rho_vac = 1.0
        errors.append("param_rho_vac is not numeric")
    if rho_vac < DEGENERATE_RHO_VAC:
        warnings.append(
            f"param_rho_vac={rho_vac} < {DEGENERATE_RHO_VAC}: DEGENERATE_GEOMETRY "
            "(Omega^2 collapses to the conformal floor); run is tagged and excluded from the main leaderboard"
        )

    # CFL guard for the explicit symplectic-Euler A integrator.
    if is_power_of_two(int(N_grid)) and float(dt) > 0 and float(L_domain) > 0:
        try:
            c_affect = float(params.get("param_c_affect", 1.0))
            dealias = float(params.get("param_dealias_fraction", 0.5))
        except (TypeError, ValueError):
            c_affect, dealias = 1.0, 0.5
        cfl = cfl_number(c_affect, float(dt), int(N_grid), float(L_domain), dealias)
        if cfl > CFL_LIMIT:
            errors.append(
                f"CFL violation: c_affect*dt*k_cut={cfl:.4f} > {CFL_LIMIT}; the A integrator "
                "(explicit symplectic Euler) will be unstable — reduce dt or param_c_affect"
            )

    return errors, warnings


def check_no_overwrite(path: str, overwrite: bool = False) -> List[str]:
    """An existing completed artifact must not be silently overwritten."""
    if path and os.path.exists(path) and not overwrite:
        return [f"output already exists and overwrite=false: {path}"]
    return []


def validate_smoke_caps(N_grid: int, T_steps: int) -> List[str]:
    errors: List[str] = []
    if int(N_grid) > SMOKE_MAX_N_GRID:
        errors.append(f"smoke N_grid={N_grid} exceeds hard cap {SMOKE_MAX_N_GRID}")
    if int(T_steps) > SMOKE_MAX_T_STEPS:
        errors.append(f"smoke T_steps={T_steps} exceeds hard cap {SMOKE_MAX_T_STEPS}")
    if not is_power_of_two(int(N_grid)):
        errors.append(f"smoke N_grid={N_grid} must be a power of 2")
    return errors
