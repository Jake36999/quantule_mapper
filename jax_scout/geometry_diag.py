"""
Emergent-geometry / SDG diagnostic layer (PASSIVE, MONITORED — not solver feedback).

Loop being monitored:  psi -> rho=|psi|^2, phase phi -> T_info / Omega^2 -> curvature.
Omega^2 is taken from the SINGLE SOURCE OF TRUTH (gravity/unified_omega) with
param_skip_topology_cap=True (matches the solver's simulation geometry path). This
module only READS the geometry the solver already produces; it never alters solver
physics. Stamp DIAG_CONTRACT_VERSION into provenance. Promote to active feedback only
after this layer is shown stable + falsifiable.

Headline question: do stable_multinode candidates show density + phase-locking driving
BOUNDED emergent curvature (geometry following resonance density), vs random tensor
noise or runaway curvature?

Native .venv (numpy + scipy; gravity.unified_omega).
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gravity"))
from unified_omega import derive_stable_conformal_factor  # noqa: E402  (single source of truth)

DIAG_CONTRACT_VERSION = "IRER-SDG-DIAG-v1"


def _grad(f, dx):
    return ((np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx),
            (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dx),
            (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2 * dx))


def _lap(f, dx):
    return sum((np.roll(f, -1, ax) - 2 * f + np.roll(f, 1, ax)) / dx ** 2 for ax in range(3))


def _corr(a, b):
    a = a.ravel() - a.ravel().mean(); b = b.ravel() - b.ravel().mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _curl(Ax, Ay, Az, dx):
    g = lambda A, ax: (np.roll(A, -1, ax) - np.roll(A, 1, ax)) / (2 * dx)
    return (g(Az, 1) - g(Ay, 2), g(Ax, 2) - g(Az, 0), g(Ay, 0) - g(Ax, 1))


def diagnose(psi, params, dx, node_sigma=3.0):
    """Passive emergent-geometry/SDG diagnostics for one field. Returns a flat metric dict."""
    psi = np.asarray(psi)
    rho = np.abs(psi) ** 2
    rho_safe = np.maximum(rho, 1e-7)
    geo_params = dict(params); geo_params["param_skip_topology_cap"] = True
    omega_sq = np.asarray(derive_stable_conformal_factor(rho_safe, geo_params), dtype=np.float64)
    omega = np.sqrt(np.maximum(omega_sq, 1e-30))
    log_om = np.log(np.maximum(omega_sq, 1e-30))
    a_c = float(params.get("param_a_coupling", 1.0))
    omin = float(params.get("param_omega_sq_min", 1e-9))
    omax = float(params.get("param_omega_sq_max", 1e6))

    node_mask = rho > (rho.mean() + node_sigma * rho.std())
    bg = ~node_mask
    has_nodes = node_mask.sum() > 0
    sm = lambda x, m: float(x[m].mean()) if m.sum() > 0 else float("nan")

    # === Group 1: resonance-density -> geometry coupling ===
    grx, gry, grz = _grad(rho, dx)
    gox, goy, goz = _grad(omega, dx)
    grm = np.sqrt(grx**2 + gry**2 + grz**2); gom = np.sqrt(gox**2 + goy**2 + goz**2)
    dot = grx*gox + gry*goy + grz*goz
    g = {
        "omega_node_correlation": _corr(np.log(rho_safe), log_om),
        "omega_gradient_alignment": float(np.sum(dot) / (np.sum(grm * gom) + 1e-30)),
        "node_omega_contrast": sm(omega_sq, node_mask) / (sm(omega_sq, bg) + 1e-30),
        "omega_saturation_fraction": float(np.mean((omega_sq <= omin * 1.01) | (omega_sq >= omax * 0.99))),
    }

    # === Group 2: phase-locking -> current coupling ===
    px, py, pz = _grad(psi, dx)
    Jx = np.imag(np.conj(psi) * px); Jy = np.imag(np.conj(psi) * py); Jz = np.imag(np.conj(psi) * pz)
    Jm = np.sqrt(Jx**2 + Jy**2 + Jz**2)            # J_info = rho * grad(phase)
    wx, wy, wz = _curl(Jx, Jy, Jz, dx)
    ph = np.exp(1j * np.angle(psi))
    lbl, nn = ndi.label(node_mask)
    if nn >= 2:
        means = np.array([np.angle(ph[lbl == i].mean()) for i in range(1, nn + 1)])
        drift = float(np.sqrt(1.0 - np.abs(np.exp(1j * means).mean()) ** 2))  # circular spread of node phases
    else:
        drift = float("nan")
    g.update({
        "J_info_l2": float(np.sqrt(np.mean(Jm**2))),
        "phase_coherence_nodes": float(np.abs(ph[node_mask].mean())) if has_nodes else float("nan"),
        "internode_phase_drift": drift,
        "current_circulation_l2": float(np.sqrt(np.mean(wx**2 + wy**2 + wz**2))),
        "current_in_node_fraction": float(Jm[node_mask].sum() / (Jm.sum() + 1e-30)) if has_nodes else float("nan"),
    })

    # === Group 3: informational stress-energy T_info ===
    gp = [Jx / rho_safe, Jy / rho_safe, Jz / rho_safe]     # grad(phase) = J/rho
    gr = [grx, gry, grz]
    T = np.empty((3, 3) + rho.shape)
    for i in range(3):
        for j in range(3):
            T[i, j] = gr[i] * gr[j] / rho_safe + rho * gp[i] * gp[j]
    trace = T[0, 0] + T[1, 1] + T[2, 2]
    p_iso = trace / 3.0
    shear2 = np.zeros_like(rho)
    for i in range(3):
        for j in range(3):
            Sij = T[i, j] - (p_iso if i == j else 0.0)
            shear2 += Sij ** 2
    Tf = np.sqrt(np.sum(T**2, axis=(0, 1)))
    g.update({
        "shear_fraction": float(np.sqrt(np.sum(shear2)) / (np.sqrt(np.sum(Tf**2)) + 1e-30)),
        "stress_node_contrast": sm(Tf, node_mask) / (sm(Tf, bg) + 1e-30),
        "stress_l2": float(np.sqrt(np.mean(Tf**2))),
        "tensor_symmetry_error": float(np.mean(np.abs(T - np.transpose(T, (1, 0, 2, 3, 4))))),
    })

    # === Group 4: emergent curvature / SDG proxy ===
    lnOm = 0.5 * log_om
    glx, gly, glz = _grad(lnOm, dx)
    # conformal scalar-curvature proxy (D=3): R ~ -(2/Omega^2)[lap(lnOmega) + 1/2 |grad lnOmega|^2]
    R = -(2.0 / np.maximum(omega_sq, 1e-30)) * (_lap(lnOm, dx) + 0.5 * (glx**2 + gly**2 + glz**2))
    h_res = _lap(omega, dx) + a_c * rho * omega    # Hamiltonian-constraint residual
    g.update({
        "curvature_l2": float(np.sqrt(np.mean(R**2))),
        "curvature_max": float(np.max(np.abs(R))),
        "curvature_node_correlation": _corr(np.abs(R), rho),
        "sdg_h_norm_l2": float(np.sqrt(np.mean(h_res**2))),
        "n_nodes": int(nn),
    })

    return {"diag_contract_version": DIAG_CONTRACT_VERSION, **g}


def curvature_max_only(psi, params, dx):
    """Lightweight max|conformal curvature| (no T-tensor/current) for inner-loop
    bounded-validity gating. Uses the real unified_omega conformal map."""
    rho_safe = np.maximum(np.abs(psi) ** 2, 1e-7)
    geo = dict(params); geo["param_skip_topology_cap"] = True
    omega_sq = np.asarray(derive_stable_conformal_factor(rho_safe, geo), dtype=np.float64)
    lnOm = 0.5 * np.log(np.maximum(omega_sq, 1e-30))
    glx, gly, glz = _grad(lnOm, dx)
    R = -(2.0 / np.maximum(omega_sq, 1e-30)) * (_lap(lnOm, dx) + 0.5 * (glx ** 2 + gly ** 2 + glz ** 2))
    return float(np.max(np.abs(R)))


def geometry_verdict(diag):
    """Heuristic read of the headline question: bounded geometry following RD vs noise/runaway."""
    follows_rd = abs(diag["omega_node_correlation"]) > 0.5 and abs(diag["curvature_node_correlation"]) > 0.3
    bounded = (diag["curvature_max"] < 1e6) and (diag["omega_saturation_fraction"] < 0.5)
    coherent_current = (diag.get("phase_coherence_nodes", 0) or 0) > 0.3
    if follows_rd and bounded and coherent_current:
        return "geometry_follows_RD_bounded"
    if not bounded:
        return "runaway_or_saturated_geometry"
    if not follows_rd:
        return "geometry_decoupled_from_RD"
    return "ambiguous"
