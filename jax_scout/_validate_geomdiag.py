"""Sanity-check geometry_diag on known fields (native .venv)."""
import numpy as np
import geometry_diag as gd

N, L = 48, 10.0
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
params = {"param_rho_vac": 1.33, "param_a_coupling": 0.5}

# 1. single coherent condensate (real, smooth) -> geometry follows rho, low current, bounded curvature
blob = np.exp(-(X**2 + Y**2 + Z**2) / 0.8).astype(np.complex128)
# 2. two coherent blobs with a phase gradient (phase-locked current between them)
two = (np.exp(-((X-2)**2+Y**2+Z**2)/0.8) + np.exp(-((X+2)**2+Y**2+Z**2)/0.8) * np.exp(1j*0.5*X)).astype(np.complex128)
# 3. random broadband noise -> geometry follows rho pointwise but incoherent phase/current
rng = np.random.default_rng(0)
noise = (0.3*(rng.standard_normal((N,N,N))+1j*rng.standard_normal((N,N,N)))).astype(np.complex128)

for name, psi in [("condensate", blob), ("two_blob_phased", two), ("noise", noise)]:
    d = gd.diagnose(psi, params, dx)
    print(f"\n=== {name} ===  verdict={gd.geometry_verdict(d)}")
    for k in ["omega_node_correlation", "omega_gradient_alignment", "node_omega_contrast",
              "omega_saturation_fraction", "J_info_l2", "phase_coherence_nodes",
              "current_circulation_l2", "shear_fraction", "stress_node_contrast",
              "curvature_l2", "curvature_max", "curvature_node_correlation", "sdg_h_norm_l2", "n_nodes"]:
        print(f"   {k:28s} {d[k]}")
