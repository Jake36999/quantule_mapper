"""
Generate ONE shared initial condition + config for the FP64 equivalence proof.

Both engines (CuPy backbone, JAX port) load the *identical* psi0.npy, so RNG
differences between cupy/numpy/jax are removed as a variable. The IC mirrors the
structure of solver/run.py:initialize_psi (Gaussian packet + 0.01 * complex noise).
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

# --- shared experiment config (small, stable; exercises all physics terms) ---
CFG = {
    "N": 32,
    "L": 10.0,
    "dt": 0.005,
    "n_steps": 40,
    "collect_every": 4,
    "ic_seed": 12345,
    "params": {
        "param_D": 1.0,
        "param_eta": 0.2,            # linear damping -> bounded trajectory
        "param_rho_vac": 1.33,
        "param_omega0": 1.0,
        "param_a": 0.1,             # cubic
        "param_s": 0.05,            # quintic
        "param_f": 0.02,            # septic
        "param_a_coupling": 1.0,    # geometry conformal exponent
        "param_c_affect": 1.0,
        "param_dealias_fraction": 0.5,
    },
}


def make_psi0(N, L, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R2 = X ** 2 + Y ** 2 + Z ** 2
    psi = np.exp(-R2 / 2.0).astype(np.complex128)
    noise = (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape)).astype(np.complex128)
    return psi + 0.01 * noise


if __name__ == "__main__":
    psi0 = make_psi0(CFG["N"], CFG["L"], CFG["ic_seed"])
    np.save(os.path.join(ART, "psi0.npy"), psi0)
    with open(os.path.join(ART, "cfg.json"), "w") as fh:
        json.dump(CFG, fh, indent=2)
    print(f"wrote {ART}/psi0.npy  shape={psi0.shape} dtype={psi0.dtype}")
    print(f"wrote {ART}/cfg.json")
    print(json.dumps(CFG, indent=2))
