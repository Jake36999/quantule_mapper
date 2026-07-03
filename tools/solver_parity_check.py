#!/usr/bin/env python3
"""Solver parity check (H4) — jax_scout FP64 mirror vs CuPy production, NON-INVASIVE.

Feeds the SAME deterministic initial field + params to both solvers and compares the output after N steps.
Isolates *solver* differences from IC-RNG differences by using one shared IC array (built once, saved). This
script only CALLS the existing public solver paths read-only — it changes no solver behaviour.

Neither backend runs on the Windows dev box (no cupy/jax). Workflow across the three boxes:
  1. anywhere (numpy):   python tools/solver_parity_check.py make-ic  --N 48 --out parity/shared_ic.npz
  2. WSL (jax_scout):    python tools/solver_parity_check.py run --backend jax  --ic parity/shared_ic.npz --steps 200 --out parity/jax_ref.npz
  3. CuPy production box: python tools/solver_parity_check.py run --backend cupy --ic parity/shared_ic.npz --steps 200 --out parity/cupy_ref.npz
  4. anywhere (numpy):   python tools/solver_parity_check.py compare parity/jax_ref.npz parity/cupy_ref.npz

Do NOT claim bit-parity until step 3 (the CuPy-box run) has actually produced cupy_ref.npz and step 4 passes.
See docs/SOLVER_PARITY_ARTIFACT.md.
"""
import os, sys, argparse, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Frozen feb params (jax_scout core_saturation_search.FEB) — the validated baseline point.
FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}
L_DEFAULT, DT_DEFAULT = 10.0, 0.005


def make_ic(N, seed, out):
    """Deterministic shared IC (numpy; backend-independent): centred Gaussian + fixed seeded complex noise."""
    x = np.linspace(-L_DEFAULT / 2, L_DEFAULT / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    psi = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2.0).astype(np.complex128)
    rng = np.random.default_rng(seed)
    psi = psi + 0.01 * (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, psi0=psi.astype(np.complex128), N=N, L=L_DEFAULT, dt=DT_DEFAULT, seed=seed)
    print(f"make-ic: N={N} seed={seed} -> {out}  (mass={float(np.sum(np.abs(psi)**2)):.4f})")


def run_jax(psi0, N, L, dt, params, steps):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_scout import physics
    ops = physics.build_operators(N, L, dt, params)
    psi_k = physics.initial_psi_k(jnp.asarray(psi0), ops)
    for _ in range(steps):
        psi_k = physics.step(psi_k, ops)     # baseline: no a_vec/q_tensor/drag (scalar local geometry)
    return np.asarray(jnp.fft.ifftn(psi_k))


def run_cupy(psi0, N, L, dt, params, steps):
    import cupy as cp
    from solver.core import ETDRK4Solver
    solver = ETDRK4Solver(N, L, dt, params)
    psi_k = solver.fft_single(cp.asarray(psi0)) * solver.dealias_mask
    for _ in range(steps):                    # mirrors solver/run.py:run_simulation (baseline gamma_A=0)
        rho_real = cp.maximum(cp.abs(solver.ifft_single(psi_k)) ** 2, cp.float64(1e-7))
        rho_k = solver.fft_single(rho_real) * solver.dealias_mask
        try:
            solver.update_field_of_affect(rho_k, dt)   # computed but uncoupled at gamma_A=0 (see audit)
        except Exception:
            pass
        psi_k = solver.step(psi_k)
    return cp.asnumpy(cp.fft.ifftn(psi_k))


def cmd_run(args):
    z = np.load(args.ic); psi0 = z["psi0"]; N = int(z["N"]); L = float(z["L"]); dt = float(z["dt"])
    params = json.loads(args.params) if args.params else FEB
    fn = {"jax": run_jax, "cupy": run_cupy}[args.backend]
    psi_fin = fn(psi0, N, L, dt, params, args.steps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, psi_fin=psi_fin.astype(np.complex128), backend=args.backend, steps=args.steps,
                        N=N, L=L, dt=dt, params=json.dumps(params))
    print(f"run[{args.backend}]: {args.steps} steps, N={N} -> {args.out}  "
          f"(mass={float(np.sum(np.abs(psi_fin)**2)):.6f}, max|psi|={float(np.max(np.abs(psi_fin))):.6f})")


def cmd_compare(args):
    a = np.load(args.a); b = np.load(args.b)
    pa, pb = a["psi_fin"], b["psi_fin"]
    if pa.shape != pb.shape:
        print(f"SHAPE MISMATCH {pa.shape} vs {pb.shape}"); return
    d = np.abs(pa - pb)
    max_abs = float(d.max()); l2_rel = float(np.linalg.norm(d) / (np.linalg.norm(pb) + 1e-30))
    print(f"compare {os.path.basename(args.a)} vs {os.path.basename(args.b)}:")
    print(f"  backends: {a['backend']} vs {b['backend']}  steps: {a['steps']}/{b['steps']}  N={a['N']}")
    print(f"  max|Δ| = {max_abs:.3e}   rel-L2 = {l2_rel:.3e}")
    tol = args.tol
    verdict = "BIT_PARITY" if max_abs == 0.0 else ("PARITY_WITHIN_TOL" if l2_rel < tol else "PARITY_FAIL")
    print(f"  tol(rel-L2)={tol:.1e}  ->  {verdict}")
    print("  NOTE: only meaningful once BOTH artifacts are from real backend runs (jax on WSL, cupy on the CuPy box).")


def main():
    ap = argparse.ArgumentParser(description="jax_scout <-> CuPy solver parity (non-invasive)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-ic"); p.add_argument("--N", type=int, default=48); p.add_argument("--seed", type=int, default=12345); p.add_argument("--out", default="parity/shared_ic.npz")
    p = sub.add_parser("run"); p.add_argument("--backend", choices=["jax", "cupy"], required=True); p.add_argument("--ic", required=True); p.add_argument("--steps", type=int, default=200); p.add_argument("--params", default=None); p.add_argument("--out", required=True)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    if args.cmd == "make-ic": make_ic(args.N, args.seed, args.out)
    elif args.cmd == "run": cmd_run(args)
    elif args.cmd == "compare": cmd_compare(args)


if __name__ == "__main__":
    main()
