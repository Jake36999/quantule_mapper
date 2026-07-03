# Solver Parity Artifact (H4)

**Goal:** close the bit-level residual from the architecture audit — confirm the jax_scout FP64 mirror and the
CuPy production solver produce the same output, not just the same source. **Non-invasive:** the check only calls
the existing public solver paths read-only; **no solver behaviour is changed.**

## What is already established (code-level)
`BASELINE_AUDIT_ARCHITECTURE.md` confirmed **RHS code-parity**: both solvers use the same local
`calculate_nonlinear_rhs(psi, rho, lap_cov, lap_flat, D, a, s, f)` with the same `splash→s/f` aliasing, and the
non-local Field of Affect is *computed but uncoupled* in both baselines (`solver/run.py:78` calls
`update_field_of_affect`, but `step` uses it only when `gamma_A ≠ 0`). What is **not yet** established is a
**bit-level output comparison** from actual runs.

## The artifact
`tools/solver_parity_check.py` — feeds ONE deterministic shared IC + the frozen feb params to both backends and
compares the field after N steps (isolates *solver* differences from IC-RNG differences). Modes: `make-ic` (pure
numpy), `run --backend {jax,cupy}`, `compare`.

## Exact procedure (three boxes)
```bash
# 1. anywhere (numpy only — runs on the Windows dev box):
python tools/solver_parity_check.py make-ic --N 48 --seed 12345 --out parity/shared_ic.npz

# 2. WSL Ubuntu (jax_scout venv):
wsl.exe -d Ubuntu -- bash -c "cd /mnt/f/quantule_mapper && source ~/jax_irer/bin/activate && \
  python tools/solver_parity_check.py run --backend jax --ic parity/shared_ic.npz --steps 200 --out parity/jax_ref.npz"

# 3. CuPy PRODUCTION box (has cupy + solver/):
python tools/solver_parity_check.py run --backend cupy --ic parity/shared_ic.npz --steps 200 --out parity/cupy_ref.npz

# 4. anywhere (numpy only):
python tools/solver_parity_check.py compare parity/jax_ref.npz parity/cupy_ref.npz
```

## Acceptance
`compare` reports `max|Δ|` and `rel-L2`, and a verdict:
- `BIT_PARITY` — max|Δ| == 0 (identical);
- `PARITY_WITHIN_TOL` — rel-L2 < tol (default 1e-6) — expected given FP64 + different FFT libraries / reduction
  orders across CuPy and JAX/XLA (small non-bit differences are normal and acceptable);
- `PARITY_FAIL` — rel-L2 ≥ tol → investigate (a real operator divergence).

## Status (2026-07-03)
- Step 1 (`make-ic`) — **done** on the dev box → `parity/shared_ic.npz` (N=48, seed 12345).
- Step 2 (`run --backend jax`) — **done** on WSL → `parity/jax_ref.npz` (200 steps, mass 672.79, max|psi| 0.965).
- Step 3 (`run --backend cupy`) — **PENDING** (needs the CuPy production box).
- Step 4 (`compare`) — **PENDING** step 3.

**Do not claim bit-parity until `cupy_ref.npz` exists and `compare` passes.** Established claim remains: *RHS
code-parity confirmed (both solvers share `calculate_nonlinear_rhs`, local, splash→s/f); the jax reference output
exists; numeric output parity pending the production-box run.* (`parity/` is gitignored — artifacts stay local.)

## Guardrails honoured
No solver behaviour change; read-only calls; the script constructs a throwaway IC and reads existing solver APIs
only. Parity outputs (`parity/*.npz`) are working artifacts — keep them out of git (add `parity/` to `.gitignore`
if desired; do not commit generated `.npz`).
