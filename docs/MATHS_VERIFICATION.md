# IRER Maths Verification — N96 Phase C pre-launch gate

**Date:** 2026-06-24
**Supersedes / updates:** [IRER_MATH_SANITY_CHECK.md](IRER_MATH_SANITY_CHECK.md) (2026-06-18)
**Purpose:** Re-verify the geometry/PDE math actually exercised by the Phase C N96/T6000
validation, and certify whether the physics/geometry state is *frozen* before any GPU launch.
**Repo state at verification:** `HEAD = ce312d5e1`; only `gravity/unified_omega.py` dirty
(unstaged, **+6 lines, 0 deletions — comments only**, see §1).

---

## 0. Why this document exists

The N96 launch was gated on one blocker: `gravity/unified_omega.py` showing dirty in the working
tree. The advice "do not touch omega" was a heuristic carried over from earlier exploratory threads
that *proposed* altering the conformal map (anisotropic-metric / A-coupling avenues). This document
checks the actual diff and the actual code path, rather than relying on that heuristic.

**Headline:** the dirty change is documentation only; the geometry math is byte-for-byte identical to
the last functional commit (`4356d06f8`, 2026-05-02). The Phase C validation path does not even import
this file — it runs a separate, equivalence-proven JAX mirror. The physics/geometry state is frozen.

---

## 1. The `gravity/unified_omega.py` working-tree diff — full content

Exactly two identical 3-line comment blocks, one in each conformal-factor function:

```diff
@@ derive_stable_conformal_factor (line 130) and
@@ derive_stable_conformal_factor_with_gradient (line 185):
     # Parameter extraction
+    # param_rho_vac here is the CONFORMAL REFERENCE DENSITY (geometry only); the
+    # vacuum oscillator frequency is the separate param_omega0 used in the solver's
+    # L_k.  Canonical default (1.0) matches orchestrator.contracts.DEFAULT_PARAM_RHO_VAC.
     rho_vac = _as_xp_float64(float(params.get("param_rho_vac", 1.0)), xp)
```

- **No executable line changes.** The `rho_vac = ...` line is unchanged diff *context*, not an
  insertion. `git diff --numstat` = `6  0`.
- The comment documents the `param_rho_vac` / `param_omega0` split that was **already implemented and
  RESOLVED on 2026-06-18** (IRER_MATH_SANITY_CHECK §7.2). It is annotation catching up to a resolved
  decision, not a new code change.
- Verified consistent: `orchestrator/contracts.py:27-28` defines `DEFAULT_PARAM_RHO_VAC = 1.0` and
  `DEFAULT_PARAM_OMEGA0 = 1.0`, exactly as the comment claims.

**Classification of the diff:** *intentional (deliberately written) but documentation-only — it does
not change the physics or geometry path.* Per the launch protocol this is the decision point reported
in §6; it is not an unintended edit to revert, and it is not a physics change.

---

## 2. Which geometry code actually runs at N96

This is the structural correction to the 2026-06-18 audit, which traced the **CuPy production**
solver. The Phase C saturation search does **not** run that path.

| Path | Module | Used by | Geometry source |
|---|---|---|---|
| CuPy/ASTE production (FP64 source of truth) | `solver/core.py`, `solver/kernels.py`, `gravity/unified_omega.py` | full hunts / validation pipeline | `gravity/unified_omega.py` (the dirty file) |
| **JAX scout (the Phase C / N96 path)** | `jax_scout/physics.py` → `sweep_probe` / `run_probe` | `core_saturation_search.py`, `core_saturation_replay.py` | **mirror inside `physics.py`** (does *not* import `gravity/unified_omega.py`) |
| Diagnostics only | `jax_scout/geometry_diag.py`, `transfer_diag.py` | post-hoc Ω²/curvature audits | imports the real `unified_omega` as single source of truth |

Confirmed by `grep`: no module under `jax_scout/` that the **solver** path touches imports
`gravity.unified_omega`; only the two diagnostic modules do. So the comment-only edit to
`gravity/unified_omega.py` is **not in the N96 execution path at all** — it cannot affect a Phase C
result even in principle.

---

## 3. Geometry formula — source vs mirror, line by line

The N96 path's geometry (`jax_scout/physics.py:_geometry_with_gradient`, lines 105–122) is an exact
transcription of the committed CuPy formula. Both compute, on the `param_skip_topology_cap=True`
(local-geometry) path:

```
ρ_safe   = max(ρ, ε)                      ε = 1e-12
ρ_capped = ρ_safe                          (cap bypassed; d ρ_capped/dρ = 1)
Ω²_raw   = (ρ_vac / ρ_capped)^a            a = param_a_coupling
∂Ω²/∂ρ   = a · (ρ_vac/ρ_capped)^(a-1) · (−ρ_vac/ρ_capped²)
Ω², d_soft = soft_clip_log(Ω²_raw, 1e-9, 1e6, β=3)
∂Ω²/∂ρ  ← d_soft · ∂Ω²/∂ρ                  (chain rule through the soft clip)
```

| Quantity | CuPy `gravity/unified_omega.py` | JAX mirror `jax_scout/physics.py` | Match |
|---|---|---|---|
| conformal law `Ω²=(ρ_vac/ρ)^a` | `:155`, `:211` | `:114` | ✅ identical |
| analytic `∂Ω²/∂ρ` | `:215` | `:116` | ✅ identical |
| log-space soft clip `[1e-9,1e6]` | `:158`, `:218` + `_soft_clip_log_with_derivative` | `:118-121` + `_soft_clip_log_with_derivative` `:82-102` | ✅ identical |
| `skip_topology_cap` local path | `:147-148`, `:202-204` | `:111` | ✅ identical |
| covariant Laplacian `Δ_g=(Δ_flat+(D-2)(∇Ω/Ω)·∇ψ)/Ω²` | `solver/kernels.py` | `_cov_laplacian` `:158-171` | ✅ identical |
| nonlinearity `a·ρ+s·ρ²+f·ρ³` | `solver/kernels.py` | `_nonlinear_rhs` `:174-177` | ✅ identical |

The geometry exponent knob `param_a_coupling` and the nonlinear cubic knob `param_a` are correctly
kept distinct in both back ends (equivalence README §Notes). `ρ_vac` is used **only** as the conformal
reference density in both — the oscillator role lives in the solver's `L_k` via `param_omega0`, exactly
as the new comment states.

---

## 4. Equivalence proof — the mirror reproduces the source to FP64 round-off

`jax_scout/equivalence/` proves the JAX port reproduces the CuPy backbone:

```
N=32³, 40 ETDRK4 steps (≈960 FFTs + full geometry/cov-Laplacian/cubic-quintic-septic pipeline)
worst rel_L2(ψ_k) = 5.78e-16   (threshold 1e-8)   →   PASS
```

`5.78e-16` is FP64 machine epsilon (~2.2e-16) and **does not grow** with steps. Bit-identity is not
achievable (different cuFFT-via-CuPy vs cuFFT-via-XLA reduction order) and is not the bar; non-growing
agreement at round-off is.

**Crucially, the precision regimes match.** The Phase C scout runs the same FP64 regime the proof
covers, not the FP32 scout regime:
- `core_saturation_search.py:19` — `jax.config.update("jax_enable_x64", True)`
- `core_saturation_search.py:270` — `physics.sweep_probe(..., jnp.float64, jnp.complex128)`

So the FP32 accuracy caveat in the equivalence README does **not** apply to the N96 saturation run —
it integrates in complex128. The geometry/PDE math the N96 run executes is the FP64-equivalence-proven
math.

---

## 5. Carry-forward of the 2026-06-18 resolutions (still valid)

| Item | 2026-06-18 verdict | Status now |
|---|---|---|
| k=0 A-field secular runaway | RESOLVED (DC-mode gated + proven in `tests/test_run_identity.py`) | unchanged; A is decoupled (γ_A=0) in the bare S-NCGL Phase C path anyway |
| `param_rho_vac` dual role | RESOLVED — split into `param_omega0` (oscillator) + `param_rho_vac` (geometry) | unchanged; the dirty comment merely documents this in-file |
| phase centering | SUPPORTED (global U(1) gauge; conservation intact) | unchanged; equivalence harness excludes it to isolate physics math |
| conformal law / covariant Laplacian / ψ-primary | SUPPORTED | re-verified line-by-line in §3 |
| A causal feedback into geometry | NOT TESTED (next phase) | still not coupled — and intentionally out of scope: Phase C is bare S-NCGL `γ_A=0`. This validation does not claim to test A-coupled IRER. |

No regression. Nothing in the resolved set was reopened by the dirty diff.

---

## 6. Pre-launch certification (the 6 required items)

1. **`gravity/unified_omega.py` status:** dirty, unstaged, **+6/-0 comment-only** lines documenting the
   already-resolved `param_rho_vac`/`param_omega0` split. Executable geometry math byte-identical to
   commit `4356d06f8` (2026-05-02). Not in the N96 code path (§2).
2. **Git commit / dirty state:** `HEAD = ce312d5e14ccd9685f3ee36cdf8f9940dc7fc42f`; working tree carries
   the comment-only omega change plus unrelated untracked sweep/log/pyc artifacts (none in the N96
   physics path).
3. **Exact Stage 1 commands:** [PHASE_C_OPTION_B_N96_VALIDATION_PLAN.md](PHASE_C_OPTION_B_N96_VALIDATION_PLAN.md) §3 (8 candidates).
4. **Output root:** `sweep_runs/PHASE_C_OPTION_B_N96_STAGE1_<timestamp>/` (fresh).
5. **Cross-resolution overrides:** every CSV-sourced replay carries an explicit
   `--target-initial-mass-override = raw_N48 × 8`; the replay refuses cross-resolution runs without it
   and auto-stamps `mass_scaling_mode=resolution_scaled_raw_target`. `feb56dc7` uses `--ref` with no
   override (per-blob norm). Verified in `core_saturation_replay.py:174-192`.
6. **No generated outputs committed:** sweep bundles / frames / logs are inspected in place and rendered
   on the Windows side; only docs/source are ever committed.

**Geometry/physics verdict:** `PHYSICS_STATE_FROZEN`. The N96 validation would run against a known,
equivalence-proven geometry/PDE, unchanged from the committed source. The dirty file is comment-only and
out of the execution path.

**Decision point (yours, per protocol "do not run N96 until we decide"):** the diff is *intentional but
non-physics*. Recommended resolution: commit the comment-only `unified_omega.py` change so the tree's
physics files are clean and the frozen state is recorded at an explicit commit, then launch Stage 1.
Acceptable alternative: leave the comment uncommitted and launch (the geometry math is identical either
way). Either way no revert is warranted — the comment is correct and matches the resolved split.

---

## 7. What this document does NOT certify

- It does not claim the A-coupled two-field IRER is under test — Phase C is bare S-NCGL (`γ_A=0`) by
  design.
- It does not upgrade any Option B branch to a topological transition, proof, molecule, ground state,
  black-hole analogue, or universal law. The accepted read stays: distributed (lower-risk) vs compact
  (higher-resolution-risk) branch families; resolution-survival is exactly what Stage 1 tests.
- It does not re-run the FP32 scout accuracy envelope (not needed — the N96 saturation path is FP64).
