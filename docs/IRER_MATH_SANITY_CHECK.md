# IRER Math Sanity Check

**Date:** 2026-06-18  
**Reviewer:** Code audit against _Declaration of Intellectual Provenance v9.txt_ and docs/IRER_MATH_REFERENCE.md  
**Status:** Pre-compute gate — do not commit large GPU time until items marked UNCLEAR or REJECTED are resolved.

---

## 1. Reconstructed PDE from Code

Tracing the actual computation in `solver/core.py`, `solver/kernels.py`, and `gravity/unified_omega.py`:

### 1.1 Linear operator (spectral space, exponentiated by ETDRK4)

```
L̂(k) = -D·k² - η + i·ρ_vac
```

Source: `solver/core.py:90`
```python
self.L_k = (-self.D_diff * self.k_sq + (-self.eta + 1j * self.rho_vac)).astype(cp.complex128)
```

The ETDRK4 exponentiates this: `E = exp(L̂·dt)`, `E₂ = exp(L̂·dt/2)`.

### 1.2 Conformal geometry

```
Ω²(x) = soft_clip[ (ρ_vac / max(ρ(x), ε))^α ]   ∈ [10⁻⁹, 10⁶]
```

where `α = param_a_coupling`, `ρ_vac = param_rho_vac`.

Source: `gravity/unified_omega.py:152`  
Note: `param_skip_topology_cap=True` in the simulation path — the μ+3σ density cap is disabled, geometry is purely local.

### 1.3 Covariant Laplacian (real space, N_op)

```
Δ_g[ψ] = (Δ_flat[ψ] + (D-2) · (∇Ω/Ω) · ∇ψ) / Ω²
```

where `∇Ω = (∂Ω/∂ρ)·∇ρ` and `∇ρ = 2·Re(ψ*·∇ψ)`.

Source: `solver/kernels.py:13-27`
```python
gx = 2.0 * cp.real(psi_conj * dx)             # = ∂ρ/∂x
g_om_x = d_omega_d_rho * gx                   # = ∂Ω/∂x
grad_omega_dot_grad_psi = g_om_x*dx + ...     # = ∇Ω · ∇ψ
cov_term = (D_spatial - 2.0) * grad_omega_dot_grad_psi / omega
return (lap_flat + cov_term) / omega_sq        # Δ_g[ψ]
```

### 1.4 Nonlinear operator N(ψ) (real space)

```
N(ψ) = D·(Δ_g[ψ] - Δ_flat[ψ]) + (a·ρ + s·ρ² + f·ρ³)·ψ
```

Source: `solver/kernels.py:30-33`
```python
nonlin = a * psi * rho + s * psi * (rho**2) + f * psi * (rho**3)
return D_diff * (lap_cov - lap_flat) + nonlin
```

The geometry correction `D·(Δ_g - Δ_flat)` is in N because the flat `-D·k²` is already exponentiated in L̂. The split is exact: L̂+N together give the full SNCGL.

### 1.5 Full reconstructed PDE (real space)

```
∂ψ/∂t = D·Δ_g[ψ] + (-η + i·ρ_vac)·ψ + (a·ρ + s·ρ² + f·ρ³)·ψ
```

where:
- `ρ = |ψ|²` (derived, not fundamental)
- `Ω² = (ρ_vac/ρ)^α` (derived from ρ)
- `Δ_g[ψ]` = covariant Laplacian on conformal metric `g_{ij} = Ω²δ_{ij}`

### 1.6 Auxiliary field A (currently passive)

```
∂²A/∂t² = -c²·∇²A + ρ      (integrated with symplectic Euler, spectral)
```

Source: `solver/core.py:306-319`. A is evolved but its output is not fed back into N_op or L̂.

---

## 2. Theory vs Implementation Comparison

### 2.1 Verdict table

| Claim | Theory (Declaration v9) | Implementation | Verdict |
|---|---|---|---|
| ψ is the primary complex field | Yes — Ontological Informational Wave | `cp.complex128` field, ETDRK4 integrates ψ_k | **SUPPORTED** |
| ρ = \|ψ\|² is derived, not fundamental | Yes — Resonance Density emerges from OIW | `fused_compute_rho(psi)` each N_op call; not stored as state | **SUPPORTED** |
| Ω² derived from ρ (not evolved independently) | Yes — geometry is emergent | `derive_stable_conformal_factor_with_gradient(self.rho, ...)` called inside N_op, not stored as a BSSN-like state | **SUPPORTED** |
| Covariant Laplacian with (D-2) cross-term | Yes — conformal curved space | Kernel implements exact formula; D=3 gives factor 1 as expected | **SUPPORTED** |
| Vacuum oscillator frequency ω₀ in L̂ | Yes — background AIS oscillation | `i·ρ_vac` in L_k. **But ρ_vac also controls geometry** — dual role means geometry and oscillation cannot be tuned independently | **SUPPORTED (with critical caveat §3.1)** |
| A as finite-speed causal propagation | Yes — Field of Affect, sequential time | Spectral wave equation ∂²A/∂t² = -c²k²A + ρ is implemented | **SUPPORTED (field exists)** |
| A feeds back into geometry | Yes — "the geometric feedback loop meant to simulate sequential time" (user) | NOT IMPLEMENTED. A is evolved but γ_A=0 and N_op does not use A | **NOT TESTED** |
| Prime-harmonic resonance as fitness metric | Yes — Quantule lattice, k-space peaks at ln(primes) | `quantulemapper_real.prime_log_sse` computes SSE to LOG_PRIME_TARGETS = ln([2,3,5,7,11,13,17]) | **SUPPORTED** |
| Collapse dynamics / Payan States | Yes — informational collapse, threshold events | `collapse_event_count` in metrics; sentinel codes 1002/1003 | **PROMISING (partial)** |
| Angular deficit / topology | Yes — topological features of Quantule lattice | TDA profiler in validation pipeline | **PROMISING (partial)** |
| Hunter optimises documented metrics | Yes | Hunter uses `log_prime_sse` from ledger, does not alter solver params mid-run | **SUPPORTED** |
| Amplitude clamping absent | Required (no ad hoc state mutation) | `collapse_threshold` terminates rather than clamps. No in-loop amplitude forcing. | **SUPPORTED** |
| Validation metrics do not feed back into solver | Required | ValidationPipeline runs post-hoc; no path into ETDRK4 loop | **SUPPORTED** |

---

## 3. Issues Requiring Resolution Before Large Compute

### 3.1 CRITICAL: param_rho_vac = 0 is degenerate (allowed in current bounds)

**Location:** `burn_in_config.json`, bounds: `"param_rho_vac": [0.0, 2.0]`  
**Effect:**  
- Geometry: Ω² = (0/ρ)^α = 0 everywhere → complete conformal collapse, only the soft conformal floor (1e-9) saves numerical stability. The field effectively evolves on a flat space with a tiny non-zero metric. This is NOT a simulation of IRER with an emergent geometry.
- Oscillator: `L̂ = -D·k² - η + i·0` → no imaginary term → the vacuum oscillator is absent.

**Recommendation:** Set lower bound of `param_rho_vac` to `≥ 0.05` so that both geometry and oscillation are active. Alternatively, split the parameter into `param_rho_vac_geo` (geometry) and `param_omega0` (oscillation frequency) to decouple the two roles.

**Verdict:** UNCLEAR — runs with ρ_vac ≈ 0 are testing a fundamentally different physical limit.

---

### 3.2 MEDIUM: Spectral phase centering every 50 steps suppresses vacuum oscillator signal

**Location:** `solver/run.py:95-97`
```python
if step % 50 == 0:
    mean_psi = psi_k[0, 0, 0] / (N_grid**3)
    mean_phase = cp.angle(mean_psi)
    psi_k *= cp.exp(-1j * mean_phase)
```

**Effect:** This removes the global phase accumulated from the `i·ρ_vac` term in L̂ every 50 steps. In gauge theory this is a valid U(1) gauge fixing. In IRER theory the vacuum oscillation is supposed to carry physical meaning (the oscillation rate ω₀ distinguishes resonant from non-resonant states).

**Note:** The centering uses `cp.angle(mean_psi)` — the phase of the DC (k=0) Fourier mode — which is not the same as `mean(cp.angle(psi_real))`. The current code uses the latter in the telemetry comment but the former in the operation. These are numerically different.

**Recommendation:** Decide whether global phase is physically meaningful in IRER. If yes, remove or loosen the centering (or change frequency). If no, document this explicitly as a gauge choice. Either way, the telemetry `mean_phase` variable is misleading.

**Verdict:** UNCLEAR

---

### 3.3 MEDIUM: A field is entirely passive — the theory is not being tested

**Location:** `solver/core.py:N_op`, `solver/run.py`  
**Effect:** The Field of Affect A is evolved correctly but does not feed back into the geometry or the PDE. The current solver is therefore testing **single-field local-density SNCGL**, not the **two-field causal-geometry SNCGL** described in the theory.

The SSE score from the current runs reflects the local-ρ variant's ability to produce prime-harmonic patterns, not the causal-A variant. These are different experiments.

**Verdict:** NOT TESTING IRER (A-coupled) — currently testing a degenerate single-field limit.

**Required action:** Implement the A-coupling (vacuum-reference modulation as decided 2026-06-18) before treating results as IRER evidence.

---

### 3.4 LOW: param_rho_vac default conflict between modules

**Location:** `gravity/unified_omega.py:131` (`default=1.0`) vs `solver/core.py:59` (`default=0.0`)  
**Effect:** If a params dict is missing `param_rho_vac`, unified_omega defaults to 1.0 but the solver defaults to 0.0. This could produce silent geometry discrepancies in validation vs simulation.

**Recommendation:** Establish a single canonical default in `orchestrator/contracts.py` and import it everywhere.

---

### 3.5 LOW: Collapse metric is a 2-term truncation of the 3-term nonlinear potential

The theory describes a cubic nonlinear potential `V(ρ) = a·ρ + s·ρ² + f·ρ³` but the collapse dynamics module (`metrics/collapse_dynamics.py`) was noted to use a 2-term approximation. If `f=0` this is exact; for `f≠0` the collapse metric may undercount events.

**Verdict:** PROMISING BUT UNPROVEN — needs inspection of collapse_dynamics.py.

---

## 4. Summary Verdict

| Component | Verdict |
|---|---|
| SNCGL PDE (linear + geometry + nonlinear) | **SUPPORTED** |
| Conformal geometry Ω²=(ρ_vac/ρ)^α | **SUPPORTED** |
| Covariant Laplacian formula | **SUPPORTED** |
| ψ primary, ρ derived | **SUPPORTED** |
| Amplitude clamping absent | **SUPPORTED** |
| Validation metrics post-hoc only | **SUPPORTED** |
| Field of Affect A (existence) | **SUPPORTED** |
| A causal feedback into geometry | **NOT TESTED** |
| Prime-harmonic fitness metric | **SUPPORTED** |
| param_rho_vac dual role | **UNCLEAR** |
| Phase centering effect | **UNCLEAR** |
| Collapse metric completeness | **PROMISING BUT UNPROVEN** |
| Overall: "Are we testing IRER?" | **PARTIALLY** — testing a degenerate single-field limit; the A-coupled two-field theory is not yet under test |

---

## 5. Falsification Tests Required Before Large Compute

The following tests would meaningfully confirm or deny IRER predictions. They are ordered by feasibility at current scale (N=128).

| # | Test | Pass criterion | Fail criterion |
|---|---|---|---|
| F-1 | **A-coupling null test**: run with γ_A=0 vs γ_A>0; compare log_prime_sse | CAUSAL-AFFECT run reaches lower SSE at same compute budget | No SSE improvement → A is not the mechanism producing prime peaks |
| F-2 | **Prime-target scramble**: replace ln(2,3,5,7,11,13,17) with random targets; check SSE | Real prime run achieves lower SSE than scrambled-target run | Scrambled targets achieve equivalent SSE → solver is not specifically selecting primes |
| F-3 | **Phase ablation**: scramble phase of ψ_final, recompute SSE | SSE worsens significantly after phase scramble | SSE unchanged → prime peaks are amplitude-only artifacts |
| F-4 | **Geometry ablation**: run with α=0 (flat space, Ω²=1) | Flat-space run achieves higher SSE | Flat space achieves same SSE → conformal geometry provides no advantage |
| F-5 | **Seed robustness**: 10 seeds at champion params, all reach SSE < 1.0 | All 10 seeds converge | High variance → result is seed-specific, not structurally determined |
| F-6 | **Grid scaling**: run N=64, 128, 256 at champion params | log_prime_sse decreases or stabilises with N | SSE diverges with N → spectral peaks are N-dependent artifacts |
| F-7 | **ρ_vac=0 sentinel**: confirm that ρ_vac=0 runs are flagged DEGENERATE | Ledger shows DEGENERATE_GEOMETRY tag | Silent pass → degenerate runs pollute leaderboard |

Tests F-2 and F-3 are already partially implemented in the validation pipeline (falsifiability module). F-1, F-4, F-5, F-6 require deliberate experimental design at the hunt level.

---

## 6. Answer to "Is the solver testing IRER as intended?"

**Short answer:** The solver correctly implements the SNCGL PDE with emergent conformal geometry and the correct fitness metric. However, it is testing **IRER without causal time** — the Field of Affect is computed but not coupled, which means the key theoretical mechanism (finite-speed causal propagation shaping geometry) is absent. The current results are evidence for a related but distinct model.

**Before large compute:** Implement A-coupling, set ρ_vac lower bound ≥ 0.05, clarify or remove the phase centering, and confirm all runs are labelled with variant identifiers so that pre-coupling and post-coupling results cannot be mixed.

---

## 7. Resolution Log — DC-v1.0 hardening (2026-06-18)

This section resolves the items flagged UNCLEAR / required-before-coupling above, with exact code references and the action taken.

### 7.1 RESOLVED — k=0 secular runaway in the A field

**Was:** §3 (skipped sentinel). The affect wave equation `∂²A/∂t² = -c²k²A + ρ` has no restoring force at k=0 (`c²k²=0`), while the source `ρ̂(k=0) = ∫ρ dV` is the (positive, ~constant) total mass M. The zero mode therefore integrated to `A_k(0,0,0) ≈ ½ M t²` — an unbounded quadratic offset in `A_real`.

**Fix (`solver/core.py:update_field_of_affect`):**
1. Project the DC source out before forcing: `rho_k_safe[0,0,0] = 0` (zero-mean causal forcing).
2. Pin the gauge each step: `A_k[0,0,0] = 0`, `A_dot_k[0,0,0] = 0`.

The constant mode of A is pure gauge (A is the response to density *inhomogeneities*), so this removes the divergence without altering any inhomogeneous mode. Canonical helper: `orchestrator.run_identity.zero_dc_mode`. **Boundedness is proven** by a GPU-independent numpy mirror in `tests/test_run_identity.py::TestK0Runaway` (DC mode stays `< 1e-9` with the gate, explodes without). **Verdict: RESOLVED — merge gate cleared.**

### 7.2 ANALYSED — param_rho_vac dual role (exact locations)

`param_rho_vac` is read in two physically independent roles from the **same** scalar:

| Role | Location | Code |
|---|---|---|
| Vacuum oscillator frequency ω₀ | `solver/core.py:90` | `self.L_k = (-self.D_diff*self.k_sq + (-self.eta + 1j*self.rho_vac))` → the `i·ρ_vac` term rotates the global phase at rate ω₀ = ρ_vac |
| Conformal reference density | `gravity/unified_omega.py:152` (and `:205` for the gradient) | `omega_sq = (rho_vac / rho_capped) ** a` → ρ_vac sets the Ω²=1 equilibrium density |

**Consequence:** you cannot tune the oscillation rate independently of the geometric equilibrium point — one knob moves two physics. At `ρ_vac=0` *both* collapse simultaneously (no oscillation **and** Ω²→0). There is also a **default conflict**: `solver/core.py:59` defaults `param_rho_vac` to `0.0`, while `gravity/unified_omega.py:131,183` default it to `1.0`, so a params dict missing the key produces different geometry in solver vs. validation.

**Recommendation (decouple before the next physics campaign):** split into two parameters —
- `param_omega0` → the `i·ω₀` term in `L_k` (oscillation),
- `param_rho_vac` → the conformal reference density in Ω² (geometry),

and set a single canonical default in `orchestrator/contracts.py` imported by both modules. Until split, set the search lower bound `param_rho_vac ≥ 0.05` so neither role degenerates.

**RESOLVED (2026-06-18):** the split is implemented.
- `param_omega0` now drives the oscillator (`solver/core.py:90` → `... + 1j * self.omega0`); `param_rho_vac` remains the geometry reference in `unified_omega.py`.
- Backward compatible: `omega0` defaults to `param_rho_vac` when unset (`solver/core.py`), so any pre-split config reproduces the historical coupled behaviour exactly — the γ_A=0 regression is unaffected.
- Default conflict fixed: canonical `DEFAULT_PARAM_RHO_VAC = DEFAULT_PARAM_OMEGA0 = 1.0` in `orchestrator/contracts.py`, imported by `solver/core.py` (was 0.0); `unified_omega.py` already used 1.0.
- `burn_in_config.json`: `param_rho_vac` lower bound raised 0.0 → 0.05 (non-degenerate geometry); `param_omega0` bounds `[0.0, 2.0]` added. Ledger `parameters` table + `result_processor` now record `param_omega0`. Tests: `test_data_contract.py::TestRhoVacOmega0Split`.

**Verdict: RESOLVED.**

### 7.3 ANALYSED — phase centering does NOT break conservation laws

**Location:** `solver/run.py:93-97`. Every 50 steps the field is multiplied by `exp(-i·θ)`, where `θ = angle(psi_k[0,0,0])` (the DC-mode phase).

**Mathematical impact — this is a global U(1) rotation `ψ → e^{-iθ}ψ`:**
- **Conservation laws are exactly preserved.** `ρ = |ψ|²` is invariant under a global phase, so `∫ρ dV` (energy), `∫ρ² dV` (C-invariant), and every ρ-derived quantity are unchanged to machine precision. *There is no conserved-quantity drift to "prevent."*
- **Every fitness/validation metric is invariant.** The SNCGL PDE is globally U(1)-covariant (the `i·ρ_vac` term, the nonlinear `a·ρ+s·ρ²+f·ρ³` terms, and the covariant Laplacian all commute with `ψ → e^{iφ}ψ`). All metrics are built from `|ψ_k|` (prime-peak SSE — phase-blind), from `ρ` (PCS, collapse), or from **phase gradients** `∇arg ψ` (a global phase adds a constant → gradient unchanged). A single uniform rotation changes none of them.
- **What it does touch:** only the unobservable absolute global phase accumulated by `i·ρ_vac`.

**So the original "suppresses the vacuum-oscillator signal" worry only bites if a metric reads the absolute global phase — none do.** Two real (cosmetic) defects remain: (a) the variable name `mean_phase` is reused for two different quantities — `angle(psi_k[0,0,0])` at line 96 vs `mean(angle(psi_real))` at line 123 — making telemetry misleading; (b) re-centering on the *noisy nonlinear DC mode* every 50 steps is a less clean gauge than the analytic rotation it is approximating.

**Mathematically sound alternative (preserves conservation laws — they were never broken):**
- *Preferred:* **remove the centering entirely** — it has zero effect on any U(1)-invariant observable. Or, if an absolute-phase diagnostic is ever wanted, subtract the **analytic** rotation `exp(-i·ρ_vac·t)` every step (a deterministic gauge) instead of reading the contaminated `angle(psi_k[0,0,0])`.
- Either way: rename the line-96 variable (e.g. `dc_mode_phase`) to end the telemetry collision, and **stamp the choice in `solver_contract` as `phase_gauge: "none" | "dc_mode_comoving" | "analytic_omega0"`** so two runs with different gauge handling are never silently compared.

**Verdict: UNCLEAR → SUPPORTED (gauge-safe).** The operation is a legitimate U(1) gauge fix; it neither violates the PDE physics nor breaks conservation. Action is documentation + naming + contract-stamping, not a physics correction.

### 7.4 Updated verdicts

| Component | Prior | Now |
|---|---|---|
| k=0 A-field secular runaway | NOT GATED | **RESOLVED** (implemented + proven) |
| Phase centering effect | UNCLEAR | **SUPPORTED** (global U(1) gauge; conservation intact) |
| param_rho_vac dual role | UNCLEAR | **RESOLVED** — split into `param_omega0` + `param_rho_vac`; default conflict fixed |
| A causal feedback into geometry | NOT TESTED | NOT TESTED (next phase; gate now cleared) |
