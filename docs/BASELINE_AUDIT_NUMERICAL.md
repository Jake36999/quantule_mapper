# Baseline Audit — Numerical (Stage 1.2)

**Central question:** *could any numerical artefact masquerade as physics?* **Descriptive only** — no redesign.
Findings tagged `CONFIRMED` (sound as-is) · `KNOWN-ISSUE` (documented limitation, not a correctness bug) ·
`ARTEFACT-CORRECTED` (an artefact that did masquerade as physics and was caught/fixed) · `RESIDUAL` (honest open
numerical question, not known to invalidate any closed claim). Reference: `jax_scout/physics.py`,
`IRER_MATH_REFERENCE.md §8–12`, `PHASE_C_METHOD_PARITY_AUDIT.md`.

## 1. Integrator — `CONFIRMED` sound
ETDRK4 (exponential time-differencing RK4) with **Kassam–Trefethen 64-point Cauchy-contour** coefficients
(`physics.py:291-317`): `E=exp(L_k·dt)`, `E2=exp(L_k·dt/2)`, `Q,f1,f2,f3` via contour integral (real part
taken). This is the standard *stable* method for stiff semilinear PDEs — it treats the stiff linear operator
`L_k` **exactly** (exponential) and avoids the Cox–Matthews cancellation errors of the naive formula. Appropriate
for the diffusive `−D·k²` stiffness. 4th-order in time. (`IRER_MATH_REFERENCE.md:150,339`.)

## 2. Spatial method + dealiasing — `CONFIRMED`
Spectral (FFT) derivatives on a periodic 3-torus (N³, L=10). Dealias mask `|k| ≤ dealias_frac·k_max` applied
**after every transform and every ETDRK4 stage** (`physics.py:265,320`). Default `dealias_frac=0.5` (half
Nyquist) — *more* conservative than the standard 2/3 rule (`IRER_MATH_REFERENCE.md:284`). **Correctly applied;
side effect: reduced effective resolution** (a conservatism, not an error). `KNOWN-ISSUE (minor)`.

## 3. Precision & timestep — `CONFIRMED` core, `RESIDUAL` on dt-convergence
- FP64 geometry path; `complex128` field (`physics.py` dtype args). CONFIRMED.
- dt=0.005; the stiff linear part is integrated exactly, so the timestep constraint is set by the **nonlinear**
  terms, not linear CFL. No documented instability at feb params. CONFIRMED for the tested regime.
- **RESIDUAL:** no explicit **dt-halving convergence study** at long T is on record. dt=0.005 is reasonable for a
  4th-order exponential integrator with the linear stiffness handled exactly, but the temporal-error bound at
  T≥72000 (~15M steps) is argued, not measured (see §7).

## 4. Boundary conditions — `CONFIRMED`
Periodic (FFT torus). Mobility probes used periodic-safe metrics (circular COM, `2πn/L`-quantised kicks) so the
box seam was handled correctly. `KNOWN-ISSUE (minor)`: finite box L=10 → possible finite-domain effects on
extended multi-node structures; not quantified, but the N128 resolution check (same L) reproduced the basin, so
box artefacts are not driving the stability results.

## 5. Guardrails / termination — `CONFIRMED` present, `KNOWN-ISSUE` permissive
Termination codes (`IRER_MATH_REFERENCE.md:300-306`): `math_explosion` (NaN/Inf or |ψ|>`collapse_threshold`),
`physics_drift` (⟨ρ⟩<1e-5), `geometry_sanity` (⟨Ω²⟩ outside (0.1,1e6)). Guardrails exist and fire. **KNOWN-ISSUE:**
`collapse_threshold=1e10` (burn-in) is very permissive — a runaway can grow far before it trips (a tighter ~1e6
would catch it earlier). Did not affect Phase C conclusions (the v3/er classifier caught growers via `er_max`
and `TRANSIENT_GROWER_REJECT` well below 1e10).

## 6. Conservation diagnostics — `KNOWN-ISSUE` (intrinsic, not a bug)
**No conserved quantity is available as a numerical invariant.** The substrate is dissipative (gain/loss +
diffusion), so mass and energy are *not* conserved — there is no Hamiltonian/energy-conservation check to
validate the integrator the way one would for a conservative system. The primary trajectory diagnostic is the
**energy-ratio `er(t)`** (a physical observable, not a conservation invariant). *Implication:* integrator
correctness rests on the method's known properties (§1) + resolution/window checks, **not** on a conservation
law. (A future conservative/Phase-D sector would, by contrast, gain a conservation check — noted, not actioned.)

## 7. The artefact that DID masquerade as physics — `ARTEFACT-CORRECTED` (canonical case)
**Validation-window artefact.** Short integration windows systematically **over-reported bound states**:
- T=6000 "TRUE_SATURATED" verdicts for K6 configs were transients (`PHASE_C_N96_OVERNIGHT_20260625`);
- revealed at T=24000 (K6 configs decay/blow up);
- and even the T=24000 "TRUE" for feb-center was itself a window artefact — it **slowly decays** by T=72000
  (`FEB_BREATHING_LONGT_T72000`).

This is the single clearest "looked like physics, wasn't" case, and it was **caught and corrected**: the
promotion criterion moved to a **late-window energy slope → 0** requirement (gate v3 + the a\* gain-ladder
criterion), validated at 2× windows (T=144000) and across seeds/resolution. **Codified lesson (standing rule for
the validation audit 1.3):** short-window "saturation" is not stability; a bound-state claim requires a
long-window slope→0 (or breathing-bounded) verdict. See `PHASE_C_GAIN_LADDER_RESULTS.md`,
`PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md`.

## 8. Long-time FP64 accumulation — `RESIDUAL` (honest open question)
T=72000/144000 ≈ 15–30M FP64 steps. Trustworthiness is **argued, not directly quantified**:
- feb was stable and steady to T=24000 as a numerical control (the earlier "accumulation caveat" was retired on
  that basis);
- the T=72000 decay is **gain-dependent** (a×1.05 decays slower than a×1.0) → physical, not a uniform numerical
  drift;
- N=128 reproduced the basin → not a resolution artefact.
These are strong *indirect* arguments. What is **not** on record: a direct dt-convergence or higher-precision
control run at the longest T. Status: no reason to doubt the closed claims, but this is the honest residual
numerical uncertainty at the extreme integration lengths.

## 9. Known diagnostic / config issues (not solver-correctness bugs) — `KNOWN-ISSUE`
- `collapse_dynamics.compute_nonlinear_balance` is a **2-term** ratio for a **3-term** (a/s/f) potential →
  understates parameter sensitivity; a diagnostic heuristic only (`IRER_MATH_REFERENCE.md:264-274`).
- `param_rho_vac=0` degeneracy (Ω²→floor) and a **default mismatch** — solver defaults `param_rho_vac=0.0`,
  `unified_omega.py` defaults `1.0`; a config omitting it makes the two modules disagree
  (`IRER_MATH_REFERENCE.md:255-262`). feb sets `param_rho_vac=1.1866`, so Phase C was unaffected.
- Non-local "Field of Affect" is *computed but not dynamically coupled* (`IRER_MATH_REFERENCE.md:342`) —
  consistent with the §1.1 finding that the jax_scout evolution is purely local.

## 10. Method parity — `CONFIRMED` (classifier/IC), `RESIDUAL` (solver-level jax↔CuPy)
`PHASE_C_METHOD_PARITY_AUDIT.md`: `CLASSIFIER_PARITY_CONFIRMED` and IC-family parity confirmed; the important
**non-parity** is IC normalization (total injected mass scales with K). **RESIDUAL:** the "jax_scout FP64 mirror
is equivalence-proven to CuPy" claim (memory) is **not backed by a located solver-level bit-comparison artifact**
— it should be produced or downgraded to "classifier+IC parity confirmed; full solver parity asserted but
un-artifacted." Connects to the open method-parity question from `BASELINE_AUDIT_PHYSICS.md §1` (does CuPy carry
a true non-local/imaginary term the mirror omits?).

## Answer to the central question
The one numerical/methodological artefact that **did** masquerade as physics — short-window "saturation" — was
found and corrected, and is now a standing validation rule. Remaining numerical exposure is bounded and honest:
**(a)** un-artifacted solver-level jax↔CuPy parity, **(b)** unquantified long-T FP64 accumulation (argued
physical), **(c)** no dt-convergence study at the longest T, **(d)** a permissive blow-up threshold and a couple
of diagnostic/config heuristics. None is currently known to invalidate a closed Phase C claim; (a)–(c) are the
items a hardening pass (Stage 2) would close. Numerics of the integrator itself are sound.
