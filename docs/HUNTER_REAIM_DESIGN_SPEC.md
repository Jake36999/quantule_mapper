# Hunter Re-Aim Design Spec (H7 — DESIGN ONLY, NOT IMPLEMENTED)

**Status:** `HUNTER_REAIM_NOT_IMPLEMENTED` · `HUNTER_REAIM_REVALIDATION_REQUIRED`.
**No code, no config, no runs.** This designs a replacement `aste_hunter` objective aligned with the *validated*
Phase C result. Nothing here changes the solver, geometry, gate, or the hunter itself.

## 0. Governing principle — the Hunter proposes, the Gate certifies
The re-aim changes only *what the search optimises toward* (what it **proposes**). The promotion **gate** stays
`core_saturation_search.classify` (v3) + the a\*-arc late-slope (`PHASE_C_GATE_CALIBRATION_SUMMARY.md`). The
re-aimed hunter should propose candidates *likely to pass that gate*, not replace it. Keep search-heuristic and
certification strictly separate.

## 1. Current objective — the problem
`aste_hunter` minimises `log_prime_sse` (fitness ≈ `1/log_prime_sse + falsifiability + …`) with NSGA-II fronts
over `[primary_harmonic_error, missing_peak_penalty, pcs, grad_phase_var, ic]`, and its directed search (SGN
gradient, ASMT GP-manifold) steers `param_a_coupling` / `param_splash_coupling`.
- **Misaligned:** `log_prime_sse` is `NON-DISCRIMINATING` for stability — 0/60 null, flat across TRUE/FAIL
  (`BASELINE_AUDIT_VALIDATION.md`). It optimises a signature the local dissipative substrate cannot generate.
- **Wrong axes:** stability is controlled by **`param_a`** (cubic gain; a\* ≈ ×1.15, ±~0.5%) balanced against
  `param_eta` / `param_rho_vac` — none of which the directed search steers.
- **Keep unchanged (reusable):** the NSGA-II machinery, SBD (DBSCAN basin detection), ASMT (GP manifold), the SGN
  gradient framework, the SQLite ledger, and the *prefilter structure* — all re-targetable onto stability
  features rather than spectral peaks.

## 2. Proposed new objective components (all from the validated gate / a\*-arc)
Minimise / maximise, as NSGA-II objectives + a scalar fitness, computed on a **long-window** run:
- **long-window boundedness** — `er` stays in-band `[≈0.5, 2.5]` to the validation T (no decay/blow-up);
- **late-slope → 0** — `|late_slope_50pct_per1k|` small (the a\*-stationarity criterion) = primary target;
- **v3 bounded-breathing status** — accept bounded breathers (floor_ratio ≥ 0.85, er_fin ≤ 0.95·er_max);
- **energy-band safety** — `er_max ≤ blowup` (reject growers early);
- **seed robustness** — stability holds across ≥2 seeds (existence robust, not an IC fluke);
- **indivisibility** — §3 (the operationalized refinement);
- **explicitly NOT** node-count-alone or morphology-alone (matched controls showed both are non-discriminating).

## 3. "Indivisibility" — operational definition (measurable, not philosophical)
> **Indivisibility = under a controlled *division* perturbation, the attractor either restores a single coherent
> state or fails as a whole — it does NOT settle into stable, evenly-divided daughter structures.**

This is a **dynamical perturbation-response** test (uses the existing solver + `transfer_diag.detect_nodes` +
`classify`; **no operator change**). It is the direct, testable form of the project's prime→indivisibility
refinement: *a structure with no clean/even partition resists division; an uneven split is energetically
unfavourable and heals or collapses.* It is consistent with what Phase C already observed — the drag test's
response was **nucleation of a NEW blob or accretion, never clean division of the existing structure**.

Candidate metrics (a settled candidate is perturbed, re-evolved, and scored):
- **split-recovery** — bisect/deplete the density along a plane, or seed a dividing node; does it **re-merge**
  to one coherent attractor (indivisible) or persist as ≥2 stable pieces (divisible)?
- **daughter-instability** — force an *even* N-way division; are the daughters individually stable (→ divisible)
  or do they collapse/re-merge (→ indivisible)?
- **deletion/depletion recovery** — remove a fraction of mass; does it heal back to the coherent attractor?
- **whole-or-nothing failure** — when it fails, does it fail *as a unit* (mass → floor) rather than fissioning
  into stable independent survivors?
An **indivisible** candidate scores high on re-merge/heal/whole-failure and low on stable-daughter persistence.

**Guardrail against the prime trap:** indivisibility must be scored *only* by these perturbation-response
measurements. If it ever reduces to "prime-like essence" or a spectral signature, it is wrong — that recreates
the `log_prime_sse` null. It must remain a **dynamical, falsifiable** response property.

## 4. Evaluation cost & staged architecture (design consideration)
Prime-SSE was cheap (one FFT on the final state). The new objective needs **long-window evolution + perturbation
sub-runs per candidate** — far more expensive. To keep the search tractable, use **staged evaluation** (mirroring
the existing `fast_harmonic_prefilter`, but for stability):
1. **Cheap prefilter** — short-time boundedness + early late-slope sign + `param_a` proximity to the known
   balance band; reject obvious decayers/growers before any long run.
2. **Surrogate guidance** — train a stability surrogate on the **accumulated `sweep_runs` data** (er traces + v3
   labels — the evidence already inventoried) to predict likely-stable `(param_a, eta, rho_vac)` regions and
   steer SGN/ASMT there (reuse the ASMT GP framework, re-featured onto stability).
3. **Expensive confirmation** — long-window + indivisibility perturbation battery only for prefilter/surrogate
   survivors. The `css.classify` v3 gate certifies the survivors.
Redirect SGN/ASMT to steer **`param_a` / `param_eta` / `param_rho_vac`** (not `a_coupling`/`splash`).

## 5. What NOT to include
No `log_prime_sse` as fitness (retired to exploratory diagnostic only); no topological-proof objective; no
matter/mobility objective; **no new PDE terms; no geometry changes; no gate changes.**

## 6. Re-validation plan (the acid test — a re-aimed hunter is judged by re-discovery, not novelty)
Before adoption, the re-aimed hunter must:
- **re-find the known `a* ≈ ×1.15` basin** (if it can't rediscover the validated result, it is wrong);
- recover the known **matched controls** (promote the stable side, reject the failed side);
- **not promote T=12000 window artifacts** (respect the long-window slope→0 rule);
- **preserve the Phase C gate** as the certifier (hunter guides; gate promotes);
- yield candidates that **pass long-window checks** (T≥24000, not short-window "saturation").
Success is measured by re-discovery + control recovery, **not** by producing impressive-looking new candidates.

## 7. Implementation boundary
Design-only. **No hunter rewrite, no runs, no config changes.** Implementation (if greenlit) is a separate,
explicitly-approved step, and must ship with its own re-validation harness (§6) before any hunt is trusted.

---
Labels: `HUNTER_PRIME_SSE_OBJECTIVE_RETIRED` (proposed), `HUNTER_GAIN_LOSS_REAIM_DESIGNED`,
`INDIVISIBILITY_OBJECTIVE_DEFINED`, `HUNTER_REAIM_NOT_IMPLEMENTED`, `HUNTER_REAIM_REVALIDATION_REQUIRED`.
