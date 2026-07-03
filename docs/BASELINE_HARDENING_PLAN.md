# Baseline Hardening Plan (Stage 2)

**Objective:** make the *existing* validated baseline as defensible and reproducible as possible. **No new
physics, no new PDE terms, no new IRER assumptions.** Draws its backlog from `BASELINE_AUDIT.md`. Each item is
tagged **docs-only** (safe to execute freely) or **code-change** (modifies the system → execute only on an
explicit go-ahead; audit discipline is over, but changes still need sign-off).

Discipline: **Phase C is preserved as the validated baseline** — hardening never alters the frozen operator
(`e8d6a78ea`) or the closed claims; it improves reproducibility, tests, and hygiene around them.

## Prioritized backlog

| # | item | type | risk | status | note |
|---|---|---|---|---|---|
| H1 | **Preserve Phase C as validated baseline** — freeze marker + canonical object name | docs-only | none | this stage | supersede the stale "T=6000" naming in `RUNBOOK_PHASE_C_AND_VISUALS.md` |
| H2 | **Baseline reproduction runbook** — reproduce the validated result from code | docs-only | none | **done this turn** → `BASELINE_REPRODUCTION_RUNBOOK.md` |
| H3 | **Evidence off-box archive** — copy-script + manifest for the ~22 load-bearing runs | docs-only (+ user copy) | none | proposed | `EVIDENCE_INVENTORY.md` is the manifest; add a copy recipe; the actual off-box copy is a user action |
| H4 | **Solver-parity artifact** — bit-level jax↔CuPy output comparison | code-change (test script; **runs on the CuPy box**) | low | proposed | RHS code-parity already confirmed (architecture audit); this closes the bit-level residual — script authorable here, runnable only where CuPy exists |
| H5 | **Gate-calibration summary** — consolidate the promotion criterion + caveats | docs-only | none | mostly-existing | `PHASE_C_STABILITY_GATE_CALIBRATION.md` + `..._GATE_V3_BREATHING...` exist; add the a\*-arc late-slope criterion + the single-exemplar caveat as a one-page summary |
| H6 | **Validation-path reconciliation** — delineate (or bridge) the two paths | docs-only, optional small adapter (code) | low | proposed | document that `css.classify` is the gate and `validation_pipeline.py` is exploratory; decide whether to build the `.npz→HDF5` adapter or leave the paths explicitly separate |
| H7 | **Hunter objective re-aim** — fitness prime-SSE → gain/loss-balance stability; redirect SGN/ASMT onto `param_a`/`eta`/`rho_vac` | **code-change** | medium | design-only until go-ahead | changes search behaviour → needs its **own re-validation** (does the re-aimed hunter re-find the a\* basin?); a tool change, not a physics change. **DESIGN NOTE (per user):** the re-aim must honour the project's refinement of the prime hypothesis from *spectral prime-harmonics* to **indivisibility** — structures do **not** divide when there is no clean/even division (an uneven split is energetically unfavourable). So the objective is not "seek prime spectra" (null) but "reward configurations whose stability reflects an indivisible/non-evenly-divisible balance"; treat prime-log-SSE as a retired proxy, not the target. |
| H8 | **Dead-script archival** — move the 25 `afield/payan/bridge/corridor/transfer` scripts to `jax_scout/_legacy/` (or a manifest) | code-change (file moves) | low | proposed | falsified-hypothesis era; keeps the core 6 + recent `feb_*` clean. Lightweight repo, so moving is fine |
| H9 | **Config/diagnostic fixes** — `param_rho_vac` default mismatch (0 vs 1.0); permissive `collapse_threshold` (1e10); `collapse_dynamics` 2-term heuristic | code-change (small, targeted) | low | design-only until go-ahead | each is an isolated fix; none touches the validated operator |
| H10 | **Test gaps** — add solver-parity + mobility-script (`feb_kick_inertia`, `feb_adiabatic_drag`) coverage | code-change (tests) | low | proposed | strengthens the suite; no behaviour change |

## Suggested execution order
1. **Foundation (docs-only, execute now):** H1, H2 (done), H3, H5, H6.
2. **Parity & tests (low-risk code):** H4 (script only), H10.
3. **Hygiene (low-risk code, on go-ahead):** H8 dead-script archival, H9 config fixes.
4. **Tool change (medium, on go-ahead + re-validation):** H7 hunter re-aim.

Items H7 and H9 are marked **design-only until go-ahead** because they change behaviour/config; H4/H8/H10 are
low-risk and can proceed on the general "free to proceed" grant. All of Stage 2 completes **before** any Stage-3
`CAPABILITY_EXPANSION_RFC.md` is written — the validated dissipative baseline must be hardened and frozen first.

## Progress (2026-07-03)
**Done:** H1/H2/H3/H5/H6 (docs). **H4** — parity script + docs done; **jax reference produced**
(`parity/jax_ref.npz`, N=48/200 steps); **CuPy-box run + `compare` PENDING** (needs the CuPy machine).
**H8** — **24 legacy scripts moved** to `jax_scout/_legacy/` with cross-imports repointed; excluded the 3
dependencies (`afield_current_coupled`, `transfer_diag` = live core deps; `afield_prototype` = test dep); live
core + `afield_prototype` import verified intact on WSL. **H9b** — `collapse_threshold` 1e10→1e6 (4 configs +
`solver/run.py`); guardrail only, validated jax_scout runs unaffected. **H9a** — the flagged `param_rho_vac`
0-vs-1.0 default mismatch is **already resolved** (`orchestrator.contracts.DEFAULT_PARAM_RHO_VAC = 1.0`;
`unified_omega`/`physics` also 1.0) — no change needed; the `IRER_MATH_REFERENCE` note is stale. **H10** —
`tests/test_solver_parity_artifact.py` (5 pass, dev box) + `tests/test_mobility_metrics.py` (logic verified on
WSL). **Held:** H4 CuPy run, H7 (hunter re-aim), H8-further, H9 (any deeper config work).
**H7 DESIGN written** (`HUNTER_REAIM_DESIGN_SPEC.md`, 2026-07-03) — design-only; operationalizes indivisibility
as a dynamical division-perturbation response (re-merge/heal/whole-failure vs stable daughters), keeps
`css.classify` as the certifier, and requires re-discovery of the a\* basin at re-validation.
**H7.1 + H7.2 done OFFLINE** (`HUNTER_REAIM_OFFLINE_RESCORE.md`): `tools/stability_objective.py` scorer (prime-SSE
retired; late-slope/boundedness/breathing/window-gate; indivisibility = pending run-hook) + offline re-score over
the real load-bearing CSVs — **ranks a\*≈×1.15 top of both the confirm and gain-ladder sets**, decayers/growers
below, short-window discounted; `tests/test_stability_objective.py` 6 pass. This is parity-INDEPENDENT (scores the
already-validated jax_scout results). **STILL NOT wired into `aste_hunter`; no hunt; no solver/gate change**
(`HUNTER_REAIM_NOT_IMPLEMENTED`). **Held (gated):** H7.1 hunter-wiring (code change), indivisibility perturbation
runs (GPU), H7.3 re-validation harness, H7.4 controlled hunt — the hunt is gated on H7.3 + the H4 CuPy parity +
explicit go. **H4 CuPy run is environment-blocked from the agent session** (no cupy in any reachable env) → the
operator must run it on the CuPy production box.
**H7.1 WIRING DONE** (`HUNTER_REAIM_IMPLEMENTATION_NOTES.md`): `aste_hunter.Hunter(objective="stability")` flag +
`_stability_fitness_from_provenance` branch — fitness from `tools.stability_objective`, prime-SSE dethroned as
steering (recorded diagnostic only), **default "prime" path unchanged**; `tests/test_hunter_stability_wiring.py`
5 pass (dev box). **Honest scope limit — FITNESS ONLY:** NSGA fronts still rank on spectral objectives (degenerate
in stability mode → falls back toward fitness-ranking); a proper stability-NSGA front + the worker emitting
`stability_metrics` into provenance are follow-ups (H7.1b), not done. No hunt, no solver/gate change, no
re-discovery claim.
**H7.3 RE-DISCOVERY DONE — PASS** (`HUNTER_REAIM_REDISCOVERY_RESULTS.md`, harness
`jax_scout/hunter_reaim_rediscovery.py`, jax_scout/WSL, **no CuPy, no prime-SSE**). Searched the validated
`param_a × eta × rho_vac` axes around the basin; scored with `tools/stability_objective`; certified with
`css.classify`. **Fresh long-T flip (T=36000, eta×1.0 a-triplet): a\* (a×1.15) → rank 0, score 0.867, late-slope
≈0, certified**; slow-decayer a×1.05 (er 1.709→1.285) below; a×1.25 `TRANSIENT_GROWER_REJECT` by css *and*
objective. The cheap T=8000 filter under-resolves a\* (ranks it 5th, all *uncertifiable*) — a documented
short-window transient, corrected once the window is certifiable; **no artifact was promoted**. Corroborated by the
existing 3D joint-basin (T=12000: css-stable ranked above css-failures, growers lowest) + the T=72000 gain-ladder
(a\* top) + T=144000 confirm (a\* seed-robust). Pre-registered PASS criteria all met, no FAIL trigger tripped.
**Still NOT production-ready:** H7.1b (stability-NSGA front + worker `stability_metrics`) and H4 CuPy parity remain
open; this validates the *objective* on the jax_scout path, not a deployed Hunter.
**PRODUCTION RE-CENTERING STARTED (2026-07-03, `PRODUCTION_ALIGNMENT_PLAN.md`, Track A).** Guards against the
split-brain risk (objective proven on the reachable jax_scout mirror; production CuPy stack must catch up).
Verified at source: **both engines are FP64** (jax `x64`/`complex128`; CuPy `complex128`, `solver/core.py:97`) —
CuPy is production authority for *orchestration* reasons, not precision — and **they already share the kinetic term**
`L_k = -D·k² - η + i·ω₀` (so no CuPy kinetic change is needed for alignment). **A3 done (code-only, no GPU):**
`solver/stability_metrics.py` (pure-numpy, mirrors `core_saturation_search.classify` er-math exactly; raw `Σ|ψ|²`,
no floor/dV) + `solver/run.py` read-only observers (ic_e_raw, per-cadence raw_energy, emit `/stability_metrics` into
HDF5 + result payload; **no physics change**, py_compile-checked) + `tests/test_stability_metrics.py` 5 pass
(exact css-parity, cadence-independence, objective feed, from_history).
**A4 done (code-only, no GPU):** `stability_metrics` carry-through wired + tested — `validation_pipeline.read_json_dataset`
reads HDF5 `/stability_metrics` into the provenance report top-level key (peer of `spectral_fidelity`), which
`aste_hunter._stability_fitness_from_provenance` already consumes. `tests/test_stability_metrics_provenance.py` 6 pass
(no cupy): emit→assemble→prov_data→consume survives verbatim; absent metrics → `NO_STABILITY_METRICS` non-promotion
(never prime fallback); grower carried + hard-rejected; prime default backward-compatible. Field contract in `PRODUCTION_ALIGNMENT_PLAN.md`.
**A4b done (code-only, no GPU): provenance-filename seam RESOLVED.** New `run_identity.resolve_provenance_report`
(single shared resolver: prefers the identity-folded path, else most-recent of plain +
`provenance_{hash}_*.json`, collision-safe, `None` when absent) + `aste_hunter.process_generation_results` rewired
to use it for **both** objectives (was a hardcoded plain name). `tests/test_provenance_resolution.py` 7 pass (no
cupy): plain-legacy resolves; identity-folded discoverable; missing→None/`failed`; no cross-config collision; Hunter
**stability** reads stability_metrics from a folded file (fitness>0); Hunter **prime** reads spectral_fidelity from a
folded file (completed). **A5 PREPARED (code-only, no GPU): production re-validation harness + runbook.** `tools/production_h7_revalidation.py`
(`build-configs` emits worker_cupy `--params` for a\*×1.15 + matched controls a×1.05/1.25 + a T=12000 window-artifact
probe, feb-frozen, N=96/dt=0.005/T=36000 for jax_scout parity; `evaluate` scores the resulting provenance via the
stability objective + shared resolver and prints PASS/REVIEW) + `docs/PRODUCTION_H7_REVALIDATION_RUNBOOK.md` (box
steps + PASS criteria + the cross-IC caveat: production single-Gaussian IC ≠ jax_scout multiseed). Evaluator
unit-tested (`tests/test_production_h7_revalidation.py` 6 pass, no cupy). **Remaining Track A (needs the CuPy box): A1
H4 bit-parity run, then run the A5 harness (worker + validate steps) → evaluate.** A2 operator audit done at code
level. **Track B (Phase D kinetic RFC) deferred** — a formalism decision, not a patch.
