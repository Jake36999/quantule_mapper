# Hunter Re-Aim — Implementation Notes (H7.1 wiring)

**Scope: wiring only.** The stability objective is now selectable in `aste_hunter` behind an explicit flag. **No
hunt was run, no GPU/CuPy job, no solver/geometry/gate/physics change, and the default (prime) path is
untouched.** Hunter proposes; `core_saturation_search.classify` (v3) still certifies.

## What changed in `aste_hunter.py`
1. `Hunter.__init__(..., objective: str = "prime")` — new opt-in flag; **default `"prime"` = legacy behaviour,
   unchanged.**
2. Module helper `_stability_fitness_from_provenance(prov_data)` — scores `prov_data["stability_metrics"]` with
   `tools.stability_objective.stability_score`; returns fitness ≥ 0. **`log_prime_sse` is NOT used as fitness
   here.**
3. `process_generation_results` — an early branch: when `objective == "stability"`, fitness comes from the
   helper; the prime prefilter and spectral fitness are **skipped**; `log_prime_sse` and peak fields are still
   **recorded as diagnostics** (provenance), never steering.

## How to select it (later — no run implied)
```python
from aste_hunter import Hunter
h = Hunter(db_file="simulation_ledger.db", objective="stability")   # opt-in
```
or a config key `hunter_objective = "stability"` wherever the Hunter is constructed. Omit it → `"prime"`
(default, unchanged).

## Tests (`tests/test_hunter_stability_wiring.py`, 5 pass, dev box, no GPU)
- default Hunter `objective == "prime"` (unchanged); stability objective **selectable**;
- stability fitness ranks a\* (`+0.894`) above a decayer (`+0.521`);
- **prime-SSE does not steer**: a config with excellent `log_prime_sse=0.001` but no stability metrics scores
  **0.0** in stability mode;
- absent `stability_metrics` → graceful `0.0` / `NO_STABILITY_METRICS`.

## Honest scope limit (what is NOT yet done)
- **Fitness only.** The NSGA-II fronts in `generate_next_generation` still rank on the *spectral* objectives
  (`primary_harmonic_error`, `pcs`, `grad_phase_var`, …), which are degenerate in stability mode → selection
  effectively falls back toward fitness-ranking. A proper **stability-NSGA front** (objectives over
  late-slope / boundedness / breathing / seed-robustness) is a follow-up (H7.1b), not done here.
- **The worker does not yet emit `stability_metrics`.** A real stability hunt needs the (CuPy) worker /
  evaluator to write `stability_metrics` (er_fin, er_max, floor_ratio, late-slope, T) into each provenance JSON.
  That is a worker change on the CuPy production path — **not made, not runnable from this session.**
- **Indivisibility** remains a run-based hook (design spec §3), unimplemented.

## Remaining gates (unchanged)
- **H4 CuPy parity** — still pending the operator's production-box run.
- **H7.1b** — stability-NSGA front + worker `stability_metrics` emission.
- **H7.3** — re-validation harness (re-find a\* via search, matched-control recovery).
- **H7.4** — a small controlled hunt — gated on H7.1b + H7.3 + H4 parity + explicit approval.

**No claim** is made that the re-aimed Hunter has rediscovered the a\* basin via search, nor that H7 is validated.
What is established: the objective is wired, selectable, prime-SSE is dethroned as steering, and it ranks the
validated a\* case top on real data (offline, `HUNTER_REAIM_OFFLINE_RESCORE.md`).
