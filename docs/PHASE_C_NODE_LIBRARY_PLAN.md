# Phase D.3 — Phase C Node-Library Expansion — Plan

**Goal.** Build `PHASE_C_NODE_LIBRARY`: a catalogue of stable / near-stable / control **node configurations** with
per-config **node descriptors**, diverse enough to test whether stress / current / phase / spacing predict *coupling*
(D.2 re-run, D.4 coupling laws) and to seed the D.6 zoom-out model. **D.2's first pass was underpowered, not
negative** — its conclusion is exactly this: we need a diverse node library before coupling metrics can be validated.

**Scope / guardrails (unchanged).** No Phase C solver/physics/gate change; no production default change; no active
stress-tensor force term; no C2 transport branch yet; no matter-like claims. Search is used **only** to populate
diverse *stable/near-stable* node families around the validated basin. The catalogue is **read-only analysis** of
saved fields.

## Strategy — harvest first, then a small targeted batch
Most descriptors need only the saved `psi` field (node detection, centroids, spacing, per-node mass/phase, current
`J=ρ∇φ`, vorticity, stress-tensor terms). So the **cheap, smart first step is to catalogue the ~200 already-saved
Phase C states** (joint-basin, param-basin, node-count families, a\*-confirm, gain-ladder, core-delineation) into the
library, assess the diversity we already have, and only **then** run a *small targeted batch* to fill genuine gaps.
This avoids spending hunt compute to re-discover morphologies we already have on disk.

## 1. Node-library schema (per config)
| field | source |
|---|---|
| `key`, `run_path`, `param_a/eta/rho_vac` (factors), `seed` | filename + dir CSV |
| `stability_verdict` (klass), `er_fin`, `late_drift`, `bounded_breathing`, `T` | dir CSV |
| `n_nodes` | `transfer_diag.detect_nodes` |
| `node_centroids` (vox) | detect_nodes |
| `nn_spacing_min/mean/max` (minimal-image pairwise, box units) | centroids |
| `node_mass[]`, `node_energy[]` (per node) | detect_nodes (M, E) |
| `node_phase[]`, `phase_spread` (circular std of node phases) | detect_nodes |
| `mass_total`, `mass_per_node_mean/cv` | ρ |
| `Jmag_mean`, `node_current[]` (‖J‖ integrated per node) | `J=ρ∇φ` |
| `vorticity_mean`, `vort_over_J` (|∇×J| vs |J|: rotational vs radial current) | curl J |
| `stress_frob_mean`, `stress_aniso`, `stress_curr_vs_dens` (current-term vs density-strain-term weight) | `info_stress_tensor` |
| `morphology` (compact vs distributed proxy: spacing/box + mass_cv) | derived |
| `layout` (isolated vs bridge-like proxy: min NN spacing + inter-node density conductance) | centroids + `corridor_pair_metrics` |
| `seed_robust` (same param cell present + matching at ≥2 seeds) | cross-reference |

## 2. Diversity axes to populate (NOT score-only)
Target *variety* among stable/near-stable configs, plus deliberate controls:
- **node count** — K3–K8 families (FEB_BASIN / FEB_BASIN_CONFIRM) already give this.
- **spacing bands** — tight vs wide NN spacing (joint-basin a×η×ρ variation).
- **compact vs distributed** — mass concentrated in few nodes vs spread.
- **bridge-like vs isolated** — high inter-node density conductance vs isolated wells.
- **current/vorticity pattern** — rotational-core vs radial current differences.
- **stability spectrum** — TRUE bound states, NEAR/marginal, and controlled failures.

## 3. Reuse the validated basin
Any *new* targeted runs (only to fill gaps) stay in the validated ranges: `param_a` around a\* (≈feb×1.0–1.25),
`param_eta`/`param_rho_vac` in the joint-basin windows, node-count via K / IC morphology + seed variation — using the
**stability** objective (`objective="stability"` / `css.classify`), never an undocumented objective.

## 4. Contrast controls (must be in the library)
a\* 4-node; seed620/621 6-node; growers (a1.16/1.175/1.20); spin-down/decay corners (joint-basin SPIN cells);
and any C1 instability states as non-stable contrast.

## 5. Outputs
`jax_scout/node_library.py` (read-only cataloguer) → `PHASE_C_NODE_LIBRARY.{csv,json}` (off-repo, gitignored) +
`docs/PHASE_C_NODE_LIBRARY_RESULTS.md` (diversity assessment). Then a **small targeted batch** only if the harvested
diversity has real gaps, and finally the **D.2 re-run** on the library (isolate current vs density-strain term,
compute `∇·T`, spacing/phase regressions, density-null + bridged-vs-isolated tests).

## Success / failure
- **Success:** a usable library with enough diversity (node count, spacing, morphology, layout, stability spectrum)
  to test whether stress/current/phase/spacing predict coupling.
- **Failure:** search only re-discovers the same feb/a\* morphology, or diversity is only achievable with unstable
  candidates (→ record as a finding about the basin, not a forced result).
