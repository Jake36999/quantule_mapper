# Phase C — feb56dc7 basin focused search (plan)

**Date:** 2026-06-25
**Premise:** the broad Option-B saturation search produced only T=6000 window-artifacts; the long-T
control showed **feb56dc7 is the one validated long-time bound state** (TRUE, bounded, stable to T=24000),
while every search-discovered K6/K4/K2 branch decays or blows up. We now have a **v2 stability gate**
(commit `6de28a2a7`, `docs/PHASE_C_STABILITY_GATE_CALIBRATION.md`). Instead of another broad hunt, we
characterize the basin **around the one known attractor**, validated under the new gate.

**Question:** is feb's stability a property of its *parameter point* (so other node-counts / IC families
are also stable there), or is it specific to feb's exact IC (6-blob, per-blob-fixed norm)?

**Discipline:** analysis/replay-grade — fixed feb parameters, only the IC family is varied; no PDE/solver
change; geometry frozen at `e8d6a78ea`. This is a *small pre-registered grid*, not a random search. All
runs validated at **N=96, T=12000** so the v2 drift gate bites. No charge/topology/proof/ground-state/
black-hole/universal-law language.

## Grid A — node-count basin (the core experiment)

Fixed `params = FEB`, `ic_norm = per_blob_fixed`, `seed = 20260619`, N=96, T=12000:

| K (IC blob count) | 3 | 4 | 5 | 6 (feb) | 8 |
|---|---|---|---|---|---|

→ which blob counts reach a v2-gated `TRUE_SATURATED_BOUND_STATE`? Tests whether feb's bound state is
K-specific or generic to its param point. (feb's IC is 6-blob; its attractor is 4-node.)

## Grid B — IC-norm / mass probe at feb's K=6

Fixed `params = FEB`, `K = 6`, seed 20260619, N=96, T=12000. Compare:
- `per_blob_fixed` (feb's family) — the reference;
- `total_mass_fixed` at {0.5×, 1×, 2×} the per-blob-natural mass.

→ does *forcing* the total mass (the Option-B family's mode) destabilize the feb point? This directly
tests the lead that total-mass-fixed pushed configs off the stable manifold into the decay/blowup
separatrix.

## Pre-registered reads

- **Only K=6/per_blob is TRUE** → feb's bound state is narrow (IC-specific); the field has very few
  genuine attractors. Record and stop broadening.
- **A band of K (e.g. 4–6) per_blob is TRUE** → the param point supports a node-count family; map it
  further (this becomes the new, principled search axis).
- **total_mass_fixed destabilizes vs per_blob at the same mass** → confirms the IC-norm/forcing
  hypothesis for why Option-B failed; future searches should use per-blob-fixed norm.
- Any new TRUE survivor → confirm with 2 extra seeds before believing it (the discovery seed-sensitivity
  we already saw).

## Outputs

`sweep_runs/FEB_BASIN_<ts>/` — `feb_basin_results.csv` (K, ic_norm, target_mass, class, er_fin, er_max,
late_slope, **late_drift**, n_fin, held_mass) + per-config `probe_data.npz` + `feb_basin_summary.json`.
Render/inspect on the Windows side afterward. Runtime ≈ 8 configs × ~12 min ≈ ~1.6 h at T=12000.

Promotion to "bound state" requires the **v2 gate** (`|late_drift| ≤ 0.15`, in-band er, stable node
count) — no config is called a bound state on a short-window pass.
