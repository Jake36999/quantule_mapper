---
run_id: "CORE_SAT_MASS_THRESHOLD_N96_20260623_202328"
date: 2026-06-23
family: "Stability"
sector: "stability"
verdict: null
complete: false
source: derived
n_csv: 1
n_plots: 1
tags: [run, stability, Stability]
---
# CORE_SAT_MASS_THRESHOLD_N96_20260623_202328

*Phase C attractor / saturation hunts* &middot; **Stability** &middot; `2026-06-23`

> [!note] No verdict recorded
> This run's summary carries no `verdict` key.

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: .
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/mass_threshold_n96_validation.png]]

*mass threshold n96 validation*

## Figures (produced by the run itself)

*Copied from the run directory so Obsidian can display them. These are the run's own rendered output, not regenerated here.*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k1_above_threshold_failure_diagnostic_frames.png]]

*`rendered/k1_above_threshold_failure_diagnostic_frames.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k1_above_threshold_failure_probe_data.png]]

*`rendered/k1_above_threshold_failure_probe_data.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k1_below_threshold_survivor_diagnostic_frames.png]]

*`rendered/k1_below_threshold_survivor_diagnostic_frames.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k1_below_threshold_survivor_probe_data.png]]

*`rendered/k1_below_threshold_survivor_probe_data.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k6_highest_mass_survivor_diagnostic_frames.png]]

*`rendered/k6_highest_mass_survivor_diagnostic_frames.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k6_highest_mass_survivor_probe_data.png]]

*`rendered/k6_highest_mass_survivor_probe_data.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k6_same_mass_survivor_diagnostic_frames.png]]

*`rendered/k6_same_mass_survivor_diagnostic_frames.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__k6_same_mass_survivor_probe_data.png]]

*`rendered/k6_same_mass_survivor_probe_data.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__ref_feb56dc7_control_diagnostic_frames.png]]

*`rendered/ref_feb56dc7_control_diagnostic_frames.png`*

![[_figures/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/rendered__ref_feb56dc7_control_probe_data.png]]

*`rendered/ref_feb56dc7_control_probe_data.png`*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/`
- **CSV data** (1): `mass_threshold_n96_validation.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Stability`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[CORE_SAT_HUNT_20260623_175018]] &middot; `2026-06-23`
- [[CORE_SAT_HUNT_20260623_174215]] &middot; `2026-06-23`
- [[CORE_SAT_HUNT_20260623_173417]] &middot; `2026-06-23`

## Associated docs

- [[PHASE_C_MASS_THRESHOLD_N96_VALIDATION]]

## Next experiment

- [[CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546]] &middot; `2026-06-23`

## Branches

- [[Main branch]] &larr; via [[Branch - Stability - Index|stability sector index]]
- Sector: `stability` &middot; family: `Stability`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

> [!important] Paired reading - fill your block before reading the other one.
> A disagreement here is the point, not a problem. Record the resolution; do not
> overwrite the disagreement.

### Reading - Claude

*(filled 2026-09-17, plan item C2. Read for a specific reason: [[../gravity_maturity/S1_STATIC_TG_GREENS_FUNCTION|S1]]
showed the load->geometry map is exactly linear and therefore cannot hold a threshold, leaving the
phi sector's own stability boundary as one of only two candidates for the saturation cliff. This run
is that boundary.)*

- **Observation:** From `mass_threshold_n96_validation.csv`, every candidate carried forward:

  | candidate | N48 class | N96 class | mode matched |
  |---|---|---|---|
  | `k1_below_threshold_survivor` | `TRUE_SATURATED_BOUND_STATE` | **`DELOCALIZED_HALO_REJECT`** | False |
  | `k6_same_mass_survivor` | `TRUE_SATURATED_BOUND_STATE` | **`SPIN_DOWN_REJECT`** | False |
  | `k6_highest_mass_survivor` | `TRUE_SATURATED_BOUND_STATE` | **`SPIN_DOWN_REJECT`** | False |
  | `k1_above_threshold_failure` | `LATE_BLOWUP_REJECT` | `LATE_BLOWUP_REJECT` | True |
  | `ref_feb56dc7_control` | `REFERENCE_CONTROL` | `TRUE_SATURATED_BOUND_STATE` | True |

  **Three of three N48 "survivors" fail at N96, and `mode_matched_N48` is False for exactly those
  three.** The only candidate whose N48 verdict survives refinement is the one that already FAILED at
  N48, plus the pre-existing reference control.

- **Reading:** The mass threshold located at N48 is **not resolution-converged**, and the survivors
  that defined it were N48 artefacts. What survives refinement is the reference control, which was
  not discovered by this search.

  The failure modes are also informative and differ: `k1` delocalises into a halo while `k6` spins
  down. Those are different mechanisms, so this is not one uniform "everything blows up at higher
  resolution" -- each candidate fails in its own way, which is what an under-resolved core looks like
  rather than a single numerical instability.

  **For the saturation cliff this is evidence against the phi-sector-boundary explanation**, or at
  least against using *this* boundary as the measured one: a threshold whose survivors vanish on
  refinement cannot support a critical exponent. Combined with S1 (the T->G map is linear, so the
  cliff cannot live there) and the [[../gravity_maturity/P1B_FROZEN_REFERENCE_REANALYSIS|P1-b
  reanalysis]] (the mass axis shows no threshold at all, just clean power laws), the remaining
  candidate is the **source self-normalisation**, which saturates by construction and is not physics.

  Consistent with the recorded `BASIN_PARTIAL at N=96 (strong-gain edge = resolution artifact)`.

- **Confidence:** High on "the N48 threshold does not survive N96" -- it is five rows of an explicit
  table, not an inference. Medium on the implication for the cliff, since this run is the
  *stability* threshold and the cliff was reported on the *load-capacity* axis; they are related but
  not identical, and I have not checked that the load-capacity sweeps show the same resolution
  sensitivity.

- **What would change my mind:** An N96-native search (rather than N48 candidates promoted to N96)
  finding survivors near the same masses -- that would make this a promotion artefact rather than a
  threshold artefact. Also: a load-capacity sweep repeated at two resolutions showing the cliff
  position stable, which would rescue the phi-sector explanation on the axis that actually matters.

> [!warning] This run carries `verdict: null`
> The frontmatter records no verdict and the note says "No verdict recorded", yet the table above is
> an unambiguous negative result about a threshold the project has used. It is exactly the kind of
> finding the catalogue exists to stop being stranded, and it should be given one.

### Reading - Jake

- **Observation:**
- **Reading:**
- **Confidence:**
- **What would change my mind:**

### Comparison

**Agree on:**

**Disagree on:**

**Resolution:**

