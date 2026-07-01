# Phase C K-Varied Saturation Hunt

## 1. Purpose

Measure whether long-time saturated node count at `N=48`, `T=4000` tracks the initial blob count `K`, or whether the field preferentially settles into a multi-node saturated state regardless of `K`.

This document treats all findings as proxy evidence only. The earned scope here is:

- saved-classification evidence at `N=48`, `T=4000`
- no claim of proof
- no claim of ground state, molecule, topological invariant, or infinite stability

## 2. Method

Completed hunt command from the runtime log:

```powershell
python jax_scout/core_saturation_search.py --hours 6 --batch 64 --ic-counts 1,2,3,4,6
```

Fixed settings from the saved summary:

- `N = 48`
- `T = 4000`
- `batch = 64`
- `K in {1,2,3,4,6}`
- regime tightened around the feb56dc7 saturation band

Round-robin sampling was used over the listed `K` values until the timebox expired.

## 3. Run Directory and Command

Completed run directory:

```text
F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605
```

Supporting runtime log:

```text
F:/quantule_mapper/runtime_logs/core_sat_hunt_20260623_004559.log
```

Excluded incomplete runs:

- `F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004423`
- `F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260622_211628`

Both excluded directories contain empty `all_evals.csv` files and no usable `summary.json`.

## 4. Artifact Contracts

Primary saved artifacts used for this analysis:

- `all_evals.csv`
- `summary.json`
- `analysis_summary.json`

Standard Phase C report pack regenerated with:

```powershell
python -m quantule_viz phase-c sweep_runs\CORE_SAT_HUNT_20260623_004605 --overwrite
```

Generated report pack:

- `summary_panel.png`
- `class_histogram.png`
- `class_counts_by_K.png`
- `node_counts_by_K.png`
- `saturation_slope_scatter.png`
- `best_candidates_table.csv`
- `analysis_summary.json`

## 5. Operational Integrity Check

Status:

```text
RUN_COMPLETE_WITH_WARNINGS
```

Evidence:

- `summary.json` reports `n_eval = 832`
- `summary.json` reports `elapsed_h = 6.30349309987492`
- `all_evals.csv` contains `832` non-empty rows with contiguous indices `0..831`
- runtime log ends with `HUNT EXIT 0`
- `all_evals.csv` and `summary.json` share the same last-write window around `2026-06-23 07:04`
- all intended `K` values appear: `1, 2, 3, 4, 6`

Warnings:

- sampling is not perfectly balanced because the 6-hour cutoff landed after `13` completed batches:
  - `K=1`: `192`
  - `K=2`: `192`
  - `K=3`: `192`
  - `K=4`: `128`
  - `K=6`: `128`
- a stale partial `analysis_summary.json` from an earlier render was present in the run directory before rerender; it was regenerated from the full `832`-row CSV with `quantule_viz`

Interpretation:

- evidence supports a clean operational finish
- the imbalance is moderate, not catastrophic, but it weakens direct K-to-K comparisons for `K=4` and `K=6`

## 6. Overall Results

Overall class counts:

| Class | Count |
| --- | ---: |
| `TRUE_SATURATED_BOUND_STATE` | 92 |
| `NEAR_SATURATED_BOUND_STATE` | 35 |
| `TRANSIENT_GROWER_REJECT` | 27 |
| `LATE_BLOWUP_REJECT` | 388 |
| `SPIN_DOWN_REJECT` | 290 |
| `FRAGMENTATION_REJECT` | 0 |
| `DELOCALIZED_HALO_REJECT` | 0 |

Global TRUE final node-count distribution:

| Final node count | TRUE count |
| --- | ---: |
| `1` | 20 |
| `2` | 41 |
| `3` | 11 |
| `4` | 16 |
| `5` | 4 |

Immediate implication:

- the completed hunt does **not** support a simple universal preference for `4-5` nodes across all tested initial conditions
- the completed hunt also does **not** support clean one-to-one IC tracking across all `K`

## 7. Results by Initial Blob Count K

Sampling counts by `K`:

| K | Rows |
| --- | ---: |
| `1` | 192 |
| `2` | 192 |
| `3` | 192 |
| `4` | 128 |
| `6` | 128 |

Class counts by `K`:

| K | TRUE | NEAR | GROWER | BLOWUP | SPIN_DOWN |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | 21 | 6 | 6 | 90 | 69 |
| `2` | 23 | 9 | 2 | 92 | 66 |
| `3` | 21 | 11 | 7 | 81 | 72 |
| `4` | 14 | 3 | 8 | 66 | 37 |
| `6` | 13 | 6 | 4 | 59 | 46 |

Low-K behavior summary:

- `K=1`:
  - TRUE outcomes exist
  - TRUE final nodes are `1` or `2`
  - no TRUE `4-5` node outcomes
- `K=2`:
  - TRUE outcomes exist
  - TRUE final nodes are mostly `1` or `2`, with one `3`-node case
- `K=3`:
  - TRUE outcomes exist
  - mixture of `2`, `3`, `4`, and one `5`-node outcome
- `K=4`:
  - TRUE outcomes exist
  - mixture of `2`, `3`, and one `4`-node outcome
- `K=6`:
  - TRUE outcomes exist
  - TRUE outcomes are mostly `4` nodes, with some `5` nodes and one `3`-node case

## 8. Final Node-Count Distribution by K

TRUE saturated final node counts by `K`:

| K | 1-node | 2-node | 3-node | 4-node | 5-node |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | 8 | 13 | 0 | 0 | 0 |
| `2` | 12 | 10 | 1 | 0 | 0 |
| `3` | 0 | 11 | 3 | 6 | 1 |
| `4` | 0 | 7 | 6 | 1 | 0 |
| `6` | 0 | 0 | 1 | 9 | 3 |

NEAR saturated final node counts by `K`:

| K | 1-node | 2-node | 3-node | 4-node | 5-node | 6-node |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | 3 | 3 | 0 | 0 | 0 | 0 |
| `2` | 2 | 7 | 0 | 0 | 0 | 0 |
| `3` | 0 | 6 | 2 | 3 | 0 | 0 |
| `4` | 0 | 0 | 0 | 3 | 0 | 0 |
| `6` | 0 | 0 | 0 | 4 | 1 | 1 |

Cross-tab `K x final_node_count x class` highlights:

- `K=1`:
  - TRUE only at `1` and `2` nodes
- `K=2`:
  - TRUE mostly at `1` and `2` nodes
- `K=3`:
  - TRUE concentrated at `2` nodes, with a substantial `4`-node tail
- `K=4`:
  - TRUE concentrated at `2` and `3` nodes
- `K=6`:
  - TRUE concentrated at `4` and `5` nodes

## 9. Best Candidates

Selection rule used here:

- "best TRUE per K" = lowest `|late_slope|` among TRUE outcomes, preferring `n_mid == n_fin`
- "closest to feb56dc7 behavior" = heuristic only, using TRUE outcomes with low normalized parameter distance to feb56dc7 and a preference for `n_fin` near `4`

Best TRUE by `K`:

| K | idx | final nodes | late slope | er_fin | core_fin | Why |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `663` | `2` | `-6.80e-06` | `0.6145` | `0.3787` | best low-K by `|late_slope|` |
| `2` | `743` | `1` | `-7.91e-06` | `0.8257` | `0.4453` | best `K=2` by `|late_slope|` |
| `3` | `154` | `2` | `+1.64e-05` | `1.0627` | `0.8587` | best `K=3` by `|late_slope|` |
| `4` | `570` | `3` | `-2.77e-05` | `1.2198` | `3.1445` | best `K=4` by `|late_slope|` and strongest localization |
| `6` | `623` | `4` | `-2.79e-05` | `0.8915` | `0.7897` | best `K=6`; also closest feb-like TRUE in this hunt |

Closest feb56dc7-like TRUE candidates:

- `K=6`, idx `623`, `n_fin=4`
- `K=3`, idx `807`, `n_fin=4`
- `K=4`, idx `573`, `n_fin=3` only weakly feb-like

Specific checklist answers:

- `K=1` produced TRUE single-node candidates: **yes**, `8`
- `K=2` produced TRUE two-node candidates: **yes**, `10`
- `K=3` produced TRUE three-node candidates: **yes**, `3`
- `K=4` reproduced feb-like `4`-node TRUEs: **weakly**, `1`
- `K=6` reproduced feb-like `4`-node TRUEs: **yes**, `9`

## 10. Near Misses

Best NEAR per `K`:

| K | idx | final nodes | late slope | er_fin | core_fin |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `50` | `1` | `+1.53e-04` | `2.3678` | `2.2601` |
| `2` | `744` | `2` | `+1.51e-04` | `1.4071` | `0.8844` |
| `3` | `490` | `4` | `+1.75e-04` | `1.6326` | `0.7755` |
| `4` | `224` | `4` | `+1.63e-04` | `1.4195` | `0.4495` |
| `6` | `317` | `4` | `+1.52e-04` | `1.8673` | `1.0452` |

Interpretation:

- low-K did not fail outright; there are TRUE and NEAR low-node saturators
- `K=4` and `K=6` also have NEAR `4`-node cases, supporting a high-K route into feb-like multiplicity

## 11. Provisional Verdict

Verdict:

```text
K_DEPENDENCE_UNCLEAR
```

Evidence:

- low-K support is real:
  - `K=1` produced TRUE `1`-node and `2`-node outcomes at `N=48/T=4000`
  - `K=2` produced TRUE `1`-node and `2`-node outcomes at `N=48/T=4000`
- the field does not collapse everything into `4-5` nodes:
  - many TRUE low-K outcomes remain at `1-2` nodes
- simple IC tracking is also not supported:
  - `K=1` tends more often to `2` than `1`
  - `K=2` is split between `1` and `2`
  - `K=3` tends more often to `2` than `3`
  - `K=4` mostly lands at `2-3`
  - `K=6` mostly lands at `4-5`

Inference:

- the current proxy evidence suggests **partial K dependence with branching multiplicities**
- high `K`, especially `K=6`, favors feb-like `4-5` node saturation
- low `K` does **not** show "no support"; it produces its own TRUE low-node attractors under tested conditions
- the first-pass `N=96/T=6000` replays reinforce that mixed picture rather than collapsing it to a single story

This is still not a proof claim. The hunt itself is `N=48`, `T=4000` evidence, and the row-targeted
replays below are a small validation subset rather than a fresh exhaustive search.

## 12. Caveats

- the main hunt is still a proxy search at `N=48`, `T=4000`
- only a shortlist, not the full population, has been replayed at `N=96`, `T=6000`
- K sampling is complete but imbalanced because the 6-hour timebox ended after `13` batches
- the IC family is **not** normalized across `K`; total injected density / mass proxy rises strongly with `K`
- the completed hunt was launched from a dirty working tree and did not stamp its own commit or seed metadata into the original outputs
- the saved classifier set contains no fragmentation or halo classes here; this may be regime-specific rather than universal
- the new replay helper now exists and was verified on both the feb reference and saved hunt rows

## 13. N=96/T=6000 Validation Outcomes

Validated controls:

| Candidate | K | N=96/T=6000 class | n_mid | n_fin | late slope | er_fin | er_max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `ref_feb56dc7` | `6` | TRUE | `4` | `4` | `-2.316e-06` | `1.5781` | `1.5956` |
| `623` | `6` | TRUE | `4` | `4` | `-2.951e-05` | `0.8313` | `0.9918` |

Validated shortlist:

| idx | K | N=48 class | N=48 n_fin | N=96/T=6000 class | N=96 n_mid | N=96 n_fin | N=96 late slope | N=96 er_fin | Note |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `663` | `1` | TRUE | `2` | TRUE | `2` | `2` | `+7.391e-06` | `0.6444` | low-K 2-node state survives |
| `743` | `2` | TRUE | `1` | TRUE | `1` | `1` | `-4.114e-06` | `0.8180` | low-K 1-node state survives |
| `154` | `3` | TRUE | `2` | TRUE | `2` | `4` | `+1.220e-05` | `1.0877` | regrows from 2 to 4 nodes by late time |
| `570` | `4` | TRUE | `3` | TRUE | `3` | `3` | `-1.163e-04` | `0.6531` | stable 3-node case survives close to TRUE cutoff |
| `623` | `6` | TRUE | `4` | TRUE | `4` | `4` | `-2.951e-05` | `0.8313` | best feb-like hunt row holds |
| `744` | `2` | NEAR | `2` | NEAR | `2` | `1` | `+1.944e-04` | `1.8312` | stays NEAR, does not upgrade |

Parameter vectors:

- idx `663`: `eta=0.0066`, `a=0.4122`, `D=1.6006`, `rho_vac=0.1779`, `omega0=1.0363`, `a_coupling=0.3922`, `s=-0.6321`, `f=-0.2392`
- idx `743`: `eta=0.0268`, `a=0.2811`, `D=1.1515`, `rho_vac=0.7851`, `omega0=0.4926`, `a_coupling=2.8826`, `s=-0.4657`, `f=-0.0922`
- idx `154`: `eta=0.0657`, `a=0.4433`, `D=2.0507`, `rho_vac=1.0972`, `omega0=0.6677`, `a_coupling=1.4095`, `s=-0.2819`, `f=-0.1753`
- idx `570`: `eta=0.1223`, `a=0.2052`, `D=1.4256`, `rho_vac=0.9599`, `omega0=0.6101`, `a_coupling=1.6701`, `s=0.2442`, `f=-0.0461`
- idx `623`: `eta=0.0795`, `a=0.4338`, `D=4.7558`, `rho_vac=1.0816`, `omega0=0.4798`, `a_coupling=3.1266`, `s=-0.0691`, `f=-0.4751`
- idx `744`: `eta=0.0239`, `a=0.2232`, `D=3.1651`, `rho_vac=0.7396`, `omega0=1.8458`, `a_coupling=3.0764`, `s=0.0181`, `f=-0.1614`
- `ref_feb56dc7`: `eta=0.0704`, `a=0.4802`, `D=2.7329`, `rho_vac=1.1866`, `omega0=0.0`, `a_coupling=2.3098`, `s=0.0129`, `f=-0.4861`

Verified validation command:

```powershell
python /mnt/f/quantule_mapper/jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605/all_evals.csv --idx <row_idx> --N 96 --T 6000 --out <outdir>
```

Interpretation of the replayed shortlist:

- `K=1` and `K=2` do retain low-node TRUE cases at higher fidelity
- `K=3` shows mixed behavior and can regrow to a 4-node TRUE late state
- `K=4` supports a 3-node TRUE case
- `K=6` cleanly reproduces feb-like 4-node TRUE behavior
- the fallback NEAR candidate `744` remains NEAR rather than automatically upgrading

These outcomes keep the honest verdict at `K_DEPENDENCE_UNCLEAR`:

- not clean IC tracking
- not universal 4-5-node convergence
- real branching structure, with both low-node and higher-node validated outcomes

## 14. Next Actions

- stamp future Phase C hunts with commit, command, classifier version, and IC seed in `summary.json` and CSV
- decide whether to rerun the K-varied hunt with IC normalization across `K` so multiplicity can be tested without the mass-scaling confound
- expand `N=96/T=6000` validation beyond the shortlist only if the next question is population-level support rather than candidate-level confirmation

Evidence:

- low-K TRUE outcomes exist
- `K=6` clearly supports feb-like `4`-node saturation
- `K=3` is a mixed branch point worth retesting at higher fidelity

Inference:

- the field is not universally forcing `4-5` nodes under this tested regime
- the field is also not cleanly mirroring the initial `K`

Caveat:

- this conclusion is still provisional until the shortlisted cases are rerun at `N=96`, `T=6000`

Proposed action:

1. add a row-targeted N=96/T=6000 replay helper for saved Phase C candidates
2. validate shortlist entries `663`, `743`, `154`, `570`, `623`, plus `ref_feb56dc7`
3. compare whether low-K TRUE cases remain low-node at higher fidelity or regrow/reclassify
