# Phase D C2.8 + C2.6 Codex Handover

Date: 2026-07-09

This is a data handover for Claude. It separates completed C2.8 two-node measurements from the independent C2.6 audit. It does not make a broad stability or matter claim.

## A. What Was Run

### C2.8 two-node interaction run

Harness:

```text
jax_scout/phase_d_c2_8_two_node.py
```

Status:

- The harness is currently untracked in git.
- A preservation snapshot was saved in the run folder as `harness_snapshot_phase_d_c2_8_two_node.py`.
- No commit was created.

Run folder:

```text
F:/quantule_mapper/sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611
```

Exact command recorded in:

```text
c2_8_handover_manifest.json
```

Primary preserved outputs:

- `run.log`
- `summary.json`
- `phi_iso.npy`
- `rho_x_t0_hold.npz`
- `rho_x_t1_headon_n2.npz`
- `rho_x_t1_headon_n4.npz`
- `rho_x_t2_inphase.npz`
- `rho_x_t2_antiphase.npz`

Analysis outputs added by Codex:

- `c2_8_analysis_report.md`
- `c2_8_summary_tables.csv`
- `c2_8_peak_trajectories.csv`
- `c2_8_separation_vs_time.csv`
- `c2_8_mass_momentum_vs_time.csv`
- `*_rho_x_spacetime.png`
- `c2_8_peak_trajectories_vs_time.png`
- `c2_8_separation_vs_time.png`
- `c2_8_mass_retention_vs_time.png`
- `c2_8_momentum_vs_time_tail_only.png`

Bounded diagnostic supplements:

```text
F:/quantule_mapper/sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/bounded_supplements
```

Supplement outputs:

- `run.log`
- `supplement_summary.json`
- `rho_x_supp_static_phase_pi2.npz`
- `rho_x_supp_headon_n2_amp_asym.npz`

### C2.6 independent audit

Audit runner:

```text
tools/c2_6_independent_audit.py
```

Audit folder:

```text
F:/quantule_mapper/sweep_runs/PHASE_D_C2_6_CODEX_AUDIT_20260709
```

Audit outputs:

- `run.log`
- `c2_6_independent_audit_report.md`
- `c2_6_independent_audit_summary.json`

## B. What Was Measured

### C2.8 T0 profile and hold gate

From `summary.json`:

- Petviashvili residual: `1.8074313340524997e-08`
- `mu_check`: `0.0704000000000001`
- amplitude: `0.988521114121002`
- mass: `5712.904401466432`
- isolated mass: `5164` from `run.log`
- pedestal fraction: `0.09608617321960596`
- hold mass retention at T=2: `0.9999`

### C2.8 head-on n=2

Neutral label:

```text
TWO_CORES_SURVIVE_CLOSE_ENCOUNTER
```

Measurements:

- existing classifier: `PASS_THROUGH_OR_REBOUND`
- `v_each`: `0.37699111843077515`
- min separation: `1.458`
- final peak count: `2`
- final mass: `0.9991`
- projected-density analysis min separation: `1.4583333333333335`
- projected-density final separation: `1.4583333333333335`
- pre-min separation slope: `-0.5986721611721616`
- post-min separation slope: approximately `0`

Caveat:

`rho_x` does not preserve soliton identity or phase labels. Do not claim elastic scattering from this output alone.

### C2.8 head-on n=4

Neutral label:

```text
TWO_CORES_SURVIVE_CLOSE_ENCOUNTER
```

Measurements:

- existing classifier: `PASS_THROUGH_OR_REBOUND`
- `v_each`: `0.7539822368615503`
- min separation: `1.458`
- final peak count: `2`
- final mass: `0.9989`
- projected-density analysis min separation: `1.4583333333333335`
- projected-density final separation: `2.0833333333333335`
- pre-min separation slope: `-0.43633449883449876`
- post-min separation slope: `0.10416666666666675`

Caveat:

The output supports two cores surviving a close encounter. It does not by itself distinguish pass-through from rebound.

### C2.8 static in-phase pair

Neutral label:

```text
STATIC_INPHASE_ATTRACTION
```

Measurements:

- separation: `5.833 -> 1.458`
- final peak count: `2`
- final mass: `0.9995`
- projected-density final separation: `1.4583333333333335`

Directly supported wording:

```text
in-phase static pair attracts
```

### C2.8 static anti-phase pair

Neutral label:

```text
STATIC_ANTIPHASE_REPULSION
```

Measurements:

- separation: `5.833 -> 8.75`
- final peak count: `2`
- final mass: `0.9994`
- projected-density final separation: `8.75`

Directly supported wording:

```text
anti-phase static pair repels
```

### Bounded supplement A: static phase midpoint

Diagnostic-only, not canonical C2.8.

Setup:

- sep `6`
- relative phase `pi/2`
- T `20`
- reused `phi_iso.npy`

Result:

- trend: `ATTRACT`
- separation: `5.625 -> 1.458`
- final peak count: `2`
- final mass: `0.9995`

Caveat:

This suggests the pi/2 midpoint is on the attracting side for this setup, but this is a single diagnostic supplement.

### Bounded supplement B: asymmetric n=2 head-on

Diagnostic-only, not canonical C2.8.

Setup:

- head-on `n=2`
- tiny amplitude asymmetry: left `1.01`, right `0.99`
- T about `23.3`
- reused `phi_iso.npy`

Result:

- neutral label: `DIAGNOSTIC_ASYMMETRIC_TWO_CORES_SURVIVE_CLOSE_ENCOUNTER`
- existing classifier: `PASS_THROUGH_OR_REBOUND`
- min separation: `1.458`
- final peak count: `2`
- final mass: `0.9991`

Caveat:

The supplement still does not cleanly resolve pass-through versus rebound. A proper identity-labelled diagnostic would be needed.

## C. What Is Directly Supported

- The C2.8 run completed successfully on WSL/JAX GPU.
- The GTX 1080 was visible as JAX `cuda:0`.
- The isolated profile held over the T0 gate with high mass retention.
- In head-on `n=2` and `n=4`, two cores survive close encounter with high mass retention.
- In-phase static pair attracts.
- Anti-phase static pair repels.
- The pi/2 static supplement also attracted in this single bounded diagnostic.
- Clean single-soliton transport was independently audited in C2.6:
  - default parity exact;
  - true geometry-off is flat;
  - old D_eff bug is reproducible;
  - linear packet transport is correct;
  - true soliton transport agrees under ETDRK4 and independent RK4;
  - geometry-off flux is near numerical zero.

## D. What Remains Ambiguous

- Head-on identity is unresolved. The available projected-density `rho_x` outputs show two peaks/cores, but they do not label which incoming soliton became which outgoing soliton.
- The head-on verdict should remain neutral:

```text
TWO_CORES_SURVIVE_CLOSE_ENCOUNTER
```

- Do not claim elastic scattering unless Claude adds or inspects identity-resolving diagnostics.
- The asymmetric supplement is useful but not sufficient as a canonical identity proof.
- No broad stability or matter claim is made.

## E. What Claude Should Inspect Next

1. Inspect the C2.8 analysis tables and spacetime heatmaps:

   - `c2_8_summary_tables.csv`
   - `c2_8_peak_trajectories.csv`
   - `c2_8_separation_vs_time.csv`
   - `*_rho_x_spacetime.png`

2. Decide whether a labelled identity diagnostic is needed for head-on cases:

   - phase tag,
   - slight frequency/velocity label,
   - overlap tracking against left/right reference profiles,
   - or full complex-field snapshots instead of only `rho_x`.

3. Treat static-pair phase response as promising but still sparse:

   - phase `0`: attraction,
   - phase `pi/2`: attraction in supplement,
   - phase `pi`: repulsion.

4. Use the C2.6 audit as the corrected-substrate confidence gate:

   - the old transport/pinning readings from the bugged substrate should not be reused;
   - corrected true-flat single-soliton transport is independently confirmed.

5. Keep future interpretation cautious:

```text
two cores survive close encounter
in-phase static pair attracts
anti-phase static pair repels
clean single-soliton transport was independently audited
no broad stability or matter claim
```

## F. C2.6 Audit Consolidation

Independent audit values:

- default parity exact:
  - `max_delta_L_k = 0.0`
  - `max_delta_E = 0.0`
  - `max_delta_f1 = 0.0`
- `param_geom_off=True` genuinely flat:
  - `geom_fac = 0.0`
  - `n_op_minus_polynomial_only_max_abs = 0.0`
- old D_eff bug reproduced:
  - ratio `0.006511027642497719`
  - close to `1/151 = 0.006622516556291391`
- linear packet moves correctly:
  - peak indices `[24, 25, 25, 26, 26, 27]`
  - expected shift over T=0.5: about `3.016` cells
- true soliton transport agrees in ETDRK4 and independent RK4:
  - ETDRK4 peak indices `[24, 25, 25, 26, 26, 27]`
  - RK4 peak indices `[24, 25, 25, 26, 26, 27]`
  - ETDRK4 mass retention `0.9999168363918189`
  - RK4 mass retention `1.0000000000000009`
- geometry-off flux near numerical zero:
  - random state flux `3.2311627426270885e-16`
  - packet flux `1.1332323302107005e-17`
- protected production/reference diff remained empty.

## G. Protected File Status

No protected production/reference diff was present for:

- `solver/core.py`
- `solver/run.py`
- `worker_cupy.py`
- `aste_hunter.py`
- `validation_pipeline.py`
- `config_utils.py`
- `tools/production_h7_revalidation.py`
- `jax_scout/physics.py`
- `jax_scout/phase_d_c2_transport.py`
- `jax_scout/phase_d_c2_soliton_scout.py`
- `jax_scout/phase_d_c2_2_loss_source.py`

New diagnostic/package files from Codex:

- `tools/analyze_c2_8_two_node_outputs.py`
- `tools/run_c2_8_bounded_supplements.py`
- `tools/c2_6_independent_audit.py`
- `docs/PHASE_D_C2_8_AND_C2_6_CODEX_HANDOVER.md`

No commit was created.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `f19b1e0` (2026-07-09) — *Phase D C2.8: two-node interaction harness + Codex analysis tools + C2.6 audit (*
**Revised since:** 12 commit(s), most recently `59b7ec3` (2026-09-12)

**Harness code changed since it was written:** 17 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 12 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[PHASE_D_C2_8_TWONODE_RESULTS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
