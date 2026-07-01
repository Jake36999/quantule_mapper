# Phase C Option B — N96 / T6000 Validation Plan (DRAFT — DO NOT LAUNCH)

Resolution-validation command matrix for the 8 Option B shortlist cases. Discovery was `N=48 / T=4000`
(see [PHASE_C_OPTION_B_ANALYSIS.md](PHASE_C_OPTION_B_ANALYSIS.md)). This plan re-runs the same IC family,
solver core, and classifier at `N=96 / T=6000` with explicit resolution-scaled raw-target overrides.

**This is a draft. Do not launch any command below until explicitly approved.** No PDE / solver /
classifier / `gravity/unified_omega.py` / Ω² / search-logic changes are implied or permitted; this is a
pure resolution replay using the existing `jax_scout/core_saturation_replay.py`.

---

## 1. Mass scaling (why the overrides look large)

The search input is a **raw grid-sum** target `sum(|psi|^2)`. The documented physical quantity is the
`dx³`-weighted integral. Holding the integral fixed across resolution:

```
target_raw_N96 = target_raw_N48 * (dx_N48 / dx_N96)^3
              = target_raw_N48 * (N96 / N48)^3
              = target_raw_N48 * (96/48)^3
              = target_raw_N48 * 8
```

`core_saturation_replay.py` confirms this contract directly: it reads `source_resolution_N=48` from the
source run's `summary.json`, computes `target_integral_mass = saved_raw * (L/48)^3`, and the expected
N96 raw target `target_integral_mass / (L/96)^3`. When `--target-initial-mass-override` equals that
value (i.e. `saved_raw * 8`), the run auto-stamps:

- `mass_scaling_mode = resolution_scaled_raw_target`
- `replay_kind = RESOLUTION_SCALED_TARGET_REPLAY`

If you pass any other override it stamps `explicit_raw_target_override` instead — that is the tell that a
number is wrong. **Cross-resolution replay is refused without the override** (the script raises rather
than silently re-using an N48 raw target at N96), so the override is mandatory for every CSV-sourced case.

Stamped override values (verified `raw_N48 * 8`):

| raw_N48 target | scaled N96 raw target (`--target-initial-mass-override`) |
|---|---|
| `1000.0`        | `8000.0` |
| `1200.0`        | `9600.0` |
| `2050.293702`   | `16402.349616` |

**feb56dc7 control is different:** it is `--ref feb56dc7`, which uses *per-blob-fixed* IC norm
(`IC_NORM_PER_BLOB_FIXED`), **not** total-mass-fixed. It takes **no** mass override — the per-blob
normalization is resolution-stable by construction. Do not apply `*8` to it.

---

## 2. Candidate matrix (priority order)

Source rows are the exact shortlisted discovery rows (`runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json`,
mirrored in `sweep_runs/PHASE_C_VISUAL_ANALYSIS_V2_*/n96_shortlist/n96_shortlist_inspection_sheet.csv`).
All source dirs confirmed `N=48` with `all_evals.csv` present.

| # | candidate | source run (`--csv .../all_evals.csv`) | `--idx` | K | seed | raw_N48 | `--target-initial-mass-override` | N48 verdict | resolution-risk | role |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | K6 high-mass TRUE | `CORE_SAT_HUNT_20260624_112918` | 32 | 6 | 20260619 | 2050.293702 | `16402.349616` | TRUE / SATURATED | low | strongest distributed branch |
| 2 | K6 mid-mass TRUE | `CORE_SAT_HUNT_20260624_142152` | 34 | 6 | 20260621 | 1000.0 | `8000.0` | TRUE / SATURATED | low | distributed branch, lower mass |
| 3 | K4 intermediate TRUE | `CORE_SAT_HUNT_20260624_124444` | 25 | 4 | 20260620 | 1200.0 | `9600.0` | TRUE / SATURATED | low | distributed intermediate branch |
| 4 | feb56dc7 control | `--ref feb56dc7` (no csv) | — | 6 | 20260619 | — | *(none — per-blob norm)* | TRUE / SATURATED | low | external anchor control |
| 5 | K2 intermediate TRUE | `CORE_SAT_HUNT_20260624_152029` | 10 | 2 | 20260621 | 2050.293702 | `16402.349616` | TRUE / SATURATED | **HIGH** | compact branch / resolution contrast |
| 6 | K1 low-mass TRUE | `CORE_SAT_HUNT_20260624_142152` | 4 | 1 | 20260621 | 1000.0 | `8000.0` | TRUE / SATURATED | **HIGH** | fragile compact pocket / resolution contrast |
| 7 | K1 high-mass failure | `CORE_SAT_HUNT_20260624_132302` | 0 | 1 | 20260620 | 2050.293702 | `16402.349616` | BLOWUP / DELOCALIZED_GROWTH | **HIGH** | failure-boundary control |
| 8 | K6 near-threshold NEAR | `CORE_SAT_HUNT_20260624_102149` | 33 | 6 | 20260619 | 1000.0 | `8000.0` | NEAR / INCONCLUSIVE | low | near-threshold probe |

Priority rationale: **distributed candidates first** (1–3) because they are the low-risk branch family
most likely to be real and most worth confirming; the feb anchor (4) calibrates the N96 classifier
against a known good bound state; the compact resolution-risk contrasts (5–6) are where N48 "TRUE" is
most likely to *change* under resolution and are the scientifically decisive cases; the two controls (7
failure, 8 near-threshold) bound the classifier from below.

---

## 3. Stage 1 — one seed / exact shortlisted row (all 8)

Run each candidate once at its **saved row seed** (already in the CSV row; shown via `--ic-seed-override`
for explicitness and reproducibility). Output root suggestion:
`sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<timestamp>/`.

WSL invocation wrapper (per `docs/VISUAL_ANALYSIS_HANDOVER.md`):

```bash
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && <python cmd>'
```

`<python cmd>` for each candidate (paths are WSL `/mnt/f/...`):

```bash
# 1. K6 high-mass TRUE  (distributed, low-risk)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_112918/all_evals.csv --idx 32 \
  --N 96 --T 6000 --ic-seed-override 20260619 \
  --target-initial-mass-override 16402.349616 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k6_high_mass_true

# 2. K6 mid-mass TRUE  (distributed, low-risk)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_142152/all_evals.csv --idx 34 \
  --N 96 --T 6000 --ic-seed-override 20260621 \
  --target-initial-mass-override 8000.0 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k6_mid_mass_true

# 3. K4 intermediate TRUE  (distributed, low-risk)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_124444/all_evals.csv --idx 25 \
  --N 96 --T 6000 --ic-seed-override 20260620 \
  --target-initial-mass-override 9600.0 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k4_intermediate_true

# 4. feb56dc7 control  (per-blob norm — NO mass override)
python jax_scout/core_saturation_replay.py \
  --ref feb56dc7 \
  --N 96 --T 6000 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/feb56dc7_control

# 5. K2 intermediate TRUE  (compact, HIGH resolution-risk)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_152029/all_evals.csv --idx 10 \
  --N 96 --T 6000 --ic-seed-override 20260621 \
  --target-initial-mass-override 16402.349616 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k2_intermediate_true

# 6. K1 low-mass TRUE  (compact/fragile, HIGH resolution-risk)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_142152/all_evals.csv --idx 4 \
  --N 96 --T 6000 --ic-seed-override 20260621 \
  --target-initial-mass-override 8000.0 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k1_low_mass_true

# 7. K1 high-mass failure control  (expect BLOWUP / delocalized)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_132302/all_evals.csv --idx 0 \
  --N 96 --T 6000 --ic-seed-override 20260620 \
  --target-initial-mass-override 16402.349616 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k1_high_mass_failure

# 8. K6 near-threshold NEAR  (probe)
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_102149/all_evals.csv --idx 33 \
  --N 96 --T 6000 --ic-seed-override 20260619 \
  --target-initial-mass-override 8000.0 \
  --trace-snaps 40 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k6_near_threshold_near
```

**Stage 1 acceptance check (per run, before trusting any verdict):** open `<out>/summary.json` and confirm
- `replay_resolution_N == 96`, `replay_kind == "RESOLUTION_SCALED_TARGET_REPLAY"` (feb: `EXACT_ROW_REPLAY`),
- `mass_scaling_mode == "resolution_scaled_raw_target"` (feb: `exact_saved_raw_target`),
- `replay_params` matches `saved_params` (no accidental param override),
- `ic_stats.initial_mass` is finite and near the intended target.

Then read `klass` + `metrics` (`n_fin`, `late_slope`, `er_fin`).

---

## 4. Cost / capture notes (read before launching)

- **The verdict does not depend on `--trace-snaps`.** `klass`/`metrics` come from the single
  `run_probe` pass (`psi_mid`, `psi_fin`, `energy`). `--trace-snaps` triggers a *second* integration
  (`capture_trajectory`) only to write the optional `frames.npz` + `diagnostic_summary.json` for the v2
  renderer — so it roughly **doubles** wallclock for that run.
- **Storage:** a 96³ complex64 frame is ≈7 MB; `--trace-snaps 40` ⇒ 41 frames ≈ **0.29 GB per case**
  (~2.3 GB for all 8). If storage- or time-constrained, two cheaper options:
  - drop to `--trace-snaps 16` (≈0.12 GB/case) — coarser timeline, same verdict; or
  - run Stage 1 **verdict-only** (omit `--trace-snaps`) for the controls (#7, #8) and keep full
    `--trace-snaps 40` only for the candidates we most want morphology on (#1–#6).
- **Wallclock is unknown a priori** (N96/T6000 = 1.2M steps, the expensive axis). Time candidate #1
  first and extrapolate before committing the full matrix; if a single run is too slow, run the matrix
  sequentially in the background rather than over-parallelizing GPU memory (96³ state is ~8× the N48
  footprint).
- These are replay outputs (sweep bundles) — **do not commit them**; they are inspected in place and
  rendered on the Windows side.

---

## 5. Stage 2 — seed expansion (survivors / ambiguous only, after approval)

Only after Stage 1 verdicts are in. Expand **2 additional seeds** for a candidate if it is either (a) a
TRUE/NEAR distributed candidate we want to confirm is seed-robust, or (b) scientifically ambiguous (e.g.
a compact case whose N48 TRUE flipped, or a near-threshold case). Keep within the discovery's seed family
`{20260619, 20260620, 20260621}` — i.e. run the two seeds the shortlisted row did **not** already use, so
Stage 2 introduces no new RNG territory.

Mechanism: identical command, only `--ic-seed-override` changes and the `--out` leaf gains a `_seedNNNN`
suffix. Example for candidate #1 (shortlisted seed was 20260619 → add 20260620, 20260621):

```bash
python jax_scout/core_saturation_replay.py \
  --csv sweep_runs/CORE_SAT_HUNT_20260624_112918/all_evals.csv --idx 32 \
  --N 96 --T 6000 --ic-seed-override 20260620 \
  --target-initial-mass-override 16402.349616 --trace-snaps 16 \
  --out sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts>/k6_high_mass_true_seed20260620
# ...and again with --ic-seed-override 20260621 --out .../k6_high_mass_true_seed20260621
```

Do **not** run Stage 2 for cases that already failed cleanly at Stage 1 (e.g. the K1 high-mass failure
control if it reproduces BLOWUP) — one seed is enough to confirm a control.

---

## 6. Rendering the N96 results (Windows side, after the run)

Each replay with `--trace-snaps` writes `frames.npz` (`psi` complex64 `[n_snap+1,96,96,96]` + `times`) and
`diagnostic_summary.json` in the same `diagnostic_summary` schema the v2 renderer already consumes. So the
existing Option B case-dynamics path renders them unchanged — point `--cases-root` at the validation root:

```
python -m quantule_viz phase-c-option-b docs/phase_c_structured_discovery_B_summary.csv \
  --shortlist runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json \
  --cases-root sweep_runs/PHASE_C_OPTION_B_VALIDATE_N96_<ts> \
  --outdir sweep_runs/PHASE_C_OPTION_B_VALIDATE_VIS_N96_<ts>
```

(case-key subdir names under `--cases-root` must match the `--out` leaf names used above:
`k6_high_mass_true`, `k6_mid_mass_true`, `k4_intermediate_true`, `feb56dc7_control`,
`k2_intermediate_true`, `k1_low_mass_true`, `k1_high_mass_failure`, `k6_near_threshold_near`).

---

## 7. Decision gate (what Stage 1 answers)

- **Distributed branches survive (1–3 still TRUE/NEAR, feb anchor TRUE):** the distributed branch family
  is resolution-robust → proceed to Stage 2 seed expansion, then decide on a longer structured search.
- **Compact contrasts flip (5–6 N48-TRUE → not-TRUE at N96):** confirms the resolution-risk read; the
  compact "branch" is an N48 artifact, not a separate physical branch. Record as a negative result; do
  not promote.
- **Failure / near-threshold controls (7–8):** #7 should reproduce delocalized BLOWUP (classifier sanity
  from below); #8 calibrates where NEAR sits at N96.

Only after this gate is there a basis to decide whether a longer / broader structured search is justified.
**Nothing in this plan launches until approved.**
