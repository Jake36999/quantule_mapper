# Codex Handoff — Figure Digitization for EMP-V5-NG (Nguyen matter-wave soliton collisions)

**Bounded, mechanical task.** Extract phase-dependent collision-outcome data from published figures into the V5
schema and record provenance. **Do not** reinterpret IRER, change any verdict, alter framing, run simulations, or
wire anything into the validation harness or Hunter. This is data prep for a **qualitative** `CLOSE ANALOGUE`
overlay only.

## Read first

- `README.md` (charter), `EMP_V5_NG_DATA_SCHEMA.md` (the exact CSV contract), `DIGITIZATION_PROVENANCE_TEMPLATE.md`.

## Target

EMP-V5-NG — Nguyen, Dyke, Luo, Malomed, Hulet, "Collisions of matter-wave solitons," *Nature Physics* 10, 918
(2014); arXiv:1407.5087. Supports internal metric **V5** (C3 collision phase×speed grid).

## Manual acquisition (do not scrape blocked pages, do not bypass paywalls)

Try in order, stop when you have usable data:
1. arXiv:1407.5087 — PDF + source package + any ancillary/supplementary files.
2. Nature Physics page (via your legitimate/library access) — supplementary PDF, movies, data-availability
   statement, figure source data, extended data.
3. Rice / Hulet group pages (atomcool.rice.edu) — source trail / repository.
4. Last resort: digitize the visible figure(s) that encode phase-dependent collision outcome, recording that it
   was a rendered/figure extraction with its uncertainty (as was done for EMP-V1-MM).

## What to extract

The **phase dependence of collision outcome** — the physical result is that in-phase collisions produce a density
spike and tend to **collapse/merge**, while out-of-phase (≈π) collisions **survive / pass through**. Extract, per
figure point, into `EMP-V5-NG_<figure>.csv` matching `EMP_V5_NG_DATA_SCHEMA.md` **exactly**:
`source_figure, relative_phase_rad, phase_class(optional), speed_value(optional), speed_units(optional),
observed_outcome, include_in_comparison, exclusion_reason(if excluded), extraction_uncertainty, digitizer_notes`.

`observed_outcome` must be one of: `PASS_THROUGH, TRAJECTORY_JUMP, NO_COLLAPSE, BOUNCE, CAPTURE, COLLAPSE, MERGE,
INCONCLUSIVE, UNKNOWN`. Do **not** invent values you cannot see; use `UNKNOWN`/`INCONCLUSIVE` and set
`include_in_comparison=false` with a reason when a point is off-regime or unreadable.

## Storage

- Off-git raw at `E:\quantule_mapper_external_data\empirical_comparison\EMP-V5-NG\` with `raw/ rendered/ digitized/
  metadata/` (source_record.md, access_log.md, checksums.sha256, file_inventory.csv, license_or_terms.txt). All of
  this is small (papers/figures = MB), so the E: HDD is fine.
- Commit **only**: the derived CSV, `provenance_EMP-V5-NG.md` (all fields filled, boxes ticked, match-level
  `CLOSE ANALOGUE`), and the comparator output. **No PDFs, figure images, or screenshots in git.**

## Run the comparator

```
python tools/external_validation/empirical_comparison/run_empirical_comparison.py \
    --metric v5 \
    --data docs/external_validation/empirical_comparison_branch/EMP-V5-NG_<figure>.csv \
    --provenance docs/external_validation/empirical_comparison_branch/provenance_EMP-V5-NG.md
```

It writes `comparison_result.json` + `comparison_report.md` (`PROVISIONAL_UNTIL_CLAUDE_REVIEW`). If it refuses,
provenance is incomplete — fix it, don't bypass it.

## Report back

Figures/tables used; what was extractable vs not; whether data are qualitative / semi-quantitative / quantitative;
uncertainty estimate; whether the comparator accepted or refused; guardrail attestation (no reinterpretation, no
verdict/framing change, no sim, no production/Hunter/config change, match-level CLOSE ANALOGUE, output provisional).

## What Claude does after

Reviews whether the experiment concentrates transmission/survival at anti-phase (the same *direction* as the C3
grid), how the collapse-vs-capture mechanism difference should be caveated, and how much the extraction uncertainty
limits the overlay — before stating anything. No conclusion is drawn before that review.
