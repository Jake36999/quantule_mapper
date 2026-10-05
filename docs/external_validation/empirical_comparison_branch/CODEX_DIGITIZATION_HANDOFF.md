# Codex Handoff — Figure Digitization for EMP-V1-MM

**Bounded, mechanical task.** Extract one published figure into the comparator's schema and record provenance.
**Do not** reinterpret IRER, change any verdict, alter framing, or wire anything into the validation harness or the
Hunter. This is data preparation for a `CLOSE ANALOGUE` context overlay only.

## Scope

Target: **EMP-V1-MM** — Mitschke & Mollenauer, "Experimental observation of interaction forces between solitons in
optical fibers," *Opt. Lett.* 12, 355 (1987). Supports internal metric **V1** (soliton phase-force law).

Only this one target. EMP-V5-NG is **blocked** until the V5 comparator path exists — do not digitize it yet.

## Steps

1. **Obtain the paper** through a legitimate institutional/library route. Store the PDF and any figure images
   **off-git** at `E:\quantule_mapper_external_data\empirical_comparison\EMP-V1-MM\` with a `metadata/` folder
   (source record, license/terms, checksums, download log) mirroring the existing `external_data/` layout. Do not
   commit the paper or figure images.
2. **Identify the extractable figure** — the one showing the soliton-pair interaction: separation (or the
   interaction/oscillation period) vs. propagation distance, and/or its dependence on relative phase (in-phase
   attraction vs. out-of-phase repulsion).
3. **Digitize to the schema** in `DATA_SCHEMA.md` (V1 section). Produce `EMP-V1-MM_<figure>.csv` with columns
   `relative_phase` (rad), `q_half_separation` (dimensionless, normalized to the reported soliton width), and
   `interaction_measure` (signed force proxy: >0 repulsive, <0 attractive; magnitude monotone in force — e.g.
   inverse-squared interaction period, or initial acceleration). Commit only this small CSV to
   `docs/external_validation/empirical_comparison_branch/`.
4. **Complete a provenance record** — copy `DIGITIZATION_PROVENANCE_TEMPLATE.md` to `provenance_EMP-V1-MM.md`, fill
   every field (extraction tool + version, per-point uncertainty, axis-calibration checks, license), tick all
   governance boxes, keep match-level `CLOSE ANALOGUE`.
5. **Run the comparator** (does not touch production):
   ```
   python tools/external_validation/empirical_comparison/run_empirical_comparison.py \
       --metric v1 --data <the CSV> --provenance provenance_EMP-V1-MM.md
   ```
   It writes `comparison_result.json` + `comparison_report.md` (both `PROVISIONAL_UNTIL_CLAUDE_REVIEW`). If the
   comparator refuses, the provenance is incomplete — fix it, don't bypass it.

## Deliverables

- `EMP-V1-MM_<figure>.csv` (small, committed)
- `provenance_EMP-V1-MM.md` (completed, committed)
- comparator output dir (committed)
- raw paper/figures + metadata off-git on E:

## Guardrails (attest in your report)

- No IRER reinterpretation, no verdict/framing change, no new simulations, no production/Hunter/config change.
- Match-level stays `CLOSE ANALOGUE`; output stays provisional.
- Nothing added to `validation_config.yaml`, the `external_data.py` manifest, or any Hunter/production config.
- Extraction uncertainty recorded; the comparison is a faithfulness-context overlay, not evidence of physical
  correspondence.

## What Claude does after

Claude reviews the comparison: whether the experimental sign law matches, whether the experimental `λ` is in the
neighbourhood of our internal C2 `λ ≈ 1.98`, and how much the digitization uncertainty widens the comparison —
then decides how (or whether) to state the overlay. No conclusion is drawn before that review.
