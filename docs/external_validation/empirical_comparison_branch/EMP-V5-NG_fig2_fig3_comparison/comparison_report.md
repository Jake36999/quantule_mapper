# Empirical V5 Comparison (Level-3, CLOSE ANALOGUE) — collision phase×outcome

Status: `PROVISIONAL_UNTIL_CLAUDE_REVIEW`. Qualitative direction overlay only — no verdict,
gate, numeric match, or IRER-framing change.

- Data source: `docs\external_validation\empirical_comparison_branch\EMP-V5-NG_fig2_fig3.csv`
- Provenance: `docs\external_validation\empirical_comparison_branch\provenance_EMP-V5-NG.md`
- Direction verdict: **`DIRECTION_CONSISTENT`** (consistent with C3 = `True`)
- Internal C3 reference: transmission only at anti-phase = `True` (generated_metrics/v5)
- Experimental transmission (survived-as-two) fraction by phase class:
    - in_phase: `0.0`  (0/2 decisive)
    - anti_phase: `1.0`  (2/2 decisive)
- Ambiguous rows dropped: `0`; decisive rows used: `4`
- Coverage scope: **`coarse_inphase_vs_antiphase_only`** (off-phase: `False`, speed axis: `False`, distinct phase points: `2`)
- Untested C3 features (NOT corroborated by this dataset):
    - anti-phase channel NARROWNESS (no off-phase points; C3 finds off-phase captures)
    - speed dependence (no speed axis; C3 finds anti-phase transmits only below ~0.5c)

Caveats:
- Qualitative phase-direction comparison only; no numeric, rate, or speed-boundary match.
- IN-PHASE side is phenomenological only: COLLAPSE (density-spike destruction) is grouped as not-transmitted but its mechanism differs from C3 CAPTURE (binding). The ANTI-PHASE side is the stronger correspondence (destructive-interference node protects the solitons in both).
- Corroborates coarse direction only; does NOT test: anti-phase channel NARROWNESS (no off-phase points; C3 finds off-phase captures); speed dependence (no speed axis; C3 finds anti-phase transmits only below ~0.5c).

The test asks only whether transmission is concentrated at anti-phase (same *direction* as the
C3 phase×speed grid), not whether speeds, boundaries, or rates match. Interpretation deferred to
Claude review; match-level capped at CLOSE ANALOGUE.
