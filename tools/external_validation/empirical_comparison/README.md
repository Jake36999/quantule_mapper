# empirical_comparison/ — quarantined Level-3 comparator (tooling)

Code for the **opt-in** empirical comparison branch. Governance charter:
`docs/external_validation/empirical_comparison_branch/README.md`. Schema: that branch's `DATA_SCHEMA.md`.

## Isolation (binding)

- **Not** imported by `run_external_validation.py` and **not** in its `METRIC_RUNNERS`. It has its own entry point
  and must be invoked explicitly.
- Reads **no** config that the Hunter or production validation pipeline reads. Adding such a link requires a
  deliberate, reviewed human change.
- Match-level is fixed at `CLOSE ANALOGUE`; output is always `PROVISIONAL_UNTIL_CLAUDE_REVIEW`; nothing here
  promotes a verdict or gates anything.
- Refuses any dataset whose provenance record is missing or has unchecked governance boxes.

## Usage

```bash
# validate the apparatus (no external data needed):
python tools/external_validation/empirical_comparison/run_empirical_comparison.py --self-test

# run a real comparison once Codex has produced the CSV + completed provenance:
python tools/external_validation/empirical_comparison/run_empirical_comparison.py \
    --metric v1 \
    --data   <EMP-id>_<figure>.csv \
    --provenance provenance_<EMP-id>.md
```

Output: `comparison_result.json` + `comparison_report.md` next to the dataset (or under `--out`).
