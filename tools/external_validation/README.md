# External Validation Metric Harnesses

Diagnostic-only scripts for computing first-pass external-validation metrics from existing Quantule Mapper run artifacts.

These tools:

- read existing `sweep_runs/` and evidence-package artifacts;
- write small CSV/JSON/Markdown/PNG summaries under `docs/external_validation/generated_metrics/`;
- skip missing data with explicit reasons;
- do not invoke simulations;
- do not change solver physics, Hunter, validation gates, configs, or project verdicts;
- keep all results `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.

Example:

```powershell
F:\quantule_mapper\.venv\Scripts\python.exe tools\external_validation\run_external_validation.py --metric all --mode internal_analytic --no-plots
```

Modes:

- `internal_analytic`: compare existing Quantule Mapper run data against analytic behavioural laws.
- `external_data`: use only machine-readable or digitised external datasets if available; otherwise skip with reason.
- `audit_depth`: classify generated results as formalism-only, behavioural, or external-data correlation.

External models are comparison instruments, not replacements for IRER formulations.

